import logging
import streamlit as st
from math import gcd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import numpy as np

logger = logging.getLogger(__name__)
DEFAULT_IV = 0.30
MIN_GAP = 0.01

def get_strategy_price(option: dict, action: str) -> float | None:
    price = option["px_ask"] if action == "buy" else option["px_bid"]
    return price if price > 0 else None

def calculate_fees(base_cost: float, commission_rate: float) -> float:
    commission = base_cost * commission_rate
    market_fees = base_cost * 0.002
    vat = (commission + market_fees) * 0.21
    return commission + market_fees + vat

def calculate_strategy_cost(options, actions, contracts, commission_rate):
    """
    Generalized cost calculation for any strategy. Returns None if invalid.
    """
    if len(options) != len(actions) or len(options) != len(contracts):
        logger.warning(f"Mismatch in options ({len(options)}), actions ({len(actions)}), contracts ({len(contracts)})")
        return None
    total_cost = 0.0
    raw_net = 0.0
    for opt, action, num in zip(options, actions, contracts):
        px_bid = opt.get("px_bid", 0)
        px_ask = opt.get("px_ask", 0)
        if px_bid <= 0 or px_ask <= 0 or px_bid > px_ask:
            logger.debug(f"Invalid bid/ask for {opt['symbol']}: bid={px_bid}, ask={px_ask}")
            return None
        mid_price = (px_bid + px_ask) / 2  # Use mid-price for realism
        spread_pct = abs(px_ask - px_bid) / mid_price
        if spread_pct > 0.2:
            logger.debug(f"Wide spread ({spread_pct:.1%}) for {opt['symbol']}")
            return None
        price = mid_price
        base_cost = price * 100 * num * (1 if action == "buy" else -1)
        if abs(base_cost) < 1e-6:
            logger.debug(f"Near-zero base cost for {opt['symbol']}: base_cost={base_cost}")
            return None
        fees = calculate_fees(abs(base_cost), commission_rate)
        raw_net += base_cost
        total_cost += base_cost + fees
    if abs(total_cost) < 1e-6:
        logger.debug(f"Total cost too small: {total_cost}")
        return None
    return {"raw_net": raw_net, "net_cost": total_cost}

def black_scholes(S, K, T, r, sigma, option_type):
    """
    Black-Scholes option pricing model.
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free rate
    sigma: Volatility
    option_type: 'call' or 'put'
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return intrinsic_value(S, K, option_type)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(price, intrinsic_value(S, K, option_type))
    except Exception as e:
        logger.warning(f"Black-Scholes error: {e}, using intrinsic value")
        return intrinsic_value(S, K, option_type)

def intrinsic_value(S, K, option_type):
    """
    Calculate intrinsic value of an option.
    """
    if option_type == "call":
        return max(0, S - K)
    return max(0, K - S)

def _calculate_iv(raw_net, current_price, expiration_days, strategy_value_func, options, option_actions, contract_ratios=None):
    """
    Calculate implied volatility for a strategy by minimizing the difference between
    market price (raw_net) and Black-Scholes price.
    """
    if not contract_ratios:
        contract_ratios = [1] * len(options)
    
    def objective_function(sigma):
        try:
            T = max(expiration_days / 365.0, 1e-9)
            bs_price = strategy_value_func(current_price, T, sigma)
            return (bs_price - raw_net) ** 2
        except Exception as e:
            logger.warning(f"IV calculation error for sigma={sigma}: {e}")
            return float('inf')

    try:
        result = minimize_scalar(
            objective_function,
            bounds=(0.01, 2.0),  # Reasonable IV range
            method='bounded',
            options={'xatol': 1e-6, 'maxiter': 100}
        )
        if result.success and result.x > 0:
            logger.debug(f"IV calculated: {result.x:.4f}")
            return result.x
        else:
            st.session_state["iv_failure_count"] = st.session_state.get("iv_failure_count", 0) + 1
            logger.warning(f"IV calibration failed, using default IV: {DEFAULT_IV}")
            return DEFAULT_IV
    except Exception as e:
        st.session_state["iv_failure_count"] = st.session_state.get("iv_failure_count", 0) + 1
        logger.warning(f"IV calibration error: {e}, using default IV: {DEFAULT_IV}")
        return DEFAULT_IV

def calculate_bull_call_spread(long_call, short_call, num_contracts, commission_rate):
    if long_call["strike"] >= short_call["strike"]: return None
    long_price = get_strategy_price(long_call, "buy")
    short_price = get_strategy_price(short_call, "sell")
    if any(p is None for p in [long_price, short_price]): return None

    base_cost = (long_price - short_price) * 100 * num_contracts
    total_fees = calculate_fees(long_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = net_cost
    max_profit = (short_call["strike"] - long_call["strike"] - (net_cost / (100 * num_contracts))) * 100 * num_contracts
    if max_profit <= 0: return None
    lower_breakeven = long_call["strike"] + (net_cost / (100 * num_contracts))
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": lower_breakeven,
        "strikes": [long_call["strike"], short_call["strike"]],
        "num_contracts": num_contracts
    }

def calculate_bull_put_spread(short_put, long_put, num_contracts, commission_rate):
    if short_put["strike"] <= long_put["strike"]: return None
    short_price = get_strategy_price(short_put, "sell")
    long_price = get_strategy_price(long_put, "buy")
    if any(p is None for p in [short_price, long_price]): return None

    base_cost = (long_price - short_price) * 100 * num_contracts
    total_fees = calculate_fees(long_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = ((short_put["strike"] - long_put["strike"]) * 100 * num_contracts) - net_cost
    max_profit = -net_cost
    if max_profit <= 0: return None
    lower_breakeven = long_put["strike"] + (net_cost / (100 * num_contracts))
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": lower_breakeven,
        "strikes": [long_put["strike"], short_put["strike"]],
        "num_contracts": num_contracts
    }

def calculate_bear_call_spread(short_call, long_call, num_contracts, commission_rate):
    if short_call["strike"] >= long_call["strike"]: 
        logger.debug(f"Invalid strike order: short={short_call['strike']}, long={long_call['strike']}")
        return None
    short_price = get_strategy_price(short_call, "sell")
    long_price = get_strategy_price(long_call, "buy")
    if any(p is None for p in [short_price, long_price]): 
        logger.debug(f"Invalid prices: short_price={short_price}, long_price={long_price}")
        return None

    base_cost = (long_price - short_price) * 100 * num_contracts
    total_fees = calculate_fees(long_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = ((long_call["strike"] - short_call["strike"]) * 100 * num_contracts) - net_cost
    max_profit = -net_cost
    upper_breakeven = short_call["strike"] + (-net_cost / (100 * num_contracts))
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": upper_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [short_call["strike"], long_call["strike"]],
        "num_contracts": num_contracts
    }

def calculate_bear_put_spread(long_put, short_put, num_contracts, commission_rate):
    if long_put["strike"] <= short_put["strike"]: return None
    long_price = get_strategy_price(long_put, "buy")
    short_price = get_strategy_price(short_put, "sell")
    if any(p is None for p in [long_price, short_price]): return None

    base_cost = (long_price - short_price) * 100 * num_contracts
    total_fees = calculate_fees(long_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = net_cost
    max_profit = (long_put["strike"] - short_put["strike"] - (net_cost / (100 * num_contracts))) * 100 * num_contracts
    if max_profit <= 0: return None
    upper_breakeven = long_put["strike"] - (net_cost / (100 * num_contracts))
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": upper_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [long_put["strike"], short_put["strike"]],
        "num_contracts": num_contracts
    }

def calculate_call_butterfly(long_low, short_mid, long_high, num_contracts, commission_rate):
    if not (long_low["strike"] < short_mid["strike"] < long_high["strike"]): return None
    long_low_price = get_strategy_price(long_low, "buy")
    short_mid_price = get_strategy_price(short_mid, "sell")
    long_high_price = get_strategy_price(long_high, "buy")
    if any(p is None for p in [long_low_price, short_mid_price, long_high_price]): return None

    base_cost = (long_low_price + long_high_price - 2 * short_mid_price) * 100 * num_contracts
    total_fees = calculate_fees(long_low_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_mid_price * 100 * num_contracts * 2, commission_rate) + \
                 calculate_fees(long_high_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = net_cost
    max_profit = (short_mid["strike"] - long_low["strike"] - (net_cost / (100 * num_contracts))) * 100 * num_contracts
    if max_profit <= 0: return None
    lower_breakeven = long_low["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = long_high["strike"] - (net_cost / (100 * num_contracts))
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [long_low["strike"], short_mid["strike"], long_high["strike"]],
        "num_contracts": num_contracts
    }

def calculate_put_butterfly(long_high, short_mid, long_low, num_contracts, commission_rate):
    if not (long_low["strike"] < short_mid["strike"] < long_high["strike"]): return None
    long_high_price = get_strategy_price(long_high, "buy")
    short_mid_price = get_strategy_price(short_mid, "sell")
    long_low_price = get_strategy_price(long_low, "buy")
    if any(p is None for p in [long_high_price, short_mid_price, long_low_price]): return None

    base_cost = (long_high_price + long_low_price - 2 * short_mid_price) * 100 * num_contracts
    total_fees = calculate_fees(long_high_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_mid_price * 100 * num_contracts * 2, commission_rate) + \
                 calculate_fees(long_low_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = net_cost
    max_profit = (long_high["strike"] - short_mid["strike"] - (net_cost / (100 * num_contracts))) * 100 * num_contracts
    if max_profit <= 0: return None
    lower_breakeven = long_low["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = long_high["strike"] - (net_cost / (100 * num_contracts))
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [long_low["strike"], short_mid["strike"], long_high["strike"]],
        "num_contracts": num_contracts
    }

def calculate_call_condor(long_low, short_mid_low, short_mid_high, long_high, num_contracts, commission_rate):
    if not (long_low["strike"] < short_mid_low["strike"] < short_mid_high["strike"] < long_high["strike"]): return None
    long_low_price = get_strategy_price(long_low, "buy")
    short_mid_low_price = get_strategy_price(short_mid_low, "sell")
    short_mid_high_price = get_strategy_price(short_mid_high, "sell")
    long_high_price = get_strategy_price(long_high, "buy")
    if any(p is None for p in [long_low_price, short_mid_low_price, short_mid_high_price, long_high_price]): return None

    base_cost = (long_low_price + long_high_price - short_mid_low_price - short_mid_high_price) * 100 * num_contracts
    total_fees = calculate_fees(long_low_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_mid_low_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_mid_high_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(long_high_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = net_cost
    max_profit = (short_mid_high["strike"] - short_mid_low["strike"] - (net_cost / (100 * num_contracts))) * 100 * num_contracts
    if max_profit <= 0: return None
    lower_breakeven = long_low["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = long_high["strike"] - (net_cost / (100 * num_contracts))
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [long_low["strike"], short_mid_low["strike"], short_mid_high["strike"], long_high["strike"]],
        "num_contracts": num_contracts
    }

def calculate_put_condor(long_high, short_mid_high, short_mid_low, long_low, num_contracts, commission_rate):
    if not (long_low["strike"] < short_mid_low["strike"] < short_mid_high["strike"] < long_high["strike"]): return None
    long_high_price = get_strategy_price(long_high, "buy")
    short_mid_high_price = get_strategy_price(short_mid_high, "sell")
    short_mid_low_price = get_strategy_price(short_mid_low, "sell")
    long_low_price = get_strategy_price(long_low, "buy")
    if any(p is None for p in [long_high_price, short_mid_high_price, short_mid_low_price, long_low_price]): return None

    base_cost = (long_high_price + long_low_price - short_mid_high_price - short_mid_low_price) * 100 * num_contracts
    total_fees = calculate_fees(long_high_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_mid_high_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(short_mid_low_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(long_low_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = net_cost
    max_profit = (short_mid_high["strike"] - short_mid_low["strike"] - (net_cost / (100 * num_contracts))) * 100 * num_contracts
    if max_profit <= 0: return None
    lower_breakeven = long_low["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = long_high["strike"] - (net_cost / (100 * num_contracts))
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [long_low["strike"], short_mid_low["strike"], short_mid_high["strike"], long_high["strike"]],
        "num_contracts": num_contracts
    }

def calculate_straddle(call_opt, put_opt, num_contracts, commission_rate):
    if call_opt["strike"] != put_opt["strike"]: return None
    call_price = get_strategy_price(call_opt, "buy")
    put_price = get_strategy_price(put_opt, "buy")
    if any(p is None for p in [call_price, put_price]): return None

    base_cost = (call_price + put_price) * 100 * num_contracts
    total_fees = calculate_fees(call_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(put_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = net_cost
    strike = call_opt["strike"]
    lower_breakeven = strike - (net_cost / (100 * num_contracts))
    upper_breakeven = strike + (net_cost / (100 * num_contracts))
    
    current_price = st.session_state.get("current_price", 5290.0)
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    max_price = current_price * (1 + plot_range_pct)
    min_price = max(0.1, current_price * (1 - plot_range_pct))
    call_profit = max(max_price - strike, 0) - call_price
    put_profit = max(strike - min_price, 0) - put_price
    max_profit = max(call_profit, put_profit) * 100 * num_contracts * (1 - commission_rate)
    if max_profit <= 0: return None
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [strike],
        "num_contracts": num_contracts
    }

def calculate_strangle(put_opt, call_opt, num_contracts, commission_rate):
    if put_opt["strike"] >= call_opt["strike"]: return None
    put_price = get_strategy_price(put_opt, "buy")
    call_price = get_strategy_price(call_opt, "buy")
    if any(p is None for p in [put_price, call_price]): return None

    base_cost = (put_price + call_price) * 100 * num_contracts
    total_fees = calculate_fees(put_price * 100 * num_contracts, commission_rate) + \
                 calculate_fees(call_price * 100 * num_contracts, commission_rate)
    net_cost = base_cost + total_fees
    max_loss = net_cost
    lower_breakeven = put_opt["strike"] - (net_cost / (100 * num_contracts))
    upper_breakeven = call_opt["strike"] + (net_cost / (100 * num_contracts))
    
    current_price = st.session_state.get("current_price", 5290.0)
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    max_price = current_price * (1 + plot_range_pct)
    min_price = max(0.1, current_price * (1 - plot_range_pct))
    call_profit = max(max_price - call_opt["strike"], 0) - call_price
    put_profit = max(put_opt["strike"] - min_price, 0) - put_price
    max_profit = max(call_profit, put_profit) * 100 * num_contracts * (1 - commission_rate)
    if max_profit <= 0: return None
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [put_opt["strike"], call_opt["strike"]],
        "num_contracts": num_contracts
    }