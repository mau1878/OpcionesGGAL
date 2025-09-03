import logging
import streamlit as st
from math import gcd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from itertools import combinations, product

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

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if S <= 0:
        return 0 if option_type == "call" else max(K - S, 0)
    if K <= 0:
        return max(S - K, 0) if option_type == "call" else 0
    if T <= 1e-6:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    if sigma <= 1e-9:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        epsilon = 1e-12
        denom = sigma * np.sqrt(T) + epsilon
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / denom
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            val = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            val = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return np.nan_to_num(val)

def intrinsic_value(S, K, option_type="call"):
    if option_type == "call":
        return max(0, S - K)
    else:
        return max(0, K - S)

def calculate_option_iv(S, K, T, r, premium, option_type="call"):
    if premium is None or premium <= 0:
        logger.warning(f"Invalid premium {premium} for S={S}, K={K}, T={T}, option_type={option_type}")
        return None
    def bs_price(sigma):
        try:
            val = black_scholes(S, K, T, r, sigma, option_type)
            return val - premium
        except Exception as e:
            logger.warning(f"Black-Scholes error for sigma={sigma}: {e}")
            return float('inf')
    
    low, high = 0.01, 5.0
    epsilon = 1e-6
    max_iter = 200
    iter_count = 0

    while iter_count < max_iter:
        mid = (low + high) / 2
        price_diff = bs_price(mid)
        if abs(price_diff) < epsilon:
            return mid
        if price_diff > 0:
            high = mid
        else:
            low = mid
        iter_count += 1

    logger.warning(f"IV calculation did not converge for S={S}, K={K}, T={T}, premium={premium}, option_type={option_type}")
    return None

def _calibrate_iv(raw_net, current_price, expiration_days, strategy_value_func, options, option_actions, contract_ratios=None):
    if not options or not option_actions:
        logger.warning("No options or actions provided for IV calibration")
        return DEFAULT_IV

    r = st.session_state.get("risk_free_rate", 0.50)
    T = max(expiration_days / 365.0, 1e-9)
    ivs = []
    weights = contract_ratios if contract_ratios else [1.0] * len(options)

    for opt, action, weight in zip(options, option_actions, weights):
        premium = get_strategy_price(opt, action)
        if premium is None or premium <= 0:
            logger.warning(f"Invalid premium {premium} for option {opt.get('symbol', 'unknown')}, strike={opt['strike']}")
            continue
        iv = calculate_option_iv(
            S=current_price,
            K=opt["strike"],
            T=T,
            r=r,
            premium=premium,
            option_type=opt["type"]
        )
        if iv and not np.isnan(iv) and iv > 0:
            ivs.append(iv * weight)
        else:
            logger.warning(f"Failed to calculate IV for option {opt.get('symbol', 'unknown')}, strike={opt['strike']}")

    if ivs:
        total_weight = sum(weights[:len(ivs)])
        calibrated_iv = sum(ivs) / total_weight if total_weight > 0 else DEFAULT_IV
        try:
            model_value = strategy_value_func(current_price, T, calibrated_iv)
            if abs(model_value - raw_net) < raw_net * 0.2:
                return calibrated_iv
            else:
                logger.warning(f"IV verification failed: model_value={model_value}, raw_net={raw_net}, tolerance={raw_net * 0.2}")
                return calibrated_iv
        except Exception as e:
            logger.warning(f"Strategy value error during IV verification: {e}")
            return calibrated_iv
    logger.warning("No valid IVs calculated. Using default IV.")
    st.session_state["iv_failure_count"] = st.session_state.get("iv_failure_count", 0) + 1
    return DEFAULT_IV

def calculate_bull_call_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] >= short_opt["strike"]: return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if any(p is None for p in [long_price, short_price]): return None

    long_base = long_price * 100 * num_contracts
    short_base = short_price * 100 * num_contracts
    base_cost = (long_price - short_price) * 100 * num_contracts
    total_fees = calculate_fees(long_base, commission_rate) + calculate_fees(short_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    max_profit = (short_opt["strike"] - long_opt["strike"]) * 100 * num_contracts - net_cost
    if max_profit <= 0: return None  # Skip unprofitable
    breakeven = long_opt["strike"] + (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "strikes": [long_opt["strike"], short_opt["strike"]],
        "num_contracts": num_contracts,
        "breakeven": breakeven
    }

def calculate_bull_put_spread(short_opt, long_opt, num_contracts, commission_rate):
    if short_opt["strike"] <= long_opt["strike"]: return None
    short_price = get_strategy_price(short_opt, "sell")
    long_price = get_strategy_price(long_opt, "buy")
    if any(p is None for p in [short_price, long_price]): return None

    short_base = short_price * 100 * num_contracts
    long_base = long_price * 100 * num_contracts
    base_credit = (short_price - long_price) * 100 * num_contracts
    if base_credit <= 0: return None
    total_fees = calculate_fees(short_base, commission_rate) + calculate_fees(long_base, commission_rate)
    net_credit = base_credit - total_fees

    max_profit = net_credit
    max_loss = (short_opt["strike"] - long_opt["strike"]) * 100 * num_contracts - net_credit
    if max_profit <= 0: return None  # Skip unprofitable
    breakeven = short_opt["strike"] - (net_credit / (100 * num_contracts))

    return {
        "raw_net": -base_credit,
        "net_cost": -net_credit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "strikes": [long_opt["strike"], short_opt["strike"]],
        "num_contracts": num_contracts,
        "breakeven": breakeven
    }

def calculate_bear_call_spread(short_opt, long_opt, num_contracts, commission_rate):
    if short_opt["strike"] >= long_opt["strike"]: return None
    short_price = get_strategy_price(short_opt, "sell")
    long_price = get_strategy_price(long_opt, "buy")
    if any(p is None for p in [short_price, long_price]): return None

    short_base = short_price * 100 * num_contracts
    long_base = long_price * 100 * num_contracts
    base_credit = (short_price - long_price) * 100 * num_contracts
    if base_credit <= 0: return None
    total_fees = calculate_fees(short_base, commission_rate) + calculate_fees(long_base, commission_rate)
    net_credit = base_credit - total_fees

    max_profit = net_credit
    max_loss = (long_opt["strike"] - short_opt["strike"]) * 100 * num_contracts - net_credit
    if max_profit <= 0: return None  # Skip unprofitable
    breakeven = short_opt["strike"] + (net_credit / (100 * num_contracts))

    return {
        "raw_net": -base_credit,
        "net_cost": -net_credit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "strikes": [short_opt["strike"], long_opt["strike"]],
        "num_contracts": num_contracts,
        "breakeven": breakeven
    }

def calculate_bear_put_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] <= short_opt["strike"]: return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if any(p is None for p in [long_price, short_price]): return None

    long_base = long_price * 100 * num_contracts
    short_base = short_price * 100 * num_contracts
    base_cost = (long_price - short_price) * 100 * num_contracts
    total_fees = calculate_fees(long_base, commission_rate) + calculate_fees(short_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    max_profit = (long_opt["strike"] - short_opt["strike"]) * 100 * num_contracts - net_cost
    if max_profit <= 0: return None  # Skip unprofitable
    breakeven = long_opt["strike"] - (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "strikes": [short_opt["strike"], long_opt["strike"]],
        "num_contracts": num_contracts,
        "breakeven": breakeven
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

    width_lower = short_mid["strike"] - long_low["strike"]
    width_upper = long_high["strike"] - short_mid["strike"]
    max_profit = min(width_lower, width_upper) * 100 * num_contracts - net_cost
    if max_profit <= 0: return None  # Skip unprofitable
    max_loss = net_cost
    lower_breakeven = long_low["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = long_high["strike"] - (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
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

    width_lower = short_mid["strike"] - long_low["strike"]
    width_upper = long_high["strike"] - short_mid["strike"]
    max_profit = min(width_lower, width_upper) * 100 * num_contracts - net_cost
    if max_profit <= 0: return None  # Skip unprofitable
    max_loss = net_cost
    lower_breakeven = long_low["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = long_high["strike"] - (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
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

    width_lower = short_mid_low["strike"] - long_low["strike"]
    width_upper = long_high["strike"] - short_mid_high["strike"]
    max_profit = min(width_lower, width_upper) * 100 * num_contracts - net_cost
    if max_profit <= 0: return None  # Skip unprofitable
    max_loss = net_cost
    lower_breakeven = long_low["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = long_high["strike"] - (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
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

    width_lower = short_mid_low["strike"] - long_low["strike"]
    width_upper = long_high["strike"] - short_mid_high["strike"]
    max_profit = min(width_lower, width_upper) * 100 * num_contracts - net_cost
    if max_profit <= 0: return None  # Skip unprofitable
    max_loss = net_cost
    lower_breakeven = long_low["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = long_high["strike"] - (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
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
    
    current_price = st.session_state.get("current_price", 100.0)
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    max_price = current_price * (1 + plot_range_pct)
    min_price = max(0.1, current_price * (1 - plot_range_pct))
    call_profit = max(max_price - strike, 0) - call_price
    put_profit = max(strike - min_price, 0) - put_price
    max_profit = max(call_profit, put_profit) * 100 * num_contracts * (1 - commission_rate)
    if max_profit <= 0: return None  # Skip unprofitable
    
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
    
    current_price = st.session_state.get("current_price", 100.0)
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    max_price = current_price * (1 + plot_range_pct)
    min_price = max(0.1, current_price * (1 - plot_range_pct))
    call_profit = max(max_price - call_opt["strike"], 0) - call_price
    put_profit = max(put_opt["strike"] - min_price, 0) - put_price
    max_profit = max(call_profit, put_profit) * 100 * num_contracts * (1 - commission_rate)
    if max_profit <= 0: return None  # Skip unprofitable
    
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