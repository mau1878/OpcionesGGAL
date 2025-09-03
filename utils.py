import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta, timezone
from itertools import combinations, product
from math import gcd
import plotly.graph_objects as go
import logging
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
STOCK_URL = "https://data912.com/live/arg_stocks"
OPTIONS_URL = "https://data912.com/live/arg_options"
DEFAULT_IV = 0.30
MIN_GAP = 0.01

def get_third_friday(year: int, month: int) -> date:
    try:
        first_day = date(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
        return first_friday + timedelta(days=14)
    except ValueError:
        return None

year = date.today().year
EXPIRATION_MAP_2025 = {
    "E": get_third_friday(year, 1), "EN": get_third_friday(year, 1),
    "F": get_third_friday(year, 2), "FE": get_third_friday(year, 2),
    "M": get_third_friday(year, 3), "MZ": get_third_friday(year, 3),
    "A": get_third_friday(year, 4), "AB": get_third_friday(year, 4),
    "MY": get_third_friday(year, 5),
    "J": get_third_friday(year, 6), "JN": get_third_friday(year, 6),
    "JL": get_third_friday(year, 7),
    "AG": get_third_friday(year, 8),
    "S": get_third_friday(year, 9), "SE": get_third_friday(year, 9),
    "O": get_third_friday(year, 10), "OC": get_third_friday(year, 10),
    "N": get_third_friday(year, 11), "NO": get_third_friday(year, 11),
    "D": get_third_friday(year, 12), "DI": get_third_friday(year, 12),
}

def fetch_data(url: str) -> list:
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching data from {url}: {e}")
        st.error(f"Error fetching data from {url}: {e}")
        return []
    finally:
        session.close()

def parse_option_symbol(symbol: str) -> tuple[str | None, float | None, date | None]:
    option_type = "call" if symbol.startswith("GFGC") else "put" if symbol.startswith("GFGV") else None
    if not option_type:
        logger.warning(f"Invalid option symbol format: {symbol}")
        return None, None, None
    data_part = symbol[4:]
    first_letter_index = next((i for i, char in enumerate(data_part) if char.isalpha()), -1)
    if first_letter_index == -1:
        logger.warning(f"No expiration suffix in symbol: {symbol}")
        return None, None, None
    numeric_part, suffix = data_part[:first_letter_index], data_part[first_letter_index:]
    if not numeric_part:
        logger.warning(f"No strike price in symbol: {symbol}")
        return None, None, None
    try:
        strike_price = float(numeric_part) / 10.0 if len(numeric_part) >= 5 and not numeric_part.startswith('1') else float(numeric_part)
    except ValueError:
        logger.warning(f"Invalid strike price in symbol: {symbol}")
        return None, None, None
    exp_date = EXPIRATION_MAP_2025.get(suffix)
    if not exp_date:
        logger.warning(f"Unmapped expiration suffix: {suffix} in symbol: {symbol}")
    return option_type, strike_price, exp_date

def get_ggal_data() -> tuple[dict | None, list]:
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)
    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock:
        logger.error("GGAL stock data not found")
        return None, []
    ggal_options = []
    for o in options_data:
        opt_type, strike, exp = parse_option_symbol(o["symbol"])
        if all([opt_type, strike, exp, o.get("px_ask", 0) > 0, o.get("px_bid", 0) > 0]):
            ggal_options.append({
                "symbol": o["symbol"], "type": opt_type, "strike": strike,
                "expiration": exp, "px_bid": o["px_bid"], "px_ask": o["px_ask"]
            })
        else:
            logger.debug(f"Skipped option {o['symbol']}: Invalid data")
    return ggal_stock, ggal_options

def get_strategy_price(option: dict, action: str) -> float | None:
    price = option["px_ask"] if action == "buy" else option["px_bid"]
    return price if price > 0 else None

def calculate_fees(base_cost: float, commission_rate: float) -> float:
    commission = base_cost * commission_rate
    market_fees = base_cost * 0.002
    vat = (commission + market_fees) * 0.21
    return commission + market_fees + vat

# --- Strategy Calculation Functions ---

def calculate_bull_call_spread(long_opt, short_opt, num_contracts, commission_rate):
    """
    Calculate metrics for a Bull Call Spread.
    
    Args:
        long_opt (dict): Long call option
        short_opt (dict): Short call option
        num_contracts (int): Number of contracts
        commission_rate (float): Commission rate as a decimal
    Returns:
        dict: Metrics (net_cost, max_profit, max_loss, breakeven, breakeven_prob, cost_to_profit)
    """
    if long_opt["strike"] >= short_opt["strike"]: return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if any(p is None for p in [long_price, short_price]): return None
    spread_threshold = st.session_state.get('bid_ask_spread_threshold', 0.5)
    if abs(long_opt["px_ask"] - long_opt["px_bid"]) / max(long_opt["px_ask"], long_opt["px_bid"]) > spread_threshold or \
       abs(short_opt["px_ask"] - short_opt["px_bid"]) / max(short_opt["px_ask"], short_opt["px_bid"]) > spread_threshold:
        logger.debug(f"Wide bid-ask spread for options {long_opt['symbol']}/{short_opt['symbol']}")
        return None

    long_base = long_price * 100 * num_contracts
    short_base = short_price * 100 * num_contracts
    base_cost = (long_price - short_price) * 100 * num_contracts
    total_fees = calculate_fees(long_base, commission_rate) + calculate_fees(short_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    max_profit = (short_opt["strike"] - long_opt["strike"]) * 100 * num_contracts - net_cost
    breakeven = long_opt["strike"] + net_cost / (100 * num_contracts)
    T_years = st.session_state.expiration_days / 365.0
    iv = _calibrate_iv(
        net_cost / (100 * num_contracts),
        st.session_state.current_price,
        st.session_state.expiration_days,
        lambda p, t, s: (black_scholes(p, long_opt["strike"], t, st.session_state.risk_free_rate, s, "call") -
                         black_scholes(p, short_opt["strike"], t, st.session_state.risk_free_rate, s, "call")),
        [long_opt, short_opt], ["buy", "sell"]
    )
    if iv == DEFAULT_IV:
        logger.warning(f"IV calibration failed for {long_opt['symbol']}/{short_opt['symbol']}")
        breakeven_prob = 0.0
    else:
        breakeven_prob = estimate_breakeven_probability(
            [long_opt, short_opt], ["buy", "sell"], [num_contracts, num_contracts],
            net_cost, T_years, iv
        )

    return {
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "breakeven_prob": breakeven_prob,
        "cost_to_profit": abs(net_cost / max_profit) if max_profit > 0 else float('inf')
    }

def calculate_bull_put_spread(long_opt, short_opt, num_contracts, commission_rate):
    """
    Calculate metrics for a Bull Put Spread.
    
    Args:
        long_opt (dict): Long put option
        short_opt (dict): Short put option
        num_contracts (int): Number of contracts
        commission_rate (float): Commission rate as a decimal
    Returns:
        dict: Metrics (net_credit, max_profit, max_loss, breakeven, breakeven_prob, cost_to_profit)
    """
    if long_opt["strike"] >= short_opt["strike"]: return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if any(p is None for p in [long_price, short_price]): return None
    spread_threshold = st.session_state.get('bid_ask_spread_threshold', 0.5)
    if abs(long_opt["px_ask"] - long_opt["px_bid"]) / max(long_opt["px_ask"], long_opt["px_bid"]) > spread_threshold or \
       abs(short_opt["px_ask"] - short_opt["px_bid"]) / max(short_opt["px_ask"], short_opt["px_bid"]) > spread_threshold:
        logger.debug(f"Wide bid-ask spread for options {long_opt['symbol']}/{short_opt['symbol']}")
        return None

    long_base = long_price * 100 * num_contracts
    short_base = short_price * 100 * num_contracts
    base_credit = (short_price - long_price) * 100 * num_contracts
    total_fees = calculate_fees(long_base, commission_rate) + calculate_fees(short_base, commission_rate)
    net_credit = base_credit - total_fees

    max_profit = net_credit
    max_loss = (short_opt["strike"] - long_opt["strike"]) * 100 * num_contracts - net_credit
    breakeven = short_opt["strike"] - net_credit / (100 * num_contracts)
    T_years = st.session_state.expiration_days / 365.0
    iv = _calibrate_iv(
        net_credit / (100 * num_contracts),
        st.session_state.current_price,
        st.session_state.expiration_days,
        lambda p, t, s: (black_scholes(p, short_opt["strike"], t, st.session_state.risk_free_rate, s, "put") -
                         black_scholes(p, long_opt["strike"], t, st.session_state.risk_free_rate, s, "put")),
        [long_opt, short_opt], ["buy", "sell"]
    )
    if iv == DEFAULT_IV:
        logger.warning(f"IV calibration failed for {long_opt['symbol']}/{short_opt['symbol']}")
        breakeven_prob = 0.0
    else:
        breakeven_prob = estimate_breakeven_probability(
            [long_opt, short_opt], ["buy", "sell"], [num_contracts, num_contracts],
            -net_credit, T_years, iv  # Negative net_credit for credit spread
        )

    return {
        "net_credit": net_credit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "breakeven_prob": breakeven_prob,
        "cost_to_profit": abs(net_credit / max_profit) if max_profit > 0 else float('inf')
    }

def calculate_bear_call_spread(long_opt, short_opt, num_contracts, commission_rate):
    """
    Calculate metrics for a Bear Call Spread.
    
    Args:
        long_opt (dict): Long call option
        short_opt (dict): Short call option
        num_contracts (int): Number of contracts
        commission_rate (float): Commission rate as a decimal
    Returns:
        dict: Metrics (net_credit, max_profit, max_loss, breakeven, breakeven_prob, cost_to_profit)
    """
    if long_opt["strike"] <= short_opt["strike"]: return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if any(p is None for p in [long_price, short_price]): return None
    spread_threshold = st.session_state.get('bid_ask_spread_threshold', 0.5)
    if abs(long_opt["px_ask"] - long_opt["px_bid"]) / max(long_opt["px_ask"], long_opt["px_bid"]) > spread_threshold or \
       abs(short_opt["px_ask"] - short_opt["px_bid"]) / max(short_opt["px_ask"], short_opt["px_bid"]) > spread_threshold:
        logger.debug(f"Wide bid-ask spread for options {long_opt['symbol']}/{short_opt['symbol']}")
        return None

    long_base = long_price * 100 * num_contracts
    short_base = short_price * 100 * num_contracts
    base_credit = (short_price - long_price) * 100 * num_contracts
    total_fees = calculate_fees(long_base, commission_rate) + calculate_fees(short_base, commission_rate)
    net_credit = base_credit - total_fees

    max_profit = net_credit
    max_loss = (long_opt["strike"] - short_opt["strike"]) * 100 * num_contracts - net_credit
    breakeven = short_opt["strike"] + net_credit / (100 * num_contracts)
    T_years = st.session_state.expiration_days / 365.0
    iv = _calibrate_iv(
        net_credit / (100 * num_contracts),
        st.session_state.current_price,
        st.session_state.expiration_days,
        lambda p, t, s: (black_scholes(p, short_opt["strike"], t, st.session_state.risk_free_rate, s, "call") -
                         black_scholes(p, long_opt["strike"], t, st.session_state.risk_free_rate, s, "call")),
        [long_opt, short_opt], ["buy", "sell"]
    )
    if iv == DEFAULT_IV:
        logger.warning(f"IV calibration failed for {long_opt['symbol']}/{short_opt['symbol']}")
        breakeven_prob = 0.0
    else:
        breakeven_prob = estimate_breakeven_probability(
            [long_opt, short_opt], ["buy", "sell"], [num_contracts, num_contracts],
            -net_credit, T_years, iv
        )

    return {
        "net_credit": net_credit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "breakeven_prob": breakeven_prob,
        "cost_to_profit": abs(net_credit / max_profit) if max_profit > 0 else float('inf')
    }

def calculate_bear_put_spread(long_opt, short_opt, num_contracts, commission_rate):
    """
    Calculate metrics for a Bear Put Spread.
    
    Args:
        long_opt (dict): Long put option
        short_opt (dict): Short put option
        num_contracts (int): Number of contracts
        commission_rate (float): Commission rate as a decimal
    Returns:
        dict: Metrics (net_cost, max_profit, max_loss, breakeven, breakeven_prob, cost_to_profit)
    """
    if long_opt["strike"] <= short_opt["strike"]: return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if any(p is None for p in [long_price, short_price]): return None
    spread_threshold = st.session_state.get('bid_ask_spread_threshold', 0.5)
    if abs(long_opt["px_ask"] - long_opt["px_bid"]) / max(long_opt["px_ask"], long_opt["px_bid"]) > spread_threshold or \
       abs(short_opt["px_ask"] - short_opt["px_bid"]) / max(short_opt["px_ask"], short_opt["px_bid"]) > spread_threshold:
        logger.debug(f"Wide bid-ask spread for options {long_opt['symbol']}/{short_opt['symbol']}")
        return None

    long_base = long_price * 100 * num_contracts
    short_base = short_price * 100 * num_contracts
    base_cost = (long_price - short_price) * 100 * num_contracts
    total_fees = calculate_fees(long_base, commission_rate) + calculate_fees(short_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    max_profit = (long_opt["strike"] - short_opt["strike"]) * 100 * num_contracts - net_cost
    breakeven = long_opt["strike"] - net_cost / (100 * num_contracts)
    T_years = st.session_state.expiration_days / 365.0
    iv = _calibrate_iv(
        net_cost / (100 * num_contracts),
        st.session_state.current_price,
        st.session_state.expiration_days,
        lambda p, t, s: (black_scholes(p, long_opt["strike"], t, st.session_state.risk_free_rate, s, "put") -
                         black_scholes(p, short_opt["strike"], t, st.session_state.risk_free_rate, s, "put")),
        [long_opt, short_opt], ["buy", "sell"]
    )
    if iv == DEFAULT_IV:
        logger.warning(f"IV calibration failed for {long_opt['symbol']}/{short_opt['symbol']}")
        breakeven_prob = 0.0
    else:
        breakeven_prob = estimate_breakeven_probability(
            [long_opt, short_opt], ["buy", "sell"], [num_contracts, num_contracts],
            net_cost, T_years, iv
        )

    return {
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "breakeven_prob": breakeven_prob,
        "cost_to_profit": abs(net_cost / max_profit) if max_profit > 0 else float('inf')
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
    if not (long_low["strike"] < short_mid["strike"] < long_high["strike"]):
        return None
    long_high_price = get_strategy_price(long_high, "buy")
    short_mid_price = get_strategy_price(short_mid, "sell")
    long_low_price = get_strategy_price(long_low, "buy")
    if any(p is None for p in [long_high_price, short_mid_price, long_low_price]):
        return None

    base_cost = (long_high_price + long_low_price - 2 * short_mid_price) * 100 * num_contracts
    total_fees = (calculate_fees(long_high_price * 100 * num_contracts, commission_rate) +
                  calculate_fees(short_mid_price * 100 * num_contracts * 2, commission_rate) +
                  calculate_fees(long_low_price * 100 * num_contracts, commission_rate))
    net_cost = base_cost + total_fees

    width_lower = short_mid["strike"] - long_low["strike"]
    width_upper = long_high["strike"] - short_mid["strike"]
    max_profit = min(width_lower, width_upper) * 100 * num_contracts - net_cost
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
    if not (long_low["strike"] < short_mid_low["strike"] < short_mid_high["strike"] < long_high["strike"]):
        return None
    long_low_price = get_strategy_price(long_low, "buy")
    short_mid_low_price = get_strategy_price(short_mid_low, "sell")
    short_mid_high_price = get_strategy_price(short_mid_high, "sell")
    long_high_price = get_strategy_price(long_high, "buy")
    if any(p is None for p in [long_low_price, short_mid_low_price, short_mid_high_price, long_high_price]):
        return None

    base_cost = (long_low_price + long_high_price - short_mid_low_price - short_mid_high_price) * 100 * num_contracts
    total_fees = (calculate_fees(long_low_price * 100 * num_contracts, commission_rate) +
                  calculate_fees(short_mid_low_price * 100 * num_contracts, commission_rate) +
                  calculate_fees(short_mid_high_price * 100 * num_contracts, commission_rate) +
                  calculate_fees(long_high_price * 100 * num_contracts, commission_rate))
    net_cost = base_cost + total_fees

    width_lower = short_mid_low["strike"] - long_low["strike"]
    width_upper = long_high["strike"] - short_mid_high["strike"]
    max_profit = min(width_lower, width_upper) * 100 * num_contracts - net_cost
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
    if not (long_low["strike"] < short_mid_low["strike"] < short_mid_high["strike"] < long_high["strike"]):
        return None
    long_high_price = get_strategy_price(long_high, "buy")
    short_mid_high_price = get_strategy_price(short_mid_high, "sell")
    short_mid_low_price = get_strategy_price(short_mid_low, "sell")
    long_low_price = get_strategy_price(long_low, "buy")
    if any(p is None for p in [long_high_price, short_mid_high_price, short_mid_low_price, long_low_price]):
        return None

    base_cost = (long_high_price + long_low_price - short_mid_high_price - short_mid_low_price) * 100 * num_contracts
    total_fees = (calculate_fees(long_high_price * 100 * num_contracts, commission_rate) +
                  calculate_fees(short_mid_high_price * 100 * num_contracts, commission_rate) +
                  calculate_fees(short_mid_low_price * 100 * num_contracts, commission_rate) +
                  calculate_fees(long_low_price * 100 * num_contracts, commission_rate))
    net_cost = base_cost + total_fees

    width_lower = short_mid_low["strike"] - long_low["strike"]
    width_upper = long_high["strike"] - short_mid_high["strike"]
    max_profit = min(width_lower, width_upper) * 100 * num_contracts - net_cost
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
    
    # Estimate max_profit at plot range boundaries
    current_price = st.session_state.get("current_price", 100.0)
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    max_price = current_price * (1 + plot_range_pct)
    min_price = max(0.1, current_price * (1 - plot_range_pct))
    call_profit = max(max_price - strike, 0) - call_price
    put_profit = max(strike - min_price, 0) - put_price
    max_profit = max(call_profit, put_profit) * 100 * num_contracts * (1 - commission_rate)
    
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit if max_profit > 0 else 0.01,  # Avoid zero
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
    
    # Estimate max_profit at plot range boundaries
    current_price = st.session_state.get("current_price", 100.0)
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    max_price = current_price * (1 + plot_range_pct)
    min_price = max(0.1, current_price * (1 - plot_range_pct))
    call_profit = max(max_price - call_opt["strike"], 0) - call_price
    put_profit = max(put_opt["strike"] - min_price, 0) - put_price
    max_profit = max(call_profit, put_profit) * 100 * num_contracts * (1 - commission_rate)
    
    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "max_profit": max_profit if max_profit > 0 else 0.01,  # Avoid zero
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [put_opt["strike"], call_opt["strike"]],
        "num_contracts": num_contracts
    }
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
    max_iter = 200  # Increased from 100
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
# --- Shared Helpers for Viz and Tables ---

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
            if abs(model_value - raw_net) < raw_net * 0.2:  # Relaxed to 20%
                return calibrated_iv
            else:
                logger.warning(f"IV verification failed: model_value={model_value}, raw_net={raw_net}, tolerance={raw_net * 0.2}")
                return calibrated_iv  # Use calibrated IV anyway
        except Exception as e:
            logger.warning(f"Strategy value error during IV verification: {e}")
            return calibrated_iv
    logger.warning("No valid IVs calculated. Using default IV.")
    return DEFAULT_IV

def estimate_breakeven_probability(options, actions, contracts, net_cost, T_years, sigma):
    """
    Estimate the probability of breaking even at expiration using a lognormal distribution.
    
    Args:
        options (list): List of option dictionaries
        actions (list): List of actions ("buy" or "sell")
        contracts (list): List of contract quantities
        net_cost (float): Net cost or credit of the strategy
        T_years (float): Time to expiration in years
        sigma (float): Implied volatility
    Returns:
        float: Probability of P&L >= 0 at expiration
    """
    import numpy as np
    from scipy.stats import norm
    def payoff_at_expiration(price):
        return calculate_strategy_value(options, actions, contracts, price, 0, sigma) - net_cost

    plot_range_pct = st.session_state.get('plot_range_pct', 0.5)
    price_range = np.linspace(
        st.session_state.current_price * (1 - plot_range_pct),
        st.session_state.current_price * (1 + plot_range_pct), 200
    )
    payoffs = [payoff_at_expiration(p) for p in price_range]
    breakeven_points = []
    for i in range(1, len(payoffs)):
        if payoffs[i-1] * payoffs[i] <= 0:
            breakeven = price_range[i-1] + (price_range[i] - price_range[i-1]) * (-payoffs[i-1]) / (payoffs[i] - payoffs[i-1])
            breakeven_points.append(breakeven)

    if not breakeven_points:
        return 0.0

    mu = np.log(st.session_state.current_price) + (st.session_state.risk_free_rate - 0.5 * sigma**2) * T_years
    sigma_t = sigma * np.sqrt(T_years)
    prob = 0.0
    sorted_breakevens = sorted(breakeven_points)
    for i in range(0, len(sorted_breakevens) + 1):
        if i == 0:
            lower = st.session_state.current_price * (1 - plot_range_pct)
            upper = sorted_breakevens[0] if sorted_breakevens else st.session_state.current_price * (1 + plot_range_pct)
        elif i == len(sorted_breakevens):
            lower = sorted_breakevens[-1]
            upper = st.session_state.current_price * (1 + plot_range_pct)
        else:
            lower = sorted_breakevens[i-1]
            upper = sorted_breakevens[i]
        mid_point = (lower + upper) / 2
        if payoff_at_expiration(mid_point) >= 0:
            lower_log = np.log(lower) if lower > 0 else -np.inf
            upper_log = np.log(upper) if upper > 0 else np.inf
            prob += norm.cdf((upper_log - mu) / sigma_t) - norm.cdf((lower_log - mu) / sigma_t)
    return prob

def calculate_strategy_value(options, actions, contracts, price, t, sigma):
    """
    Calculate the total value of an options strategy at a given price and time.
    
    Args:
        options (list): List of option dictionaries
        actions (list): List of actions ("buy" or "sell")
        contracts (list): List of contract quantities
        price (float): Underlying stock price
        t (float): Time to expiration in years
        sigma (float): Implied volatility
    Returns:
        float: Total strategy value
    """
    total_value = 0.0
    for opt, action, num_contracts in zip(options, actions, contracts):
        opt_type = opt["type"]
        strike = opt["strike"]
        if t == 0:  # At expiration, use intrinsic value
            if opt_type == "call":
                value = max(0, price - strike)
            else:  # put
                value = max(0, strike - price)
        else:  # Use Black-Scholes for t > 0
            value = utils.black_scholes(price, strike, t, st.session_state.risk_free_rate, sigma, opt_type)
        multiplier = 1 if action == "buy" else -1
        total_value += value * num_contracts * 100 * multiplier
    return total_value

def has_limited_loss(options, actions, contracts):
    """
    Check if a strategy has limited loss potential.
    
    Args:
        options (list): List of option dictionaries
        actions (list): List of actions ("buy" or "sell")
        contracts (list): List of contract quantities
    Returns:
        bool: True if losses are limited, False otherwise
    """
    net_position = {"call": {}, "put": {}}
    for opt, action, num in zip(options, actions, contracts):
        strike = opt["strike"]
        opt_type = opt["type"]
        multiplier = num if action == "buy" else -num
        net_position[opt_type][strike] = net_position[opt_type].get(strike, 0) + multiplier
    
    for opt_type, positions in net_position.items():
        if not positions: continue
        max_strike = max(positions.keys())
        min_strike = min(positions.keys())
        net_pos = sum(positions.values())
        if (opt_type == "call" and positions.get(max_strike, 0) < 0) or \
           (opt_type == "put" and positions.get(min_strike, 0) < 0) or \
           (net_pos < 0):
            return False
    return True
def _compute_payoff_grid(strategy_value, current_price, expiration_days, iv, net_entry):
    """
    Compute the payoff grid for 3D visualization.
    
    Args:
        strategy_value (callable): Function to compute strategy value
        current_price (float): Current stock price
        expiration_days (int): Days to expiration
        iv (float): Implied volatility
        net_entry (float): Net entry cost
    Returns:
        tuple: X, Y, Z arrays for the 3D surface plot
    """
    import numpy as np
    plot_range_pct = st.session_state.get('plot_range_pct', 0.5)  # Fallback to 50%
    min_price = current_price * (1 - plot_range_pct)
    max_price = current_price * (1 + plot_range_pct)
    days = np.linspace(0, expiration_days, 20)
    prices = np.linspace(min_price, max_price, 20)
    X, Y = np.meshgrid(prices, days)
    Z = np.zeros_like(X)
    T_years = expiration_days / 365.0
    for i in range(len(days)):
        t = max(0, (expiration_days - days[i]) / 365.0)
        for j in range(len(prices)):
            Z[i, j] = strategy_value(prices[j], t, iv) - net_entry
    return X, Y, Z

def _create_3d_figure(X, Y, Z, title, current_price):
    """
    Create a 3D surface plot for strategy payoff.
    
    Args:
        X, Y, Z: Arrays for the 3D surface
        title (str): Title of the strategy
        current_price (float): Current stock price
    Returns:
        plotly.graph_objects.Figure: 3D plot
    """
    import plotly.graph_objects as go
    import numpy as np
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, 'rgb(255, 0, 0)'], [0.5, 'rgb(255, 255, 255)'], [1, 'rgb(0, 128, 0)']],
            showscale=False,
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[current_price], y=[0], z=[0],
            mode='markers', marker=dict(size=5, color='red'),
            name='Precio Actual'
        )
    )
    z_min = np.min(Z)
    z_max = np.max(Z)
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=np.zeros_like(Z),
            colorscale=[[0, 'rgba(0, 0, 0, 0.2)'], [1, 'rgba(0, 0, 0, 0.2)']],
            showscale=False
        )
    )
    fig.update_layout(
        title=f"3D P&L for {title} (IV: {iv:.1%})",
        scene=dict(
            xaxis_title="Precio de GGAL (ARS)",
            yaxis_title="Días hasta Vencimiento",
            zaxis_title="P&L (ARS)",
            xaxis=dict(tickvals=[np.min(X), current_price, np.max(X)], 
                      ticktext=[f"{np.min(X):.2f}", f"{current_price:.2f}", f"{np.max(X):.2f}"]),
            yaxis=dict(tickvals=[0, np.max(Y)], ticktext=["0", f"{int(np.max(Y))}"]),
            zaxis=dict(range=[z_min, z_max], tickvals=[z_min, 0, z_max], 
                      ticktext=[f"{z_min:.2f}", "0", f"{z_max:.2f}"])
        ),
        height=600,
        margin=dict(l=50, r=50, b=50, t=50)
    )
    return fig

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

# --- Split Table Creation Functions ---

def create_bullish_spread_table(options, calc_func, num_contracts, commission_rate, is_debit=True):
    """
    Create a table for bullish spread strategies.
    
    Args:
        options (list): List of option dictionaries
        calc_func (callable): Function to calculate spread metrics
        num_contracts (int): Number of contracts
        commission_rate (float): Commission rate as a decimal
        is_debit (bool): True for debit spreads, False for credit spreads
    Returns:
        pd.DataFrame: Table of spread metrics
    """
    import pandas as pd
    from itertools import combinations
    include_breakeven_prob = st.session_state.get('include_breakeven_prob', True)
    results = []
    for long_opt, short_opt in combinations(options, 2):
        result = calc_func(long_opt, short_opt, num_contracts, commission_rate)
        if result:
            results.append({
                "Long Strike": long_opt["strike"],
                "Short Strike": short_opt["strike"],
                "Strikes": f"{long_opt['strike']}-{short_opt['strike']}",
                "Net Cost" if is_debit else "Net Credit": result["net_cost"] if is_debit else result["net_credit"],
                "Max Profit": result["max_profit"],
                "Max Loss": result["max_loss"],
                "Breakeven": result["breakeven"],
                "Breakeven Probability": result["breakeven_prob"] if include_breakeven_prob else None,
                "Cost-to-Profit Ratio": result["cost_to_profit"]
            })
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.set_index(["Long Strike", "Short Strike"])

def create_bearish_spread_table(options, calc_func, num_contracts, commission_rate, is_debit=True):
    """
    Create a table for bearish spread strategies.
    
    Args:
        options (list): List of option dictionaries
        calc_func (callable): Function to calculate spread metrics
        num_contracts (int): Number of contracts
        commission_rate (float): Commission rate as a decimal
        is_debit (bool): True for debit spreads, False for credit spreads
    Returns:
        pd.DataFrame: Table of spread metrics
    """
    import pandas as pd
    from itertools import combinations
    include_breakeven_prob = st.session_state.get('include_breakeven_prob', True)
    results = []
    for long_opt, short_opt in combinations(options, 2):
        result = calc_func(long_opt, short_opt, num_contracts, commission_rate)
        if result:
            results.append({
                "Long Strike": long_opt["strike"],
                "Short Strike": short_opt["strike"],
                "Strikes": f"{long_opt['strike']}-{short_opt['strike']}",
                "Net Cost" if is_debit else "Net Credit": result["net_cost"] if is_debit else result["net_credit"],
                "Max Profit": result["max_profit"],
                "Max Loss": result["max_loss"],
                "Breakeven": result["breakeven"],
                "Breakeven Probability": result["breakeven_prob"] if include_breakeven_prob else None,
                "Cost-to-Profit Ratio": result["cost_to_profit"]
            })
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.set_index(["Long Strike", "Short Strike"])

def create_neutral_table(options, calc_func, num_contracts, commission_rate, num_legs):
    data = []
    options_sorted = sorted(options, key=lambda o: o["strike"])
    for combo in combinations(options_sorted, num_legs):
        if all(combo[i]["strike"] < combo[i+1]["strike"] for i in range(num_legs-1)):
            if num_legs == 3:
                result = calc_func(combo[0], combo[1], combo[2], num_contracts, commission_rate)
            elif num_legs == 4:
                result = calc_func(combo[0], combo[1], combo[2], combo[3], num_contracts, commission_rate)
            if result:
                row = {
                    "net_cost": result["net_cost"],
                    "max_profit": result["max_profit"],
                    "max_loss": result["max_loss"],
                    "lower_breakeven": result["lower_breakeven"],
                    "upper_breakeven": result["upper_breakeven"],
                    "Cost-to-Profit Ratio": result["max_loss"] / result["max_profit"] if result["max_profit"] > 0 else float('inf')
                }
                data.append((tuple(c["strike"] for c in combo), row))
    df = pd.DataFrame.from_dict(dict(data), orient='index') if data else pd.DataFrame()
    return df

def create_volatility_table(leg1_options, leg2_options, calc_func, num_contracts, commission_rate):
    data = []
    for opt1, opt2 in product(leg1_options, leg2_options):
        result = calc_func(opt1, opt2, num_contracts, commission_rate)
        if result:
            row = {
                "net_cost": result["net_cost"],
                "max_loss": result["max_loss"],
                "lower_breakeven": result["lower_breakeven"],
                "upper_breakeven": result["upper_breakeven"],
                "Cost-to-Profit Ratio": result["max_loss"] / result.get("max_profit", 0.01) if result.get("max_profit", 0.01) > 0 else float('inf')
            }
            strikes = (opt1["strike"], opt2["strike"]) if opt1["strike"] != opt2["strike"] else opt1["strike"]
            data.append((strikes, row))
    df = pd.DataFrame.from_dict(dict(data), orient='index') if data else pd.DataFrame()
    return df

def create_spread_matrix(options, calc_func, num_contracts, commission_rate, is_debit=True):
    options_sorted = sorted(options, key=lambda o: o["strike"])
    strikes = [o["strike"] for o in options_sorted]
    profit_df = pd.DataFrame(index=strikes, columns=strikes, dtype=float)
    
    for i in range(len(strikes)):
        for j in range(len(strikes)):
            if i == j:
                continue
            opt1 = options_sorted[i]
            opt2 = options_sorted[j]
            result = calc_func(opt1, opt2, num_contracts, commission_rate)
            if result is not None:
                if is_debit:
                    val = result["net_cost"]
                else:
                    val = -result["net_cost"]  # Make credit positive
                profit_df.at[opt1["strike"], opt2["strike"]] = val
    
    # Placeholder for additional matrices (max_profit, etc.) if needed in the future
    return profit_df, None, None, None

# --- Split 3D Visualization Functions ---

def visualize_bullish_3d(result, current_price, expiration_days, iv, key, options, option_actions):
    raw_net = result["raw_net"]
    net_entry = result["net_cost"]
    num_contracts = result["num_contracts"]
    r = st.session_state.get("risk_free_rate", 0.50)
    
    def strategy_value(price, T, sigma):
        strikes = result["strikes"]
        k1, k2 = min(strikes), max(strikes)
        use_intrinsic = T <= 1e-6
        
        if "call" in key.lower():
            if use_intrinsic:
                return (intrinsic_value(price, k1, "call") - intrinsic_value(price, k2, "call"))
            else:
                return (black_scholes(price, k1, T, r, sigma, "call") - 
                        black_scholes(price, k2, T, r, sigma, "call"))
        elif "put" in key.lower():
            if use_intrinsic:
                return (-intrinsic_value(price, k2, "put") + intrinsic_value(price, k1, "put"))
            else:
                return (-black_scholes(price, k2, T, r, sigma, "put") + 
                        black_scholes(price, k1, T, r, sigma, "put"))
        return 0.0
    
    iv = _calibrate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions)
    iv = max(iv, 1e-9)
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        lambda p, t, s: strategy_value(p, t, s) * 100 * num_contracts, 
        current_price, expiration_days, iv, net_entry
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)

def visualize_bearish_3d(result, current_price, expiration_days, iv, key, options, option_actions):
    raw_net = result["raw_net"]
    net_entry = result["net_cost"]
    num_contracts = result["num_contracts"]
    r = st.session_state.get("risk_free_rate", 0.50)
    
    def strategy_value(price, T, sigma):
        strikes = result["strikes"]
        k1, k2 = min(strikes), max(strikes)
        use_intrinsic = T <= 1e-6
        
        if "call" in key.lower():
            if use_intrinsic:
                return (-intrinsic_value(price, k1, "call") + intrinsic_value(price, k2, "call"))
            else:
                return (-black_scholes(price, k1, T, r, sigma, "call") + 
                        black_scholes(price, k2, T, r, sigma, "call"))
        elif "put" in key.lower():
            if use_intrinsic:
                return (intrinsic_value(price, k2, "put") - intrinsic_value(price, k1, "put"))
            else:
                return (black_scholes(price, k2, T, r, sigma, "put") - 
                        black_scholes(price, k1, T, r, sigma, "put"))
        return 0.0
    
    iv = _calibrate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions)
    iv = max(iv, 1e-9)
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        lambda p, t, s: strategy_value(p, t, s) * 100 * num_contracts, 
        current_price, expiration_days, iv, net_entry
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)

def visualize_neutral_3d(result, current_price, expiration_days, iv, key, options, option_actions):
    raw_net = result["raw_net"]
    net_entry = result["net_cost"]
    num_contracts = result["num_contracts"]
    ratios = result.get("contract_ratios", [])
    strikes = result["strikes"]
    opt_type = "call" if "call" in key.lower() else "put"
    r = st.session_state.get("risk_free_rate", 0.50)
    
    def strategy_value(price, T, sigma):
        use_intrinsic = T <= 1e-6
        if use_intrinsic:
            vals = [intrinsic_value(price, k, opt_type) for k in strikes]
        else:
            vals = [black_scholes(price, k, T, r, sigma, opt_type) for k in strikes]
        return sum(r * v for r, v in zip(ratios, vals)) * 100 * num_contracts
    
    iv = _calibrate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions, contract_ratios=ratios)
    iv = max(iv, 1e-9)
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        strategy_value, current_price, expiration_days, iv, net_entry
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)

def visualize_volatility_3d(result, current_price, expiration_days, iv, key, options, option_actions):
    raw_net = result["raw_net"]
    net_entry = result["net_cost"]
    num_contracts = result["num_contracts"]
    strikes = result["strikes"]
    r = st.session_state.get("risk_free_rate", 0.50)
    
    def strategy_value(price, T, sigma):
        use_intrinsic = T <= 1e-6
        if len(strikes) == 1:
            k = strikes[0]
            if use_intrinsic:
                call_val = intrinsic_value(price, k, "call")
                put_val = intrinsic_value(price, k, "put")
            else:
                call_val = black_scholes(price, k, T, r, sigma, "call")
                put_val = black_scholes(price, k, T, r, sigma, "put")
            return (call_val + put_val) * 100 * num_contracts
        else:
            put_k, call_k = min(strikes), max(strikes)
            if use_intrinsic:
                call_val = intrinsic_value(price, call_k, "call")
                put_val = intrinsic_value(price, put_k, "put")
            else:
                call_val = black_scholes(price, call_k, T, r, sigma, "call")
                put_val = black_scholes(price, put_k, T, r, sigma, "put")
            return (call_val + put_val) * 100 * num_contracts
        return 0.0
    
    iv = _calibrate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions)
    iv = max(iv, 1e-9)
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        strategy_value, current_price, expiration_days, iv, net_entry
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)