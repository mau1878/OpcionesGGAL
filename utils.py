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

@st.cache_data(ttl=300)
def fetch_data(url: str) -> list:
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.error(f"Timeout fetching data from {url}")
        st.error(f"Timeout fetching data from {url}")
        return []
    except requests.HTTPError as e:
        logger.error(f"HTTP error from {url}: {e}")
        st.error(f"HTTP error from {url}: {e}")
        return []
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

@st.cache_data(ttl=300)
def get_ggal_data() -> tuple[dict | None, list]:
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)
    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock:
        logger.error("GGAL stock data not found")
        return None, []
    ggal_options = []
    for o in options_data:
        px_bid = o.get("px_bid")
        px_ask = o.get("px_ask")
        logger.debug(f"Raw option data: symbol={o.get('symbol')}, px_bid={px_bid}, px_ask={px_ask}")
        if not (isinstance(px_bid, (int, float)) and px_bid > 0 and isinstance(px_ask, (int, float)) and px_ask > 0):
            logger.warning(f"Invalid bid/ask for option {o['symbol']}: bid={px_bid}, ask={px_ask}")
            continue
        opt_type, strike, exp = parse_option_symbol(o["symbol"])
        if all([opt_type, strike, exp]):
            ggal_options.append({
                "symbol": o["symbol"], "type": opt_type, "strike": strike,
                "expiration": exp, "px_bid": px_bid, "px_ask": px_ask
            })
        else:
            logger.debug(f"Skipped option {o['symbol']}: Invalid data")
    return ggal_stock, ggal_options

def get_strategy_price(option: dict, action: str) -> float | None:
    price = option["px_ask"] if action == "buy" else option["px_bid"]
    return price if isinstance(price, (int, float)) and price > 0 else None

def calculate_fees(base_cost: float, commission_rate: float) -> float:
    commission = base_cost * commission_rate
    market_fees = base_cost * 0.002
    vat = (commission + market_fees) * 0.21
    return commission + market_fees + vat

# --- Strategy Calculation Functions ---

def calculate_bull_call_spread(options: list, actions: list, contracts: list, commission_rate: float) -> dict:
    if len(options) != 2 or options[0]["type"] != "call" or options[1]["type"] != "call":
        return None
    net_cost = calculate_strategy_cost(options, actions, contracts)
    if net_cost is None:
        return None
    k1, k2 = options[0]["strike"], options[1]["strike"]
    max_profit = (k2 - k1 - net_cost / (100 * contracts[0])) * 100 * contracts[0]
    max_loss = net_cost
    breakeven = k1 + net_cost / (100 * contracts[0])
    T = st.session_state.get("expiration_days", 30) / 365.0
    iv = _calibrate_iv(net_cost / (100 * contracts[0]), st.session_state.get("current_price", 100.0), st.session_state.get("expiration_days", 30),
                       lambda p, t, s: black_scholes(p, k1, t, st.session_state.get("risk_free_rate", 0.50), s, "call") -
                       black_scholes(p, k2, t, st.session_state.get("risk_free_rate", 0.50), s, "call"), options, actions)
    breakeven_prob = estimate_breakeven_probability(options, actions, contracts, net_cost, T, iv)
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "breakeven": breakeven, "breakeven_prob": breakeven_prob}

def calculate_bull_put_spread(options: list, actions: list, contracts: list, commission_rate: float) -> dict:
    if len(options) != 2 or options[0]["type"] != "put" or options[1]["type"] != "put":
        return None
    net_cost = calculate_strategy_cost(options, actions, contracts)
    if net_cost is None:
        return None
    k1, k2 = options[0]["strike"], options[1]["strike"]
    max_profit = -net_cost
    max_loss = (k2 - k1) * 100 * contracts[0] + net_cost
    breakeven = k2 + net_cost / (100 * contracts[0])
    T = st.session_state.get("expiration_days", 30) / 365.0
    iv = _calibrate_iv(-net_cost / (100 * contracts[0]), st.session_state.get("current_price", 100.0), st.session_state.get("expiration_days", 30),
                       lambda p, t, s: black_scholes(p, k2, t, st.session_state.get("risk_free_rate", 0.50), s, "put") -
                       black_scholes(p, k1, t, st.session_state.get("risk_free_rate", 0.50), s, "put"), options, actions)
    breakeven_prob = estimate_breakeven_probability(options, actions, contracts, net_cost, T, iv)
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "breakeven": breakeven, "breakeven_prob": breakeven_prob}

def calculate_bear_call_spread(options: list, actions: list, contracts: list, commission_rate: float) -> dict:
    if len(options) != 2 or options[0]["type"] != "call" or options[1]["type"] != "call":
        return None
    net_cost = calculate_strategy_cost(options, actions, contracts)
    if net_cost is None:
        return None
    k1, k2 = options[0]["strike"], options[1]["strike"]
    max_profit = -net_cost
    max_loss = (k2 - k1) * 100 * contracts[0] + net_cost
    breakeven = k1 - net_cost / (100 * contracts[0])
    T = st.session_state.get("expiration_days", 30) / 365.0
    iv = _calibrate_iv(-net_cost / (100 * contracts[0]), st.session_state.get("current_price", 100.0), st.session_state.get("expiration_days", 30),
                       lambda p, t, s: black_scholes(p, k1, t, st.session_state.get("risk_free_rate", 0.50), s, "call") -
                       black_scholes(p, k2, t, st.session_state.get("risk_free_rate", 0.50), s, "call"), options, actions)
    breakeven_prob = estimate_breakeven_probability(options, actions, contracts, net_cost, T, iv)
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "breakeven": breakeven, "breakeven_prob": breakeven_prob}

def calculate_bear_put_spread(options: list, actions: list, contracts: list, commission_rate: float) -> dict:
    if len(options) != 2 or options[0]["type"] != "put" or options[1]["type"] != "put":
        return None
    net_cost = calculate_strategy_cost(options, actions, contracts)
    if net_cost is None:
        return None
    k1, k2 = options[0]["strike"], options[1]["strike"]
    max_profit = (k1 - k2 - net_cost / (100 * contracts[0])) * 100 * contracts[0]
    max_loss = net_cost
    breakeven = k1 - net_cost / (100 * contracts[0])
    T = st.session_state.get("expiration_days", 30) / 365.0
    iv = _calibrate_iv(net_cost / (100 * contracts[0]), st.session_state.get("current_price", 100.0), st.session_state.get("expiration_days", 30),
                       lambda p, t, s: black_scholes(p, k1, t, st.session_state.get("risk_free_rate", 0.50), s, "put") -
                       black_scholes(p, k2, t, st.session_state.get("risk_free_rate", 0.50), s, "put"), options, actions)
    breakeven_prob = estimate_breakeven_probability(options, actions, contracts, net_cost, T, iv)
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "breakeven": breakeven, "breakeven_prob": breakeven_prob}

def calculate_potential_metrics(options: list, actions: list, contracts: list, min_price: float, max_price: float, net_cost: float, iv: float) -> float:
    prices = np.linspace(min_price, max_price, 100)
    payoffs = [calculate_strategy_value(options, actions, contracts, p, 0, iv) - net_cost for p in prices]
    return max(max(payoffs), 0.0)

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

logger = logging.getLogger(__name__)

DEFAULT_IV = 0.3  # Default implied volatility (30%)

def calculate_strategy_cost(options: list, actions: list, contracts: list) -> float | None:
    if len(options) != len(actions) or len(options) != len(contracts):
        return None
    total_cost = 0.0
    for opt, action, num in zip(options, actions, contracts):
        price = get_strategy_price(opt, action)
        if price is None:
            return None
        total_cost += price * 100 * num
    fees = calculate_fees(abs(total_cost), st.session_state.get("commission_rate", 0.005))
    return total_cost + fees if total_cost >= 0 else total_cost - fees

def _calibrate_iv(raw_net: float, current_price: float, expiration_days: float, strategy_value: callable, options: list, actions: list, contract_ratios: list = None) -> float:
    if contract_ratios is None:
        contract_ratios = [1] * len(options)
    T = expiration_days / 365.0
    def objective(sigma):
        val = strategy_value(current_price, T, sigma)
        return (val - raw_net) ** 2
    try:
        result = minimize_scalar(objective, bounds=(0, 2), method="bounded")
        if result.success:
            iv = result.x
            if iv <= 0:
                logger.warning(f"Negative IV calculated: {iv}. Returning DEFAULT_IV.")
                return DEFAULT_IV
            return iv
        else:
            logger.warning(f"IV calibration failed: {result.message}")
            return DEFAULT_IV
    except Exception as e:
        logger.warning(f"IV calibration exception: {e}")
        return DEFAULT_IV

logger = logging.getLogger(__name__)

def estimate_breakeven_probability(options: list, actions: list, contracts: list, net_cost: float, T: float, iv: float) -> float:
    if len(options) != len(actions) or len(options) != len(contracts):
        return 0.0
    r = st.session_state.get("risk_free_rate", 0.50)
    S = st.session_state.get("current_price", 100.0)
    min_price = S * 0.5
    max_price = S * 1.5
    def payoff(p): return calculate_strategy_value(options, actions, contracts, p, 0, iv) - net_cost
    prices = np.linspace(min_price, max_price, 100)
    payoffs = [payoff(p) for p in prices]
    prob = 0.0
    for i in range(len(prices) - 1):
        if payoffs[i] * payoffs[i + 1] <= 0:
            d1 = (np.log(prices[i] / S) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
            prob += norm.cdf(d1)
    return min(prob, 1.0)

def calculate_strategy_value(options: list, actions: list, contracts: list, S: float, T: float, sigma: float) -> float:
    if len(options) != len(actions) or len(options) != len(contracts):
        return 0.0
    total_value = 0.0
    r = st.session_state.get("risk_free_rate", 0.50)
    for opt, action, num in zip(options, actions, contracts):
        price = black_scholes(S, opt["strike"], T / 365.0, r, sigma, opt["type"]) if T > 0 else intrinsic_value(S, opt["strike"], opt["type"])
        total_value += price * num * (1 if action == "buy" else -1)
    return total_value * 100



def has_limited_loss(options: list, actions: list, contracts: list) -> bool:
    if len(options) != len(actions) or len(options) != len(contracts):
        return False
    net_exposure = {"call": {}, "put": {}}
    for opt, action, num in zip(options, actions, contracts):
        if opt["type"] not in net_exposure:
            return False
        strike = opt["strike"]
        qty = num if action == "buy" else -num
        net_exposure[opt["type"]][strike] = net_exposure[opt["type"]].get(strike, 0) + qty
    for opt_type in ["call", "put"]:
        strikes = sorted(net_exposure[opt_type].keys())
        net_qty = 0
        for strike in strikes:
            net_qty += net_exposure[opt_type][strike]
            if opt_type == "call" and net_qty < 0 and strike != max(strikes):
                return False
            if opt_type == "put" and net_qty < 0 and strike != min(strikes):
                return False
    return True
def _compute_payoff_grid(strategy_value: callable, current_price: float, expiration_days: float, iv: float, net_cost: float):
    min_price = current_price * (1 - st.session_state.get("plot_range_pct", 0.5))
    max_price = current_price * (1 + st.session_state.get("plot_range_pct", 0.5))
    times = np.linspace(0, expiration_days, 50)  # Reduced from 100
    prices = np.linspace(min_price, max_price, 50)  # Reduced from 100
    X, Y = np.meshgrid(prices, times)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = strategy_value(X[i, j], Y[i, j], iv) - (net_cost if Y[i, j] == 0 else 0)
    return X, Y, Z, min_price, max_price, times

def _create_3d_figure(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str, current_price: float):
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Precio de GGAL (ARS)",
            yaxis_title="Días hasta el Vencimiento",
            zaxis_title="P&L (ARS)",
            xaxis=dict(tickvals=[X.min(), current_price, X.max()], ticktext=[f"{X.min():.2f}", f"{current_price:.2f}", f"{X.max():.2f}"]),
            yaxis=dict(autorange="reversed"),
            zaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=2)
        ),
        height=600
    )
    return fig


def intrinsic_value(S: float, K: float, option_type: str) -> float:
    if S <= 0 or K <= 0:
        return 0.0
    return max(0, S - K if option_type == "call" else K - S)

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    if S <= 0 or K <= 0 or T < 0 or sigma < 0:
        logger.warning(f"Invalid inputs for Black-Scholes: S={S}, K={K}, T={T}, sigma={sigma}")
        return 0.0
    if T <= 1e-6:
        return intrinsic_value(S, K, option_type)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0.0)
# --- Split Table Creation Functions ---

def create_bullish_spread_table(options: list, calculate_spread: callable, num_contracts: int, commission_rate: float, is_debit: bool = True) -> pd.DataFrame:
    results = []
    for i, long_opt in enumerate(options):
        for short_opt in options[i + 1:]:
            if long_opt["strike"] >= short_opt["strike"]:
                continue
            options_pair = [long_opt, short_opt]
            actions = ["buy", "sell"]
            contracts = [num_contracts, num_contracts]
            metrics = calculate_spread(options_pair, actions, contracts, commission_rate)
            if metrics:
                results.append({
                    "Long Strike": long_opt["strike"],
                    "Short Strike": short_opt["strike"],
                    "Strikes": f"{long_opt['strike']}-{short_opt['strike']}",
                    "Net Cost" if is_debit else "Net Credit": metrics["net_cost"],
                    "Max Profit": metrics["max_profit"],
                    "Max Loss": metrics["max_loss"],
                    "Breakeven": metrics["breakeven"],
                    "Breakeven Probability": metrics["breakeven_prob"],
                    "Cost-to-Profit Ratio": abs(metrics["net_cost"] / metrics["max_profit"]) if metrics["max_profit"] > 0 else float('inf'),
                    "raw_net": metrics["net_cost"]
                })
    return pd.DataFrame(results)

def create_bearish_spread_table(options: list, calculate_spread: callable, num_contracts: int, commission_rate: float, is_debit: bool = False) -> pd.DataFrame:
    results = []
    for i, long_opt in enumerate(options):
        for short_opt in options[i + 1:]:
            if long_opt["strike"] <= short_opt["strike"]:
                continue
            options_pair = [long_opt, short_opt]
            actions = ["buy", "sell"]
            contracts = [num_contracts, num_contracts]
            metrics = calculate_spread(options_pair, actions, contracts, commission_rate)
            if metrics:
                results.append({
                    "Long Strike": long_opt["strike"],
                    "Short Strike": short_opt["strike"],
                    "Strikes": f"{long_opt['strike']}-{short_opt['strike']}",
                    "Net Cost" if is_debit else "Net Credit": metrics["net_cost"],
                    "Max Profit": metrics["max_profit"],
                    "Max Loss": metrics["max_loss"],
                    "Breakeven": metrics["breakeven"],
                    "Breakeven Probability": metrics["breakeven_prob"],
                    "Cost-to-Profit Ratio": abs(metrics["net_cost"] / metrics["max_profit"]) if metrics["max_profit"] > 0 else float('inf'),
                    "raw_net": metrics["net_cost"]
                })
    return pd.DataFrame(results)

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
    net_cost = result["net_cost"] if "net_cost" in result else result["net_credit"]
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
                return (-intrinsic_value(price, k1, "put") + intrinsic_value(price, k2, "put"))
            else:
                return (-black_scholes(price, k1, T, r, sigma, "put") + 
                        black_scholes(price, k2, T, r, sigma, "put"))
        return 0.0
    
    iv = _calibrate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions)
    iv = max(iv, 1e-9)
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        lambda p, t, s: strategy_value(p, t / 365.0, s) * 100 * num_contracts, 
        current_price, expiration_days, iv, net_cost
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)

def visualize_bearish_3d(result, current_price, expiration_days, iv, key, options, option_actions):
    raw_net = result["raw_net"]
    net_entry = result["net_cost"] if "net_cost" in result else result["net_credit"]
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
        lambda p, t, s: strategy_value(p, t / 365.0, s) * 100 * num_contracts, 
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
        lambda p, t, s: strategy_value(p, t / 365.0, s), 
        current_price, expiration_days, iv, net_entry
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
    
    iv = _calibrate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions)
    iv = max(iv, 1e-9)
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        lambda p, t, s: strategy_value(p, t / 365.0, s), 
        current_price, expiration_days, iv, net_entry
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)