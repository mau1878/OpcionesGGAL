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
    lower_breakeven = call_opt["strike"] - (net_cost / (100 * num_contracts))
    upper_breakeven = call_opt["strike"] + (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [call_opt["strike"]],
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

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [put_opt["strike"], call_opt["strike"]],
        "num_contracts": num_contracts
    }
def calculate_option_iv(S, K, T, r, premium, option_type="call"):
    """Calculate implied volatility for a single option using bisection."""
    def bs_price(sigma):
        val = black_scholes(S, K, T, r, sigma, option_type)
        return val - premium

    low, high = 0.01, 5.0  # IV bounds
    epsilon = 1e-6
    max_iter = 100
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

    logger.warning(f"IV calculation did not converge for S={S}, K={K}, T={T}, premium={premium}")
    return DEFAULT_IV
# --- Shared Helpers for Viz and Tables ---

def _calibrate_iv(raw_net, current_price, expiration_days, strategy_value_func, options, option_actions, contract_ratios=None):
    """Calculate implied volatility for a strategy by averaging IVs of individual options."""
    if not options or not option_actions:
        logger.warning("No options or actions provided for IV calibration")
        return DEFAULT_IV

    r = st.session_state.get("risk_free_rate", 0.50)
    T = expiration_days / 365.0
    ivs = []
    weights = contract_ratios if contract_ratios else [1.0] * len(options)

    for opt, action, weight in zip(options, option_actions, weights):
        premium = get_strategy_price(opt, action)
        if premium is None:
            logger.warning(f"Invalid premium for option {opt['symbol']}")
            continue
        iv = calculate_option_iv(
            S=current_price,
            K=opt["strike"],
            T=T,
            r=r,
            premium=premium,
            option_type=opt["type"]
        )
        if iv and not np.isnan(iv):
            ivs.append(iv * weight)

    if ivs:
        total_weight = sum(weights[:len(ivs)])
        calibrated_iv = sum(ivs) / total_weight if total_weight > 0 else DEFAULT_IV
        # Verify the strategy value with the averaged IV
        try:
            model_value = strategy_value_func(current_price, T, calibrated_iv)
            if abs(model_value - raw_net) < raw_net * 0.1:
                return calibrated_iv
        except Exception:
            pass
    logger.warning("Direct IV calibration failed or verification not met. Using default IV.")
    return DEFAULT_IV

def _compute_payoff_grid(strategy_value_func, current_price, expiration_days, iv, net_entry):
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    min_price = max(0.1, current_price * (1 - plot_range_pct))
    max_price = current_price * (1 + plot_range_pct)
    
    prices = np.linspace(min_price, max_price, 50)
    times = np.linspace(0, expiration_days, 20)
    X, Y = np.meshgrid(prices, times)
    Z = np.zeros_like(X)

    for i in range(len(times)):
        for j in range(len(prices)):
            price = X[i, j]
            T = max((expiration_days - Y[i, j]) / 365.0, 1e-9)
            try:
                Z[i, j] = strategy_value_func(price, T, iv) - net_entry
            except Exception as e:
                logger.warning(f"Error computing payoff at price={price}, T={T}: {e}")
                Z[i, j] = 0
    return X, Y, Z, min_price, max_price, times

def _create_3d_figure(X, Y, Z, title, current_price):
    # Estimate z-axis range for the zero-profit plane and current price plane
    z_min = np.min(Z) * 1.1 if np.min(Z) < 0 else -np.max(Z) * 0.1
    z_max = np.max(Z) * 1.1 if np.max(Z) > 0 else -np.min(Z) * 0.1
    
    # Blue plane at z=0 (profit/loss = 0)
    z_zero = np.zeros_like(X)
    blue_plane = go.Surface(
        x=X, y=Y, z=z_zero,
        colorscale=[[0, 'rgba(0, 0, 255, 0.2)'], [1, 'rgba(0, 0, 255, 0.2)']],
        showscale=False,
        name="Zero Profit/Loss"
    )
    
    # Red plane at x=current_price using Mesh3d
    y_min, y_max = np.min(Y), np.max(Y)
    # Define vertices for the rectangle plane
    vertices = [
        [current_price, y_min, z_min],
        [current_price, y_max, z_min],
        [current_price, y_min, z_max],
        [current_price, y_max, z_max]
    ]
    # Define triangles for the plane
    i = [0, 0]
    j = [1, 2]
    k = [2, 3]
    red_plane = go.Mesh3d(
        x=[v[0] for v in vertices],
        y=[v[1] for v in vertices],
        z=[v[2] for v in vertices],
        i=i,
        j=j,
        k=k,
        opacity=0.2,
        color='red',
        flatshading=True,
        name="Current GGAL Price"
    )
    
    # Main payoff surface
    payoff_surface = go.Surface(
        x=X, y=Y, z=Z,
        colorscale='RdYlGn',
        showscale=True,
        name="Payoff Surface"
    )
    
    fig = go.Figure(data=[payoff_surface, blue_plane, red_plane])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Underlying Price",
            yaxis_title="Days from Now",
            zaxis_title="Profit / Loss",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        autosize=False,  # Disable autosize to use custom dimensions
        width=800,      # Set desired width in pixels
        height=600,     # Set desired height in pixels
        margin=dict(l=65, r=50, b=65, t=90),
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
    data = []
    options_sorted = sorted(options, key=lambda o: o["strike"])
    for long_opt, short_opt in combinations(options_sorted, 2):
        result = calc_func(long_opt, short_opt, num_contracts, commission_rate) if is_debit else calc_func(short_opt, long_opt, num_contracts, commission_rate)
        if result:
            row = {
                "Net Cost" if is_debit else "Net Credit": result["net_cost"] if is_debit else -result["net_cost"],
                "Max Profit": result["max_profit"],
                "Max Loss": result["max_loss"],
                "Cost-to-Profit Ratio": result["max_loss"] / result["max_profit"] if result["max_profit"] > 0 else float('inf'),
                "Breakeven": result["breakeven"]
            }
            data.append(((long_opt["strike"], short_opt["strike"]), row))
    df = pd.DataFrame.from_dict(dict(data), orient='index') if data else pd.DataFrame()
    return df

def create_bearish_spread_table(options, calc_func, num_contracts, commission_rate, is_debit=True):
    data = []
    options_sorted = sorted(options, key=lambda o: o["strike"])
    for short_opt, long_opt in combinations(options_sorted, 2):
        result = calc_func(short_opt, long_opt, num_contracts, commission_rate) if not is_debit else calc_func(long_opt, short_opt, num_contracts, commission_rate)
        if result:
            row = {
                "Net Credit" if not is_debit else "Net Cost": -result["net_cost"] if not is_debit else result["net_cost"],
                "Max Profit": result["max_profit"],
                "Max Loss": result["max_loss"],
                "Cost-to-Profit Ratio": result["max_loss"] / result["max_profit"] if result["max_profit"] > 0 else float('inf'),
                "Breakeven": result["breakeven"]
            }
            data.append(((short_opt["strike"], long_opt["strike"]), row))
    df = pd.DataFrame.from_dict(dict(data), orient='index') if data else pd.DataFrame()
    return df

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
                    "upper_breakeven": result["upper_breakeven"]
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
                "upper_breakeven": result["upper_breakeven"]
            }
            strikes = (opt1["strike"], opt2["strike"]) if opt1["strike"] != opt2["strike"] else opt1["strike"]
            data.append((strikes, row))
    df = pd.DataFrame.from_dict(dict(data), orient='index') if data else pd.DataFrame()
    return df

def create_spread_matrix(options, calc_func, num_contracts, commission_rate, is_debit=True):
    options_sorted = sorted(options, key=lambda o: o["strike"])
    strikes = [o["strike"] for o in options_sorted]
    profit_df = pd.DataFrame(index=strikes, columns=strikes)
    return profit_df, None, None, None  # Placeholder for full impl

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