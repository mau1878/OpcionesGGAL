import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta, timezone
from itertools import combinations
from math import gcd
from functools import reduce
import plotly.graph_objects as go
import logging
from scipy.stats import norm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Endpoints
STOCK_URL = "https://data912.com/live/arg_stocks"
OPTIONS_URL = "https://data912.com/live/arg_options"

# Constants
DEFAULT_IV = 0.30
MIN_GAP = 0.01  # Minimum gap to prevent zero division


# Helper Functions
def get_third_friday(year: int, month: int) -> date:
    try:
        first_day = date(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
        return first_friday + timedelta(days=14)
    except ValueError:
        return None


EXPIRATION_MAP_2025 = {
    "O": get_third_friday(2025, 10), "OC": get_third_friday(2025, 10),
    "N": get_third_friday(2025, 11), "NO": get_third_friday(2025, 11),
    "D": get_third_friday(2025, 12), "DI": get_third_friday(2025, 12)
}


@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_data(url: str) -> list:
    try:
        response = requests.get(url, headers={"accept": "*/*"}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching data from {url}: {e}")
        logger.error(f"Fetch error: {e}")
        return []


def parse_option_symbol(symbol: str) -> tuple[str | None, float | None, date | None]:
    if symbol.startswith("GFGC"):
        option_type = "call"
    elif symbol.startswith("GFGV"):
        option_type = "put"
    else:
        return None, None, None

    data_part = symbol[4:]
    first_letter_index = -1
    for i, char in enumerate(data_part):
        if char.isalpha():
            first_letter_index = i
            break

    if first_letter_index == -1:
        return None, None, None

    numeric_part = data_part[:first_letter_index]
    suffix = data_part[first_letter_index:]

    if not numeric_part:
        return None, None, None

    try:
        if len(numeric_part) == 5 and not numeric_part.startswith('1'):
            strike_price = float(numeric_part) / 10.0
        elif len(numeric_part) > 5:
            strike_price = float(numeric_part) / 10.0
        else:
            strike_price = float(numeric_part)
    except ValueError:
        return None, None, None

    expiration = EXPIRATION_MAP_2025.get(suffix, None)
    return option_type, strike_price, expiration


def get_ggal_data() -> tuple[dict | None, list]:
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)
    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock:
        return None, []
    ggal_options = []
    for o in options_data:
        opt_type, strike, exp = parse_option_symbol(o["symbol"])
        if opt_type and strike is not None and exp and o.get("px_ask", 0) > 0 and o.get("px_bid", 0) > 0:
            ggal_options.append({
                "symbol": o["symbol"], "type": opt_type, "strike": strike,
                "expiration": exp, "px_bid": o["px_bid"], "px_ask": o["px_ask"], "c": o["c"]
            })
    return ggal_stock, ggal_options


def get_strategy_price(option: dict, action: str) -> float | None:
    price = option["px_ask"] if action == "buy" else option["px_bid"]
    return price if price > 0 else None


def calculate_fees(base_cost: float, commission_rate: float) -> tuple[float, float, float]:
    commission = base_cost * commission_rate
    market_fees = base_cost * 0.002
    vat = (commission + market_fees) * 0.21
    return commission, market_fees, vat


# --- All your calculate_* and create_* functions go here ---
# (calculate_bull_call_spread, calculate_straddle, create_spread_matrix, etc.)
# ... (The full code for these functions is omitted for brevity, but you should paste them all here)


# --- Visualization Function ---
def visualize_3d_payoff(strategy_result, current_price, expiration_days, iv=DEFAULT_IV, key=None):
    if not strategy_result or "strikes" not in strategy_result:
        st.warning("No valid strategy selected for visualization.")
        return

    net_cost = strategy_result.get("Net Cost", strategy_result.get("net_cost", 0))
    if net_cost is None: return

    def black_scholes(S, K, T, r, sigma, option_type="call"):
        if T <= 1e-6: return max(0, S - K) if option_type == "call" else max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    prices = np.linspace(max(0.1, current_price * 0.5), current_price * 1.5, 50)
    times = np.linspace(0, expiration_days, 20)
    X, Y = np.meshgrid(prices, times)
    Z = np.zeros_like(X)
    strikes = strategy_result["strikes"]
    strategy_key = key.lower() if key else ""
    r = 0.05

    for i in range(len(times)):
        for j in range(len(prices)):
            price = X[i, j]
            T = (expiration_days - Y[i, j]) / 365.0
            position_value = 0.0

            # Calculation logic for each strategy
            if "bull call spread" in strategy_key or "bear call spread" in strategy_key:
                k1, k2 = strikes
                val1 = black_scholes(price, k1, T, r, iv, "call")
                val2 = black_scholes(price, k2, T, r, iv, "call")
                position_value = (val1 - val2) * 100
            elif "bull put spread" in strategy_key or "bear put spread" in strategy_key:
                k1, k2 = strikes
                val1 = black_scholes(price, k1, T, r, iv, "put")
                val2 = black_scholes(price, k2, T, r, iv, "put")
                position_value = (val1 - val2) * 100
            elif "straddle" in strategy_key:
                k1 = strikes[0]
                val_call = black_scholes(price, k1, T, r, iv, "call")
                val_put = black_scholes(price, k1, T, r, iv, "put")
                position_value = (val_call + val_put) * 100
            elif "strangle" in strategy_key:
                put_k, call_k = strikes
                val_call = black_scholes(price, call_k, T, r, iv, "call")
                val_put = black_scholes(price, put_k, T, r, iv, "put")
                position_value = (val_call + val_put) * 100
            elif "butterfly" in strategy_key:
                low_k, mid_k, high_k = strikes
                opt_type = "call" if "call" in strategy_key else "put"
                val1 = black_scholes(price, low_k, T, r, iv, opt_type)
                val2 = black_scholes(price, mid_k, T, r, iv, opt_type)
                val3 = black_scholes(price, high_k, T, r, iv, opt_type)
                position_value = (val1 - 2 * val2 + val3) * 100
            elif "condor" in strategy_key:
                k1, k2, k3, k4 = strikes
                opt_type = "call" if "call" in strategy_key else "put"
                val1 = black_scholes(price, k1, T, r, iv, opt_type)
                val2 = black_scholes(price, k2, T, r, iv, opt_type)
                val3 = black_scholes(price, k3, T, r, iv, opt_type)
                val4 = black_scholes(price, k4, T, r, iv, opt_type)
                position_value = (val1 - val2 - val3 + val4) * 100

            Z[i, j] = position_value - net_cost

    payoff_surface = go.Surface(z=Z, x=X, y=Y, colorscale='RdYlGn', cmin=Z.min(), cmax=Z.max(),
                                colorbar=dict(title='Profit/Loss'))
    breakeven_plane = go.Surface(z=np.zeros_like(X), x=X, y=Y, opacity=0.7, showscale=False,
                                 colorscale=[[0, '#0000FF'], [1, '#0000FF']])
    fig = go.Figure(data=[payoff_surface, breakeven_plane])
    fig.update_layout(title=f"Strategy Value Over Time: {key.replace('_', ' ').title()}",
                      scene=dict(xaxis_title='Underlying Price (ARS)', yaxis_title='Days Elapsed',
                                 zaxis_title='Profit/Loss (ARS)'))
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key}")