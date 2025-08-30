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

def calculate_bull_call_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] >= short_opt["strike"]:
        return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if long_price is None or short_price is None:
        return None
    base_cost = long_price * num_contracts * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (long_price - short_price) * num_contracts * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (short_opt["strike"] - long_opt["strike"]) * num_contracts * 100 - net_cost
    max_loss = net_cost
    breakeven = long_opt["strike"] + (net_cost / (num_contracts * 100))
    return {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "long_strike": long_opt["strike"],
        "short_strike": short_opt["strike"],
        "strikes": [long_opt["strike"], short_opt["strike"]]
    } if max_profit > 0 else None

def calculate_bull_put_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] >= short_opt["strike"]:
        return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if long_price is None or short_price is None:
        return None
    base_cost = long_price * num_contracts * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (long_price - short_price) * num_contracts * 100 + commission + market_fees + vat
    if net_cost >= 0:
        return None
    max_profit = -net_cost
    max_loss = (short_opt["strike"] - long_opt["strike"]) * num_contracts * 100 + net_cost
    breakeven = short_opt["strike"] + (net_cost / (num_contracts * 100))
    return {
        "max_profit": max_profit,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "long_strike": long_opt["strike"],
        "short_strike": short_opt["strike"],
        "strikes": [long_opt["strike"], short_opt["strike"]]
    }

def calculate_bear_call_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] <= short_opt["strike"]:
        return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if long_price is None or short_price is None:
        return None
    base_cost = long_price * num_contracts * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (long_price - short_price) * num_contracts * 100 + commission + market_fees + vat
    if net_cost >= 0:
        return None
    max_profit = -net_cost
    max_loss = (long_opt["strike"] - short_opt["strike"]) * num_contracts * 100 + net_cost
    breakeven = short_opt["strike"] + (-net_cost / (num_contracts * 100))
    return {
        "max_profit": max_profit,
        "net_cost": net_cost,
        "max_loss": max(0, max_loss),
        "breakeven": breakeven,
        "long_strike": long_opt["strike"],
        "short_strike": short_opt["strike"],
        "strikes": [short_opt["strike"], long_opt["strike"]]
    } if max_profit > 0 and max_loss > 0 else None

def calculate_bear_put_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] <= short_opt["strike"]:
        return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if long_price is None or short_price is None:
        return None
    base_cost = long_price * num_contracts * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (long_price - short_price) * num_contracts * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (long_opt["strike"] - short_opt["strike"]) * num_contracts * 100 - net_cost
    max_loss = net_cost
    breakeven = long_opt["strike"] - (net_cost / (num_contracts * 100))
    return {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "long_strike": long_opt["strike"],
        "short_strike": short_opt["strike"],
        "strikes": [short_opt["strike"], long_opt["strike"]]
    }


def calculate_call_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_opt["strike"] < high_opt["strike"]):
        return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_price = get_strategy_price(mid_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if any(p is None for p in [low_price, mid_price, high_price]):
        return None

    # --- LOGIC FOR BALANCED/UNBALANCED BUTTERFLY ---
    gap1 = mid_opt["strike"] - low_opt["strike"]
    gap2 = high_opt["strike"] - mid_opt["strike"]
    if gap1 < MIN_GAP or gap2 < MIN_GAP: return None

    scale = 100
    g = gcd(int(gap1 * scale), int(gap2 * scale)) / scale
    if g < MIN_GAP: g = MIN_GAP

    gap1_units = round(gap1 / g)
    gap2_units = round(gap2 / g)

    low_contracts = num_contracts * gap2_units
    mid_contracts = -num_contracts * (gap1_units + gap2_units)
    high_contracts = num_contracts * gap1_units

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (
                           low_price * low_contracts + mid_price * mid_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat

    if net_cost <= 0: return None

    max_profit = (mid_opt["strike"] - low_opt["strike"]) * abs(low_contracts) * 100 - net_cost
    max_loss = net_cost

    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts} : {mid_contracts} : {high_contracts}",
        "strikes": [low_opt["strike"], mid_opt["strike"], high_opt["strike"]],
        # --- FIX: Pass the contract ratios to the visualizer ---
        "contract_ratios": [low_contracts, mid_contracts, high_contracts]
    }
    return result if max_profit > 0 else None


# REPLACE this function in utils.py
def calculate_put_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_opt["strike"] < high_opt["strike"]):
        return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_price = get_strategy_price(mid_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if any(p is None for p in [low_price, mid_price, high_price]):
        return None

    # --- LOGIC FOR BALANCED/UNBALANCED BUTTERFLY (Copied from Call version) ---
    gap1 = mid_opt["strike"] - low_opt["strike"]
    gap2 = high_opt["strike"] - mid_opt["strike"]
    if gap1 < MIN_GAP or gap2 < MIN_GAP: return None

    scale = 100
    g = gcd(int(gap1 * scale), int(gap2 * scale)) / scale
    if g < MIN_GAP: g = MIN_GAP

    gap1_units = round(gap1 / g)
    gap2_units = round(gap2 / g)

    low_contracts = num_contracts * gap2_units
    mid_contracts = -num_contracts * (gap1_units + gap2_units)
    high_contracts = num_contracts * gap1_units

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (
                           low_price * low_contracts + mid_price * mid_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat

    if net_cost <= 0: return None

    # Note: Max profit calculation for put butterfly is slightly different
    max_profit = (mid_opt["strike"] - low_opt["strike"]) * abs(low_contracts) * 100 - net_cost
    max_loss = net_cost

    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts} : {mid_contracts} : {high_contracts}",
        "strikes": [low_opt["strike"], mid_opt["strike"], high_opt["strike"]],
        # --- FIX: Pass the contract ratios to the visualizer ---
        "contract_ratios": [low_contracts, mid_contracts, high_contracts]
    }
    return result if max_profit > 0 else None
def lcm(a, b):
    return abs(a * b) / gcd(int(a * 100), int(b * 100)) * 100 if a and b else 0


def calculate_call_condor(low_opt, mid_low_opt, mid_high_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_low_opt["strike"] < mid_high_opt["strike"] < high_opt["strike"]):
        return None
    prices = [get_strategy_price(opt, action) for opt, action in
              zip([low_opt, mid_low_opt, mid_high_opt, high_opt], ["buy", "sell", "sell", "buy"])]
    if any(p is None for p in prices): return None

    gap1 = mid_low_opt["strike"] - low_opt["strike"]
    gap3 = high_opt["strike"] - mid_high_opt["strike"]
    if gap1 < MIN_GAP or gap3 < MIN_GAP: return None

    # --- FIX: Calculate contract ratios ---
    scale = 100
    g = gcd(int(gap1 * scale), int(gap3 * scale)) / scale
    if g < MIN_GAP: g = MIN_GAP

    low_units = round(gap3 / g)
    high_units = round(gap1 / g)

    ratios = [
        num_contracts * low_units,
        -num_contracts * low_units,
        -num_contracts * high_units,
        num_contracts * high_units
    ]

    base_cost = (prices[0] * abs(ratios[0]) + prices[3] * abs(ratios[3])) * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = sum(p * r for p, r in zip(prices, ratios)) * 100 + commission + market_fees + vat

    if net_cost <= 0: return None

    max_profit = (mid_low_opt["strike"] - low_opt["strike"]) * abs(ratios[0]) * 100 - net_cost
    max_loss = net_cost

    result = {
        "max_profit": max(0, max_profit), "net_cost": net_cost, "max_loss": max_loss,
        "contracts": f"{ratios[0]:.2f} : {ratios[1]:.2f} : {ratios[2]:.2f} : {ratios[3]:.2f}",
        "strikes": [low_opt["strike"], mid_low_opt["strike"], mid_high_opt["strike"], high_opt["strike"]],
        "contract_ratios": ratios
    }
    return result if max_profit > 0 else None


# REPLACE this function in utils.py
def calculate_put_condor(low_opt, mid_low_opt, mid_high_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_low_opt["strike"] < mid_high_opt["strike"] < high_opt["strike"]):
        return None
    prices = [get_strategy_price(opt, action) for opt, action in
              zip([low_opt, mid_low_opt, mid_high_opt, high_opt], ["buy", "sell", "sell", "buy"])]
    if any(p is None for p in prices): return None

    gap1 = mid_low_opt["strike"] - low_opt["strike"]
    gap3 = high_opt["strike"] - mid_high_opt["strike"]
    if gap1 < MIN_GAP or gap3 < MIN_GAP: return None

    # --- FIX: Calculate contract ratios ---
    scale = 100
    g = gcd(int(gap1 * scale), int(gap3 * scale)) / scale
    if g < MIN_GAP: g = MIN_GAP

    low_units = round(gap3 / g)
    high_units = round(gap1 / g)

    ratios = [
        num_contracts * low_units,
        -num_contracts * low_units,
        -num_contracts * high_units,
        num_contracts * high_units
    ]

    base_cost = (prices[0] * abs(ratios[0]) + prices[3] * abs(ratios[3])) * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = sum(p * r for p, r in zip(prices, ratios)) * 100 + commission + market_fees + vat

    if net_cost <= 0: return None

    max_profit = (mid_low_opt["strike"] - low_opt["strike"]) * abs(ratios[0]) * 100 - net_cost
    max_loss = net_cost

    result = {
        "max_profit": max(0, max_profit), "net_cost": net_cost, "max_loss": max_loss,
        "contracts": f"{ratios[0]:.2f} : {ratios[1]:.2f} : {ratios[2]:.2f} : {ratios[3]:.2f}",
        "strikes": [low_opt["strike"], mid_low_opt["strike"], mid_high_opt["strike"], high_opt["strike"]],
        "contract_ratios": ratios
    }
    return result if max_profit > 0 else None
def calculate_straddle(call_opt, put_opt, num_contracts, commission_rate):
    if call_opt["strike"] != put_opt["strike"]:
        return None
    call_price = get_strategy_price(call_opt, "buy")
    put_price = get_strategy_price(put_opt, "buy")
    if call_price is None or put_price is None:
        return None
    base_cost = (call_price + put_price) * num_contracts * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = base_cost + commission + market_fees + vat
    max_loss = net_cost
    strike = call_opt["strike"]
    net_premium = net_cost / (num_contracts * 100)
    breakeven_upper = strike + net_premium
    breakeven_lower = strike - net_premium
    return {
        "max_profit": "Unlimited",
        "net_cost": net_cost,
        "max_loss": max_loss,
        "breakeven_upper": breakeven_upper,
        "breakeven_lower": breakeven_lower,
        "strikes": [strike]
    }

def calculate_strangle(call_opt, put_opt, num_contracts, commission_rate):
    if call_opt["strike"] <= put_opt["strike"]:
        return None
    call_price = get_strategy_price(call_opt, "buy")
    put_price = get_strategy_price(put_opt, "buy")
    if call_price is None or put_price is None:
        return None
    base_cost = (call_price + put_price) * num_contracts * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = base_cost + commission + market_fees + vat
    max_loss = net_cost
    net_premium = net_cost / (num_contracts * 100)
    breakeven_upper = call_opt["strike"] + net_premium
    breakeven_lower = put_opt["strike"] - net_premium
    return {
        "max_profit": "Unlimited",
        "net_cost": net_cost,
        "max_loss": max_loss,
        "breakeven_upper": breakeven_upper,
        "breakeven_lower": breakeven_lower,
        "strikes": [put_opt["strike"], call_opt["strike"]]
    }

def create_spread_matrix(options: list, strategy_func, num_contracts: int, commission_rate: float, is_bullish: bool):
    strikes = sorted(options, key=lambda x: x["strike"])
    profit_matrix, cost_matrix, ratio_matrix, breakeven_matrix = [], [], [], []
    for long_opt in strikes:
        profit_row, cost_row, ratio_row, breakeven_row = [], [], [], []
        for short_opt in strikes:
            if "bull_call" in strategy_func.__name__ and long_opt["strike"] < short_opt["strike"]:
                result = strategy_func(long_opt, short_opt, num_contracts, commission_rate)
            elif "bull_put" in strategy_func.__name__ and long_opt["strike"] < short_opt["strike"]:
                result = strategy_func(long_opt, short_opt, num_contracts, commission_rate)
            elif "bear_call" in strategy_func.__name__ and long_opt["strike"] > short_opt["strike"]:
                result = strategy_func(long_opt, short_opt, num_contracts, commission_rate)
            elif "bear_put" in strategy_func.__name__ and long_opt["strike"] > short_opt["strike"]:
                result = strategy_func(long_opt, short_opt, num_contracts, commission_rate)
            else:
                result = None
            if result:
                profit_row.append(result["max_profit"])
                cost_row.append(result["net_cost"])
                breakeven_row.append(result.get("breakeven", np.nan))
                if is_bullish:
                    ratio = result["net_cost"] / result["max_profit"] if result["max_profit"] > 0 else float('inf')
                else:
                    ratio = -result["net_cost"] / result["max_loss"] if result["net_cost"] < 0 and result["max_loss"] > 0 else float('inf')
                ratio_row.append(ratio)
            else:
                profit_row.append(np.nan)
                cost_row.append(np.nan)
                ratio_row.append(np.nan)
                breakeven_row.append(np.nan)
        profit_matrix.append(profit_row)
        cost_matrix.append(cost_row)
        ratio_matrix.append(ratio_row)
        breakeven_matrix.append(breakeven_row)
    strike_labels = [f"{s['strike']:.2f}" for s in strikes]
    return (
        pd.DataFrame(profit_matrix, columns=strike_labels, index=strike_labels),
        pd.DataFrame(cost_matrix, columns=strike_labels, index=strike_labels),
        pd.DataFrame(ratio_matrix, columns=strike_labels, index=strike_labels),
        pd.DataFrame(breakeven_matrix, columns=strike_labels, index=strike_labels)
    )

def create_complex_strategy_table(options: list, strategy_func, num_contracts: int, commission_rate: float, combo_size: int) -> pd.DataFrame:
    strikes = sorted(options, key=lambda x: x["strike"])
    combos = list(combinations(strikes, combo_size))
    data = []
    for combo in combos:
        if all(combo[i]["strike"] < combo[i + 1]["strike"] for i in range(len(combo) - 1)):
            result = strategy_func(*combo, num_contracts, commission_rate)
            if result and result.get("max_profit", 0) > 0:
                ratio = result["net_cost"] / result["max_profit"] if result["max_profit"] > 0 else float('inf')
                entry = {
                    "Strikes": " - ".join(f"{opt['strike']:.2f}" for opt in combo),
                    "Net Cost": result["net_cost"],
                    "Max Profit": result["max_profit"],
                    "Max Loss": result["max_loss"],
                    "Cost-to-Profit Ratio": ratio,
                    "Contracts": result.get("contracts", "N/A"),
                    "strikes": result["strikes"],
                    # --- FIX: Add contract ratios to the DataFrame row for ALL complex strategies ---
                    "contract_ratios": result.get("contract_ratios")
                }
                data.append(entry)
    return pd.DataFrame(data)
def create_vol_strategy_table(calls: list, puts: list, strategy_func, num_contracts: int, commission_rate: float):
    data = []
    call_strikes = sorted(set(o["strike"] for o in calls))
    put_strikes = sorted(set(o["strike"] for o in puts))
    if strategy_func == calculate_straddle:
        for strike in call_strikes:
            if strike in put_strikes:
                call = next(o for o in calls if o["strike"] == strike)
                put = next(o for o in puts if o["strike"] == strike)
                result = strategy_func(call, put, num_contracts, commission_rate)
                if result:
                    ratio = result["net_cost"] / result["max_loss"]
                    data.append({
                        "Strikes": f"{strike:.2f} (Call & Put)",
                        "Net Cost": result["net_cost"],
                        "Max Profit": result["max_profit"],
                        "Max Loss": result["max_loss"],
                        "Breakeven Upper": result["breakeven_upper"],
                        "Breakeven Lower": result["breakeven_lower"],
                        "Cost-to-Loss Ratio": ratio,
                        "strikes": result["strikes"]
                    })
    elif strategy_func == calculate_strangle:
        for call_strike in call_strikes:
            for put_strike in put_strikes:
                if call_strike > put_strike:
                    call = next(o for o in calls if o["strike"] == call_strike)
                    put = next(o for o in puts if o["strike"] == put_strike)
                    result = strategy_func(call, put, num_contracts, commission_rate)
                    if result:
                        ratio = result["net_cost"] / result["max_loss"]
                        data.append({
                            "Strikes": f"Put {put_strike:.2f} - Call {call_strike:.2f}",
                            "Net Cost": result["net_cost"],
                            "Max Profit": result["max_profit"],
                            "Max Loss": result["max_loss"],
                            "Breakeven Upper": result["breakeven_upper"],
                            "Breakeven Lower": result["breakeven_lower"],
                            "Cost-to-Loss Ratio": ratio,
                            "strikes": result["strikes"]
                        })
    return pd.DataFrame(data)
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
        if T <= 1e-9: return max(0, S - K) if option_type == "call" else max(0, K - S)
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

            if "butterfly" in strategy_key:
                ratios = strategy_result.get("contract_ratios", [1, -2, 1])
                low_k, mid_k, high_k = strikes
                opt_type = "call" if "call" in strategy_key else "put"
                vals = [black_scholes(price, k, T, r, iv, opt_type) for k in [low_k, mid_k, high_k]]
                position_value = (ratios[0] * vals[0] + ratios[1] * vals[1] + ratios[2] * vals[2]) * 100
            # --- FIX: Add corrected logic for condor visualization ---
            elif "condor" in strategy_key:
                ratios = strategy_result.get("contract_ratios", [1, -1, -1, 1])
                k1, k2, k3, k4 = strikes
                opt_type = "call" if "call" in strategy_key else "put"
                vals = [black_scholes(price, k, T, r, iv, opt_type) for k in strikes]
                position_value = (ratios[0] * vals[0] + ratios[1] * vals[1] + ratios[2] * vals[2] + ratios[3] * vals[
                    3]) * 100
            elif "bull call spread" in strategy_key or "bear call spread" in strategy_key:
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