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

    # Isolate the part of the symbol that contains the strike and expiration
    data_part = symbol[4:]
    
    # Find the first letter of the expiration code to separate numbers from letters
    first_letter_index = -1
    for i, char in enumerate(data_part):
        if char.isalpha():
            first_letter_index = i
            break
    
    if first_letter_index == -1:
        logger.error(f"Could not find expiration suffix in symbol: {symbol}")
        return None, None, None

    numeric_part = data_part[:first_letter_index]
    suffix = data_part[first_letter_index:]

    if not numeric_part:
        return None, None, None

    try:
        # --- NEW CORRECTED LOGIC ---
        # This logic correctly handles the special convention for Argentinian options.
        
        # Convention for strikes < 10000 with decimals (e.g., "38785" -> 3878.5)
        # This also works for strikes >= 10000 with decimals (e.g., a hypothetical "123456" -> 12345.6)
        if len(numeric_part) == 5 and not numeric_part.startswith('1'):
             strike_price = float(numeric_part) / 10.0
        elif len(numeric_part) > 5:
             strike_price = float(numeric_part) / 10.0
        # Convention for whole number strikes (e.g., "4200" -> 4200.0 or "10079" -> 10079.0)
        else:
             strike_price = float(numeric_part)

    except ValueError:
        logger.error(f"Failed to parse strike from symbol: {symbol}")
        return None, None, None

    expiration = EXPIRATION_MAP_2025.get(suffix, None)
    return option_type, strike_price, expiration
def get_ggal_data() -> tuple[dict | None, list]:
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)
    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock:
        st.error("No se encontraron datos de la acción GGAL.")
        logger.error("GGAL stock data not found")
        return None, []
    ggal_options = []
    for o in options_data:
        opt_type, strike, exp = parse_option_symbol(o["symbol"])
        if opt_type and strike is not None and exp and o.get("px_ask", 0) > 0 and o.get("px_bid", 0) > 0:
            ggal_options.append({
                "symbol": o["symbol"],
                "type": opt_type,
                "strike": strike,
                "expiration": exp,
                "px_bid": o["px_bid"],
                "px_ask": o["px_ask"],
                "c": o["c"]
            })
    logger.info(f"Fetched {len(ggal_options)} options")
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

    gap1 = mid_opt["strike"] - low_opt["strike"]
    gap2 = high_opt["strike"] - mid_opt["strike"]
    if gap1 < MIN_GAP or gap2 < MIN_GAP:
        logger.warning(f"Invalid gaps: gap1={gap1}, gap2={gap2} for strikes {low_opt['strike']}, {mid_opt['strike']}, {high_opt['strike']}")
        return None

    # Scale for gcd to handle decimals
    scale = 100
    g = gcd(int(gap1 * scale), int(gap2 * scale)) / scale
    if g < MIN_GAP:
        g = MIN_GAP  # Prevent zero division
    gap1_units = gap1 / g
    gap2_units = gap2 / g

    if abs(gap1 - gap2) < MIN_GAP:
        low_contracts = num_contracts
        mid_contracts = -2 * num_contracts
        high_contracts = num_contracts
        min_contracts = 1
    else:
        low_contracts = num_contracts * int(gap2_units)
        mid_contracts = -num_contracts * int(gap1_units + gap2_units)
        high_contracts = num_contracts * int(gap1_units)
        min_contracts = int(gcd(int(gap1_units), int(gap2_units)))

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (low_price * low_contracts + mid_price * mid_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (mid_opt["strike"] - low_opt["strike"]) * abs(low_contracts) * 100 - net_cost
    max_loss = net_cost
    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts} : {mid_contracts} : {high_contracts}",
        "min_contracts": min_contracts,
        "strikes": [low_opt["strike"], mid_opt["strike"], high_opt["strike"]]
    }
    return result if max_profit > 0 else None

def calculate_put_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_opt["strike"] < high_opt["strike"]):
        return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_price = get_strategy_price(mid_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if any(p is None for p in [low_price, mid_price, high_price]):
        return None

    gap1 = mid_opt["strike"] - low_opt["strike"]
    gap2 = high_opt["strike"] - mid_opt["strike"]
    if gap1 < MIN_GAP or gap2 < MIN_GAP:
        logger.warning(f"Invalid gaps: gap1={gap1}, gap2={gap2} for strikes {low_opt['strike']}, {mid_opt['strike']}, {high_opt['strike']}")
        return None

    scale = 100
    g = gcd(int(gap1 * scale), int(gap2 * scale)) / scale
    if g < MIN_GAP:
        g = MIN_GAP
    gap1_units = gap1 / g
    gap2_units = gap2 / g

    if abs(gap1 - gap2) < MIN_GAP:
        low_contracts = num_contracts
        mid_contracts = -2 * num_contracts
        high_contracts = num_contracts
        min_contracts = 1
    else:
        low_contracts = num_contracts * int(gap2_units)
        mid_contracts = -num_contracts * int(gap1_units + gap2_units)
        high_contracts = num_contracts * int(gap1_units)
        min_contracts = int(gcd(int(gap1_units), int(gap2_units)))

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (low_price * low_contracts + mid_price * mid_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (high_opt["strike"] - mid_opt["strike"]) * abs(high_contracts) * 100 - net_cost
    max_loss = net_cost
    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts} : {mid_contracts} : {high_contracts}",
        "min_contracts": min_contracts,
        "strikes": [low_opt["strike"], mid_opt["strike"], high_opt["strike"]]
    }
    return result if max_profit > 0 else None

def lcm(a, b):
    return abs(a * b) / gcd(int(a * 100), int(b * 100)) * 100 if a and b else 0

def calculate_call_condor(low_opt, mid_low_opt, mid_high_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_low_opt["strike"] < mid_high_opt["strike"] < high_opt["strike"]):
        return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_low_price = get_strategy_price(mid_low_opt, "sell")
    mid_high_price = get_strategy_price(mid_high_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if any(p is None for p in [low_price, mid_low_price, mid_high_price, high_price]):
        return None

    gap1 = mid_low_opt["strike"] - low_opt["strike"]
    gap2 = mid_high_opt["strike"] - mid_low_opt["strike"]
    gap3 = high_opt["strike"] - mid_high_opt["strike"]
    if any(g < MIN_GAP for g in [gap1, gap2, gap3]):
        logger.warning(f"Invalid gaps: gap1={gap1}, gap2={gap2}, gap3={gap3}")
        return None

    gaps = [gap1, gap2, gap3]
    scale = 100
    lcm_gaps = reduce(lcm, gaps) if all(gaps) else 1
    low_contracts = num_contracts * (lcm_gaps / gap1)
    mid_low_contracts = -num_contracts * (lcm_gaps / gap1)
    mid_high_contracts = -num_contracts * (lcm_gaps / gap3)
    high_contracts = num_contracts * (lcm_gaps / gap3)
    min_contracts = lcm_gaps / min(gaps) if gaps else 1

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (low_price * low_contracts + mid_low_price * mid_low_contracts + mid_high_price * mid_high_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (mid_high_opt["strike"] - mid_low_opt["strike"]) * abs(mid_low_contracts) * 100 - net_cost
    max_loss = net_cost
    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts:.2f} : {mid_low_contracts:.2f} : {mid_high_contracts:.2f} : {high_contracts:.2f}",
        "min_contracts": min_contracts,
        "strikes": [low_opt["strike"], mid_low_opt["strike"], mid_high_opt["strike"], high_opt["strike"]]
    }
    return result if max_profit > 0 else None

def calculate_put_condor(low_opt, mid_low_opt, mid_high_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_low_opt["strike"] < mid_high_opt["strike"] < high_opt["strike"]):
        return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_low_price = get_strategy_price(mid_low_opt, "sell")
    mid_high_price = get_strategy_price(mid_high_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if any(p is None for p in [low_price, mid_low_price, mid_high_price, high_price]):
        return None

    gap1 = mid_low_opt["strike"] - low_opt["strike"]
    gap2 = mid_high_opt["strike"] - mid_low_opt["strike"]
    gap3 = high_opt["strike"] - mid_high_opt["strike"]
    if any(g < MIN_GAP for g in [gap1, gap2, gap3]):
        logger.warning(f"Invalid gaps: gap1={gap1}, gap2={gap2}, gap3={gap3}")
        return None

    gaps = [gap1, gap2, gap3]
    scale = 100
    lcm_gaps = reduce(lcm, gaps) if all(gaps) else 1
    low_contracts = num_contracts * (lcm_gaps / gap1)
    mid_low_contracts = -num_contracts * (lcm_gaps / gap1)
    mid_high_contracts = -num_contracts * (lcm_gaps / gap3)
    high_contracts = num_contracts * (lcm_gaps / gap3)
    min_contracts = lcm_gaps / min(gaps) if gaps else 1

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (low_price * low_contracts + mid_low_price * mid_low_contracts + mid_high_price * mid_high_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (mid_high_opt["strike"] - mid_low_opt["strike"]) * abs(mid_low_contracts) * 100 - net_cost
    max_loss = net_cost
    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts:.2f} : {mid_low_contracts:.2f} : {mid_high_contracts:.2f} : {high_contracts:.2f}",
        "min_contracts": min_contracts,
        "strikes": [low_opt["strike"], mid_low_opt["strike"], mid_high_opt["strike"], high_opt["strike"]]
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
            if result and result["max_profit"] > 0:
                ratio = result["net_cost"] / result["max_profit"]
                entry = {
                    "Strikes": " - ".join(f"{opt['strike']:.2f}" for opt in combo),
                    "Net Cost": result["net_cost"],
                    "Max Profit": result["max_profit"],
                    "Max Loss": result["max_loss"],
                    "Cost-to-Profit Ratio": ratio,
                    "Contracts": result["contracts"],
                    "strikes": result["strikes"]
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

def visualize_3d_payoff(strategy_result, current_price, expiration_days, iv=DEFAULT_IV, key=None):
    if not strategy_result or "strikes" not in strategy_result:
        st.warning("No valid strategy selected for visualization.")
        logger.warning("No strikes in strategy_result")
        return

    # --- FIX: Use the correct key "Net Cost" from the DataFrame ---
    net_cost = strategy_result.get("Net Cost", strategy_result.get("net_cost", 0))
    
    if net_cost is None:
        st.error("Could not retrieve net cost for visualization.")
        return

    # Black-Scholes option pricing model
    def black_scholes(S, K, T, r, sigma, option_type="call"):
        if T <= 1e-6: # Handle expiration day
            return max(0, S - K) if option_type == "call" else max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else: # put
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    prices = np.linspace(max(0.1, current_price * 0.5), current_price * 1.5, 50)
    times = np.linspace(0, expiration_days, 20)
    X, Y = np.meshgrid(prices, times)
    Z = np.zeros_like(X)

    strikes = strategy_result["strikes"]
    strategy_key = key.lower() if key else ""
    r = 0.05  # Assumed risk-free rate

    # Calculate P/L for each point in time and price using Black-Scholes
    for i in range(len(times)):
        for j in range(len(prices)):
            price = X[j, i]
            time_to_exp_days = expiration_days - Y[j, i]
            T = time_to_exp_days / 365.0
            
            position_value = 0.0

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

            # For credit spreads, net_cost is negative (a credit received)
            # For debit spreads, net_cost is positive (a debit paid)
            # The P/L is always the current value minus the initial cost.
            Z[j, i] = position_value - net_cost

    # Create the main payoff surface
    payoff_surface = go.Surface(z=Z, x=X, y=Y, colorscale='RdYlGn', cmin=Z.min(), cmax=Z.max(), colorbar=dict(title='Profit/Loss'))
    
    # Create a transparent plane at z=0 for breakeven
    breakeven_plane = go.Surface(z=np.zeros_like(X), x=X, y=Y, opacity=0.7, showscale=False, colorscale=[[0, '#0000FF'], [1, '#0000FF']])

    fig = go.Figure(data=[payoff_surface, breakeven_plane])
    fig.update_layout(
        title=f"Strategy Value Over Time: {key.replace('_', ' ').title()}",
        scene=dict(xaxis_title='Underlying Price (ARS)', yaxis_title='Days Elapsed', zaxis_title='Profit/Loss (ARS)')
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key}")
def display_spread_matrix(tab, strategy_name, options, strategy_func, is_bullish):
    with tab:
        st.subheader(f"Matriz de {strategy_name}")
        st.write(f"""
        **¿Qué es {strategy_name}?**  
        - **Bull Call Spread**: Estrategia alcista con calls. Compra call baja, vende call alta.
        - **Bull Put Spread**: Estrategia alcista con puts. Vende put alta, compra put baja.
        - **Bear Call Spread**: Estrategia bajista con calls. Vende call baja, compra call alta.
        - **Bear Put Spread**: Estrategia bajista con puts. Compra put alta, vende put baja.
        """)

        if is_bullish:
            filter_ratio = st.slider("Relación máxima de costo a ganancia (%)", 0.0, 500.0, 50.0, key=f"filter_{strategy_name}") / 100
            label = "Relación costo a ganancia"
        else:
            filter_ratio = st.slider("Relación mínima de crédito a pérdida (%)", 0.0, 100.0, 50.0, key=f"filter_{strategy_name}") / 100
            label = "Relación crédito a pérdida"

        profit_df, cost_df, ratio_df, breakeven_df = create_spread_matrix(options, strategy_func, st.session_state.num_contracts, st.session_state.commission_rate, is_bullish)
        
        mask = ratio_df <= filter_ratio if is_bullish else ratio_df >= filter_ratio
        filtered_profit_df = profit_df.where(mask, np.nan)

        st.write("**Matriz de ganancia máxima (ARS)**")
        st.dataframe(filtered_profit_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn'))

        # --- NEW VISUALIZATION UI ADDED BELOW ---
        st.subheader("Visualización 3D para una combinación específica")
        
        strike_prices = sorted([opt['strike'] for opt in options])
        
        col1, col2 = st.columns(2)
        with col1:
            long_strike_price = st.selectbox(
                "Selecciona el strike de COMPRA (Long)", 
                strike_prices, 
                key=f"long_select_{strategy_name}"
            )
        with col2:
            short_strike_price = st.selectbox(
                "Selecciona el strike de VENTA (Short)", 
                strike_prices, 
                key=f"short_select_{strategy_name}"
            )

        if long_strike_price and short_strike_price:
            long_opt = next((opt for opt in options if opt['strike'] == long_strike_price), None)
            short_opt = next((opt for opt in options if opt['strike'] == short_strike_price), None)

            if long_opt and short_opt:
                # The strategy function itself contains the logic for valid strike combinations
                result = strategy_func(long_opt, short_opt, st.session_state.num_contracts, st.session_state.commission_rate)
                
                if result:
                    st.success("Combinación válida encontrada. Generando gráfico...")
                    expiration_days = max(1, (st.session_state.selected_exp - date.today()).days)
                    # Pass the strategy_name to the key so the visualization function knows what to draw
                    visualize_3d_payoff(result, st.session_state.current_price, expiration_days, st.session_state.iv, key=strategy_name)
                else:
                    st.warning("La combinación de strikes seleccionada no es válida para esta estrategia (ej: en un Bull Call, el strike de compra debe ser menor que el de venta).")

def display_complex_strategy(tab, strategy_name, options, strategy_func, combo_size):
    with tab:
        st.subheader(f"Análisis de {strategy_name}")
        st.write(f"""
        **¿Qué es {strategy_name}?**  
        - **Call Butterfly**: Compra call baja y alta, vende 2 calls media.
        - **Put Butterfly**: Similar con puts.
        - **Call Condor**: Compra call baja y alta, vende calls media baja y media alta.
        - **Put Condor**: Similar con puts.
        """)

        filter_ratio = st.slider("Relación máxima de costo a ganancia (%)", 0.0, 500.0, 50.0, key=f"filter_{strategy_name}") / 100
        df = create_complex_strategy_table(options, strategy_func, st.session_state.num_contracts, st.session_state.commission_rate, combo_size)
        if not df.empty:
            filtered_df = df[df["Cost-to-Profit Ratio"] <= filter_ratio]
            st.dataframe(filtered_df.style.format({"Net Cost": "{:.2f}", "Max Profit": "{:.2f}", "Max Loss": "{:.2f}", "Cost-to-Profit Ratio": "{:.2f}"}))
            selected_row = st.selectbox("Selecciona una combinación para visualizar en 3D", filtered_df.index, key=f"select_{strategy_name}")
            if selected_row is not None:
                result = filtered_df.iloc[selected_row].to_dict()
                expiration_days = max(1, (st.session_state.selected_exp - date.today()).days)
                visualize_3d_payoff(result, st.session_state.current_price, expiration_days, st.session_state.iv, key=f"chart_{strategy_name}")
        else:
            st.write("No se encontraron combinaciones válidas.")

def display_vol_strategy(tab, strategy_name, calls, puts, strategy_func):
    with tab:
        st.subheader(f"Análisis de {strategy_name}")
        st.write(f"""
        **¿Qué es {strategy_name}?**  
        - **Straddle**: Compra call y put al mismo strike. Profita de volatilidad.
        - **Strangle**: Compra call alta y put baja. Similar, pero rango más amplio.
        """)

        filter_ratio = st.slider("Relación máxima de costo a pérdida (%)", 0.0, 500.0, 50.0, key=f"filter_{strategy_name}") / 100
        df = create_vol_strategy_table(calls, puts, strategy_func, st.session_state.num_contracts, st.session_state.commission_rate)
        if not df.empty:
            filtered_df = df[df["Cost-to-Loss Ratio"] <= filter_ratio]
            st.dataframe(filtered_df.style.format({"Net Cost": "{:.2f}", "Max Loss": "{:.2f}", "Breakeven Upper": "{:.2f}", "Breakeven Lower": "{:.2f}", "Cost-to-Loss Ratio": "{:.2f}"}))
            selected_row = st.selectbox("Selecciona una combinación para visualizar en 3D", filtered_df.index, key=f"select_vol_{strategy_name}")
            if selected_row is not None:
                result = filtered_df.iloc[selected_row].to_dict()
                expiration_days = max(1, (st.session_state.selected_exp - date.today()).days)
                visualize_3d_payoff(result, st.session_state.current_price, expiration_days, st.session_state.iv, key=f"chart_vol_{strategy_name}")
        else:
            st.write("No se encontraron combinaciones válidas.")

def main():
    st.title("Analizador de Estrategias de Opciones para GGAL (Mejorado)")
    st.write("¡Bienvenido! Analiza estrategias para GGAL con straddles, strangles y visualización 3D.")

    if 'ggal_stock' not in st.session_state or 'ggal_options' not in st.session_state:
        with st.spinner("Cargando datos..."):
            st.session_state['ggal_stock'], st.session_state['ggal_options'] = get_ggal_data()
            st.session_state['last_updated'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if st.button("Actualizar"):
        with st.spinner("Actualizando..."):
            st.session_state['ggal_stock'], st.session_state['ggal_options'] = get_ggal_data()
            st.session_state['last_updated'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.success("Datos actualizados.")

    ggal_stock = st.session_state['ggal_stock']
    ggal_options = st.session_state['ggal_options']
    last_updated = st.session_state['last_updated']

    if not ggal_stock or not ggal_options:
        return

    st.session_state.current_price = float(ggal_stock["c"])
    st.write(f"**Precio actual de GGAL:** {st.session_state.current_price:.2f} ARS")
    st.write(f"**Última actualización:** {last_updated}")

    expirations = sorted(list(set(o["expiration"] for o in ggal_options if o["expiration"] is not None)))
    st.session_state.selected_exp = st.selectbox(
        "Selecciona la fecha de vencimiento",
        expirations,
        format_func=lambda x: x.strftime("%Y-%m-%d")
    )

    st.session_state.num_contracts = st.number_input("Número de contratos", min_value=1, value=1, step=1)
    st.session_state.commission_rate = st.number_input("Porcentaje de comisión (%)", min_value=0.0, value=0.5, step=0.1) / 100
    st.session_state.iv = st.number_input("Volatilidad implícita (%) para simulación", min_value=0.0, value=DEFAULT_IV * 100, step=1.0) / 100
    strike_percentage = st.slider("Rango de precios de ejercicio (% del precio actual)", 0.0, 100.0, 20.0) / 100

    min_strike = st.session_state.current_price * (1 - strike_percentage)
    max_strike = st.session_state.current_price * (1 + strike_percentage)

    calls = [o for o in ggal_options if o["type"] == "call" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike]
    puts = [o for o in ggal_options if o["type"] == "put" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike]

    if not calls or not puts:
        st.warning("No hay opciones dentro del rango seleccionado.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Bull Call Spread",
        "Bull Put Spread",
        "Bear Call Spread",
        "Bear Put Spread",
        "Butterfly & Condor",
        "Straddle & Strangle"
    ])
    st.session_state.disable_filter = st.checkbox("Desactivar filtro para mostrar todas las estrategias", value=False)

    display_spread_matrix(tab1, "Bull Call Spread", calls, calculate_bull_call_spread, True)
    display_spread_matrix(tab2, "Bull Put Spread", puts, calculate_bull_put_spread, False)
    display_spread_matrix(tab3, "Bear Call Spread", calls, calculate_bear_call_spread, False)
    display_spread_matrix(tab4, "Bear Put Spread", puts, calculate_bear_put_spread, True)

    display_complex_strategy(tab5, "Call Butterfly", calls, calculate_call_butterfly, 3)
    display_complex_strategy(tab5, "Put Butterfly", puts, calculate_put_butterfly, 3)
    display_complex_strategy(tab5, "Call Condor", calls, calculate_call_condor, 4)
    display_complex_strategy(tab5, "Put Condor", puts, calculate_put_condor, 4)

    display_vol_strategy(tab6, "Straddle", calls, puts, calculate_straddle)
    display_vol_strategy(tab6, "Strangle", calls, puts, calculate_strangle)

if __name__ == "__main__":
    main()
