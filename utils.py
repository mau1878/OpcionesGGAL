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

# --- SPREAD CALCULATIONS ---
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
        "strikes": [long_opt["strike"], short_opt["strike"]],
        "num_contracts": num_contracts,
        "breakeven": breakeven
    }

def calculate_call_butterfly(opt1, opt2, opt3, num_contracts, commission_rate):
    if not (opt1["strike"] < opt2["strike"] < opt3["strike"]): return None
    buy1_price = get_strategy_price(opt1, "buy")
    sell2_price = get_strategy_price(opt2, "sell")
    buy3_price = get_strategy_price(opt3, "buy")
    if any(p is None for p in [buy1_price, sell2_price, buy3_price]): return None

    buy1_base = buy1_price * 100 * num_contracts
    sell2_base = sell2_price * 100 * num_contracts * 2
    buy3_base = buy3_price * 100 * num_contracts
    base_cost = (buy1_price + buy3_price - 2 * sell2_price) * 100 * num_contracts
    total_fees = calculate_fees(buy1_base, commission_rate) + calculate_fees(sell2_base, commission_rate) + calculate_fees(buy3_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    max_profit = (opt2["strike"] - opt1["strike"]) * 100 * num_contracts - net_cost
    lower_breakeven = opt1["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = opt3["strike"] - (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "strikes": [opt1["strike"], opt2["strike"], opt3["strike"]],
        "num_contracts": num_contracts,
        "contract_ratios": [1, -2, 1],
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven
    }

def calculate_put_butterfly(opt1, opt2, opt3, num_contracts, commission_rate):
    if not (opt1["strike"] < opt2["strike"] < opt3["strike"]): return None
    buy1_price = get_strategy_price(opt1, "buy")
    sell2_price = get_strategy_price(opt2, "sell")
    buy3_price = get_strategy_price(opt3, "buy")
    if any(p is None for p in [buy1_price, sell2_price, buy3_price]): return None

    buy1_base = buy1_price * 100 * num_contracts
    sell2_base = sell2_price * 100 * num_contracts * 2
    buy3_base = buy3_price * 100 * num_contracts
    base_cost = (buy1_price + buy3_price - 2 * sell2_price) * 100 * num_contracts
    total_fees = calculate_fees(buy1_base, commission_rate) + calculate_fees(sell2_base, commission_rate) + calculate_fees(buy3_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    max_profit = (opt3["strike"] - opt2["strike"]) * 100 * num_contracts - net_cost
    lower_breakeven = opt1["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = opt3["strike"] - (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "strikes": [opt1["strike"], opt2["strike"], opt3["strike"]],
        "num_contracts": num_contracts,
        "contract_ratios": [1, -2, 1],
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven
    }

def calculate_call_condor(opt1, opt2, opt3, opt4, num_contracts, commission_rate):
    if not (opt1["strike"] < opt2["strike"] < opt3["strike"] < opt4["strike"]): return None
    buy1_price = get_strategy_price(opt1, "buy")
    sell2_price = get_strategy_price(opt2, "sell")
    sell3_price = get_strategy_price(opt3, "sell")
    buy4_price = get_strategy_price(opt4, "buy")
    if any(p is None for p in [buy1_price, sell2_price, sell3_price, buy4_price]): return None

    buy1_base = buy1_price * 100 * num_contracts
    sell2_base = sell2_price * 100 * num_contracts
    sell3_base = sell3_price * 100 * num_contracts
    buy4_base = buy4_price * 100 * num_contracts
    base_cost = (buy1_price - sell2_price - sell3_price + buy4_price) * 100 * num_contracts
    total_fees = calculate_fees(buy1_base, commission_rate) + calculate_fees(sell2_base, commission_rate) + calculate_fees(sell3_base, commission_rate) + calculate_fees(buy4_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    max_profit = (opt3["strike"] - opt2["strike"]) * 100 * num_contracts - net_cost
    lower_breakeven = opt1["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = opt4["strike"] - (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "strikes": [opt1["strike"], opt2["strike"], opt3["strike"], opt4["strike"]],
        "num_contracts": num_contracts,
        "contract_ratios": [1, -1, -1, 1],
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven
    }

def calculate_put_condor(opt1, opt2, opt3, opt4, num_contracts, commission_rate):
    if not (opt1["strike"] < opt2["strike"] < opt3["strike"] < opt4["strike"]): return None
    buy1_price = get_strategy_price(opt1, "buy")
    sell2_price = get_strategy_price(opt2, "sell")
    sell3_price = get_strategy_price(opt3, "sell")
    buy4_price = get_strategy_price(opt4, "buy")
    if any(p is None for p in [buy1_price, sell2_price, sell3_price, buy4_price]): return None

    buy1_base = buy1_price * 100 * num_contracts
    sell2_base = sell2_price * 100 * num_contracts
    sell3_base = sell3_price * 100 * num_contracts
    buy4_base = buy4_price * 100 * num_contracts
    base_cost = (buy1_price - sell2_price - sell3_price + buy4_price) * 100 * num_contracts
    total_fees = calculate_fees(buy1_base, commission_rate) + calculate_fees(sell2_base, commission_rate) + calculate_fees(sell3_base, commission_rate) + calculate_fees(buy4_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    max_profit = (opt3["strike"] - opt2["strike"]) * 100 * num_contracts - net_cost
    lower_breakeven = opt1["strike"] + (net_cost / (100 * num_contracts))
    upper_breakeven = opt4["strike"] - (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "strikes": [opt1["strike"], opt2["strike"], opt3["strike"], opt4["strike"]],
        "num_contracts": num_contracts,
        "contract_ratios": [1, -1, -1, 1],
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven
    }

def calculate_straddle(call_opt, put_opt, num_contracts, commission_rate):
    if call_opt["strike"] != put_opt["strike"]: return None
    call_price = get_strategy_price(call_opt, "buy")
    put_price = get_strategy_price(put_opt, "buy")
    if any(p is None for p in [call_price, put_price]): return None

    call_base = call_price * 100 * num_contracts
    put_base = put_price * 100 * num_contracts
    base_cost = (call_price + put_price) * 100 * num_contracts
    total_fees = calculate_fees(call_base, commission_rate) + calculate_fees(put_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    lower_breakeven = call_opt["strike"] - (net_cost / (100 * num_contracts))
    upper_breakeven = call_opt["strike"] + (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "strikes": [call_opt["strike"]],
        "num_contracts": num_contracts,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven
    }

def calculate_strangle(put_opt, call_opt, num_contracts, commission_rate):
    if put_opt["strike"] >= call_opt["strike"]: return None
    put_price = get_strategy_price(put_opt, "buy")
    call_price = get_strategy_price(call_opt, "buy")
    if any(p is None for p in [put_price, call_price]): return None

    put_base = put_price * 100 * num_contracts
    call_base = call_price * 100 * num_contracts
    base_cost = (put_price + call_price) * 100 * num_contracts
    total_fees = calculate_fees(put_base, commission_rate) + calculate_fees(call_base, commission_rate)
    net_cost = base_cost + total_fees

    max_loss = net_cost
    lower_breakeven = put_opt["strike"] - (net_cost / (100 * num_contracts))
    upper_breakeven = call_opt["strike"] + (net_cost / (100 * num_contracts))

    return {
        "raw_net": base_cost,
        "net_cost": net_cost,
        "max_loss": max_loss,
        "strikes": [put_opt["strike"], call_opt["strike"]],
        "num_contracts": num_contracts,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven
    }

# --- TABLE CREATION ---
def create_spread_table(options, strategy_func, num_contracts, commission_rate, is_debit):
    data = []
    func_name = strategy_func.__name__
    for opt1, opt2 in combinations(options, 2):
        if opt1["strike"] > opt2["strike"]:
            opt1, opt2 = opt2, opt1  # Ensure opt1 < opt2
        if func_name == "calculate_bull_call_spread":
            result = strategy_func(long_opt=opt1, short_opt=opt2, num_contracts=num_contracts, commission_rate=commission_rate)
            long_strike = opt1["strike"]
            short_strike = opt2["strike"]
        elif func_name == "calculate_bear_call_spread":
            result = strategy_func(short_opt=opt1, long_opt=opt2, num_contracts=num_contracts, commission_rate=commission_rate)
            long_strike = opt2["strike"]
            short_strike = opt1["strike"]
        elif func_name == "calculate_bull_put_spread":
            result = strategy_func(short_opt=opt2, long_opt=opt1, num_contracts=num_contracts, commission_rate=commission_rate)
            long_strike = opt1["strike"]
            short_strike = opt2["strike"]
        elif func_name == "calculate_bear_put_spread":
            result = strategy_func(long_opt=opt2, short_opt=opt1, num_contracts=num_contracts, commission_rate=commission_rate)
            long_strike = opt2["strike"]
            short_strike = opt1["strike"]
        if result and "net_cost" in result and result["net_cost"] is not None:
            cost = result["net_cost"]
            max_profit = result["max_profit"]
            max_loss = result["max_loss"]
            cost_to_profit = abs(cost / max_profit) if is_debit and max_profit != 0 else 0 if not is_debit else float('inf')
            data.append({
                "Long Strike": long_strike,
                "Short Strike": short_strike,
                "Net Cost" if is_debit else "Net Credit": -cost if not is_debit else cost,
                "Max Profit": max_profit,
                "Max Loss": max_loss,
                "Cost-to-Profit Ratio": cost_to_profit
            })
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.set_index(["Long Strike", "Short Strike"], inplace=True)
    return df


def create_complex_strategy_table(options, strategy_func, num_contracts, commission_rate, num_legs):
    data = []
    if len(options) < num_legs:
        return pd.DataFrame()
    for combo in combinations(options, num_legs):
        combo = sorted(combo, key=lambda o: o["strike"])
        result = strategy_func(*combo, num_contracts, commission_rate)
        if result and all(k in result for k in ["net_cost", "max_profit", "max_loss", "strikes"]):
            strikes = tuple(result["strikes"])
            data.append({
                "net_cost": result["net_cost"],
                "max_profit": result["max_profit"],
                "max_loss": result["max_loss"],
                "strikes": strikes
            })
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Set MultiIndex with strike names
    strike_labels = [f"Strike {i+1}" for i in range(num_legs)]
    df.set_index(pd.MultiIndex.from_tuples(df["strikes"], names=strike_labels), inplace=True)
    df.drop(columns=["strikes"], inplace=True)
    return df

def create_vol_strategy_table(calls, puts, strategy_func, num_contracts, commission_rate):
    data = []
    for call in calls:
        for put in puts:
            if strategy_func.__name__ == "calculate_straddle" and call["strike"] != put["strike"]:
                continue
            if strategy_func.__name__ == "calculate_strangle" and put["strike"] >= call["strike"]:
                continue
            result = strategy_func(call, put, num_contracts, commission_rate)
            if result:
                strikes = result["strikes"]
                key = tuple(strikes) if len(strikes) > 1 else strikes[0]
                data.append({
                    "Strikes": key,
                    "Net Cost": result["net_cost"],
                    "Max Loss": result["max_loss"],
                    "Lower Breakeven": result["lower_breakeven"],
                    "Upper Breakeven": result["upper_breakeven"]
                })
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.set_index("Strikes", inplace=True)
    return df


# REPLACE the broken create_spread_matrix function with this new one.

def create_spread_matrix(options: list, strategy_func, num_contracts: int, commission_rate: float, is_debit: bool) -> tuple:
    """
    Creates a matrix of net cost or net credit for a given spread strategy.
    Rows and columns represent the strikes of the two legs of the spread.
    """
    strikes = sorted(options, key=lambda x: x["strike"])
    strike_labels = [f"{s['strike']:.2f}" for s in strikes]
    
    matrix_data = np.full((len(strikes), len(strikes)), np.nan)
    func_name = strategy_func.__name__

    for i, row_opt in enumerate(strikes):
        for j, col_opt in enumerate(strikes):
            result = None
            
            # The logic here maps the row/column to the correct leg (long/short)
            # based on the strategy and the convention (e.g., "Buy on Row, Sell on Col").
            if "bull_call" in func_name and row_opt["strike"] < col_opt["strike"]:
                # Buy low (row), Sell high (col)
                result = strategy_func(long_opt=row_opt, short_opt=col_opt, num_contracts=num_contracts, commission_rate=commission_rate)
            elif "bull_put" in func_name and row_opt["strike"] > col_opt["strike"]:
                # Sell high (row), Buy low (col)
                result = strategy_func(short_opt=row_opt, long_opt=col_opt, num_contracts=num_contracts, commission_rate=commission_rate)
            elif "bear_call" in func_name and row_opt["strike"] < col_opt["strike"]:
                # Sell low (row), Buy high (col)
                result = strategy_func(short_opt=row_opt, long_opt=col_opt, num_contracts=num_contracts, commission_rate=commission_rate)
            elif "bear_put" in func_name and row_opt["strike"] > col_opt["strike"]:
                # Buy high (row), Sell low (col)
                result = strategy_func(long_opt=row_opt, short_opt=col_opt, num_contracts=num_contracts, commission_rate=commission_rate)

            if result and "net_cost" in result and result["net_cost"] is not None:
                cost = result["net_cost"]
                # For credit spreads (is_debit=False), display the credit as a positive number.
                # For debit spreads (is_debit=True), display the cost as a positive number.
                matrix_data[i, j] = -cost if not is_debit else cost

    df = pd.DataFrame(matrix_data, index=strike_labels, columns=strike_labels)
    
    # Return a tuple of 4 for compatibility with the page code that unpacks it.
    return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# --- 3D VISUALIZATION ---
# REPLACE the entire visualize_3d_payoff function with this one.

# REPLACE the entire visualize_3d_payoff function with this corrected version.

# DELETE the old visualize_3d_payoff function and REPLACE it with this one.

# REPLACE the entire visualize_3d_payoff function in utils.py with this corrected and enhanced version.

# REPLACE the entire visualize_3d_payoff function in utils.py with this corrected and enhanced version that includes IV calibration.

# REPLACE the entire visualize_3d_payoff function in utils.py with this version that uses the correct variable name in the Z assignment.

def visualize_3d_payoff(strategy_result, current_price, expiration_days, iv=DEFAULT_IV, key=None):
    import streamlit as st
    from scipy.stats import norm
    from scipy.optimize import minimize_scalar
    import numpy as np
    import logging

    if not strategy_result or "strikes" not in strategy_result:
        st.warning("No valid strategy selected for visualization.")
        return

    net_entry = strategy_result.get("net_cost", 0)
    strikes = strategy_result.get("strikes", [])
    num_contracts = strategy_result.get("num_contracts", 1)
    max_profit = strategy_result.get("max_profit", 0)
    max_loss = strategy_result.get("max_loss", 0)

    # Determine option type and fetch corresponding options
    options = st.session_state.get("filtered_calls", []) if "call" in key.lower() else st.session_state.get("filtered_puts", [])
    if not options:
        st.warning("No options data available for visualization.")
        return

    # Map strikes to options
    matched_options = []
    for strike in strikes:
        matched = next((opt for opt in options if opt["strike"] == strike and opt["expiration"] == st.session_state.selected_exp), None)
        if matched:
            matched_options.append(matched)
        else:
            logger.warning(f"No matching option found for strike {strike}")
            matched_options.append({"px_ask": 0, "px_bid": 0})  # Fallback to avoid crash

    # Calculate mid-price
    action = "buy" if "buy" in key.lower() else "sell"
    prices = [get_strategy_price(opt, action) for opt in matched_options if get_strategy_price(opt, action) is not None]
    mid_price = sum(prices) / len(prices) if prices else 0
    raw_net = mid_price * 100 * num_contracts if mid_price > 0 else net_entry

    def black_scholes(S, K, T, r, sigma, option_type="call"):
        try:
            S = float(S)
            K = float(K)
            T = float(T)
            r = float(r)
            sigma = float(sigma)
        except (TypeError, ValueError):
            return 0.0
        if S <= 0:
            return 0 if option_type == "call" else max(K - S, 0)
        if K <= 0:
            return max(S - K, 0) if option_type == "call" else 0
        if T <= 1e-9 or sigma <= 1e-9:
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

    def strategy_value(price, T, sigma):
        position_value = 0.0
        strategy_key = key.lower() if key else ""
        r = st.session_state.get("risk_free_rate", 0.50)  # 50% as per your input

        if "spread" in strategy_key:
            strikes = strategy_result["strikes"]
            k1, k2 = strikes[0], strikes[1]

            if "bull call" in strategy_key:
                position_value = (black_scholes(price, k1, T, r, sigma, "call") - black_scholes(price, k2, T, r, sigma, "call"))
            elif "bull put" in strategy_key:
                position_value = (black_scholes(price, k1, T, r, sigma, "put") - black_scholes(price, k2, T, r, sigma, "put"))
            elif "bear call" in strategy_key:
                position_value = (black_scholes(price, k2, T, r, sigma, "call") - black_scholes(price, k1, T, r, sigma, "call"))
            elif "bear put" in strategy_key:
                position_value = (black_scholes(price, k1, T, r, sigma, "put") - black_scholes(price, k2, T, r, sigma, "put"))

            position_value *= 100 * num_contracts

        elif "butterfly" in strategy_key or "condor" in strategy_key:
            ratios = strategy_result.get("contract_ratios", [])
            strikes = strategy_result.get("strikes", [])
            opt_type = "call" if "call" in strategy_key else "put"
            if ratios and strikes:
                vals = [black_scholes(price, k, T, r, sigma, opt_type) for k in strikes]
                position_value = sum(r * v for r, v in zip(ratios, vals)) * 100 * num_contracts

        elif "straddle" in strategy_key:
            k = strategy_result["strikes"][0]
            call_val = black_scholes(price, k, T, r, sigma, "call")
            put_val = black_scholes(price, k, T, r, sigma, "put")
            position_value = (call_val + put_val) * 100 * num_contracts

        elif "strangle" in strategy_key:
            put_k, call_k = strategy_result["strikes"]
            call_val = black_scholes(price, call_k, T, r, sigma, "call")
            put_val = black_scholes(price, put_k, T, r, sigma, "put")
            position_value = (call_val + put_val) * 100 * num_contracts

        return position_value

    # Calibrate IV
    strategy_key = key.lower() if key else ""
    r = st.session_state.get("risk_free_rate", 0.50)
    def objective(sigma):
        if sigma <= 0: return float('inf')
        return abs(strategy_value(current_price, expiration_days / 365.0, sigma) - raw_net)

    try:
        result = minimize_scalar(objective, bounds=(0.01, 10.0), method='bounded')
        calibrated_iv = result.x
        if calibrated_iv > 0 and not np.isnan(calibrated_iv) and result.success:
            iv = calibrated_iv
        else:
            raise ValueError("Optimization failed")
    except Exception as e:
        logger.error(f"IV calibration failed for {strategy_key}: {e}")
        iv = DEFAULT_IV

    iv = max(iv, 1e-9)

    # Compute grid
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    prices = np.linspace(max(0.1, current_price * (1 - plot_range_pct)), current_price * (1 + plot_range_pct), 50)
    times = np.linspace(0, expiration_days, 20)
    X, Y = np.meshgrid(prices, times)
    Z = np.zeros_like(X)

    # Improved scaling factor: use the maximum of absolute values to reflect full range
    scale_factor = max(abs(max_profit), abs(max_loss), abs(net_entry)) if max(abs(max_profit), abs(max_loss), abs(net_entry)) > 0 else 1.0

    for i in range(len(times)):
        for j in range(len(prices)):
            price = X[i, j]
            T = (expiration_days - Y[i, j]) / 365.0
            T = max(T, 1e-9)
            Z[i, j] = (strategy_value(price, T, iv) - net_entry) / scale_factor

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(
        title="3D Payoff Visualization",
        scene=dict(
            xaxis_title="Underlying Price",
            yaxis_title="Days from Now",
            zaxis_title="Profit / Loss (Scaled)",
        ),
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    st.plotly_chart(fig, use_container_width=True, key=key)