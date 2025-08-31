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
# ADD the following import to utils.py if not already present
from scipy.optimize import minimize_scalar

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
STOCK_URL = "https://data912.com/live/arg_stocks"
OPTIONS_URL = "https://data912.com/live/arg_options"
DEFAULT_IV = 0.30
MIN_GAP = 0.01


# --- DATE & EXPIRATION HELPERS ---
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


# --- DATA FETCHING & PARSING ---
@st.cache_data(ttl=300)
def fetch_data(url: str) -> list:
    try:
        response = requests.get(url, headers={"accept": "*/*"}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching data from {url}: {e}")
        return []


def parse_option_symbol(symbol: str) -> tuple[str | None, float | None, date | None]:
    option_type = "call" if symbol.startswith("GFGC") else "put" if symbol.startswith("GFGV") else None
    if not option_type: return None, None, None

    data_part = symbol[4:]
    first_letter_index = next((i for i, char in enumerate(data_part) if char.isalpha()), -1)
    if first_letter_index == -1: return None, None, None

    numeric_part, suffix = data_part[:first_letter_index], data_part[first_letter_index:]
    if not numeric_part: return None, None, None

    try:
        strike_price = float(numeric_part) / 10.0 if len(numeric_part) >= 5 and not numeric_part.startswith(
            '1') else float(numeric_part)
    except ValueError:
        return None, None, None

    return option_type, strike_price, EXPIRATION_MAP_2025.get(suffix)


def get_ggal_data() -> tuple[dict | None, list]:
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)
    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock: return None, []

    ggal_options = []
    for o in options_data:
        opt_type, strike, exp = parse_option_symbol(o["symbol"])
        if all([opt_type, strike, exp, o.get("px_ask", 0) > 0, o.get("px_bid", 0) > 0]):
            ggal_options.append({
                "symbol": o["symbol"], "type": opt_type, "strike": strike,
                "expiration": exp, "px_bid": o["px_bid"], "px_ask": o["px_ask"]
            })
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

    base_cost = (long_price - short_price) * num_contracts * 100
    # --- FIX: Added the missing commission_rate parameter ---
    total_fees = calculate_fees(base_cost, commission_rate) if base_cost > 0 else 0
    net_cost = base_cost + total_fees
    
    max_loss = net_cost
    max_profit = (short_opt["strike"] - long_opt["strike"]) * num_contracts * 100 - net_cost
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "strikes": [long_opt["strike"], short_opt["strike"]],"num_contracts": num_contracts}


def calculate_bull_put_spread(short_opt, long_opt, num_contracts, commission_rate):
    # For a Bull Put, you sell a higher strike and buy a lower strike.
    if short_opt["strike"] <= long_opt["strike"]:
        return None

    short_price = get_strategy_price(short_opt, "sell")
    long_price = get_strategy_price(long_opt, "buy")
    if any(p is None for p in [short_price, long_price]):
        return None

    # You receive a credit because the higher strike put you sell is worth more.
    base_credit = (short_price - long_price) * num_contracts * 100

    # If the credit is negative (a debit), it's not a valid bull put spread.
    if base_credit <= 0:
        return None

    total_fees = calculate_fees(abs(base_credit), commission_rate)
    net_credit = base_credit - total_fees

    # Your max profit is the net credit you receive.
    max_profit = net_credit

    # Your max loss is the difference in strikes minus the credit received.
    max_loss = (short_opt["strike"] - long_opt["strike"]) * num_contracts * 100 - net_credit

    return {
        "net_cost": -net_credit,  # Stored as negative cost
        "max_profit": max_profit,
        "max_loss": max_loss,
        "strikes": [long_opt["strike"], short_opt["strike"]],"num_contracts": num_contracts  # Convention: long, short
    }


def calculate_bear_call_spread(short_opt, long_opt, num_contracts, commission_rate):
    if short_opt["strike"] >= long_opt["strike"]: return None
    short_price = get_strategy_price(short_opt, "sell")
    long_price = get_strategy_price(long_opt, "buy")
    if any(p is None for p in [short_price, long_price]): return None

    base_credit = (short_price - long_price) * num_contracts * 100
    total_fees = calculate_fees(abs(base_credit), commission_rate)
    net_credit = base_credit - total_fees

    max_profit = net_credit
    max_loss = (long_opt["strike"] - short_opt["strike"]) * num_contracts * 100 - max_profit
    return {"net_cost": -net_credit, "max_profit": max_profit, "max_loss": max_loss,
            "strikes": [short_opt["strike"], long_opt["strike"]],"num_contracts": num_contracts}


def calculate_bear_put_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] <= short_opt["strike"]: return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if any(p is None for p in [long_price, short_price]): return None

    base_cost = (long_price - short_price) * num_contracts * 100
    # --- FIX: Added the missing commission_rate parameter ---
    total_fees = calculate_fees(base_cost, commission_rate) if base_cost > 0 else 0
    net_cost = base_cost + total_fees
    
    max_loss = net_cost
    max_profit = (long_opt["strike"] - short_opt["strike"]) * num_contracts * 100 - net_cost
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "strikes": [long_opt["strike"], short_opt["strike"]],"num_contracts": num_contracts}


# --- OTHER STRATEGY CALCULATIONS ---
# (This section includes Butterfly, Condor, Straddle, Strangle calculations)
def calculate_call_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate):
    strikes = [low_opt["strike"], mid_opt["strike"], high_opt["strike"]]
    if strikes[0] >= strikes[1] or strikes[1] >= strikes[2]: return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_price = get_strategy_price(mid_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if None in (low_price, mid_price, high_price): return None
    base_cost = (low_price + high_price - 2 * mid_price) * 100 * num_contracts
    fees = calculate_fees(abs(base_cost), commission_rate)
    net_cost = base_cost + fees if base_cost > 0 else base_cost - fees
    lower_spread = strikes[1] - strikes[0]
    upper_spread = strikes[2] - strikes[1]
    max_profit = min(lower_spread, upper_spread) * 100 * num_contracts - net_cost
    max_loss = net_cost
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "strikes": strikes, "contract_ratios": [1, -2, 1], "num_contracts": num_contracts}

def calculate_put_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate):
    strikes = [low_opt["strike"], mid_opt["strike"], high_opt["strike"]]
    if strikes[0] >= strikes[1] or strikes[1] >= strikes[2]: return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_price = get_strategy_price(mid_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if None in (low_price, mid_price, high_price): return None
    base_cost = (low_price + high_price - 2 * mid_price) * 100 * num_contracts
    fees = calculate_fees(abs(base_cost), commission_rate)
    net_cost = base_cost + fees if base_cost > 0 else base_cost - fees
    lower_spread = strikes[1] - strikes[0]
    upper_spread = strikes[2] - strikes[1]
    max_profit = min(lower_spread, upper_spread) * 100 * num_contracts - net_cost
    max_loss = net_cost
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "strikes": strikes, "contract_ratios": [1, -2, 1], "num_contracts": num_contracts}

def calculate_call_condor(low_opt, mid1_opt, mid2_opt, high_opt, num_contracts, commission_rate):
    strikes = [low_opt["strike"], mid1_opt["strike"], mid2_opt["strike"], high_opt["strike"]]
    if not all(strikes[i] < strikes[i+1] for i in range(3)): return None
    low_price = get_strategy_price(low_opt, "buy")
    mid1_price = get_strategy_price(mid1_opt, "sell")
    mid2_price = get_strategy_price(mid2_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if None in (low_price, mid1_price, mid2_price, high_price): return None
    base_cost = (low_price + high_price - mid1_price - mid2_price) * 100 * num_contracts
    fees = calculate_fees(abs(base_cost), commission_rate)
    net_cost = base_cost + fees if base_cost > 0 else base_cost - fees
    spread_width = mid2_opt["strike"] - mid1_opt["strike"]
    max_profit = spread_width * 100 * num_contracts - net_cost
    max_loss = net_cost
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "strikes": strikes, "contract_ratios": [1, -1, -1, 1], "num_contracts": num_contracts}

def calculate_put_condor(low_opt, mid1_opt, mid2_opt, high_opt, num_contracts, commission_rate):
    strikes = [low_opt["strike"], mid1_opt["strike"], mid2_opt["strike"], high_opt["strike"]]
    if not all(strikes[i] < strikes[i+1] for i in range(3)): return None
    low_price = get_strategy_price(low_opt, "buy")
    mid1_price = get_strategy_price(mid1_opt, "sell")
    mid2_price = get_strategy_price(mid2_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if None in (low_price, mid1_price, mid2_price, high_price): return None
    base_cost = (low_price + high_price - mid1_price - mid2_price) * 100 * num_contracts
    fees = calculate_fees(abs(base_cost), commission_rate)
    net_cost = base_cost + fees if base_cost > 0 else base_cost - fees
    spread_width = mid2_opt["strike"] - mid1_opt["strike"]
    max_profit = spread_width * 100 * num_contracts - net_cost
    max_loss = net_cost
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "strikes": strikes, "contract_ratios": [1, -1, -1, 1], "num_contracts": num_contracts}


def calculate_straddle(call, put, num_contracts, commission_rate):
    if call["strike"] != put["strike"]: return None
    call_price = get_strategy_price(call, "buy")
    put_price = get_strategy_price(put, "buy")
    if None in (call_price, put_price): return None
    base_cost = (call_price + put_price) * 100 * num_contracts
    fees = calculate_fees(base_cost, commission_rate)
    net_cost = base_cost + fees
    k = call["strike"]
    max_loss = net_cost
    lower_breakeven = k - net_cost / (100 * num_contracts)
    upper_breakeven = k + net_cost / (100 * num_contracts)
    return {"net_cost": net_cost, "max_loss": max_loss, "strikes": [k], "lower_breakeven": lower_breakeven, "upper_breakeven": upper_breakeven, "num_contracts": num_contracts}

def calculate_strangle(put, call, num_contracts, commission_rate):
    if put["strike"] >= call["strike"]: return None
    put_price = get_strategy_price(put, "buy")
    call_price = get_strategy_price(call, "buy")
    if None in (put_price, call_price): return None
    base_cost = (put_price + call_price) * 100 * num_contracts
    fees = calculate_fees(base_cost, commission_rate)
    net_cost = base_cost + fees
    max_loss = net_cost
    lower_breakeven = put["strike"] - net_cost / (100 * num_contracts)
    upper_breakeven = call["strike"] + net_cost / (100 * num_contracts)
    return {"net_cost": net_cost, "max_loss": max_loss, "strikes": [put["strike"], call["strike"]], "lower_breakeven": lower_breakeven, "upper_breakeven": upper_breakeven, "num_contracts": num_contracts}


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
            cost_to_profit = abs(cost / max_profit) if max_profit != 0 else float('inf')
            data.append({
                "Long Strike": long_strike,
                "Short Strike": short_strike,
                "Net Cost" if is_debit else "Net Credit": abs(cost) if is_debit else -cost,
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
    for combo in combinations(options, num_legs):
        combo = sorted(combo, key=lambda o: o["strike"])
        result = strategy_func(*combo, num_contracts, commission_rate)
        if result:
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
    df.set_index("strikes", inplace=True)
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

    if not strategy_result or "strikes" not in strategy_result:
        st.warning("No valid strategy selected for visualization.")
        return

    net_entry = strategy_result.get("net_cost", 0)
    if net_entry is None: return

    num_contracts = strategy_result.get("num_contracts", 1)

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
        r = st.session_state.get("risk_free_rate", 0.50)

        if "spread" in strategy_key:
            strikes = strategy_result["strikes"]
            k1, k2 = strikes[0], strikes[1]

            if "bull call" in strategy_key:  # Long Call Spread
                position_value = (black_scholes(price, k1, T, r, sigma, "call") - black_scholes(price, k2, T, r, sigma, "call"))
            elif "bull put" in strategy_key:  # Short Put Spread
                position_value = (black_scholes(price, k1, T, r, sigma, "put") - black_scholes(price, k2, T, r, sigma, "put"))
            elif "bear call" in strategy_key:  # Short Call Spread
                position_value = (black_scholes(price, k2, T, r, sigma, "call") - black_scholes(price, k1, T, r, sigma, "call"))
            elif "bear put" in strategy_key:  # Long Put Spread
                position_value = (black_scholes(price, k1, T, r, sigma, "put") - black_scholes(price, k2, T, r, sigma, "put"))

            position_value *= 100 * num_contracts

        elif "butterfly" in strategy_key or "condor" in strategy_key:
            ratios = strategy_result.get("contract_ratios")
            strikes = strategy_result.get("strikes")
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

    # Calibrate IV to minimize |strategy_value - net_entry| at current price and initial time
    strategy_key = key.lower() if key else ""
    r = st.session_state.get("risk_free_rate", 0.50)
    def objective(sigma):
        if sigma <= 0: return float('inf')
        return abs(strategy_value(current_price, expiration_days / 365.0, sigma) - net_entry)

    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(objective, bounds=(0.01, 10.0), method='bounded')
        calibrated_iv = result.x
        if calibrated_iv > 0 and not np.isnan(calibrated_iv) and result.success:
            iv = calibrated_iv
    except:
        pass  # Use original IV if optimization fails

    iv = max(iv, 1e-9) # Ensure iv is always positive

    # Compute the grid with calibrated IV
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    prices = np.linspace(max(0.1, current_price * (1 - plot_range_pct)), current_price * (1 + plot_range_pct), 50)
    times = np.linspace(0, expiration_days, 20)
    X, Y = np.meshgrid(prices, times)
    Z = np.zeros_like(X)

    # Dynamic scale factor to emphasize time decay
    scale_factor = abs(net_entry) if abs(net_entry) > 0 else 1
    if "straddle" in strategy_key or "strangle" in strategy_key:
        scale_factor *= 1.5  # Amplify for high-theta strategies

    for i in range(len(times)):
        for j in range(len(prices)):
            price = X[i, j]
            T = (expiration_days - Y[i, j]) / 365.0
            T = max(T, 1e-9)  # Ensure T is always positive
            Z[i, j] = (strategy_value(price, T, iv) - net_entry) / scale_factor

    # Plotting with plotly
    import plotly.graph_objects as go

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