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
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "strikes": [long_opt["strike"], short_opt["strike"]]}


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
        "strikes": [long_opt["strike"], short_opt["strike"]]  # Convention: long, short
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
            "strikes": [short_opt["strike"], long_opt["strike"]]}


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
    return {"net_cost": net_cost, "max_profit": max_profit, "max_loss": max_loss, "strikes": [long_opt["strike"], short_opt["strike"]]}


# --- OTHER STRATEGY CALCULATIONS ---
# (This section includes Butterfly, Condor, Straddle, Strangle calculations)
def calculate_call_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_opt["strike"] < high_opt["strike"]): return None
    prices = [get_strategy_price(opt, action) for opt, action in
              zip([low_opt, mid_opt, high_opt], ["buy", "sell", "buy"])]
    if any(p is None for p in prices): return None

    gap1, gap2 = mid_opt["strike"] - low_opt["strike"], high_opt["strike"] - mid_opt["strike"]
    if gap1 < MIN_GAP or gap2 < MIN_GAP: return None

    g = gcd(int(gap1 * 100), int(gap2 * 100)) / 100
    if g < MIN_GAP: g = MIN_GAP

    ratios = [num_contracts * round(gap2 / g), -num_contracts * round((gap1 + gap2) / g),
              num_contracts * round(gap1 / g)]

    base_cost = (prices[0] * abs(ratios[0]) + prices[2] * abs(ratios[2])) * 100
    net_cost = sum(p * r for p, r in zip(prices, ratios)) * 100 + calculate_fees(base_cost, commission_rate)

    max_profit = (mid_opt["strike"] - low_opt["strike"]) * abs(ratios[0]) * 100 - net_cost
    return {"max_profit": max_profit, "net_cost": net_cost, "max_loss": net_cost,
            "contracts": f"{ratios[0]} : {ratios[1]} : {ratios[2]}",
            "strikes": [low_opt["strike"], mid_opt["strike"], high_opt["strike"]], "contract_ratios": ratios}


def calculate_put_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate):
    # This function is identical to the call version, just with puts.
    return calculate_call_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate)


def calculate_call_condor(k1_opt, k2_opt, k3_opt, k4_opt, num_contracts, commission_rate):
    if not (k1_opt["strike"] < k2_opt["strike"] < k3_opt["strike"] < k4_opt["strike"]): return None
    prices = [get_strategy_price(opt, action) for opt, action in
              zip([k1_opt, k2_opt, k3_opt, k4_opt], ["buy", "sell", "sell", "buy"])]
    if any(p is None for p in prices): return None

    gap1, gap3 = k2_opt["strike"] - k1_opt["strike"], k4_opt["strike"] - k3_opt["strike"]
    if gap1 < MIN_GAP or gap3 < MIN_GAP: return None

    g = gcd(int(gap1 * 100), int(gap3 * 100)) / 100
    if g < MIN_GAP: g = MIN_GAP

    ratios = [num_contracts * round(gap3 / g), -num_contracts * round(gap3 / g), -num_contracts * round(gap1 / g),
              num_contracts * round(gap1 / g)]

    base_cost = (prices[0] * abs(ratios[0]) + prices[3] * abs(ratios[3])) * 100
    net_cost = sum(p * r for p, r in zip(prices, ratios)) * 100 + calculate_fees(base_cost, commission_rate)

    max_profit = (k2_opt["strike"] - k1_opt["strike"]) * abs(ratios[0]) * 100 - net_cost
    return {"max_profit": max_profit, "net_cost": net_cost, "max_loss": net_cost,
            "contracts": f"{ratios[0]} : {ratios[1]} : {ratios[2]} : {ratios[3]}",
            "strikes": [k1_opt["strike"], k2_opt["strike"], k3_opt["strike"], k4_opt["strike"]],
            "contract_ratios": ratios}


def calculate_put_condor(k1_opt, k2_opt, k3_opt, k4_opt, num_contracts, commission_rate):
    # This function is identical to the call version, just with puts.
    return calculate_call_condor(k1_opt, k2_opt, k3_opt, k4_opt, num_contracts, commission_rate)


def calculate_straddle(call_opt, put_opt, num_contracts, commission_rate):
    if call_opt["strike"] != put_opt["strike"]: return None
    prices = [get_strategy_price(call_opt, "buy"), get_strategy_price(put_opt, "buy")]
    if any(p is None for p in prices): return None

    base_cost = sum(prices) * num_contracts * 100
    net_cost = base_cost + calculate_fees(base_cost, commission_rate)
    return {"net_cost": net_cost, "max_loss": net_cost, "max_profit": float('inf'), "strikes": [call_opt["strike"]]}


def calculate_strangle(put_opt, call_opt, num_contracts, commission_rate):
    if put_opt["strike"] >= call_opt["strike"]: return None
    prices = [get_strategy_price(put_opt, "buy"), get_strategy_price(call_opt, "buy")]
    if any(p is None for p in prices): return None

    base_cost = sum(prices) * num_contracts * 100
    net_cost = base_cost + calculate_fees(base_cost, commission_rate)
    return {"net_cost": net_cost, "max_loss": net_cost, "max_profit": float('inf'),
            "strikes": [put_opt["strike"], call_opt["strike"]]}


# --- TABLE CREATION ---
def create_spread_table(options: list, strategy_func, num_contracts: int, commission_rate: float, is_debit: bool) -> pd.DataFrame:
    strikes = sorted(options, key=lambda x: x["strike"])
    combos = list(combinations(strikes, 2))
    data = []
    cost_label = "Net Cost" if is_debit else "Net Credit"

    for opt1, opt2 in combos:  # In combos, opt1.strike is always < opt2.strike
        func_name = strategy_func.__name__
        result, strike_label = (None, "")

        # --- FIX: Correctly assign long/short legs based on strategy ---
        if "bull_put" in func_name:
            # Sell higher strike (opt2), Buy lower strike (opt1)
            result = strategy_func(short_opt=opt2, long_opt=opt1, num_contracts=num_contracts, commission_rate=commission_rate)
            strike_label = f"{opt1['strike']:.2f} - {opt2['strike']:.2f}"
        elif "bear_put" in func_name:
            # Buy higher strike (opt2), Sell lower strike (opt1)
            result = strategy_func(long_opt=opt2, short_opt=opt1, num_contracts=num_contracts, commission_rate=commission_rate)
            strike_label = f"{opt2['strike']:.2f} - {opt1['strike']:.2f}"
        else: # bull_call, bear_call
            # For calls, opt1 is long/short and opt2 is the other leg
            result = strategy_func(opt1, opt2, num_contracts, commission_rate)
            strike_label = f"{opt1['strike']:.2f} - {opt2['strike']:.2f}"

        if result:
            cost, profit, loss = result.get("net_cost"), result.get("max_profit"), result.get("max_loss")
            if any(v is None for v in [cost, profit, loss]): continue

            if is_debit:
                # For debit spreads, ratio is Cost / Profit
                ratio = cost / profit if profit > 0 else float('inf')
                display_cost = cost
            else:  # Credit Spread
                # For credit spreads, ratio is Risk (Max Loss) / Reward (Max Profit)
                ratio = loss / profit if profit > 0 else float('inf')
                display_cost = -cost  # Show credit as a positive number

            data.append({
                "Strikes": strike_label, cost_label: display_cost, "Max Profit": profit,
                "Max Loss": loss, "Cost-to-Profit Ratio": ratio, "strikes": result["strikes"]
            })

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by="Cost-to-Profit Ratio", ascending=True)
    return df


def create_complex_strategy_table(options: list, strategy_func, num_contracts: int, commission_rate: float,
                                  combo_size: int) -> pd.DataFrame:
    strikes = sorted(options, key=lambda x: x["strike"])
    combos = list(combinations(strikes, combo_size))
    data = []
    for combo in combos:
        if not all(combo[i]["strike"] < combo[i + 1]["strike"] for i in range(len(combo) - 1)): continue
        result = strategy_func(*combo, num_contracts, commission_rate)
        if result:
            cost, profit, loss = result.get("net_cost"), result.get("max_profit"), result.get("max_loss")
            if any(v is None for v in [cost, profit, loss]): continue
            ratio = cost / profit if profit > 0 else float('inf')
            data.append({
                "Strikes": " - ".join(f"{opt['strike']:.2f}" for opt in combo), "Net Cost": cost,
                "Max Profit": profit, "Max Loss": loss, "Cost-to-Profit Ratio": ratio,
                "Contracts": result.get("contracts", "N/A"), "strikes": result["strikes"],
                "contract_ratios": result.get("contract_ratios")
            })
    return pd.DataFrame(data)


# REPLACE the entire create_vol_strategy_table function in utils.py with this version that includes breakeven points instead of the cost-to-profit ratio.

# REPLACE the entire create_vol_strategy_table function in utils.py with this version that returns the DataFrame without styling.

def create_vol_strategy_table(calls, puts, calc_func, num_contracts, commission_rate):
    data = []
    sorted_calls = sorted(calls, key=lambda x: x['strike'])
    sorted_puts = sorted(puts, key=lambda x: x['strike'])
    
    if 'straddle' in calc_func.__name__.lower():
        for call in sorted_calls:
            put = next((p for p in sorted_puts if p['strike'] == call['strike']), None)
            if put:
                result = calc_func(call, put, num_contracts, commission_rate)
                if result:
                    net_cost = result['net_cost']
                    per_contract = net_cost / (100 * num_contracts)
                    lower_be = call['strike'] - per_contract
                    upper_be = call['strike'] + per_contract
                    result.update({
                        'lower_breakeven': lower_be,
                        'upper_breakeven': upper_be
                    })
                    data.append(result)
    elif 'strangle' in calc_func.__name__.lower():
        for put, call in product(sorted_puts, sorted_calls):
            if put['strike'] < call['strike']:
                result = calc_func(put, call, num_contracts, commission_rate)
                if result:
                    net_cost = result['net_cost']
                    per_contract = net_cost / (100 * num_contracts)
                    lower_be = put['strike'] - per_contract
                    upper_be = call['strike'] + per_contract
                    result.update({
                        'lower_breakeven': lower_be,
                        'upper_breakeven': upper_be
                    })
                    data.append(result)
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df[['strikes', 'net_cost', 'max_loss', 'max_profit', 'lower_breakeven', 'upper_breakeven']]
        df.columns = ['Strikes', 'Net Cost', 'Max Loss', 'Max Profit', 'Lower Breakeven', 'Upper Breakeven']
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
    if not strategy_result or "Strikes" not in strategy_result:
        st.warning("No valid strategy selected for visualization.")
        return

    net_debit = strategy_result.get("Net Cost", 0)
    net_credit = strategy_result.get("net_credit", 0)
    net_entry = net_debit - net_credit
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
        if T <= 1e-9: return max(0, S - K) if option_type == "call" else max(0, K - S)
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                val = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                val = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return np.nan_to_num(val)

    def strategy_value(price, T, sigma):
        position_value = 0.0
        strategy_key = key.lower() if key else ""
        
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
                position_value = (black_scholes(price, k2, T, r, sigma, "put") - black_scholes(price, k1, T, r, sigma, "put"))
            
            position_value *= 100 * num_contracts

        elif "butterfly" in strategy_key or "condor" in strategy_key:
            ratios = strategy_result.get("contract_ratios")
            strikes = strategy_result.get("Strikes")
            opt_type = "call" if "call" in strategy_key else "put"
            if ratios and strikes:
                vals = [black_scholes(price, k, T, r, sigma, opt_type) for k in strikes]
                position_value = sum(r * v for r, v in zip(ratios, vals)) * 100

        elif "straddle" in strategy_key:
            k = strategy_result["Strikes"][0]
            call_val = black_scholes(price, k, T, r, sigma, "call")
            put_val = black_scholes(price, k, T, r, sigma, "put")
            position_value = (call_val + put_val) * 100 * num_contracts

        elif "strangle" in strategy_key:
            put_k, call_k = strategy_result["Strikes"]
            call_val = black_scholes(price, call_k, T, r, sigma, "call")
            put_val = black_scholes(price, put_k, T, r, sigma, "put")
            position_value = (call_val + put_val) * 100 * num_contracts

        return position_value

    # Calibrate IV to minimize |strategy_value - net_entry| at current price and initial time
    strategy_key = key.lower() if key else ""
    r = st.session_state.get("risk_free_rate", 0.50)
    def objective(sigma):
        if sigma <= 0: return float('inf')
        return strategy_value(current_price, expiration_days / 365.0, sigma) - net_entry

    try:
        result = minimize_scalar(lambda sigma: abs(objective(sigma)), bounds=(0.01, 10.0), method='bounded')
        calibrated_iv = result.x
        if calibrated_iv > 0 and not np.isnan(calibrated_iv) and result.success:
            iv = calibrated_iv
    except:
        pass  # Use original IV if optimization fails

    # Now compute the grid with calibrated IV
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    # Removed scaling factor to show actual profit/loss values

    prices = np.linspace(max(0.1, current_price * (1 - plot_range_pct)), current_price * (1 + plot_range_pct), 50)
    times = np.linspace(0, expiration_days, 20)
    X, Y = np.meshgrid(prices, times)
    Z = np.zeros_like(X)

    for i in range(len(times)):
        for j in range(len(prices)):
            price = X[i, j]
            T = (expiration_days - Y[i, j]) / 365.0
            Z[i, j] = (strategy_value(price, T, iv) - net_entry) / scale_factor  # Scale values

    net_entry_scaled = net_entry / scale_factor

    # Debug prints
    print(f"Payoff surface Z min: {Z.min()}, max: {Z.max()}")
    print(f"Net entry scaled: {net_entry_scaled}")
    Z[i, j] = (strategy_value(price, T, iv) - net_entry) / scale_factor  # Scale values for visualization

    # Force Z range to include 0 with margin
    z_min = min(Z.min(), 0)
    z_max = max(Z.max(), 0)
    z_margin = 0.1 * (z_max - z_min) if z_max - z_min > 0 else 1.0
    z_min -= z_margin
    z_max += z_margin

    payoff_surface = go.Surface(z=Z, x=X, y=Y, colorscale='RdYlGn', cmin=z_min, cmax=z_max,
                                colorbar=dict(title='Profit/Loss'))

    breakeven_plane = go.Surface(z=np.zeros_like(X), x=X, y=Y, opacity=0.7, showscale=False,
                                 colorscale=[[0, '#0000FF'], [1, '#0000FF']])

    yy, zz = np.meshgrid([Y.min(), Y.max()], [z_min, z_max])
    xx = np.full_like(yy, current_price)
    current_price_plane = go.Surface(x=xx, y=yy, z=zz, opacity=0.7, showscale=False,
                                     colorscale=[[0, '#FF0000'], [1, '#FF0000']])

    fig = go.Figure(data=[payoff_surface, breakeven_plane, current_price_plane])
    fig.update_layout(title=f"Strategy Value Over Time: {key.replace('_', ' ').title()}",
                      scene=dict(xaxis_title='Underlying Price (ARS)', yaxis_title='Days Elapsed',
                                 zaxis_title='Profit/Loss (ARS)',
                                 zaxis=dict(range=[z_min, z_max])))
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key}")