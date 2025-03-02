import requests
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta, timezone
from itertools import combinations
import pandas as pd
from math import comb

# API Endpoints
STOCK_URL = "https://data912.com/live/arg_stocks"
OPTIONS_URL = "https://data912.com/live/arg_options"

# Third Friday calculation
def get_third_friday(year, month):
    first_day = date(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
    return first_friday + timedelta(days=14)

# Expiration Mapping for 2025
EXPIRATION_MAP_2025 = {
    "M": get_third_friday(2025, 3), "MA": get_third_friday(2025, 3),
    "A": get_third_friday(2025, 4), "AB": get_third_friday(2025, 4),
    "J": get_third_friday(2025, 6), "JU": get_third_friday(2025, 6)
}

def fetch_data(url):
    try:
        response = requests.get(url, headers={"accept": "*/*"}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error al obtener datos de {url}: {e}")
        return []

def parse_option_symbol(symbol):
    if symbol.startswith("GFGC"):
        option_type = "call"
    elif symbol.startswith("GFGV"):
        option_type = "put"
    else:
        return None, None, None
    numeric_part = "".join(filter(str.isdigit, symbol[4:]))
    strike_price = float(numeric_part) / 10
    suffix = symbol[4 + len(numeric_part):]
    expiration = EXPIRATION_MAP_2025.get(suffix, None)
    return option_type, strike_price, expiration

def get_ggal_data():
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)

    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock:
        st.error("No se encontraron datos de la acción GGAL.")
        return None, []

    ggal_options = []
    for option in options_data:
        symbol = option["symbol"]
        if symbol.startswith("GFGC") or symbol.startswith("GFGV"):
            opt_type, strike, exp = parse_option_symbol(symbol)
            if exp and strike >= 4900:
                ggal_options.append({
                    "symbol": symbol,
                    "type": opt_type,
                    "strike": strike,
                    "expiration": exp,
                    "px_bid": option["px_bid"],
                    "px_ask": option["px_ask"],
                    "c": option["c"]
                })
    return ggal_stock, ggal_options

# Time check for Argentine market hours (GMT-3, 11:00–17:00, weekdays)
def is_market_hours():
    now_utc = datetime.now(timezone.utc)
    ar_tz_offset = timedelta(hours=-3)  # GMT-3
    now_ar = now_utc + ar_tz_offset
    hour = now_ar.hour
    minute = now_ar.minute
    weekday = now_ar.weekday()  # 0 = Monday, 6 = Sunday
    time_in_minutes = hour * 60 + minute
    market_open = 11 * 60  # 11:00
    market_close = 17 * 60  # 17:00
    return (weekday < 5 and market_open <= time_in_minutes <= market_close)  # Weekdays only

# Updated price function based on market hours and buy/sell action
def get_strategy_price(option, action, current_stock_price=None):
    market_open = is_market_hours()
    if market_open:
        if action == "buy":
            price = option["px_ask"]
            if price == 0 or price is None:
                return None  # Exclude if no valid ask during market hours
            return price
        elif action == "sell":
            price = option["px_bid"]
            if price == 0 or price is None:
                return None  # Exclude if no valid bid during market hours
            return price
    else:
        price = option["c"]
        if price == 0 or price is None:
            st.warning(f"No hay precio de cierre válido para {option['symbol']}. Usando 0 como fallback.")
            return 0
        return price

# Strategy Calculations (updated for buy/sell pricing logic)
def calculate_bull_spread(long_call, short_call, current_price, num_contracts, debug=False):
    long_price = get_strategy_price(long_call, "buy", current_price)
    short_price = get_strategy_price(short_call, "sell", current_price)
    if long_price is None or short_price is None:
        return None  # Exclude strategy if prices are invalid during market hours
    net_cost = (long_price - short_price) * num_contracts * 100
    net_cost_per_share = long_price - short_price

    if debug:
        st.write(f"Bull Spread - Long Call: {long_price}, Short Call: {short_price}, Net Cost per Share: {net_cost_per_share}")

    max_profit = (short_call["strike"] - long_call["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven = long_call["strike"] + net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = max(price - long_call["strike"], 0) - max(price - short_call["strike"], 0)
        if price <= long_call["strike"]:
            profit = -net_cost
        elif price <= short_call["strike"]:
            profit = (payoff_per_share - net_cost_per_share) * num_contracts * 100
        else:
            profit = max_profit
        profits.append(profit)

    details = f"Compra {num_contracts} Call {long_call['strike']} ARS, Vende {num_contracts} Call {short_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Bull Spread (Calls)", details

def calculate_bear_spread(long_put, short_put, current_price, num_contracts, debug=False):
    long_price = get_strategy_price(long_put, "buy", current_price)
    short_price = get_strategy_price(short_put, "sell", current_price)
    if long_price is None or short_price is None:
        return None
    net_cost = (long_price - short_price) * num_contracts * 100
    net_cost_per_share = long_price - short_price

    if debug:
        st.write(f"Bear Spread - Long Put: {long_price}, Short Put: {short_price}, Net Cost per Share: {net_cost_per_share}")

    max_profit = (long_put["strike"] - short_put["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven = long_put["strike"] - net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = max(long_put["strike"] - price, 0) - max(short_put["strike"] - price, 0)
        profit = (payoff_per_share - net_cost_per_share) * num_contracts * 100
        profit = min(max(profit, -net_cost), max_profit)
        profits.append(profit)

    details = f"Compra {num_contracts} Put {long_put['strike']} ARS, Vende {num_contracts} Put {short_put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Bear Spread (Puts)", details

def calculate_butterfly_spread(low_call, mid_call, high_call, current_price, num_contracts, debug=False):
    low_price = get_strategy_price(low_call, "buy", current_price)
    mid_price = get_strategy_price(mid_call, "sell", current_price)
    high_price = get_strategy_price(high_call, "buy", current_price)
    if low_price is None or mid_price is None or high_price is None:
        return None
    net_cost = (low_price + high_price - 2 * mid_price) * num_contracts * 100
    net_cost_per_share = low_price + high_price - 2 * mid_price

    if debug:
        st.write(f"Butterfly Spread - Low Call: {low_price}, Mid Call: {mid_price}, High Call: {high_price}, Net Cost per Share: {net_cost_per_share}")

    max_profit = (mid_call["strike"] - low_call["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven1 = low_call["strike"] + net_cost_per_share
    breakeven2 = high_call["strike"] - net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = (max(price - low_call["strike"], 0) - 2 * max(price - mid_call["strike"], 0) +
                            max(price - high_call["strike"], 0))
        profit = (payoff_per_share - net_cost_per_share) * num_contracts * 100
        profit = min(max(profit, -net_cost), max_profit)
        profits.append(profit)

    details = f"Compra {num_contracts} Call {low_call['strike']} ARS, Vende {2 * num_contracts} Call {mid_call['strike']} ARS, Compra {num_contracts} Call {high_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Mariposa (Calls)", details

def calculate_condor_spread(low_call, mid_low_call, mid_high_call, high_call, current_price, num_contracts, debug=False):
    low_price = get_strategy_price(low_call, "buy", current_price)
    mid_low_price = get_strategy_price(mid_low_call, "sell", current_price)
    mid_high_price = get_strategy_price(mid_high_call, "sell", current_price)
    high_price = get_strategy_price(high_call, "buy", current_price)
    if any(p is None for p in [low_price, mid_low_price, mid_high_price, high_price]):
        return None
    net_cost = (low_price + high_price - mid_low_price - mid_high_price) * num_contracts * 100
    net_cost_per_share = low_price + high_price - mid_low_price - mid_high_price

    if debug:
        st.write(f"Condor Spread - Low Call: {low_price}, Mid Low Call: {mid_low_price}, Mid High Call: {mid_high_price}, High Call: {high_price}, Net Cost per Share: {net_cost_per_share}")

    payoff_at_high_S = (mid_low_call["strike"] - low_call["strike"]) + (mid_high_call["strike"] - high_call["strike"])
    max_loss = (-payoff_at_high_S * 100 * num_contracts) + net_cost if payoff_at_high_S < 0 else net_cost
    max_payoff_per_share = min(mid_low_call["strike"] - low_call["strike"], high_call["strike"] - mid_high_call["strike"])
    max_profit = (max_payoff_per_share - net_cost_per_share) * 100 * num_contracts

    breakeven1 = low_call["strike"] + net_cost_per_share
    breakeven2 = high_call["strike"] - net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = (max(price - low_call["strike"], 0) - max(price - mid_low_call["strike"], 0) -
                            max(price - mid_high_call["strike"], 0) + max(price - high_call["strike"], 0))
        profit = (payoff_per_share - net_cost_per_share) * 100 * num_contracts
        profit = min(max(profit, -max_loss), max_profit)
        profits.append(profit)

    details = f"Compra {num_contracts} Call {low_call['strike']} ARS, Vende {num_contracts} Call {mid_low_call['strike']} ARS, Vende {num_contracts} Call {mid_high_call['strike']} ARS, Compra {num_contracts} Call {high_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Cóndor (Calls)", details

def calculate_straddle(call, put, current_price, num_contracts, debug=False):
    call_price = get_strategy_price(call, "buy", current_price)
    put_price = get_strategy_price(put, "buy", current_price)
    if call_price is None or put_price is None:
        return None
    net_cost = (call_price + put_price) * num_contracts * 100
    net_cost_per_share = call_price + put_price

    if debug:
        st.write(f"Straddle - Call: {call_price}, Put: {put_price}, Net Cost per Share: {net_cost_per_share}")

    max_profit = float('inf')
    max_loss = net_cost
    breakeven1 = call["strike"] + net_cost_per_share
    breakeven2 = call["strike"] - net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = (max(price - call["strike"], 0) + max(put["strike"] - price, 0))
        profit = (payoff_per_share - net_cost_per_share) * num_contracts * 100
        profit = min(profit, max_profit) if profit > 0 else max(profit, -max_loss)
        profits.append(profit)

    details = f"Compra {num_contracts} Call {call['strike']} ARS y {num_contracts} Put {put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Straddle", details

def calculate_strangle(call, put, current_price, num_contracts, debug=False):
    call_price = get_strategy_price(call, "buy", current_price)
    put_price = get_strategy_price(put, "buy", current_price)
    if call_price is None or put_price is None:
        return None
    net_cost = (call_price + put_price) * num_contracts * 100
    net_cost_per_share = call_price + put_price

    if debug:
        st.write(f"Strangle - Call: {call_price}, Put: {put_price}, Net Cost per Share: {net_cost_per_share}")

    max_profit = float('inf')
    max_loss = net_cost
    breakeven1 = call["strike"] + net_cost_per_share
    breakeven2 = put["strike"] - net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = (max(price - call["strike"], 0) + max(put["strike"] - price, 0))
        profit = (payoff_per_share - net_cost_per_share) * num_contracts * 100
        profit = min(profit, max_profit) if profit > 0 else max(profit, -max_loss)
        profits.append(profit)

    details = f"Compra {num_contracts} Call {call['strike']} ARS y {num_contracts} Put {put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Strangle", details

def calculate_put_butterfly(low_put, mid_put, high_put, current_price, num_contracts, debug=False):
    low_price = get_strategy_price(low_put, "buy", current_price)
    mid_price = get_strategy_price(mid_put, "sell", current_price)
    high_price = get_strategy_price(high_put, "buy", current_price)
    if low_price is None or mid_price is None or high_price is None:
        return None
    net_cost = (low_price + high_price - 2 * mid_price) * num_contracts * 100
    net_cost_per_share = low_price + high_price - 2 * mid_price

    if debug:
        st.write(f"Put Butterfly - Low Put: {low_price}, Mid Put: {mid_price}, High Put: {high_price}, Net Cost per Share: {net_cost_per_share}")

    max_profit = (mid_put["strike"] - low_put["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven1 = low_put["strike"] - net_cost_per_share
    breakeven2 = high_put["strike"] + net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = (max(low_put["strike"] - price, 0) - 2 * max(mid_put["strike"] - price, 0) +
                            max(high_put["strike"] - price, 0))
        profit = (payoff_per_share - net_cost_per_share) * num_contracts * 100
        profit = min(max(profit, -net_cost), max_profit)
        profits.append(profit)

    details = f"Compra {num_contracts} Put {low_put['strike']} ARS, Vende {2 * num_contracts} Put {mid_put['strike']} ARS, Compra {num_contracts} Put {high_put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Mariposa (Puts)", details

def calculate_iron_condor(call_short, call_long, put_short, put_long, current_price, num_contracts, debug=False):
    call_short_price = get_strategy_price(call_short, "sell", current_price)
    call_long_price = get_strategy_price(call_long, "buy", current_price)
    put_short_price = get_strategy_price(put_short, "sell", current_price)
    put_long_price = get_strategy_price(put_long, "buy", current_price)
    if any(p is None for p in [call_short_price, call_long_price, put_short_price, put_long_price]):
        return None
    net_cost = (call_long_price + put_long_price - call_short_price - put_short_price) * num_contracts * 100
    net_cost_per_share = call_long_price + put_long_price - call_short_price - put_short_price

    if debug:
        st.write(f"Iron Condor - Short Call: {call_short_price}, Long Call: {call_long_price}, Short Put: {put_short_price}, Long Put: {put_long_price}, Net Cost per Share: {net_cost_per_share}")

    max_profit = -net_cost if net_cost < 0 else 0
    max_loss = abs(net_cost) if net_cost > 0 else (call_long["strike"] - call_short["strike"] + put_short["strike"] - put_long["strike"]) * 100 * num_contracts
    breakeven1 = put_long["strike"] + net_cost_per_share
    breakeven2 = call_short["strike"] - net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = (max(price - call_short["strike"], 0) - max(price - call_long["strike"], 0) -
                            max(put_short["strike"] - price, 0) + max(put_long["strike"] - price, 0))
        profit = (payoff_per_share - net_cost_per_share) * num_contracts * 100
        profit = min(max(profit, -max_loss), max_profit)
        profits.append(profit)

    details = f"Vende {num_contracts} Call {call_short['strike']} ARS, Compra {num_contracts} Call {call_long['strike']} ARS, Vende {num_contracts} Put {put_short['strike']} ARS, Compra {num_contracts} Put {put_long['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Iron Condor", details

def calculate_put_bull_spread(long_put, short_put, current_price, num_contracts, debug=False):
    long_price = get_strategy_price(long_put, "buy", current_price)
    short_price = get_strategy_price(short_put, "sell", current_price)
    if long_price is None or short_price is None:
        return None
    net_cost = (long_price - short_price) * num_contracts * 100
    net_cost_per_share = long_price - short_price

    if debug:
        st.write(f"Put Bull Spread - Long Put: {long_price}, Short Put: {short_price}, Net Cost per Share: {net_cost_per_share}")

    max_profit = (short_put["strike"] - long_put["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven = long_put["strike"] + net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = max(short_put["strike"] - price, 0) - max(long_put["strike"] - price, 0)
        if price >= short_put["strike"]:
            profit = max_profit
        elif price >= long_put["strike"]:
            profit = (payoff_per_share - net_cost_per_share) * num_contracts * 100
        else:
            profit = -net_cost
        profits.append(profit)

    details = f"Compra {num_contracts} Put {long_put['strike']} ARS, Vende {num_contracts} Put {short_put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Put Bull Spread", details

def calculate_call_bear_spread(long_call, short_call, current_price, num_contracts, debug=False):
    long_price = get_strategy_price(long_call, "buy", current_price)
    short_price = get_strategy_price(short_call, "sell", current_price)
    if long_price is None or short_price is None:
        return None
    net_cost = (long_price - short_price) * num_contracts * 100
    net_cost_per_share = long_price - short_price

    if debug:
        st.write(f"Call Bear Spread - Long Call: {long_price}, Short Call: {short_price}, Net Cost per Share: {net_cost_per_share}")

    max_profit = (long_call["strike"] - short_call["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven = long_call["strike"] - net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        payoff_per_share = max(long_call["strike"] - price, 0) - max(short_call["strike"] - price, 0)
        if price <= short_call["strike"]:
            profit = max_profit
        elif price <= long_call["strike"]:
            profit = (payoff_per_share - net_cost_per_share) * num_contracts * 100
        else:
            profit = -net_cost
        profits.append(profit)

    details = f"Compra {num_contracts} Call {long_call['strike']} ARS, Vende {num_contracts} Call {short_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Call Bear Spread", details

def count_combinations(iterable, r):
    return comb(len(iterable), r) if len(iterable) >= r else 0


def analyze_all_strategies(calls, puts, current_price, lower_percentage, upper_percentage,
                           max_loss_to_profit_ratio=0.3, min_credit_to_loss_ratio=0.7,
                           exclude_loss_to_profit=True, exclude_credit_to_loss=True,
                           included_strategies=None, exclude_bullish=False, exclude_bearish=False, debug=False):
    if included_strategies is None:
        included_strategies = {
            "Bull Spread (Calls)", "Bear Spread (Puts)", "Mariposa (Calls)", "Cóndor (Calls)",
            "Straddle", "Strangle", "Mariposa (Puts)", "Iron Condor", "Put Bull Spread", "Call Bear Spread"
        }

    # Define bullish and bearish strategies
    bullish_strategies = {"Bull Spread (Calls)", "Put Bull Spread", "Straddle", "Strangle"}
    bearish_strategies = {"Bear Spread (Puts)", "Call Bear Spread"}

    # Apply filters based on user selection
    filtered_strategies = included_strategies.copy()
    if exclude_bullish:
        filtered_strategies -= bullish_strategies
    if exclude_bearish:
        filtered_strategies -= bearish_strategies

    results = []
    max_contracts = 10

    min_strike = current_price * (1 - lower_percentage)
    max_strike = current_price * (1 + upper_percentage)
    filtered_calls = [c for c in calls if min_strike <= c["strike"] <= max_strike]
    filtered_puts = [p for p in puts if min_strike <= p["strike"] <= max_strike]

    progress = st.progress(0)
    total_tasks = (
            (count_combinations(filtered_calls, 2) * 10 if "Bull Spread (Calls)" in filtered_strategies else 0) +
            (count_combinations(filtered_puts, 2) * 10 if "Bear Spread (Puts)" in filtered_strategies else 0) +
            (count_combinations(filtered_calls, 3) * 10 if "Mariposa (Calls)" in filtered_strategies else 0) +
            (count_combinations(filtered_calls, 4) * 10 if "Cóndor (Calls)" in filtered_strategies else 0) +
            (len(set(c["strike"] for c in filtered_calls) & set(
                p["strike"] for p in filtered_puts)) * 10 if "Straddle" in filtered_strategies else 0) +
            (len([(c, p) for c in filtered_calls for p in filtered_puts if
                  c["strike"] > p["strike"]]) * 10 if "Strangle" in filtered_strategies else 0) +
            (count_combinations(filtered_puts, 3) * 10 if "Mariposa (Puts)" in filtered_strategies else 0) +
            (count_combinations(filtered_calls, 2) * count_combinations(filtered_puts,
                                                                        2) * 10 if "Iron Condor" in filtered_strategies else 0) +
            (count_combinations(filtered_puts, 2) * 10 if "Put Bull Spread" in filtered_strategies else 0) +
            (count_combinations(filtered_calls, 2) * 10 if "Call Bear Spread" in filtered_strategies else 0)
    )
    task_count = 0

    # Rest of the function remains the same, just replace `included_strategies` with `filtered_strategies` in all conditions
    if "Bull Spread (Calls)" in filtered_strategies:
        for long_call, short_call in combinations(filtered_calls, 2):
            if long_call["strike"] < short_call["strike"]:
                for num_contracts in range(1, max_contracts + 1):
                    result = calculate_bull_spread(long_call, short_call, current_price, num_contracts, debug)
                    if result is not None:
                        net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                        if net_cost > 0:
                            loss_to_profit_ratio = max_loss / max_profit if max_profit > 0 else float('inf')
                            meets_criteria = loss_to_profit_ratio <= max_loss_to_profit_ratio
                            if meets_criteria or not exclude_loss_to_profit:
                                result_dict = {
                                    "strategy": strat_name,
                                    "strikes": [long_call["strike"], short_call["strike"]],
                                    "num_contracts": num_contracts,
                                    "net_cost": net_cost,
                                    "max_profit": max_profit,
                                    "max_loss": max_loss,
                                    "profit_potential": max_profit / net_cost if max_profit > 0 else 0,
                                    "breakevens": breakevens,
                                    "prices": prices,
                                    "profits": profits,
                                    "details": details,
                                    "warning": "Pérdida/Ganancia excede el límite" if not meets_criteria else None
                                }
                                if not exclude_loss_to_profit or meets_criteria:
                                    results.append(result_dict)
                    task_count += 1
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    if "Bear Spread (Puts)" in filtered_strategies:
        for long_put, short_put in combinations(filtered_puts, 2):
            if long_put["strike"] > short_put["strike"]:
                for num_contracts in range(1, max_contracts + 1):
                    result = calculate_bear_spread(long_put, short_put, current_price, num_contracts, debug)
                    if result is not None:
                        net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                        if net_cost > 0:
                            loss_to_profit_ratio = max_loss / max_profit if max_profit > 0 else float('inf')
                            meets_criteria = loss_to_profit_ratio <= max_loss_to_profit_ratio
                            if meets_criteria or not exclude_loss_to_profit:
                                result_dict = {
                                    "strategy": strat_name,
                                    "strikes": [long_put["strike"], short_put["strike"]],
                                    "num_contracts": num_contracts,
                                    "net_cost": net_cost,
                                    "max_profit": max_profit,
                                    "max_loss": max_loss,
                                    "profit_potential": max_profit / net_cost if max_profit > 0 else 0,
                                    "breakevens": breakevens,
                                    "prices": prices,
                                    "profits": profits,
                                    "details": details,
                                    "warning": "Pérdida/Ganancia excede el límite" if not meets_criteria else None
                                }
                                if not exclude_loss_to_profit or meets_criteria:
                                    results.append(result_dict)
                    task_count += 1
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    if "Mariposa (Calls)" in filtered_strategies:
        for low, mid, high in combinations(filtered_calls, 3):
            if low["strike"] < mid["strike"] < high["strike"] and (mid["strike"] - low["strike"]) == (
                    high["strike"] - mid["strike"]):
                for num_contracts in range(1, max_contracts + 1):
                    result = calculate_butterfly_spread(low, mid, high, current_price, num_contracts, debug)
                    if result is not None:
                        net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                        if net_cost > 0:
                            results.append({
                                "strategy": strat_name,
                                "strikes": [low["strike"], mid["strike"], high["strike"]],
                                "num_contracts": num_contracts,
                                "net_cost": net_cost,
                                "max_profit": max_profit,
                                "max_loss": max_loss,
                                "profit_potential": max_profit / net_cost if max_profit > 0 else 0,
                                "breakevens": breakevens,
                                "prices": prices,
                                "profits": profits,
                                "details": details,
                                "warning": None
                            })
                    task_count += 1
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    if "Cóndor (Calls)" in filtered_strategies:
        for low, mid_low, mid_high, high in combinations(filtered_calls, 4):
            if (low["strike"] < mid_low["strike"] < mid_high["strike"] < high["strike"] and
                    mid_low["strike"] - low["strike"] == high["strike"] - mid_high["strike"]):
                for num_contracts in range(1, max_contracts + 1):
                    result = calculate_condor_spread(low, mid_low, mid_high, high, current_price, num_contracts, debug)
                    if result is not None:
                        net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                        if net_cost > 0:
                            results.append({
                                "strategy": strat_name,
                                "strikes": [low["strike"], mid_low["strike"], mid_high["strike"], high["strike"]],
                                "num_contracts": num_contracts,
                                "net_cost": net_cost,
                                "max_profit": max_profit,
                                "max_loss": max_loss,
                                "profit_potential": max_profit / net_cost if max_profit > 0 else 0,
                                "breakevens": breakevens,
                                "prices": prices,
                                "profits": profits,
                                "details": details,
                                "warning": None
                            })
                    task_count += 1
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    if "Straddle" in filtered_strategies:
        for strike in set(c["strike"] for c in filtered_calls) & set(p["strike"] for p in filtered_puts):
            call = next(c for c in filtered_calls if c["strike"] == strike)
            put = next(p for p in filtered_puts if p["strike"] == strike)
            for num_contracts in range(1, max_contracts + 1):
                result = calculate_straddle(call, put, current_price, num_contracts, debug)
                if result is not None:
                    net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                    if net_cost > 0:
                        results.append({
                            "strategy": strat_name,
                            "strikes": [strike, strike],
                            "num_contracts": num_contracts,
                            "net_cost": net_cost,
                            "max_profit": max_profit,
                            "max_loss": max_loss,
                            "profit_potential": max_profit / net_cost if max_profit > 0 else 0,
                            "breakevens": breakevens,
                            "prices": prices,
                            "profits": profits,
                            "details": details,
                            "warning": None
                        })
                task_count += 1
                progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    if "Strangle" in filtered_strategies:
        for call, put in [(c, p) for c in filtered_calls for p in filtered_puts if c["strike"] > p["strike"]]:
            for num_contracts in range(1, max_contracts + 1):
                result = calculate_strangle(call, put, current_price, num_contracts, debug)
                if result is not None:
                    net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                    if net_cost > 0:
                        results.append({
                            "strategy": strat_name,
                            "strikes": [put["strike"], call["strike"]],
                            "num_contracts": num_contracts,
                            "net_cost": net_cost,
                            "max_profit": max_profit,
                            "max_loss": max_loss,
                            "profit_potential": max_profit / net_cost if max_profit > 0 else 0,
                            "breakevens": breakevens,
                            "prices": prices,
                            "profits": profits,
                            "details": details,
                            "warning": None
                        })
                task_count += 1
                progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    if "Mariposa (Puts)" in filtered_strategies:
        for low, mid, high in combinations(filtered_puts, 3):
            if low["strike"] < mid["strike"] < high["strike"] and (mid["strike"] - low["strike"]) == (
                    high["strike"] - mid["strike"]):
                for num_contracts in range(1, max_contracts + 1):
                    result = calculate_put_butterfly(low, mid, high, current_price, num_contracts, debug)
                    if result is not None:
                        net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                        if net_cost > 0:
                            results.append({
                                "strategy": strat_name,
                                "strikes": [low["strike"], mid["strike"], high["strike"]],
                                "num_contracts": num_contracts,
                                "net_cost": net_cost,
                                "max_profit": max_profit,
                                "max_loss": max_loss,
                                "profit_potential": max_profit / net_cost if max_profit > 0 else 0,
                                "breakevens": breakevens,
                                "prices": prices,
                                "profits": profits,
                                "details": details,
                                "warning": None
                            })
                    task_count += 1
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    if "Iron Condor" in filtered_strategies:
        for call_short, call_long in combinations(filtered_calls, 2):
            if call_short["strike"] < call_long["strike"]:
                for put_short, put_long in combinations(filtered_puts, 2):
                    if put_short["strike"] > put_long["strike"] and put_long["strike"] < call_short["strike"]:
                        for num_contracts in range(1, max_contracts + 1):
                            result = calculate_iron_condor(call_short, call_long, put_short, put_long, current_price,
                                                           num_contracts, debug)
                            if result is not None:
                                net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                                if net_cost > 0 or net_cost < 0:
                                    results.append({
                                        "strategy": strat_name,
                                        "strikes": [call_short["strike"], call_long["strike"], put_short["strike"],
                                                    put_long["strike"]],
                                        "num_contracts": num_contracts,
                                        "net_cost": net_cost,
                                        "max_profit": max_profit,
                                        "max_loss": max_loss,
                                        "profit_potential": max_profit / abs(net_cost) if net_cost != 0 else 0,
                                        "breakevens": breakevens,
                                        "prices": prices,
                                        "profits": profits,
                                        "details": details,
                                        "warning": None
                                    })
                            task_count += 1
                            progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    if "Put Bull Spread" in filtered_strategies:
        for long_put, short_put in combinations(filtered_puts, 2):
            if long_put["strike"] < short_put["strike"]:
                for num_contracts in range(1, max_contracts + 1):
                    result = calculate_put_bull_spread(long_put, short_put, current_price, num_contracts, debug)
                    if result is not None:
                        net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                        if net_cost < 0:
                            credit_received = -net_cost
                            credit_to_loss_ratio = credit_received / max_loss if max_loss > 0 else float('inf')
                            meets_criteria = credit_to_loss_ratio >= min_credit_to_loss_ratio
                            if meets_criteria or not exclude_credit_to_loss:
                                result_dict = {
                                    "strategy": strat_name,
                                    "strikes": [long_put["strike"], short_put["strike"]],
                                    "num_contracts": num_contracts,
                                    "net_cost": net_cost,
                                    "max_profit": max_profit,
                                    "max_loss": max_loss,
                                    "profit_potential": max_profit / abs(net_cost) if net_cost != 0 else 0,
                                    "breakevens": breakevens,
                                    "prices": prices,
                                    "profits": profits,
                                    "details": details,
                                    "warning": "Crédito/Pérdida menor al límite" if not meets_criteria else None
                                }
                                if not exclude_credit_to_loss or meets_criteria:
                                    results.append(result_dict)
                        elif net_cost > 0:
                            results.append({
                                "strategy": strat_name,
                                "strikes": [long_put["strike"], short_put["strike"]],
                                "num_contracts": num_contracts,
                                "net_cost": net_cost,
                                "max_profit": max_profit,
                                "max_loss": max_loss,
                                "profit_potential": max_profit / net_cost if max_profit > 0 else 0,
                                "breakevens": breakevens,
                                "prices": prices,
                                "profits": profits,
                                "details": details,
                                "warning": None
                            })
                    task_count += 1
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    if "Call Bear Spread" in filtered_strategies:
        for long_call, short_call in combinations(filtered_calls, 2):
            if long_call["strike"] > short_call["strike"]:
                for num_contracts in range(1, max_contracts + 1):
                    result = calculate_call_bear_spread(long_call, short_call, current_price, num_contracts, debug)
                    if result is not None:
                        net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
                        if net_cost < 0:
                            credit_received = -net_cost
                            credit_to_loss_ratio = credit_received / max_loss if max_loss > 0 else float('inf')
                            meets_criteria = credit_to_loss_ratio >= min_credit_to_loss_ratio
                            if meets_criteria or not exclude_credit_to_loss:
                                result_dict = {
                                    "strategy": strat_name,
                                    "strikes": [long_call["strike"], short_call["strike"]],
                                    "num_contracts": num_contracts,
                                    "net_cost": net_cost,
                                    "max_profit": max_profit,
                                    "max_loss": max_loss,
                                    "profit_potential": max_profit / abs(net_cost) if net_cost != 0 else 0,
                                    "breakevens": breakevens,
                                    "prices": prices,
                                    "profits": profits,
                                    "details": details,
                                    "warning": "Crédito/Pérdida menor al límite" if not meets_criteria else None
                                }
                                if not exclude_credit_to_loss or meets_criteria:
                                    results.append(result_dict)
                        elif net_cost > 0:
                            results.append({
                                "strategy": strat_name,
                                "strikes": [long_call["strike"], short_call["strike"]],
                                "num_contracts": num_contracts,
                                "net_cost": net_cost,
                                "max_profit": max_profit,
                                "max_loss": max_loss,
                                "profit_potential": max_profit / net_cost if max_profit > 0 else 0,
                                "breakevens": breakevens,
                                "prices": prices,
                                "profits": profits,
                                "details": details,
                                "warning": None
                            })
                    task_count += 1
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    progress.empty()
    return sorted(results, key=lambda x: x["profit_potential"], reverse=True)[:10]

def plot_strategy(prices, profits, current_price, breakevens, strategy_name, expiration, num_contracts):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=profits, mode="lines", name="Ganancia/Pérdida"))
    fig.add_vline(x=current_price, line_dash="dash", line_color="gray", annotation_text="Precio Actual")
    for b in breakevens:
        fig.add_vline(x=b, line_dash="dash", line_color="green", annotation_text="Punto de Equilibrio")
    fig.update_layout(
        title=f"{strategy_name} - Ganancia/Pérdida (Vencimiento: {expiration.strftime('%Y-%m-%d')}, {num_contracts} Contratos)",
        xaxis_title="Precio de la Acción (ARS)",
        yaxis_title="Ganancia/Pérdida (ARS)"
    )
    return fig

def create_spread_matrix(calls, puts, current_price, strategy_type="Bull Spread (Calls)", num_contracts=1):
    filtered_calls = sorted([c for c in calls if c["strike"] >= 4900], key=lambda x: x["strike"])
    filtered_puts = sorted([p for p in puts if p["strike"] >= 4900], key=lambda x: x["strike"])

    strikes = filtered_calls if strategy_type.startswith("Bull") or strategy_type.startswith("Call") else filtered_puts
    profit_matrix = []
    cost_matrix = []
    ratio_matrix = []

    for long_strike in strikes:
        profit_row = []
        cost_row = []
        ratio_row = []
        for short_strike in strikes:
            if (strategy_type == "Bull Spread (Calls)" and long_strike["strike"] < short_strike["strike"]) or \
               (strategy_type == "Bear Spread (Puts)" and long_strike["strike"] > short_strike["strike"]) or \
               (strategy_type == "Put Bull Spread" and long_strike["strike"] < short_strike["strike"]) or \
               (strategy_type == "Call Bear Spread" and long_strike["strike"] > short_strike["strike"]):
                if strategy_type == "Bull Spread (Calls)":
                    result = calculate_bull_spread(long_strike, short_strike, current_price, num_contracts, debug=False)
                elif strategy_type == "Bear Spread (Puts)":
                    result = calculate_bear_spread(long_strike, short_strike, current_price, num_contracts, debug=False)
                elif strategy_type == "Put Bull Spread":
                    result = calculate_put_bull_spread(long_strike, short_strike, current_price, num_contracts, debug=False)
                elif strategy_type == "Call Bear Spread":
                    result = calculate_call_bear_spread(long_strike, short_strike, current_price, num_contracts, debug=False)

                if result is not None:
                    net_cost, max_profit, max_loss, _, prices, profits, _, _ = result
                    profit_row.append(max_profit)
                    cost_row.append(net_cost)
                    if net_cost > 0 and max_profit > 0:
                        ratio = max_profit / net_cost
                        ratio_row.append(min(ratio, 100))
                    else:
                        ratio_row.append(np.nan)
                else:
                    profit_row.append(np.nan)
                    cost_row.append(np.nan)
                    ratio_row.append(np.nan)
            else:
                profit_row.append(np.nan)
                cost_row.append(np.nan)
                ratio_row.append(np.nan)
        profit_matrix.append(profit_row)
        cost_matrix.append(cost_row)
        ratio_matrix.append(ratio_row)

    strike_labels = [f"{s['strike']:.1f}" for s in strikes]
    profit_df = pd.DataFrame(profit_matrix, columns=strike_labels, index=strike_labels).fillna(np.nan)
    cost_df = pd.DataFrame(cost_matrix, columns=strike_labels, index=strike_labels).fillna(np.nan)
    ratio_df = pd.DataFrame(ratio_matrix, columns=strike_labels, index=strike_labels).fillna(np.nan)

    return profit_df, cost_df, ratio_df

def main():
    st.title("Visualizador de Estrategias de Opciones GGAL (Strikes ≥ 4900 ARS)")

    ggal_stock, ggal_options = get_ggal_data()
    if not ggal_stock or not ggal_options:
        return

    current_price = float(ggal_stock["c"])
    now_ar = datetime.now(timezone.utc) + timedelta(hours=-3)
    st.write(f"Precio Actual de la Acción GGAL: {current_price} ARS (al {now_ar.strftime('%Y-%m-%d %H:%M:%S')} AR)")
    st.write(f"Mercado {'abierto' if is_market_hours() else 'cerrado'} - Usando {'bid/ask' if is_market_hours() else 'precios de cierre'} para cálculos.")

    if st.button("Actualizar Datos"):
        ggal_stock, ggal_options = get_ggal_data()
        current_price = float(ggal_stock["c"]) if ggal_stock else current_price
        now_ar = datetime.now(timezone.utc) + timedelta(hours=-3)
        st.write(f"Datos actualizados. Precio Actual: {current_price} ARS (al {now_ar.strftime('%Y-%m-%d %H:%M:%S')} AR)")

    debug_mode = st.checkbox("Modo Depuración (mostrar detalles técnicos)", value=False)

    expirations = sorted(list(set(o["expiration"] for o in ggal_options)))
    strikes_by_expiration = {}
    all_strikes = set()
    for exp in expirations:
        calls = [o for o in ggal_options if o["type"] == "call" and o["expiration"] == exp]
        puts = [o for o in ggal_options if o["type"] == "put" and o["expiration"] == exp]
        strikes_by_expiration[exp] = {
            "calls": sorted(calls, key=lambda x: x["strike"]),
            "puts": sorted(puts, key=lambda x: x["strike"])
        }
        all_strikes.update(c["strike"] for c in calls)
        all_strikes.update(p["strike"] for p in puts)

    min_strike = float(min(all_strikes))
    max_strike = float(max(all_strikes))

    tab1, tab2, tab3 = st.tabs(["Selección Manual de Estrategia", "Análisis Automático", "Matrices de Spreads"])

    with tab1:
        strategies = ["Bull Spread (Calls)", "Bear Spread (Puts)", "Mariposa (Calls)", "Cóndor (Calls)",
                      "Straddle", "Strangle", "Mariposa (Puts)", "Iron Condor",
                      "Put Bull Spread", "Call Bear Spread"]
        selected_strategy = st.selectbox("Seleccioná una Estrategia", strategies)
        selected_exp = st.selectbox("Seleccioná Fecha de Vencimiento", expirations,
                                    format_func=lambda x: x.strftime("%Y-%m-%d"))
        num_contracts = st.number_input("Número de Contratos (1 contrato = 100 opciones)", min_value=1, value=1, step=1)

        calls = strikes_by_expiration[selected_exp]["calls"]
        puts = strikes_by_expiration[selected_exp]["puts"]
        call_strikes = [c["strike"] for c in calls]
        put_strikes = [p["strike"] for p in puts]

        result = None
        if selected_strategy == "Bull Spread (Calls)":
            if len(call_strikes) < 2:
                st.error("No hay suficientes opciones call ≥ 4900 ARS.")
            else:
                long_strike = st.selectbox("Strike Menor (Compra Call)", call_strikes)
                short_strike_idx = call_strikes.index(long_strike) + 1
                short_strike = st.selectbox("Strike Mayor (Venta Call)", call_strikes[short_strike_idx:], index=0)
                long_call = next(c for c in calls if c["strike"] == long_strike)
                short_call = next(c for c in calls if c["strike"] == short_strike)
                result = calculate_bull_spread(long_call, short_call, current_price, num_contracts, debug_mode)

        elif selected_strategy == "Bear Spread (Puts)":
            if len(put_strikes) < 2:
                st.error("No hay suficientes opciones put ≥ 4900 ARS.")
            else:
                long_strike = st.selectbox("Strike Mayor (Compra Put)", put_strikes[::-1])
                short_strike_idx = len(put_strikes) - put_strikes[::-1].index(long_strike)
                short_strike = st.selectbox("Strike Menor (Venta Put)", put_strikes[:short_strike_idx - 1:-1], index=0)
                long_put = next(p for p in puts if p["strike"] == long_strike)
                short_put = next(p for p in puts if p["strike"] == short_strike)
                result = calculate_bear_spread(long_put, short_put, current_price, num_contracts, debug_mode)

        elif selected_strategy == "Mariposa (Calls)":
            if len(call_strikes) < 3:
                st.error("No hay suficientes opciones call ≥ 4900 ARS.")
            else:
                low_strike = st.selectbox("Strike Menor", call_strikes)
                mid_idx = call_strikes.index(low_strike) + 1
                mid_strike = st.selectbox("Strike Medio", call_strikes[mid_idx:], index=0)
                high_idx = call_strikes.index(mid_strike) + 1
                high_strike = st.selectbox("Strike Mayor", call_strikes[high_idx:], index=0)
                low_call = next(c for c in calls if c["strike"] == low_strike)
                mid_call = next(c for c in calls if c["strike"] == mid_strike)
                high_call = next(c for c in calls if c["strike"] == high_strike)
                result = calculate_butterfly_spread(low_call, mid_call, high_call, current_price, num_contracts, debug_mode)

        elif selected_strategy == "Cóndor (Calls)":
            if len(call_strikes) < 4:
                st.error("No hay suficientes opciones call ≥ 4900 ARS.")
            else:
                low_strike = st.selectbox("Strike Menor", call_strikes)
                mid_low_idx = call_strikes.index(low_strike) + 1
                mid_low_strike = st.selectbox("Strike Medio-Bajo", call_strikes[mid_low_idx:], index=0)
                mid_high_idx = call_strikes.index(mid_low_strike) + 1
                mid_high_strike = st.selectbox("Strike Medio-Alto", call_strikes[mid_high_idx:], index=0)
                high_idx = call_strikes.index(mid_high_strike) + 1
                high_strike = st.selectbox("Strike Mayor", call_strikes[high_idx:], index=0)
                low_call = next(c for c in calls if c["strike"] == low_strike)
                mid_low_call = next(c for c in calls if c["strike"] == mid_low_strike)
                mid_high_call = next(c for c in calls if c["strike"] == mid_high_strike)
                high_call = next(c for c in calls if c["strike"] == high_strike)
                result = calculate_condor_spread(low_call, mid_low_call, mid_high_call, high_call, current_price, num_contracts, debug_mode)

        elif selected_strategy == "Straddle":
            if not (call_strikes and put_strikes):
                st.error("No hay suficientes opciones call y put ≥ 4900 ARS.")
            else:
                strike = st.selectbox("Strike (Compra Call y Put)", sorted(set(call_strikes) & set(put_strikes)))
                call = next(c for c in calls if c["strike"] == strike)
                put = next(p for p in puts if p["strike"] == strike)
                result = calculate_straddle(call, put, current_price, num_contracts, debug_mode)

        elif selected_strategy == "Strangle":
            if not (call_strikes and put_strikes):
                st.error("No hay suficientes opciones call y put ≥ 4900 ARS.")
            else:
                put_strike = st.selectbox("Strike Menor (Compra Put)", sorted(put_strikes))
                call_strike_idx = call_strikes.index(next(s for s in call_strikes if s > put_strike)) if any(s > put_strike for s in call_strikes) else -1
                if call_strike_idx == -1:
                    st.error("No hay un strike de call mayor que el strike de put seleccionado.")
                else:
                    call_strike = st.selectbox("Strike Mayor (Compra Call)", call_strikes[call_strike_idx:], index=0)
                    call = next(c for c in calls if c["strike"] == call_strike)
                    put = next(p for p in puts if p["strike"] == put_strike)
                    result = calculate_strangle(call, put, current_price, num_contracts, debug_mode)

        elif selected_strategy == "Mariposa (Puts)":
            if len(put_strikes) < 3:
                st.error("No hay suficientes opciones put ≥ 4900 ARS.")
            else:
                low_strike = st.selectbox("Strike Menor", put_strikes)
                mid_idx = put_strikes.index(low_strike) + 1
                mid_strike = st.selectbox("Strike Medio", put_strikes[mid_idx:], index=0)
                high_idx = put_strikes.index(mid_strike) + 1
                high_strike = st.selectbox("Strike Mayor", put_strikes[high_idx:], index=0)
                low_put = next(p for p in puts if p["strike"] == low_strike)
                mid_put = next(p for p in puts if p["strike"] == mid_strike)
                high_put = next(p for p in puts if p["strike"] == high_strike)
                result = calculate_put_butterfly(low_put, mid_put, high_put, current_price, num_contracts, debug_mode)

        elif selected_strategy == "Iron Condor":
            if len(call_strikes) < 2 or len(put_strikes) < 2:
                st.error("No hay suficientes opciones call y put ≥ 4900 ARS.")
            else:
                call_short_strike = st.selectbox("Strike Call Venta (Mayores)", sorted(call_strikes, reverse=True))
                call_long_idx = call_strikes.index(call_short_strike) - 1 if call_strikes.index(call_short_strike) > 0 else -1
                if call_long_idx == -1:
                    st.error("No hay un strike de call menor que el strike de venta seleccionado.")
                else:
                    call_long_strike = st.selectbox("Strike Call Compra (Menores)", call_strikes[:call_long_idx + 1], index=0)
                    put_short_strike = st.selectbox("Strike Put Venta (Menores)", sorted(put_strikes))
                    put_long_idx = put_strikes.index(put_short_strike) - 1 if put_strikes.index(put_short_strike) > 0 else -1
                    if put_long_idx == -1:
                        st.error("No hay un strike de put mayor que el strike de venta seleccionado.")
                    else:
                        put_long_strike = st.selectbox("Strike Put Compra (Mayores)", put_strikes[:put_long_idx + 1], index=0, key="put_long")
                        call_short = next(c for c in calls if c["strike"] == call_short_strike)
                        call_long = next(c for c in calls if c["strike"] == call_long_strike)
                        put_short = next(p for p in puts if p["strike"] == put_short_strike)
                        put_long = next(p for p in puts if p["strike"] == put_long_strike)
                        result = calculate_iron_condor(call_short, call_long, put_short, put_long, current_price, num_contracts, debug_mode)

        elif selected_strategy == "Put Bull Spread":
            if len(put_strikes) < 2:
                st.error("No hay suficientes opciones put ≥ 4900 ARS.")
            else:
                long_strike = st.selectbox("Strike Menor (Compra Put)", put_strikes)
                short_strike_idx = put_strikes.index(long_strike) + 1
                short_strike = st.selectbox("Strike Mayor (Venta Put)", put_strikes[short_strike_idx:], index=0)
                long_put = next(p for p in puts if p["strike"] == long_strike)
                short_put = next(p for p in puts if p["strike"] == short_strike)
                result = calculate_put_bull_spread(long_put, short_put, current_price, num_contracts, debug_mode)

        elif selected_strategy == "Call Bear Spread":
            if len(call_strikes) < 2:
                st.error("No hay suficientes opciones call ≥ 4900 ARS.")
            else:
                long_strike = st.selectbox("Strike Mayor (Compra Call)", sorted(call_strikes, reverse=True))
                short_strike_idx = call_strikes.index(long_strike) - 1 if call_strikes.index(long_strike) > 0 else -1
                if short_strike_idx == -1:
                    st.error("No hay un strike de call menor que el strike de compra seleccionado.")
                else:
                    short_strike = st.selectbox("Strike Menor (Venta Call)", call_strikes[:short_strike_idx + 1], index=0)
                    long_call = next(c for c in calls if c["strike"] == long_strike)
                    short_call = next(c for c in calls if c["strike"] == short_strike)
                    result = calculate_call_bear_spread(long_call, short_call, current_price, num_contracts, debug_mode)

        if result is None:
            st.error("No se puede calcular la estrategia: faltan precios bid/ask válidos durante el horario de mercado.")
        else:
            net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
            st.write(f"Detalles: {details}")
            st.write(f"Costo Neto: {net_cost:.2f} ARS")
            st.write(f"Ganancia Máxima: {max_profit:.2f} ARS" if max_profit != float('inf') else "Ganancia Máxima: Ilimitada ARS")
            st.write(f"Pérdida Máxima: {max_loss:.2f} ARS")
            st.write(f"Punto(s) de Equilibrio: {', '.join([f'{b:.2f}' for b in breakevens])} ARS")
            fig = plot_strategy(prices, profits, current_price, breakevens, strat_name, selected_exp, num_contracts)
            st.plotly_chart(fig, key=f"manual_chart_{selected_strategy}_{selected_exp.strftime('%Y%m%d')}")

    with tab2:
        st.subheader("Top 10 Combinaciones de Opciones Más Rentables (Strikes ≥ 4900 ARS)")
        selected_exp_auto = st.selectbox("Seleccioná Fecha de Vencimiento para el Análisis", expirations,
                                         format_func=lambda x: x.strftime("%Y-%m-%d"), key="auto_exp")

        lower_percentage = st.number_input("Porcentaje Inferior del Precio Actual", min_value=0.0, max_value=100.0,
                                           value=5.0, step=0.1) / 100
        upper_percentage = st.number_input("Porcentaje Superior del Precio Actual", min_value=0.0, max_value=100.0,
                                           value=5.0, step=0.1) / 100
        min_strike = current_price * (1 - lower_percentage)
        max_strike = current_price * (1 + upper_percentage)
        st.write(f"Incluyendo solo strikes entre {min_strike:.2f} y {max_strike:.2f} ARS")

        max_loss_to_profit = st.number_input("Máxima Razón Pérdida/Ganancia (Bull Call/Bear Put)", min_value=0.01,
                                             value=0.3, step=0.01)
        exclude_loss_to_profit = st.checkbox("Excluir estrategias que excedan la razón Pérdida/Ganancia", value=True,
                                             key="exclude_loss_profit")
        st.write("Si no se excluyen, las estrategias serán marcadas con una advertencia.")

        min_credit_to_loss = st.number_input("Mínima Razón Crédito/Pérdida (Bear Call/Bull Put)", min_value=0.01,
                                             value=0.7, step=0.01)
        exclude_credit_to_loss = st.checkbox("Excluir estrategias por debajo de la razón Crédito/Pérdida", value=True,
                                             key="exclude_credit_loss")
        st.write("Si no se excluyen, las estrategias serán marcadas con una advertencia.")

        # New filter options for bullish/bearish strategies
        exclude_bullish = st.checkbox("Excluir Estrategias Alcistas (Bull Spread, Put Bull Spread, Straddle, Strangle)", value=False, key="exclude_bullish")
        exclude_bearish = st.checkbox("Excluir Estrategias Bajistas (Bear Spread, Call Bear Spread)", value=False, key="exclude_bearish")

        all_strategies = ["Bull Spread (Calls)", "Bear Spread (Puts)", "Mariposa (Calls)", "Cóndor (Calls)",
                          "Straddle", "Strangle", "Mariposa (Puts)", "Iron Condor", "Put Bull Spread", "Call Bear Spread"]
        excluded_strategies = st.multiselect("Excluir Estrategias del Análisis", all_strategies, default=[])
        included_strategies = set(all_strategies) - set(excluded_strategies)
        st.write(f"Estrategias incluidas: {', '.join(included_strategies) if included_strategies else 'Ninguna'}")

        calls_auto = strikes_by_expiration[selected_exp_auto]["calls"]
        puts_auto = strikes_by_expiration[selected_exp_auto]["puts"]

        with st.spinner("Analizando estrategias..."):
            top_strategies = analyze_all_strategies(
                calls_auto, puts_auto, current_price, lower_percentage, upper_percentage,
                max_loss_to_profit, min_credit_to_loss, exclude_loss_to_profit, exclude_credit_to_loss,
                included_strategies, exclude_bullish, exclude_bearish, debug_mode
            )

        if not top_strategies:
            st.warning("No se encontraron estrategias válidas con los criterios seleccionados.")
        else:
            for i, strat in enumerate(top_strategies, 1):
                st.write(f"**#{i}: {strat['strategy']}**")
                st.write(f"Detalles: {strat['details']}")
                st.write(f"Strikes: {strat['strikes']}")
                st.write(f"Número de Contratos: {strat['num_contracts']}")
                st.write(f"Costo Neto: {strat['net_cost']:.2f} ARS")
                st.write(f"Ganancia Máxima: {strat['max_profit']:.2f} ARS" if strat['max_profit'] != float('inf') else "Ganancia Máxima: Ilimitada ARS")
                st.write(f"Pérdida Máxima: {strat['max_loss']:.2f} ARS")
                st.write(f"Potencial de Ganancia: {strat['profit_potential']:.2f}x" if strat['profit_potential'] > 0 else "Potencial de Ganancia: N/A")
                st.write(f"Punto(s) de Equilibrio: {', '.join([f'{b:.2f}' for b in strat['breakevens']])} ARS")
                if strat["warning"]:
                    st.warning(f"Advertencia: {strat['warning']}")
                if st.button(f"Ver Gráfico para Estrategia #{i}", key=f"plot_button_{i}"):
                    fig = plot_strategy(strat["prices"], strat["profits"], current_price, strat["breakevens"],
                                        strat["strategy"], selected_exp_auto, strat["num_contracts"])
                    st.plotly_chart(fig, key=f"auto_chart_{i}_{selected_exp_auto.strftime('%Y%m%d')}")
                st.write("---")

    with tab3:
        st.subheader("Matrices de Ganancia/Pérdida para Spreads")
        selected_exp_matrix = st.selectbox("Seleccioná Fecha de Vencimiento para la Matriz", expirations,
                                           format_func=lambda x: x.strftime("%Y-%m-%d"), key="matrix_exp")
        strategy_types = ["Bull Spread (Calls)", "Bear Spread (Puts)", "Put Bull Spread", "Call Bear Spread"]
        selected_strategy_type = st.selectbox("Seleccioná Tipo de Spread", strategy_types)
        num_contracts_matrix = st.number_input("Número de Contratos para la Matriz (1 contrato = 100 opciones)",
                                               min_value=1, value=1, step=1)

        if st.button("Generar Matrices"):
            with st.spinner("Generando matrices..."):
                calls_matrix = strikes_by_expiration[selected_exp_matrix]["calls"]
                puts_matrix = strikes_by_expiration[selected_exp_matrix]["puts"]
                profit_df, cost_df, ratio_df = create_spread_matrix(calls_matrix, puts_matrix, current_price,
                                                                    selected_strategy_type, num_contracts_matrix)
                st.write(f"**Matriz de Ganancia Máxima ({selected_strategy_type}) en ARS**")
                st.dataframe(profit_df.style.background_gradient(cmap='RdYlGn', axis=None,
                                                                 vmin=profit_df.min().min(),
                                                                 vmax=profit_df.max().max()))
                st.write(f"**Matriz de Costo Neto ({selected_strategy_type}) en ARS**")
                st.dataframe(cost_df.style.background_gradient(cmap='RdYlGn_r', axis=None,
                                                               vmin=cost_df.min().min(),
                                                               vmax=cost_df.max().max()))
                st.write(f"**Matriz de Relación Ganancia/Costo ({selected_strategy_type})**")
                st.dataframe(ratio_df.style.background_gradient(cmap='RdYlGn', axis=None,
                                                                vmin=ratio_df.min().min(),
                                                                vmax=ratio_df.max().max()))

if __name__ == "__main__":
    main()
