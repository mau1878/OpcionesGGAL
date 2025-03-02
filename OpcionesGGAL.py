import requests
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta, timezone
from itertools import combinations
import pandas as pd
from math import comb
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import newton
from tenacity import retry, stop_after_attempt, wait_fixed
import json

# API Endpoints
STOCK_URL = "https://data912.com/live/arg_stocks"
OPTIONS_URL = "https://data912.com/live/arg_options"

# Third Friday calculation
def get_third_friday(year: int, month: int) -> date:
    first_day = date(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
    return first_friday + timedelta(days=14)

# Expiration Mapping for 2025
EXPIRATION_MAP_2025 = {
    "M": get_third_friday(2025, 3), "MA": get_third_friday(2025, 3),
    "A": get_third_friday(2025, 4), "AB": get_third_friday(2025, 4),
    "J": get_third_friday(2025, 6), "JU": get_third_friday(2025, 6)
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@st.cache_data
def fetch_data(url: str) -> list:
    try:
        response = requests.get(url, headers={"accept": "*/*"}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error al obtener datos de {url}: {e}")
        return []

def parse_option_symbol(symbol: str) -> tuple[str | None, float | None, date | None]:
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

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(S: float, K: float, T: float, r: float, option_price: float, option_type: str = "call") -> float:
    def objective(sigma):
        if option_type == "call":
            return black_scholes_call(S, K, T, r, sigma) - option_price
        else:
            return black_scholes_put(S, K, T, r, sigma) - option_price
    try:
        return newton(objective, 0.2, tol=1e-6, maxiter=100)
    except RuntimeError:
        return np.nan

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2) / 100
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

def get_ggal_data(risk_free_rate: float) -> tuple[dict | None, list]:
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)
    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock:
        st.error("No se encontraron datos de la acción GGAL.")
        return None, []
    current_date = datetime.now(timezone.utc).date()
    ggal_options = []
    for option in options_data:
        symbol = option["symbol"]
        if symbol.startswith("GFGC") or symbol.startswith("GFGV"):
            opt_type, strike, exp = parse_option_symbol(symbol)
            if exp:
                mid_price = (option["px_bid"] + option["px_ask"]) / 2 if option["px_bid"] > 0 and option["px_ask"] > 0 else option["c"]
                if st.session_state.get('debug_mode', False):
                    st.write(f"Option {symbol}: bid={option['px_bid']}, ask={option['px_ask']}, close={option['c']}, mid={mid_price}")
                T = (exp - current_date).days / 365.0
                iv = implied_volatility(float(ggal_stock["c"]), strike, T, risk_free_rate, mid_price, opt_type)
                greeks = calculate_greeks(float(ggal_stock["c"]), strike, T, risk_free_rate, iv, opt_type)
                ggal_options.append({
                    "symbol": symbol,
                    "type": opt_type,
                    "strike": strike,
                    "expiration": exp,
                    "px_bid": option["px_bid"],
                    "px_ask": option["px_ask"],
                    "c": option["c"],
                    "iv": iv,
                    **greeks
                })
    return ggal_stock, ggal_options

def get_historical_volatility(ticker: str = "GGAL.BA", period: str = "1y", days: int = 252) -> float:
    data = yf.download(ticker, period=period, multi_level_index=False)
    if data.empty:
        return np.nan
    returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    return np.std(returns) * np.sqrt(days)

def is_market_hours() -> bool:
    now_utc = datetime.now(timezone.utc)
    ar_tz_offset = timedelta(hours=-3)
    now_ar = now_utc + ar_tz_offset
    hour, minute, weekday = now_ar.hour, now_ar.minute, now_ar.weekday()
    time_in_minutes = hour * 60 + minute
    market_open, market_close = 11 * 60, 17 * 60
    return weekday < 5 and market_open <= time_in_minutes <= market_close

def get_strategy_price(option: dict, action: str, current_stock_price: float | None = None) -> float | None:
    market_open = is_market_hours()
    if market_open:
        price = option["px_ask"] if action == "buy" else option["px_bid"]
    else:
        price = option["c"]

    if price is None:
        if st.session_state.get('debug_mode', False):
            st.write(f"Invalid price for {option['symbol']}: {price} (action: {action}, market_open: {market_open})")
        return None

    if price <= 0 and st.session_state.get('debug_mode', False):
        st.write(f"Warning: Zero or negative price for {option['symbol']}: {price}")

    return price

def calculate_bull_spread(long_call: dict, short_call: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
    long_price = get_strategy_price(long_call, "buy", current_price)
    short_price = get_strategy_price(short_call, "sell", current_price)
    if long_price is None or short_price is None:
        return None
    net_cost = (long_price - short_price) * num_contracts * 100
    net_cost_per_share = long_price - short_price
    if debug:
        st.write(f"Bull Spread - Long Call: {long_price}, Short Call: {short_price}, Net Cost per Share: {net_cost_per_share}")
    max_profit = (short_call["strike"] - long_call["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven = long_call["strike"] + net_cost_per_share
    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = [min(max((max(price - long_call["strike"], 0) - max(price - short_call["strike"], 0) - net_cost_per_share) * num_contracts * 100, -net_cost), max_profit) for price in prices]
    details = f"Compra {num_contracts} Call {long_call['strike']} ARS, Vende {num_contracts} Call {short_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Bull Spread (Calls)", details

def calculate_bear_spread(long_put: dict, short_put: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
    if long_put["strike"] <= short_put["strike"]:
        if debug:
            st.write(f"Invalid bear spread: long put strike ({long_put['strike']}) must be > short put strike ({short_put['strike']})")
        return None

    long_price = get_strategy_price(long_put, "buy", current_price)
    short_price = get_strategy_price(short_put, "sell", current_price)
    if long_price is None or short_price is None:
        return None

    net_cost = (long_price - short_price) * num_contracts * 100
    net_cost_per_share = long_price - short_price

    if debug:
        st.write(f"Bear Spread - Long Put: {long_price} at strike {long_put['strike']}, Short Put: {short_price} at strike {short_put['strike']}, Net Cost per Share: {net_cost_per_share}")

    max_profit = (long_put["strike"] - short_put["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven = long_put["strike"] - net_cost_per_share

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        if price <= short_put["strike"]:
            profit = (long_put["strike"] - short_put["strike"] - net_cost_per_share) * num_contracts * 100
        elif price >= long_put["strike"]:
            profit = -net_cost
        else:
            profit = ((long_put["strike"] - price) - net_cost_per_share) * num_contracts * 100
        profits.append(profit)

    details = f"Compra {num_contracts} Put {long_put['strike']} ARS, Vende {num_contracts} Put {short_put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Bear Spread (Puts)", details

def calculate_butterfly_spread(low_call: dict, mid_call: dict, high_call: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
    low_price = get_strategy_price(low_call, "buy", current_price)
    mid_price = get_strategy_price(mid_call, "sell", current_price)
    high_price = get_strategy_price(high_call, "buy", current_price)
    if any(p is None for p in [low_price, mid_price, high_price]):
        return None
    net_cost = (low_price + high_price - 2 * mid_price) * num_contracts * 100
    net_cost_per_share = low_price + high_price - 2 * mid_price
    if debug:
        st.write(f"Butterfly Spread - Low Call: {low_price}, Mid Call: {mid_price}, High Call: {high_price}, Net Cost per Share: {net_cost_per_share}")
    max_profit = (mid_call["strike"] - low_call["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven1, breakeven2 = low_call["strike"] + net_cost_per_share, high_call["strike"] - net_cost_per_share
    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = [min(max(((max(price - low_call["strike"], 0) - 2 * max(price - mid_call["strike"], 0) + max(price - high_call["strike"], 0)) - net_cost_per_share) * num_contracts * 100, -net_cost), max_profit) for price in prices]
    details = f"Compra {num_contracts} Call {low_call['strike']} ARS, Vende {2 * num_contracts} Call {mid_call['strike']} ARS, Compra {num_contracts} Call {high_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Mariposa (Calls)", details

def calculate_condor_spread(low_call: dict, mid_low_call: dict, mid_high_call: dict, high_call: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
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
    max_profit = min((max_payoff_per_share - net_cost_per_share) * 100 * num_contracts, (high_call["strike"] - low_call["strike"]) * 100 * num_contracts)
    breakeven1, breakeven2 = low_call["strike"] + net_cost_per_share, high_call["strike"] - net_cost_per_share
    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = [min(max(((max(price - low_call["strike"], 0) - max(price - mid_low_call["strike"], 0) - max(price - mid_high_call["strike"], 0) + max(price - high_call["strike"], 0)) - net_cost_per_share) * 100 * num_contracts, -max_loss), max_profit) for price in prices]
    details = f"Compra {num_contracts} Call {low_call['strike']} ARS, Vende {num_contracts} Call {mid_low_call['strike']} ARS, Vende {num_contracts} Call {mid_high_call['strike']} ARS, Compra {num_contracts} Call {high_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Cóndor (Calls)", details

def calculate_straddle(call: dict, put: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
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
    breakeven1, breakeven2 = call["strike"] + net_cost_per_share, call["strike"] - net_cost_per_share
    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = [max(((max(price - call["strike"], 0) + max(put["strike"] - price, 0)) - net_cost_per_share) * num_contracts * 100, -max_loss) for price in prices]
    details = f"Compra {num_contracts} Call {call['strike']} ARS y {num_contracts} Put {put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Straddle", details

def calculate_strangle(call: dict, put: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
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
    breakeven1, breakeven2 = call["strike"] + net_cost_per_share, put["strike"] - net_cost_per_share
    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = [max(((max(price - call["strike"], 0) + max(put["strike"] - price, 0)) - net_cost_per_share) * num_contracts * 100, -max_loss) for price in prices]
    details = f"Compra {num_contracts} Call {call['strike']} ARS y {num_contracts} Put {put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Strangle", details

def calculate_put_butterfly(low_put: dict, mid_put: dict, high_put: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
    low_price = get_strategy_price(low_put, "buy", current_price)
    mid_price = get_strategy_price(mid_put, "sell", current_price)
    high_price = get_strategy_price(high_put, "buy", current_price)
    if any(p is None for p in [low_price, mid_price, high_price]):
        return None
    net_cost = (low_price + high_price - 2 * mid_price) * num_contracts * 100
    net_cost_per_share = low_price + high_price - 2 * mid_price
    if debug:
        st.write(f"Put Butterfly - Low Put: {low_price}, Mid Put: {mid_price}, High Put: {high_price}, Net Cost per Share: {net_cost_per_share}")
    max_profit = (mid_put["strike"] - low_put["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven1, breakeven2 = low_put["strike"] - net_cost_per_share, high_put["strike"] + net_cost_per_share
    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = [min(max(((max(low_put["strike"] - price, 0) - 2 * max(mid_put["strike"] - price, 0) + max(high_put["strike"] - price, 0)) - net_cost_per_share) * num_contracts * 100, -net_cost), max_profit) for price in prices]
    details = f"Compra {num_contracts} Put {low_put['strike']} ARS, Vende {2 * num_contracts} Put {mid_put['strike']} ARS, Compra {num_contracts} Put {high_put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Mariposa (Puts)", details

def calculate_iron_condor(call_short: dict, call_long: dict, put_short: dict, put_long: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
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
    breakeven1, breakeven2 = put_long["strike"] + net_cost_per_share, call_short["strike"] - net_cost_per_share
    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = [min(max(((max(price - call_short["strike"], 0) - max(price - call_long["strike"], 0) - max(put_short["strike"] - price, 0) + max(put_long["strike"] - price, 0)) - net_cost_per_share) * num_contracts * 100, -max_loss), max_profit) for price in prices]
    details = f"Vende {num_contracts} Call {call_short['strike']} ARS, Compra {num_contracts} Call {call_long['strike']} ARS, Vende {num_contracts} Put {put_short['strike']} ARS, Compra {num_contracts} Put {put_long['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven1, breakeven2], prices, profits, "Iron Condor", details

def calculate_put_bull_spread(long_put: dict, short_put: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
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
    profits = [min(max(((max(short_put["strike"] - price, 0) - max(long_put["strike"] - price, 0)) - net_cost_per_share) * num_contracts * 100, -net_cost), max_profit) for price in prices]
    details = f"Compra {num_contracts} Put {long_put['strike']} ARS, Vende {num_contracts} Put {short_put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Put Bull Spread", details

def calculate_call_bear_spread(long_call: dict, short_call: dict, current_price: float, num_contracts: int, debug: bool = False) -> tuple | None:
    if long_call["strike"] <= short_call["strike"]:
        if debug:
            st.write(f"Invalid call bear spread: long strike ({long_call['strike']}) must be > short strike ({short_call['strike']})")
        return None

    long_price = get_strategy_price(long_call, "buy", current_price)
    short_price = get_strategy_price(short_call, "sell", current_price)
    if long_price is None or short_price is None:
        return None

    net_cost = (long_price - short_price) * num_contracts * 100
    net_cost_per_share = long_price - short_price

    if debug:
        st.write(f"Call Bear Spread - Long Call: {long_price} at strike {long_call['strike']}, Short Call: {short_price} at strike {short_call['strike']}, Net Cost per Share: {net_cost_per_share}")

    max_profit = (long_call["strike"] - short_call["strike"] - net_cost_per_share) * num_contracts * 100
    max_loss = net_cost
    breakeven = short_call["strike"] + (long_call["strike"] - short_call["strike"] - net_cost_per_share)

    prices = np.linspace(current_price - 5000, current_price + 5000, 200)
    profits = []
    for price in prices:
        if price <= short_call["strike"]:
            profit = -net_cost
        elif price >= long_call["strike"]:
            profit = (long_call["strike"] - short_call["strike"] - net_cost_per_share) * num_contracts * 100
        else:
            profit = ((short_call["strike"] - price) - net_cost_per_share) * num_contracts * 100
        profits.append(profit)

    details = f"Compra {num_contracts} Call {long_call['strike']} ARS, Vende {num_contracts} Call {short_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Call Bear Spread", details

def calculate_covered_call(stock_price: float, short_call: dict, num_contracts: int, debug: bool = False) -> tuple | None:
    if not isinstance(short_call, dict):
        st.error(f"Error: short_call debe ser un diccionario, se recibió {type(short_call)}: {short_call}")
        return None
    stock_cost = stock_price * 100 * num_contracts
    call_price = get_strategy_price(short_call, "sell", stock_price)
    if call_price is None:
        return None
    premium_received = call_price * 100 * num_contracts
    if debug:
        st.write(f"Covered Call - Stock Cost: {stock_cost}, Short Call Premium: {call_price}")
    net_cost = stock_cost - premium_received
    max_profit = premium_received
    max_loss = net_cost
    breakeven = stock_price - call_price
    prices = np.linspace(stock_price - 5000, stock_price + 5000, 200)
    profits = [(min(price, short_call["strike"]) - stock_price + call_price) * 100 * num_contracts for price in prices]
    details = f"Compra {num_contracts * 100} acciones GGAL a {stock_price} ARS, Vende {num_contracts} Call {short_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Covered Call", details

def calculate_protective_put(stock_price: float, long_put: dict, num_contracts: int, debug: bool = False) -> tuple | None:
    stock_cost = stock_price * 100 * num_contracts
    put_price = get_strategy_price(long_put, "buy", stock_price)
    if put_price is None:
        return None
    premium_paid = put_price * 100 * num_contracts
    if debug:
        st.write(f"Protective Put - Stock Cost: {stock_cost}, Long Put Premium: {put_price}")
    net_cost = stock_cost + premium_paid
    max_profit = float('inf')
    max_loss = net_cost - (long_put["strike"] * 100 * num_contracts)
    breakeven = stock_price + put_price
    prices = np.linspace(stock_price - 5000, stock_price + 5000, 200)
    profits = [(max(price, long_put["strike"]) - stock_price - put_price) * 100 * num_contracts for price in prices]
    details = f"Compra {num_contracts * 100} acciones GGAL a {stock_price} ARS, Compra {num_contracts} Put {long_put['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Protective Put", details

def calculate_collar(stock_price: float, long_put: dict, short_call: dict, num_contracts: int, debug: bool = False) -> tuple | None:
    stock_cost = stock_price * 100 * num_contracts
    put_price = get_strategy_price(long_put, "buy", stock_price)
    call_price = get_strategy_price(short_call, "sell", stock_price)
    if put_price is None or call_price is None:
        return None
    premium_paid = put_price * 100 * num_contracts
    premium_received = call_price * 100 * num_contracts
    if debug:
        st.write(f"Collar - Stock Cost: {stock_cost}, Long Put: {put_price}, Short Call: {call_price}")
    net_cost = stock_cost + premium_paid - premium_received
    max_profit = (short_call["strike"] - stock_price + call_price - put_price) * 100 * num_contracts
    max_loss = (stock_price - long_put["strike"] + put_price - call_price) * 100 * num_contracts
    breakeven = stock_price + put_price - call_price
    prices = np.linspace(stock_price - 5000, stock_price + 5000, 200)
    profits = [(min(max(price, long_put["strike"]), short_call["strike"]) - stock_price - put_price + call_price) * 100 * num_contracts for price in prices]
    details = f"Compra {num_contracts * 100} acciones GGAL a {stock_price} ARS, Compra {num_contracts} Put {long_put['strike']} ARS, Vende {num_contracts} Call {short_call['strike']} ARS"
    return net_cost, max_profit, max_loss, [breakeven], prices, profits, "Collar", details

def count_combinations(iterable: list, r: int) -> int:
    return comb(len(iterable), r) if len(iterable) >= r else 0

def analyze_all_strategies(calls: list, puts: list, current_price: float, lower_percentage: float, upper_percentage: float,
                          max_loss_to_profit_ratio: float = 0.3, min_credit_to_loss_ratio: float = 0.7,
                          exclude_loss_to_profit: bool = True, exclude_credit_to_loss: bool = True,
                          included_strategies: set | None = None, exclude_bullish: bool = False,
                          exclude_bearish: bool = False, exclude_neutral: bool = False, debug: bool = False) -> list:
    if included_strategies is None:
        included_strategies = {
            "Bull Spread (Calls)", "Bear Spread (Puts)", "Mariposa (Calls)", "Cóndor (Calls)",
            "Straddle", "Strangle", "Mariposa (Puts)", "Iron Condor", "Put Bull Spread", "Call Bear Spread",
            "Covered Call", "Protective Put", "Collar"
        }

    # Define strategy categories
    bullish_strategies = {"Bull Spread (Calls)", "Put Bull Spread", "Straddle", "Strangle", "Covered Call", "Protective Put"}
    bearish_strategies = {"Bear Spread (Puts)", "Call Bear Spread"}
    neutral_strategies = {"Mariposa (Calls)", "Mariposa (Puts)", "Cóndor (Calls)", "Iron Condor"}
    # Optionally include Straddle and Strangle as neutral if desired:
    # neutral_strategies = {"Mariposa (Calls)", "Mariposa (Puts)", "Cóndor (Calls)", "Iron Condor", "Straddle", "Strangle"}

    # Apply filters based on user selections
    filtered_strategies = included_strategies.copy()
    if exclude_bullish:
        filtered_strategies -= bullish_strategies
    if exclude_bearish:
        filtered_strategies -= bearish_strategies
    if exclude_neutral:
        filtered_strategies -= neutral_strategies

    # Initialize results and tracking variables
    results = []
    max_contracts = 10
    min_strike = current_price * (1 - lower_percentage)
    max_strike = current_price * (1 + upper_percentage)

    # Track exclusion reasons
    exclusion_counts = {strat: {"price_invalid": 0, "condition_failed": 0, "loss_ratio": 0, "credit_ratio": 0, "other": 0}
                        for strat in included_strategies}

    # Filter options by strike range
    filtered_calls = [c for c in calls if isinstance(c, dict) and "strike" in c and min_strike <= c["strike"] <= max_strike]
    filtered_puts = [p for p in puts if isinstance(p, dict) and "strike" in p and min_strike <= p["strike"] <= max_strike]

    if debug:
        st.write(f"Filtered Calls: {len(filtered_calls)}, Filtered Puts: {len(filtered_puts)}")
        st.write(f"Call Strikes: {[c['strike'] for c in filtered_calls]}")
        st.write(f"Put Strikes: {[p['strike'] for p in filtered_puts]}")
        st.write(f"Strategies to analyze: {filtered_strategies}")

    # Calculate total tasks for progress bar
    progress = st.progress(0)
    total_tasks = sum([
        count_combinations(filtered_calls, 2) * 10 if "Bull Spread (Calls)" in filtered_strategies else 0,
        count_combinations(filtered_puts, 2) * 10 if "Bear Spread (Puts)" in filtered_strategies else 0,
        count_combinations(filtered_calls, 3) * 10 if "Mariposa (Calls)" in filtered_strategies else 0,
        count_combinations(filtered_calls, 4) * 10 if "Cóndor (Calls)" in filtered_strategies else 0,
        len(set(c["strike"] for c in filtered_calls) & set(p["strike"] for p in filtered_puts)) * 10 if "Straddle" in filtered_strategies else 0,
        len([(c, p) for c in filtered_calls for p in filtered_puts if c["strike"] > p["strike"]]) * 10 if "Strangle" in filtered_strategies else 0,
        count_combinations(filtered_puts, 3) * 10 if "Mariposa (Puts)" in filtered_strategies else 0,
        count_combinations(filtered_calls, 2) * count_combinations(filtered_puts, 2) * 10 if "Iron Condor" in filtered_strategies else 0,
        count_combinations(filtered_puts, 2) * 10 if "Put Bull Spread" in filtered_strategies else 0,
        count_combinations(filtered_calls, 2) * 10 if "Call Bear Spread" in filtered_strategies else 0,
        len(filtered_calls) * 10 if "Covered Call" in filtered_strategies else 0,
        len(filtered_puts) * 10 if "Protective Put" in filtered_strategies else 0,
        len(filtered_calls) * len(filtered_puts) * 10 if "Collar" in filtered_strategies else 0
    ])

    task_count = 0

    # Define strategy configurations
    strategy_configs = [
        ("Bull Spread (Calls)", calculate_bull_spread, combinations(filtered_calls, 2), lambda lc, sc: lc["strike"] < sc["strike"]),
        ("Bear Spread (Puts)", calculate_bear_spread, [(p2, p1) for p1, p2 in combinations(filtered_puts, 2) if p1["strike"] < p2["strike"]], lambda lp, sp: lp["strike"] > sp["strike"]),
        ("Mariposa (Calls)", calculate_butterfly_spread, combinations(filtered_calls, 3), lambda lc, mc, hc: lc["strike"] < mc["strike"] < hc["strike"] and (mc["strike"] - lc["strike"]) == (hc["strike"] - mc["strike"])),
        ("Cóndor (Calls)", calculate_condor_spread, combinations(filtered_calls, 4), lambda lc, mlc, mhc, hc: lc["strike"] < mlc["strike"] < mhc["strike"] < hc["strike"] and mlc["strike"] - lc["strike"] == hc["strike"] - mhc["strike"]),
        ("Straddle", calculate_straddle, [(next(c for c in filtered_calls if c["strike"] == s), next(p for p in filtered_puts if p["strike"] == s)) for s in set(c["strike"] for c in filtered_calls) & set(p["strike"] for p in filtered_puts)], lambda c, p: True),
        ("Strangle", calculate_strangle, [(c, p) for c in filtered_calls for p in filtered_puts if c["strike"] > p["strike"]], lambda c, p: True),
        ("Mariposa (Puts)", calculate_put_butterfly, combinations(filtered_puts, 3), lambda lp, mp, hp: lp["strike"] < mp["strike"] < hp["strike"] and (mp["strike"] - lp["strike"]) == (hp["strike"] - mp["strike"])),
        ("Iron Condor", calculate_iron_condor, [(cs, cl, ps, pl) for cs, cl in combinations(filtered_calls, 2) for ps, pl in combinations(filtered_puts, 2) if cs["strike"] < cl["strike"] and ps["strike"] > pl["strike"] and pl["strike"] < cs["strike"]], lambda cs, cl, ps, pl: True),
        ("Put Bull Spread", calculate_put_bull_spread, combinations(filtered_puts, 2), lambda lp, sp: lp["strike"] < sp["strike"]),
        ("Call Bear Spread", calculate_call_bear_spread, [(c2, c1) for c1, c2 in combinations(filtered_calls, 2) if c1["strike"] < c2["strike"]], lambda lc, sc: lc["strike"] > sc["strike"]),
        ("Covered Call", calculate_covered_call, [(current_price, c) for c in filtered_calls], lambda sp, sc: True),
        ("Protective Put", calculate_protective_put, [(current_price, p) for p in filtered_puts], lambda sp, lp: True),
        ("Collar", calculate_collar, [(current_price, p, c) for p in filtered_puts for c in filtered_calls if p["strike"] < c["strike"]], lambda sp, lp, sc: True)
    ]

    # Process each strategy
    for strategy, func, combos, condition in strategy_configs:
        if strategy not in filtered_strategies:
            if debug:
                st.write(f"Skipping {strategy} (not in filtered_strategies)")
            continue

        combo_list = list(combos)
        if debug:
            st.write(f"{strategy}: {len(combo_list)} combinations found")

        for combo in combo_list:
            try:
                if not condition(*combo):
                    exclusion_counts[strategy]["condition_failed"] += max_contracts
                    task_count += max_contracts
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))
                    continue
            except Exception as e:
                if debug:
                    st.write(f"Error checking condition for {strategy}: {e}")
                exclusion_counts[strategy]["condition_failed"] += max_contracts
                task_count += max_contracts
                progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))
                continue

            for num_contracts in range(1, max_contracts + 1):
                try:
                    if strategy == "Covered Call":
                        result = func(combo[0], combo[1], num_contracts, debug)
                    elif strategy == "Protective Put":
                        result = func(combo[0], combo[1], num_contracts, debug)
                    elif strategy == "Collar":
                        result = func(combo[0], combo[1], combo[2], num_contracts, debug)
                    else:
                        args = list(combo)
                        args.append(current_price)
                        args.append(num_contracts)
                        args.append(debug)
                        result = func(*args)
                except Exception as e:
                    if debug:
                        st.write(f"Error calculating {strategy}: {e}")
                    exclusion_counts[strategy]["other"] += 1
                    task_count += 1
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))
                    continue

                if result is None:
                    exclusion_counts[strategy]["price_invalid"] += 1
                    if debug:
                        st.write(f"{strategy} failed for combo {tuple(c['strike'] if isinstance(c, dict) else c for c in combo)}, contracts: {num_contracts}")
                    task_count += 1
                    progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))
                    continue

                net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result

                if net_cost > 0 or strategy in ["Covered Call", "Protective Put", "Collar"]:
                    loss_to_profit_ratio = max_loss / max_profit if max_profit > 0 and max_profit != float('inf') else float('inf')
                    credit_received = -net_cost if net_cost < 0 else 0
                    credit_to_loss_ratio = credit_received / max_loss if max_loss > 0 else float('inf')

                    meets_loss_criteria = loss_to_profit_ratio <= max_loss_to_profit_ratio or max_profit == float('inf')
                    meets_credit_criteria = credit_to_loss_ratio >= min_credit_to_loss_ratio or net_cost >= 0

                    warning = None
                    if not meets_loss_criteria:
                        warning = "Pérdida/Ganancia excede el límite"
                        exclusion_counts[strategy]["loss_ratio"] += 1
                    elif not meets_credit_criteria:
                        warning = "Crédito/Pérdida menor al límite"
                        exclusion_counts[strategy]["credit_ratio"] += 1

                    if (not exclude_loss_to_profit or meets_loss_criteria) and (not exclude_credit_to_loss or meets_credit_criteria):
                        profit_potential = max_profit / abs(net_cost) if net_cost != 0 and max_profit != float('inf') else 0
                        results.append({
                            "strategy": strat_name,
                            "strikes": [c["strike"] for c in combo[1:] if isinstance(c, dict)] if strategy in ["Covered Call", "Protective Put"] else [c["strike"] for c in combo if isinstance(c, dict)],
                            "num_contracts": num_contracts,
                            "net_cost": net_cost,
                            "max_profit": max_profit,
                            "max_loss": max_loss,
                            "profit_potential": profit_potential,
                            "breakevens": breakevens,
                            "prices": prices.tolist(),
                            "profits": profits,
                            "details": details,
                            "warning": warning
                        })

                        if debug:
                            st.write(f"Added {strat_name} with strikes {results[-1]['strikes']}, contracts: {num_contracts}, profit_potential: {results[-1]['profit_potential']:.2f}")

                task_count += 1
                progress.progress(min(1.0, task_count / total_tasks if total_tasks > 0 else 1.0))

    progress.empty()

    if debug:
        st.write("### Strategy Exclusion Statistics")
        exclusion_df = pd.DataFrame({
            "Strategy": list(exclusion_counts.keys()),
            "Price Invalid": [exclusion_counts[s]["price_invalid"] for s in exclusion_counts],
            "Condition Failed": [exclusion_counts[s]["condition_failed"] for s in exclusion_counts],
            "Loss Ratio": [exclusion_counts[s]["loss_ratio"] for s in exclusion_counts],
            "Credit Ratio": [exclusion_counts[s]["credit_ratio"] for s in exclusion_counts],
            "Other Errors": [exclusion_counts[s]["other"] for s in exclusion_counts],
            "Total": [sum(exclusion_counts[s].values()) for s in exclusion_counts]
        })
        st.dataframe(exclusion_df)

    return sorted(results, key=lambda x: x["profit_potential"], reverse=True)[:10]

def plot_strategy(prices: list, profits: list, current_price: float, breakevens: list, strategy_name: str, expiration: date, num_contracts: int) -> go.Figure:
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

def plot_comparison(strategies: list, current_price: float, expiration: date) -> go.Figure:
    fig = go.Figure()
    for strat in strategies:
        fig.add_trace(go.Scatter(x=strat["prices"], y=strat["profits"], mode="lines", name=f"{strat['strategy']} ({strat['num_contracts']} contratos)"))
    fig.add_vline(x=current_price, line_dash="dash", line_color="gray", annotation_text="Precio Actual")
    fig.update_layout(
        title=f"Comparación de Estrategias - Vencimiento: {expiration.strftime('%Y-%m-%d')}",
        xaxis_title="Precio de la Acción (ARS)",
        yaxis_title="Ganancia/Pérdida (ARS)",
        legend_title="Estrategias"
    )
    return fig

def create_spread_matrix(calls: list, puts: list, current_price: float, strategy_type: str = "Bull Spread (Calls)", num_contracts: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strikes = sorted(calls if strategy_type.startswith("Bull") or strategy_type.startswith("Call") else puts, key=lambda x: x["strike"])
    profit_matrix, cost_matrix, ratio_matrix = [], [], []
    for long_strike in strikes:
        profit_row, cost_row, ratio_row = [], [], []
        for short_strike in strikes:
            if (strategy_type == "Bull Spread (Calls)" and long_strike["strike"] < short_strike["strike"]) or \
               (strategy_type == "Bear Spread (Puts)" and long_strike["strike"] > short_strike["strike"]) or \
               (strategy_type == "Put Bull Spread" and long_strike["strike"] < short_strike["strike"]) or \
               (strategy_type == "Call Bear Spread" and long_strike["strike"] > short_strike["strike"]):
                func = {
                    "Bull Spread (Calls)": calculate_bull_spread,
                    "Bear Spread (Puts)": calculate_bear_spread,
                    "Put Bull Spread": calculate_put_bull_spread,
                    "Call Bear Spread": calculate_call_bear_spread
                }[strategy_type]
                result = func(long_strike, short_strike, current_price, num_contracts, debug=False)
                if result is not None:
                    net_cost, max_profit, max_loss, _, prices, profits, _, _ = result
                    profit_row.append(max_profit)
                    cost_row.append(net_cost)
                    ratio_row.append(max_profit / net_cost if net_cost > 0 and max_profit > 0 else np.nan)
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
    return (pd.DataFrame(profit_matrix, columns=strike_labels, index=strike_labels),
            pd.DataFrame(cost_matrix, columns=strike_labels, index=strike_labels),
            pd.DataFrame(ratio_matrix, columns=strike_labels, index=strike_labels))

def main():
    st.title("Visualizador de Estrategias de Opciones GGAL")

    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    debug_mode = st.checkbox("Modo Depuración", value=st.session_state.debug_mode, key="debug_mode")

    risk_free_rate = st.number_input("Tasa Libre de Riesgo Anualizada (%)", min_value=0.0, max_value=100.0, value=25.0,
                                     step=0.1) / 100
    min_strike_filter = st.number_input("Strike Mínimo", min_value=0.0, value=4900.0, step=100.0)
    max_strike_filter = st.number_input("Strike Máximo", min_value=0.0, value=10000.0, step=100.0)

    ggal_stock, ggal_options = get_ggal_data(risk_free_rate)
    if not ggal_stock or not ggal_options:
        return

    ggal_options = [o for o in ggal_options if min_strike_filter <= o["strike"] <= max_strike_filter]
    current_price = float(ggal_stock["c"])
    now_ar = datetime.now(timezone.utc) + timedelta(hours=-3)
    st.write(f"Precio Actual de la Acción GGAL: {current_price} ARS (al {now_ar.strftime('%Y-%m-%d %H:%M:%S')} AR)")
    st.write(
        f"Mercado {'abierto' if is_market_hours() else 'cerrado'} - Usando {'bid/ask' if is_market_hours() else 'precios de cierre'} para cálculos.")
    hist_vol = get_historical_volatility()
    st.write(f"Volatilidad Histórica Anualizada (1 año): {hist_vol:.2%}" if not np.isnan(
        hist_vol) else "Volatilidad Histórica no disponible")

    if st.button("Actualizar Datos"):
        ggal_stock, ggal_options = get_ggal_data(risk_free_rate)
        ggal_options = [o for o in ggal_options if min_strike_filter <= o["strike"] <= max_strike_filter]
        current_price = float(ggal_stock["c"]) if ggal_stock else current_price
        now_ar = datetime.now(timezone.utc) + timedelta(hours=-3)
        st.write(
            f"Datos actualizados. Precio Actual: {current_price} ARS (al {now_ar.strftime('%Y-%m-%d %H:%M:%S')} AR)")

    show_greeks = st.checkbox("Mostrar Greeks de las Opciones", value=False)
    expirations = sorted(list(set(o["expiration"] for o in ggal_options)))
    strikes_by_expiration = {exp: {
        "calls": sorted([o for o in ggal_options if o["type"] == "call" and o["expiration"] == exp],
                        key=lambda x: x["strike"]),
        "puts": sorted([o for o in ggal_options if o["type"] == "put" and o["expiration"] == exp],
                       key=lambda x: x["strike"])}
                             for exp in expirations}

    if show_greeks:
        for exp in expirations:
            with st.expander(f"Opciones con Vencimiento {exp.strftime('%Y-%m-%d')}"):
                calls, puts = strikes_by_expiration[exp]["calls"], strikes_by_expiration[exp]["puts"]
                st.write("Calls:")
                st.dataframe(pd.DataFrame([{k: o[k] for k in
                                            ["symbol", "strike", "px_bid", "px_ask", "iv", "delta", "gamma", "theta",
                                             "vega", "rho"]} for o in calls]))
                st.write("Puts:")
                st.dataframe(pd.DataFrame([{k: o[k] for k in
                                            ["symbol", "strike", "px_bid", "px_ask", "iv", "delta", "gamma", "theta",
                                             "vega", "rho"]} for o in puts]))

    tab1, tab2, tab3 = st.tabs(["Selección Manual de Estrategia", "Análisis Automático", "Matrices de Spreads"])

    with tab1:
        strategies = ["Bull Spread (Calls)", "Bear Spread (Puts)", "Mariposa (Calls)", "Cóndor (Calls)",
                      "Straddle", "Strangle", "Mariposa (Puts)", "Iron Condor", "Put Bull Spread", "Call Bear Spread",
                      "Covered Call", "Protective Put", "Collar"]
        selected_strategy = st.selectbox("Seleccioná una Estrategia", strategies)
        selected_exp = st.selectbox("Seleccioná Fecha de Vencimiento", expirations,
                                    format_func=lambda x: x.strftime("%Y-%m-%d"))
        num_contracts = st.number_input("Número de Contratos (1 contrato = 100 opciones)", min_value=1, max_value=100,
                                        value=1, step=1)
        calls, puts = strikes_by_expiration[selected_exp]["calls"], strikes_by_expiration[selected_exp]["puts"]
        call_strikes, put_strikes = [c["strike"] for c in calls], [p["strike"] for p in puts]
        result = None
        if selected_strategy == "Bull Spread (Calls)":
            if len(call_strikes) < 2: st.error("No hay suficientes opciones call."); return
            long_strike = st.selectbox("Strike Menor (Compra Call)", call_strikes)
            short_strike = st.selectbox("Strike Mayor (Venta Call)", [s for s in call_strikes if s > long_strike])
            long_call, short_call = next(c for c in calls if c["strike"] == long_strike), next(
                c for c in calls if c["strike"] == short_strike)
            result = calculate_bull_spread(long_call, short_call, current_price, num_contracts, debug_mode)
        elif selected_strategy == "Bear Spread (Puts)":
            if len(put_strikes) < 2: st.error("No hay suficientes opciones put."); return
            long_strike = st.selectbox("Strike Mayor (Compra Put)", put_strikes[::-1])
            short_strike = st.selectbox("Strike Menor (Venta Put)", [s for s in put_strikes if s < long_strike])
            long_put, short_put = next(p for p in puts if p["strike"] == long_strike), next(
                p for p in puts if p["strike"] == short_strike)
            result = calculate_bear_spread(long_put, short_put, current_price, num_contracts, debug_mode)
        elif selected_strategy == "Mariposa (Calls)":
            if len(call_strikes) < 3: st.error("No hay suficientes opciones call."); return
            low_strike = st.selectbox("Strike Menor", call_strikes)
            mid_strike = st.selectbox("Strike Medio", [s for s in call_strikes if s > low_strike])
            high_strike = st.selectbox("Strike Mayor", [s for s in call_strikes if s > mid_strike])
            low_call, mid_call, high_call = next(c for c in calls if c["strike"] == low_strike), next(
                c for c in calls if c["strike"] == mid_strike), next(c for c in calls if c["strike"] == high_strike)
            result = calculate_butterfly_spread(low_call, mid_call, high_call, current_price, num_contracts, debug_mode)
        elif selected_strategy == "Cóndor (Calls)":
            if len(call_strikes) < 4: st.error("No hay suficientes opciones call."); return
            low_strike = st.selectbox("Strike Menor", call_strikes)
            mid_low_strike = st.selectbox("Strike Medio-Bajo", [s for s in call_strikes if s > low_strike])
            mid_high_strike = st.selectbox("Strike Medio-Alto", [s for s in call_strikes if s > mid_low_strike])
            high_strike = st.selectbox("Strike Mayor", [s for s in call_strikes if s > mid_high_strike])
            low_call, mid_low_call, mid_high_call, high_call = next(
                c for c in calls if c["strike"] == low_strike), next(
                c for c in calls if c["strike"] == mid_low_strike), next(
                c for c in calls if c["strike"] == mid_high_strike), next(
                c for c in calls if c["strike"] == high_strike)
            result = calculate_condor_spread(low_call, mid_low_call, mid_high_call, high_call, current_price,
                                             num_contracts, debug_mode)
        elif selected_strategy == "Straddle":
            if not (call_strikes and put_strikes): st.error("No hay suficientes opciones call y put."); return
            strike = st.selectbox("Strike (Compra Call y Put)", sorted(set(call_strikes) & set(put_strikes)))
            call, put = next(c for c in calls if c["strike"] == strike), next(p for p in puts if p["strike"] == strike)
            result = calculate_straddle(call, put, current_price, num_contracts, debug_mode)
        elif selected_strategy == "Strangle":
            if not (call_strikes and put_strikes): st.error("No hay suficientes opciones call y put."); return
            put_strike = st.selectbox("Strike Menor (Compra Put)", sorted(put_strikes))
            call_strike = st.selectbox("Strike Mayor (Compra Call)", [s for s in call_strikes if s > put_strike])
            call, put = next(c for c in calls if c["strike"] == call_strike), next(
                p for p in puts if p["strike"] == put_strike)
            result = calculate_strangle(call, put, current_price, num_contracts, debug_mode)
        elif selected_strategy == "Mariposa (Puts)":
            if len(put_strikes) < 3: st.error("No hay suficientes opciones put."); return
            low_strike = st.selectbox("Strike Menor", put_strikes)
            mid_strike = st.selectbox("Strike Medio", [s for s in put_strikes if s > low_strike])
            high_strike = st.selectbox("Strike Mayor", [s for s in put_strikes if s > mid_strike])
            low_put, mid_put, high_put = next(p for p in puts if p["strike"] == low_strike), next(
                p for p in puts if p["strike"] == mid_strike), next(p for p in puts if p["strike"] == high_strike)
            result = calculate_put_butterfly(low_put, mid_put, high_put, current_price, num_contracts, debug_mode)
        elif selected_strategy == "Iron Condor":
            if len(call_strikes) < 2 or len(put_strikes) < 2: st.error(
                "No hay suficientes opciones call y put."); return
            put_long_strike = st.selectbox("Strike Put Compra (Menor)", sorted(put_strikes))
            put_short_strike = st.selectbox("Strike Put Venta", [s for s in put_strikes if s > put_long_strike])
            call_short_strike = st.selectbox("Strike Call Venta", [s for s in call_strikes if s > put_short_strike])
            call_long_strike = st.selectbox("Strike Call Compra (Mayor)",
                                            [s for s in call_strikes if s > call_short_strike])
            call_short, call_long, put_short, put_long = next(
                c for c in calls if c["strike"] == call_short_strike), next(
                c for c in calls if c["strike"] == call_long_strike), next(
                p for p in puts if p["strike"] == put_short_strike), next(
                p for p in puts if p["strike"] == put_long_strike)
            result = calculate_iron_condor(call_short, call_long, put_short, put_long, current_price, num_contracts,
                                           debug_mode)
        elif selected_strategy == "Put Bull Spread":
            if len(put_strikes) < 2: st.error("No hay suficientes opciones put."); return
            long_strike = st.selectbox("Strike Menor (Compra Put)", put_strikes)
            short_strike = st.selectbox("Strike Mayor (Venta Put)", [s for s in put_strikes if s > long_strike])
            long_put, short_put = next(p for p in puts if p["strike"] == long_strike), next(
                p for p in puts if p["strike"] == short_strike)
            result = calculate_put_bull_spread(long_put, short_put, current_price, num_contracts, debug_mode)
        elif selected_strategy == "Call Bear Spread":
            if len(call_strikes) < 2: st.error("No hay suficientes opciones call."); return
            long_strike = st.selectbox("Strike Mayor (Compra Call)", sorted(call_strikes, reverse=True))
            short_strike = st.selectbox("Strike Menor (Venta Call)", [s for s in call_strikes if s < long_strike])
            long_call, short_call = next(c for c in calls if c["strike"] == long_strike), next(
                c for c in calls if c["strike"] == short_strike)
            result = calculate_call_bear_spread(long_call, short_call, current_price, num_contracts, debug_mode)
        elif selected_strategy == "Covered Call":
            if not call_strikes: st.error("No hay opciones call."); return
            short_strike = st.selectbox("Strike (Venta Call)", call_strikes)
            short_call = next(c for c in calls if c["strike"] == short_strike)
            result = calculate_covered_call(current_price, short_call, num_contracts, debug_mode)
        elif selected_strategy == "Protective Put":
            if not put_strikes: st.error("No hay opciones put."); return
            long_strike = st.selectbox("Strike (Compra Put)", put_strikes)
            long_put = next(p for p in puts if p["strike"] == long_strike)
            result = calculate_protective_put(current_price, long_put, num_contracts, debug_mode)
        elif selected_strategy == "Collar":
            if not (call_strikes and put_strikes): st.error("No hay suficientes opciones call y put."); return
            long_strike = st.selectbox("Strike (Compra Put)", put_strikes)
            short_strike = st.selectbox("Strike (Venta Call)", [s for s in call_strikes if s > long_strike])
            long_put, short_call = next(p for p in puts if p["strike"] == long_strike), next(
                c for c in calls if c["strike"] == short_strike)
            result = calculate_collar(current_price, long_put, short_call, num_contracts, debug_mode)
        if result:
            net_cost, max_profit, max_loss, breakevens, prices, profits, strat_name, details = result
            st.write(f"Detalles: {details}")
            st.write(f"Costo Neto: {net_cost:.2f} ARS")
            st.write(f"Ganancia Máxima: {max_profit:.2f} ARS" if max_profit != float(
                'inf') else "Ganancia Máxima: Ilimitada ARS")
            st.write(f"Pérdida Máxima: {max_loss:.2f} ARS")
            st.write(f"Punto(s) de Equilibrio: {', '.join([f'{b:.2f}' for b in breakevens])} ARS")
            fig = plot_strategy(prices, profits, current_price, breakevens, strat_name, selected_exp, num_contracts)
            st.plotly_chart(fig)

    with tab2:
        st.subheader("Top 10 Combinaciones de Opciones Más Rentables")
        selected_exp_auto = st.selectbox("Seleccioná Fecha de Vencimiento para el Análisis", expirations,
                                         format_func=lambda x: x.strftime("%Y-%m-%d"), key="auto_exp")
        lower_percentage = st.number_input("Porcentaje Inferior del Precio Actual", min_value=0.0, max_value=100.0,
                                           value=15.0, step=0.1) / 100
        upper_percentage = st.number_input("Porcentaje Superior del Precio Actual", min_value=0.0, max_value=100.0,
                                           value=15.0, step=0.1) / 100
        min_strike = max(min_strike_filter, current_price * (1 - lower_percentage))
        max_strike = min(max_strike_filter, current_price * (1 + upper_percentage))
        st.write(f"Incluyendo solo strikes entre {min_strike:.2f} y {max_strike:.2f} ARS")
        max_loss_to_profit = st.number_input("Máxima Razón Pérdida/Ganancia", min_value=0.01, value=1.0, step=0.01)
        exclude_loss_to_profit = st.checkbox("Excluir estrategias que excedan la razón Pérdida/Ganancia", value=False)
        min_credit_to_loss = st.number_input("Mínima Razón Crédito/Pérdida", min_value=0.01, value=0.01, step=0.01)
        exclude_credit_to_loss = st.checkbox("Excluir estrategias por debajo de la razón Crédito/Pérdida", value=False)
        exclude_bullish = st.checkbox("Excluir Estrategias Alcistas", value=False)
        exclude_bearish = st.checkbox("Excluir Estrategias Bajistas", value=False)
        exclude_neutral = st.checkbox("Excluir Estrategias Neutrales", value=False)
        all_strategies = ["Bull Spread (Calls)", "Bear Spread (Puts)", "Mariposa (Calls)", "Cóndor (Calls)",
                          "Straddle", "Strangle", "Mariposa (Puts)", "Iron Condor", "Put Bull Spread",
                          "Call Bear Spread", "Covered Call", "Protective Put", "Collar"]
        excluded_strategies = st.multiselect("Excluir Estrategias del Análisis", all_strategies,
                                             default=["Collar", "Covered Call", "Protective Put"])
        included_strategies = set(all_strategies) - set(excluded_strategies)
        calls_auto, puts_auto = strikes_by_expiration[selected_exp_auto]["calls"], \
        strikes_by_expiration[selected_exp_auto]["puts"]
        with st.spinner("Analizando estrategias..."):
            top_strategies = analyze_all_strategies(calls_auto, puts_auto, current_price, lower_percentage,
                                                    upper_percentage, max_loss_to_profit, min_credit_to_loss,
                                                    exclude_loss_to_profit, exclude_credit_to_loss,
                                                    included_strategies, exclude_bullish, exclude_bearish,
                                                    exclude_neutral, debug_mode)
        if not top_strategies:
            st.warning("No se encontraron estrategias válidas con los criterios seleccionados.")
        else:
            for i, strat in enumerate(top_strategies, 1):
                with st.expander(f"#{i}: {strat['strategy']} - Potencial: {strat['profit_potential']:.2f}x"):
                    st.write(f"Detalles: {strat['details']}")
                    st.write(f"Strikes: {strat['strikes']}")
                    st.write(f"Número de Contratos: {strat['num_contracts']}")
                    st.write(f"Costo Neto: {strat['net_cost']:.2f} ARS")
                    st.write(f"Ganancia Máxima: {strat['max_profit']:.2f} ARS" if strat['max_profit'] != float(
                        'inf') else "Ganancia Máxima: Ilimitada ARS")
                    st.write(f"Pérdida Máxima: {strat['max_loss']:.2f} ARS")
                    st.write(f"Punto(s) de Equilibrio: {', '.join([f'{b:.2f}' for b in strat['breakevens']])} ARS")
                    if strat["warning"]:
                        st.warning(f"Advertencia: {strat['warning']}")
                    if st.button(f"Ver Gráfico", key=f"plot_{i}"):
                        fig = plot_strategy(strat["prices"], strat["profits"], current_price, strat["breakevens"],
                                            strat["strategy"], selected_exp_auto, strat["num_contracts"])
                        st.plotly_chart(fig)
            if st.button("Guardar Estrategias"):
                with open("top_strategies.json", "w") as f:
                    json.dump(top_strategies, f)
                st.success("Estrategias guardadas en 'top_strategies.json'.")
            uploaded_file = st.file_uploader("Cargar Estrategias", type="json")
            if uploaded_file:
                top_strategies = json.load(uploaded_file)
                st.success("Estrategias cargadas exitosamente.")
            compare_strategies = st.multiselect("Seleccioná Estrategias para Comparar",
                                                [f"#{i}: {s['strategy']}" for i, s in enumerate(top_strategies, 1)])
            if compare_strategies:
                selected = [top_strategies[int(s.split(":")[0][1:]) - 1] for s in compare_strategies]
                fig = plot_comparison(selected, current_price, selected_exp_auto)
                st.plotly_chart(fig)

    with tab3:
        st.subheader("Matrices de Ganancia/Pérdida para Spreads")
        selected_exp_matrix = st.selectbox("Seleccioná Fecha de Vencimiento para la Matriz", expirations,
                                           format_func=lambda x: x.strftime("%Y-%m-%d"), key="matrix_exp")
        strategy_types = ["Bull Spread (Calls)", "Bear Spread (Puts)", "Put Bull Spread", "Call Bear Spread"]
        selected_strategy_type = st.selectbox("Seleccioná Tipo de Spread", strategy_types)
        num_contracts_matrix = st.number_input("Número de Contratos para la Matriz", min_value=1, max_value=100,
                                               value=1, step=1)
        if st.button("Generar Matrices"):
            with st.spinner("Generando matrices..."):
                calls_matrix, puts_matrix = strikes_by_expiration[selected_exp_matrix]["calls"], \
                strikes_by_expiration[selected_exp_matrix]["puts"]
                profit_df, cost_df, ratio_df = create_spread_matrix(calls_matrix, puts_matrix, current_price,
                                                                    selected_strategy_type, num_contracts_matrix)
                st.write(f"**Matriz de Ganancia Máxima ({selected_strategy_type}) en ARS**")
                st.dataframe(profit_df.style.background_gradient(cmap='RdYlGn'))
                st.write(f"**Matriz de Costo Neto ({selected_strategy_type}) en ARS**")
                st.dataframe(cost_df.style.background_gradient(cmap='RdYlGn_r'))
                st.write(f"**Matriz de Relación Ganancia/Costo ({selected_strategy_type})**")
                st.dataframe(ratio_df.style.background_gradient(cmap='RdYlGn'))

if __name__ == "__main__":
    main()
