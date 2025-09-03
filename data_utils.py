import requests
from datetime import date, timedelta
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STOCK_URL = "https://data912.com/live/arg_stocks"
OPTIONS_URL = "https://data912.com/live/arg_options"

def get_third_friday(year: int, month: int) -> date:
    try:
        first_day = date(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
        return first_friday + timedelta(days=14)
    except ValueError:
        return None

def get_expiration_map():
    current_year = date.today().year
    months = ["E", "F", "M", "A", "MY", "J", "JL", "AG", "S", "O", "N", "D"]
    suffix_variations = ["", "N", "Z", "B", "", "N", "", "", "E", "C", "O", "I"]
    exp_map = {}
    for i, (base, var) in enumerate(zip(months, suffix_variations)):
        exp_date = get_third_friday(current_year, i + 1)
        if exp_date and exp_date >= date.today():
            exp_map[base] = exp_date
            if var:
                exp_map[base + var] = exp_date
    if date.today().month > 11:
        for i, (base, var) in enumerate(zip(months, suffix_variations)):
            exp_date = get_third_friday(current_year + 1, i + 1)
            if exp_date:
                exp_map[base] = exp_date
                if var:
                    exp_map[base + var] = exp_date
    return exp_map

EXPIRATION_MAP = get_expiration_map()

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
    exp_date = EXPIRATION_MAP.get(suffix)
    if not exp_date:
        logger.warning(f"Unmapped expiration suffix: {suffix} in symbol: {symbol}")
        return None, None, None
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
        px_bid, px_ask = o.get("px_bid", 0), o.get("px_ask", 0)
        if all([opt_type, strike, exp, px_ask > 0, px_bid > 0, abs(px_ask - px_bid) / max(px_ask, px_bid) < 0.2]):
            ggal_options.append({
                "symbol": o["symbol"], "type": opt_type, "strike": strike,
                "expiration": exp, "px_bid": px_bid, "px_ask": px_ask
            })
        else:
            logger.debug(f"Skipped option {o['symbol']}: Invalid data or wide spread")
    return ggal_stock, ggal_options