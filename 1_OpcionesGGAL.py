import streamlit as st
from datetime import datetime, date, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import QuantLib as ql
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Logging setup
logging.basicConfig(level=logging.DEBUG, filename='debug.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# utils module content
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

@st.cache_data(ttl=300)
def fetch_data(url: str) -> list:
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.error(f"Timeout fetching data from {url}")
        st.error(f"Timeout fetching data from {url}")
        return []
    except requests.HTTPError as e:
        logger.error(f"HTTP error from {url}: {e}")
        st.error(f"HTTP error from {url}: {e}")
        return []
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

@st.cache_data(ttl=300)
def get_ggal_data() -> tuple[dict | None, list]:
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)
    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock:
        logger.error("GGAL stock data not found")
        return None, []
    logger.info(f"ggal_stock: {ggal_stock}")
    ggal_options = []
    for o in options_data:
        px_bid = o.get("px_bid")
        px_ask = o.get("px_ask")
        logger.debug(f"Raw option data: symbol={o.get('symbol')}, px_bid={px_bid}, px_ask={px_ask}")
        if not (isinstance(px_bid, (int, float)) and px_bid > 0 and isinstance(px_ask, (int, float)) and px_ask > 0):
            logger.warning(f"Invalid bid/ask for option {o['symbol']}: bid={px_bid}, ask={px_ask}")
            continue
        opt_type, strike, exp = parse_option_symbol(o["symbol"])
        if all([opt_type, strike, exp]):
            option = {"symbol": o["symbol"], "type": opt_type, "strike": strike, "expiration": exp, "px_bid": px_bid, "px_ask": px_ask}
            ggal_options.append(option)
            logger.info(f"Option added: {option}")
        else:
            logger.debug(f"Skipped option {o['symbol']}: Invalid data")
    return ggal_stock, ggal_options

def get_strategy_price(option: dict, action: str) -> float | None:
    price = option["px_ask"] if action == "buy" else option["px_bid"]
    return price if isinstance(price, (int, float)) and price > 0 else None

def calculate_fees(base_cost: float, commission_rate: float) -> float:
    commission = base_cost * commission_rate
    market_fees = base_cost * 0.002
    vat = (commission + market_fees) * 0.21
    return commission + market_fees + vat

@st.cache_data(ttl=300)
def fetch_cauciones_table(url: str, headers: dict = None) -> pd.DataFrame:
    session = requests.Session()
    if headers:
        session.headers.update(headers)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table') or soup.find('div', class_='tabla')
        if not table:
            raise RuntimeError("Unable to locate the Cauciones table in the HTML.")
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        if not headers:
            first_row = table.find('tr')
            headers = [td.get_text(strip=True) for td in first_row.find_all(['th', 'td'])]
        rows = []
        for tr in table.find_all('tr')[1:]:
            cells = tr.find_all(['td', 'th'])
            if cells:
                row = [c.get_text(strip=True) for c in cells]
                rows.append(row)
        df = pd.DataFrame(rows, columns=headers if headers and len(headers) == len(rows[0]) else None)
        df.dropna(subset=['Plazo', 'Tasa Tomadora', 'Moneda'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching Cauciones table: {e}")
        return pd.DataFrame()

def get_risk_free_rate():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/Cauciones"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TableScraper/1.0)"}
    try:
        df = fetch_cauciones_table(url, headers)
        if df.empty:
            st.warning("No se pudo obtener la tabla de Cauciones. Usando tasa predeterminada de 35%.")
            return 0.35
        df['Tasa Tomadora'] = df['Tasa Tomadora'].str.replace('%', '').str.replace(',', '.').str.strip()
        df['Tasa Tomadora'] = pd.to_numeric(df['Tasa Tomadora'], errors='coerce') / 100
        df['Plazo'] = pd.to_numeric(df['Plazo'], errors='coerce')
        df_pesos = df[(df['Moneda'] == 'PESOS') & (df['Tasa Tomadora'].notna()) & (df['Tasa Tomadora'] > 0)]
        for plazo in [30, 31, 32, 33, 34, 35]:
            tasa = df_pesos[df_pesos['Plazo'] == plazo]['Tasa Tomadora']
            if not tasa.empty:
                return tasa.iloc[0]
        st.warning("No se encontró una Tasa Tomadora válida para Plazo 30-35 días en PESOS. Usando tasa predeterminada de 35%.")
        return 0.35
    except Exception as e:
        st.error(f"Error al obtener la Tasa Tomadora: {e}. Usando tasa predeterminada de 35%.")
        return 0.35

# QuantLib option pricing
def calculate_option_price(option: Dict, spot_price: float, risk_free_rate: float, volatility: float, eval_date: date, expiration_date: date, use_quantlib: bool = False) -> float:
    try:
        logger.debug(f"Calculating option price: symbol={option['symbol']}, strike={option['strike']}, spot={spot_price}, eval_date={eval_date}, expiration={expiration_date}, risk_free={risk_free_rate}, vol={volatility}, use_quantlib={use_quantlib}")
        evaluation_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
        expiration = ql.Date(expiration_date.day, expiration_date.month, expiration_date.year)
        if evaluation_date >= expiration:
            logger.debug("Evaluation date is on or after expiration; using intrinsic value")
            strike = float(option['strike'])
            if option['type'].lower() == 'call':
                return max(spot_price - strike, 0)
            else:  # put
                return max(strike - spot_price, 0)

        ql.Settings.instance().evaluationDate = evaluation_date

        strike = float(option['strike'])
        option_type = ql.Option.Call if option['type'].lower() == 'call' else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = ql.EuropeanExercise(expiration)

        # Ensure scalar inputs
        spot_price = float(spot_price)
        risk_free_rate = float(risk_free_rate)
        volatility = float(volatility)

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluation_date, risk_free_rate, ql.Actual365Fixed()))
        volatility_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(evaluation_date, ql.NullCalendar(), volatility, ql.Actual365Fixed()))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluation_date, 0.0, ql.Actual365Fixed()))

        logger.debug(f"BlackScholesProcess inputs: spot={spot_price}, strike={strike}, risk_free={risk_free_rate}, volatility={volatility}, types: spot_handle={type(spot_handle).__name__}, dividend_ts={type(dividend_ts).__name__}, risk_free_ts={type(risk_free_ts).__name__}, volatility_ts={type(volatility_ts).__name__}")

        if not use_quantlib:
            st.warning(f"Using intrinsic value for {option['symbol']} due to QuantLib issue")
            if option['type'].lower() == 'call':
                return max(spot_price - strike, 0)
            else:  # put
                return max(strike - spot_price, 0)

        process = ql.BlackScholesProcess(spot_handle, dividend_ts, risk_free_ts, volatility_ts)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        price = option.NPV()
        logger.debug(f"Calculated price: {price}")
        return price
    except Exception as e:
        logger.error(f"Error calculating price for symbol {option['symbol']}, strike {option['strike']}: {e}")
        st.warning(f"Error calculating price for symbol {option['symbol']}, strike {option['strike']}: {e}")
        strike = float(option['strike'])
        if option['type'].lower() == 'call':
            return max(spot_price - strike, 0)
        else:  # put
            return max(strike - spot_price, 0)

# Strategy P&L over time and price
def calculate_strategy_pnl(strategy: List[Dict], spot_range: np.ndarray, time_points: List[date], num_contracts: int, commission_rate: float, risk_free_rate: float, volatility: float, expiration_date: date, use_quantlib: bool) -> tuple[np.ndarray, float]:
    logger.debug(f"Calculating P&L for strategy: {strategy}")
    pnl = np.zeros((len(time_points), len(spot_range)))
    initial_cost = 0.0
    for leg in strategy:
        if leg['type'] == 'stock':
            if leg['position'] == 'long':
                initial_cost += leg['strike'] * num_contracts * 100
            else:  # short
                initial_cost -= leg['strike'] * num_contracts * 100
        else:  # option
            initial_premium = get_strategy_price(leg, 'buy' if leg['position'] == 'long' else 'sell') or 0
            fees = calculate_fees(initial_premium * num_contracts * 100, commission_rate)
            if leg['position'] == 'long':
                initial_cost += (initial_premium * num_contracts * 100 + fees)
            else:  # short
                initial_cost -= (initial_premium * num_contracts * 100 - fees)
        
        for t_idx, eval_date in enumerate(time_points):
            for s_idx, spot_price in enumerate(spot_range):
                logger.debug(f"P&L calc: leg={leg['symbol']}, eval_date={eval_date}, spot_price={spot_price}")
                if leg['type'] == 'stock':
                    if leg['position'] == 'long':
                        pnl[t_idx, s_idx] += (spot_price - leg['strike']) * num_contracts * 100
                    else:
                        pnl[t_idx, s_idx] -= (spot_price - leg['strike']) * num_contracts * 100
                else:
                    option_price = calculate_option_price(leg, spot_price, risk_free_rate, volatility, eval_date, expiration_date, use_quantlib)
                    if leg['position'] == 'long':
                        pnl[t_idx, s_idx] += option_price * num_contracts * 100
                    else:
                        pnl[t_idx, s_idx] -= option_price * num_contracts * 100
    pnl -= initial_cost
    logger.debug(f"P&L calculated, initial_cost={initial_cost}")
    return pnl, initial_cost

# Streamlit app setup
st.set_page_config(page_title="Opciones GGAL", layout="wide")
st.title("Análisis de Opciones de GGAL")

# Initialize session state
if 'ggal_stock' not in st.session_state:
    with st.spinner("Cargando datos por primera vez..."):
        st.session_state.ggal_stock, st.session_state.ggal_options = get_ggal_data()
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

if st.button("Actualizar Datos"):
    with st.spinner("Actualizando..."):
        st.session_state.ggal_stock, st.session_state.ggal_options = get_ggal_data()
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.success("Datos actualizados.")

ggal_stock = st.session_state.ggal_stock
ggal_options = st.session_state.ggal_options

if not ggal_stock or not ggal_options:
    st.error("No se pudieron cargar los datos de GGAL. Intente actualizar.")
    st.stop()

st.session_state.current_price = float(ggal_stock["c"])
if not st.session_state.current_price or st.session_state.current_price <= 0:
    st.error("El precio actual de GGAL es inválido o no está disponible. Intente actualizar los datos.")
    st.stop()

st.metric(label="Precio Actual de GGAL (ARS)", value=f"{st.session_state.current_price:.2f}")
st.caption(f"Última actualización: {st.session_state.last_updated}")

# Sidebar configuration
st.sidebar.header("Configuración de Análisis")
expirations = sorted(list(set(
    o["expiration"] for o in ggal_options
    if o["expiration"] is not None and o["expiration"] >= date.today()
)))

default_exp = date(2025, 10, 17)
default_index = expirations.index(default_exp) if default_exp in expirations else (1 if len(expirations) > 1 else 0)

st.session_state.selected_exp = st.sidebar.selectbox(
    "Selecciona la fecha de vencimiento",
    expirations,
    index=default_index,
    format_func=lambda x: x.strftime("%Y-%m-%d")
)

st.session_state.num_contracts = st.sidebar.number_input("Número de contratos", min_value=1, value=1, step=1)
st.session_state.commission_rate = st.sidebar.number_input("Comisión (%)", min_value=0.0, value=0.5, step=0.1) / 100
st.session_state.use_quantlib = st.sidebar.checkbox("Usar QuantLib para cálculos (experimental)", value=False)
st.sidebar.caption("Las tarifas son estimaciones; las reales pueden variar.")
st.session_state.risk_free_rate = get_risk_free_rate()
st.session_state.iv = DEFAULT_IV
strike_percentage = st.sidebar.slider("Rango de Strikes (% del precio actual)", 0.0, 100.0, 20.0) / 100
st.session_state.plot_range_pct = st.sidebar.slider("Rango de Precios en Gráficos (% del precio actual)", 10.0, 200.0, 50.0) / 100
st.sidebar.metric("Tasa Libre de Riesgo (%)", f"{st.session_state.risk_free_rate * 100:.2f}")
st.session_state.debug_plot_range = {
    "min_price": st.session_state.current_price * (1 - st.session_state.plot_range_pct),
    "max_price": st.session_state.current_price * (1 + st.session_state.plot_range_pct)
}

min_strike = st.session_state.current_price * (1 - strike_percentage)
max_strike = st.session_state.current_price * (1 + strike_percentage)

st.session_state.filtered_calls = sorted(
    [o for o in ggal_options if o["type"] == "call" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike],
    key=lambda x: x['strike']
)
st.session_state.filtered_puts = sorted(
    [o for o in ggal_options if o["type"] == "put" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike],
    key=lambda x: x['strike']
)
st.session_state.expiration_days = max(1, (st.session_state.selected_exp - date.today()).days)

if not st.session_state.filtered_calls and not st.session_state.filtered_puts:
    st.warning(
        f"No hay opciones disponibles para la fecha de vencimiento {st.session_state.selected_exp.strftime('%Y-%m-%d')} "
        f"en el rango de strikes [{min_strike:.2f}, {max_strike:.2f}]. "
        "Ajuste el rango de strikes o seleccione otra fecha."
    )
elif not st.session_state.filtered_calls:
    st.warning(
        f"No hay calls disponibles en el rango de strikes [{min_strike:.2f}, {max_strike:.2f}] "
        f"para la fecha de vencimiento {st.session_state.selected_exp.strftime('%Y-%m-%d')}. "
        "Ajuste el rango de strikes."
    )
elif not st.session_state.filtered_puts:
    st.warning(
        f"No hay puts disponibles en el rango de strikes [{min_strike:.2f}, {max_strike:.2f}] "
        f"para la fecha de vencimiento {st.session_state.selected_exp.strftime('%Y-%m-%d')}. "
        "Ajuste el rango de strikes."
    )

# Time points for 3D plot
time_points = [date.today() + timedelta(days=x) for x in np.linspace(0, st.session_state.expiration_days, 10) if date.today() + timedelta(days=x) <= st.session_state.selected_exp]
time_points = time_points or [date.today(), st.session_state.selected_exp]
spot_range = np.linspace(st.session_state.debug_plot_range['min_price'], st.session_state.debug_plot_range['max_price'], 20)

# Strategy selection page
st.sidebar.header("Selecciona una Estrategia")
strategy_page = st.sidebar.radio("Estrategias", ["Opciones Individuales", "Covered Call", "Protective Put"])

# Options table with QuantLib metrics
if strategy_page == "Opciones Individuales":
    st.header("Opciones Individuales")
    calls_data = []
    puts_data = []
    for option in st.session_state.filtered_calls:
        price = calculate_option_price(option, st.session_state.current_price, st.session_state.risk_free_rate, st.session_state.iv, date.today(), st.session_state.selected_exp, st.session_state.use_quantlib)
        calls_data.append({
            'Strike': option['strike'],
            'Bid': option.get('px_bid', 0.0),
            'Ask': option.get('px_ask', 0.0),
            'Theoretical Price': float(price) if price is not None else 0.0
        })
    for option in st.session_state.filtered_puts:
        price = calculate_option_price(option, st.session_state.current_price, st.session_state.risk_free_rate, st.session_state.iv, date.today(), st.session_state.selected_exp, st.session_state.use_quantlib)
        puts_data.append({
            'Strike': option['strike'],
            'Bid': option.get('px_bid', 0.0),
            'Ask': option.get('px_ask', 0.0),
            'Theoretical Price': float(price) if price is not None else 0.0
        })

    # Convert to DataFrame with numeric types
    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)
    for df in [calls_df, puts_df]:
        for col in ['Strike', 'Bid', 'Ask', 'Theoretical Price']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    st.subheader("Calls")
    st.dataframe(calls_df.round(4))
    st.subheader("Puts")
    st.dataframe(puts_df.round(4))

# Covered Call strategy
elif strategy_page == "Covered Call":
    st.header("Covered Call")
    if not st.session_state.filtered_calls:
        st.error("No hay calls disponibles para la fecha de vencimiento seleccionada.")
        st.stop()
    selected_call = st.selectbox(
        "Selecciona un Call para vender",
        st.session_state.filtered_calls,
        format_func=lambda x: f"Strike {x['strike']:.2f} (Bid: {x.get('px_bid', 0.0):.2f}, Ask: {x.get('px_ask', 0.0):.2f})"
    )
    strategy = [
        {'type': 'call', 'strike': selected_call['strike'], 'position': 'short', 'px_bid': selected_call.get('px_bid', 0.0), 'px_ask': selected_call.get('px_ask', 0.0), 'symbol': selected_call['symbol'], 'expiration': selected_call['expiration']},
        {'type': 'stock', 'strike': st.session_state.current_price, 'position': 'long', 'symbol': 'GGAL'}
    ]
    
    # Calculate P&L
    with st.spinner("Calculando P&L para Covered Call..."):
        pnl, total_cost = calculate_strategy_pnl(
            strategy, spot_range, time_points, st.session_state.num_contracts,
            st.session_state.commission_rate, st.session_state.risk_free_rate, st.session_state.iv, st.session_state.selected_exp, st.session_state.use_quantlib
        )
    
    # Create 3D plot
    X, Y = np.meshgrid([t.toordinal() for t in time_points], spot_range)
    Z = pnl.T
    
    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', name='P&L', showscale=True))
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(Z),
        colorscale=[[0, 'rgba(255, 0, 0, 0.3)'], [1, 'rgba(255, 0, 0, 0.3)']],
        showscale=False, name='Breakeven'
    ))
    current_price_plane = np.full_like(Z, st.session_state.current_price)
    fig.add_trace(go.Surface(
        x=X, y=current_price_plane, z=Z,
        colorscale=[[0, 'rgba(0, 0, 255, 0.3)'], [1, 'rgba(0, 0, 255, 0.3)']],
        showscale=False, name='Current Price'
    ))
    
    logger.debug("Configuring Plotly figure for Covered Call")
    fig.update_layout(
        title="Covered Call P&L: Time vs Underlying Price",
        scene=dict(
            xaxis_title="Time (Ordinal Date)",
            yaxis_title="Underlying Price (ARS)",
            zaxis_title="P&L (ARS)",
            xaxis=dict(
                tickvals=[t.toordinal() for t in time_points],
                ticktext=[t.strftime('%Y-%m-%d') for t in time_points]
            ),
            yaxis=dict(
                range=[st.session_state.debug_plot_range['min_price'], st.session_state.debug_plot_range['max_price']]
            ),
            zaxis=dict()
        ),
        showlegend=True,
        width=800,
        height=600
    )
    logger.debug("Plotly figure configured successfully")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Costo Total Estimado (ARS)", f"{total_cost:.2f}")

# Protective Put strategy
elif strategy_page == "Protective Put":
    st.header("Protective Put")
    if not st.session_state.filtered_puts:
        st.error("No hay puts disponibles para la fecha de vencimiento seleccionada.")
        st.stop()
    selected_put = st.selectbox(
        "Selecciona un Put para comprar",
        st.session_state.filtered_puts,
        format_func=lambda x: f"Strike {x['strike']:.2f} (Bid: {x.get('px_bid', 0.0):.2f}, Ask: {x.get('px_ask', 0.0):.2f})"
    )
    strategy = [
        {'type': 'put', 'strike': selected_put['strike'], 'position': 'long', 'px_bid': selected_put.get('px_bid', 0.0), 'px_ask': selected_put.get('px_ask', 0.0), 'symbol': selected_put['symbol'], 'expiration': selected_put['expiration']},
        {'type': 'stock', 'strike': st.session_state.current_price, 'position': 'long', 'symbol': 'GGAL'}
    ]
    
    # Calculate P&L
    with st.spinner("Calculando P&L para Protective Put..."):
        pnl, total_cost = calculate_strategy_pnl(
            strategy, spot_range, time_points, st.session_state.num_contracts,
            st.session_state.commission_rate, st.session_state.risk_free_rate, st.session_state.iv, st.session_state.selected_exp, st.session_state.use_quantlib
        )
    
    # Create 3D plot
    X, Y = np.meshgrid([t.toordinal() for t in time_points], spot_range)
    Z = pnl.T
    
    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', name='P&L', showscale=True))
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(Z),
        colorscale=[[0, 'rgba(255, 0, 0, 0.3)'], [1, 'rgba(255, 0, 0, 0.3)']],
        showscale=False, name='Breakeven'
    ))
    current_price_plane = np.full_like(Z, st.session_state.current_price)
    fig.add_trace(go.Surface(
        x=X, y=current_price_plane, z=Z,
        colorscale=[[0, 'rgba(0, 0, 255, 0.3)'], [1, 'rgba(0, 0, 255, 0.3)']],
        showscale=False, name='Current Price'
    ))
    
    logger.debug("Configuring Plotly figure for Protective Put")
    fig.update_layout(
        title="Protective Put P&L: Time vs Underlying Price",
        scene=dict(
            xaxis_title="Time (Ordinal Date)",
            yaxis_title="Underlying Price (ARS)",
            zaxis_title="P&L (ARS)",
            xaxis=dict(
                tickvals=[t.toordinal() for t in time_points],
                ticktext=[t.strftime('%Y-%m-%d') for t in time_points]
            ),
            yaxis=dict(
                range=[st.session_state.debug_plot_range['min_price'], st.session_state.debug_plot_range['max_price']]
            ),
            zaxis=dict()
        ),
        showlegend=True,
        width=800,
        height=600
    )
    logger.debug("Plotly figure configured successfully")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Costo Total Estimado (ARS)", f"{total_cost:.2f}")

st.info("La configuración ha sido guardada. Por favor, seleccione una página de estrategia en la barra lateral.")