import streamlit as st
from datetime import datetime, date
import utils
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to fetch Cauciones table
def fetch_cauciones_table(url: str, headers: dict = None) -> pd.DataFrame:
    """
    Fetches the 'Cauciones' table from the given URL and returns it as a pandas DataFrame.
    """
    session = requests.Session()
    if headers:
        session.headers.update(headers)
    
    response = session.get(url)
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
    return df

# Function to get Tasa Tomadora for Plazo 30, 31, 32, or 33
def get_risk_free_rate():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/Cauciones"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TableScraper/1.0)"}
    try:
        df = fetch_cauciones_table(url, headers)
        # Clean and convert Tasa Tomadora to float
        df['Tasa Tomadora'] = df['Tasa Tomadora'].str.replace('%', '').str.replace(',', '.').str.strip()
        df['Tasa Tomadora'] = pd.to_numeric(df['Tasa Tomadora'], errors='coerce') / 100
        df['Plazo'] = pd.to_numeric(df['Plazo'], errors='coerce')
        
        # Filter for PESOS and valid Tasa Tomadora
        df_pesos = df[(df['Moneda'] == 'PESOS') & (df['Tasa Tomadora'].notna()) & (df['Tasa Tomadora'] > 0)]
        
        # Try Plazo 30, 31, 32, 33 in order
        for plazo in [30, 31, 32, 33, 34, 35]:
            tasa = df_pesos[df_pesos['Plazo'] == plazo]['Tasa Tomadora']
            if not tasa.empty:
                return tasa.iloc[0]
        # Fallback to default if no valid Plazo found
        st.warning("No se encontró una Tasa Tomadora válida para Plazo 30-33 días en PESOS. Usando tasa predeterminada de 35%.")
        return 0.35
    except Exception as e:
        st.error(f"Error al obtener la Tasa Tomadora: {e}. Usando tasa predeterminada de 35%.")
        return 0.35

st.set_page_config(page_title="Opciones GGAL", layout="wide")
st.title("Análisis de Opciones de GGAL")

# --- Data Loading ---
if 'ggal_stock' not in st.session_state:
    with st.spinner("Cargando datos por primera vez..."):
        st.session_state.ggal_stock, st.session_state.ggal_options = utils.get_ggal_data()
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

if st.button("Actualizar Datos"):
    with st.spinner("Actualizando..."):
        st.session_state.ggal_stock, st.session_state.ggal_options = utils.get_ggal_data()
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

# --- Sidebar for Inputs ---
st.sidebar.header("Configuración de Análisis")

expirations = sorted(list(set(
    o["expiration"] for o in ggal_options 
    if o["expiration"] is not None and o["expiration"] >= date.today()
)))

# Set default expiration to October 17, 2025
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
st.sidebar.caption("Las tarifas son estimaciones; las reales pueden variar.")
# Fetch risk-free rate dynamically
st.session_state.risk_free_rate = get_risk_free_rate()
st.session_state.iv = utils.DEFAULT_IV  # 0.30
strike_percentage = st.sidebar.slider("Rango de Strikes (% del precio actual)", 0.0, 100.0, 20.0) / 100
st.session_state.plot_range_pct = st.sidebar.slider("Rango de Precios en Gráficos 3D (% del precio actual)", 10.0, 200.0, 50.0) / 100

# Display risk-free rate in sidebar
st.sidebar.metric("Tasa Libre de Riesgo (%)", f"{st.session_state.risk_free_rate * 100:.2f}")

# Debug plot range
st.session_state.debug_plot_range = {
    "min_price": st.session_state.current_price * (1 - st.session_state.plot_range_pct),
    "max_price": st.session_state.current_price * (1 + st.session_state.plot_range_pct)
}

min_strike = st.session_state.current_price * (1 - strike_percentage)
max_strike = st.session_state.current_price * (1 + strike_percentage)

st.session_state.filtered_calls = [o for o in ggal_options if o["type"] == "call" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike]
st.session_state.filtered_puts = [o for o in ggal_options if o["type"] == "put" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike]
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

st.info("La configuración ha sido guardada. Por favor, seleccione una página de estrategia en la barra lateral.")