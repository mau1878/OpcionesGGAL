import streamlit as st
from datetime import datetime, timezone, date
import utils

st.set_page_config(
    page_title="GGAL Options Analyzer",
    layout="wide"
)

st.title("Analizador de Estrategias de Opciones para GGAL")
st.write("¡Bienvenido! Use la barra lateral para cargar datos y configurar los parámetros. Luego, navegue a las páginas de estrategias para el análisis.")

# --- Data Loading ---
if 'ggal_stock' not in st.session_state:
    with st.spinner("Cargando datos por primera vez..."):
        st.session_state.ggal_stock, st.session_state.ggal_options = utils.get_ggal_data()
        st.session_state.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

if st.button("Actualizar Datos"):
    with st.spinner("Actualizando..."):
        st.session_state.ggal_stock, st.session_state.ggal_options = utils.get_ggal_data()
        st.session_state.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.success("Datos actualizados.")

ggal_stock = st.session_state.ggal_stock
ggal_options = st.session_state.ggal_options

if not ggal_stock or not ggal_options:
    st.error("No se pudieron cargar los datos de GGAL. Intente actualizar.")
    st.stop()

st.session_state.current_price = float(ggal_stock["c"])
st.metric(label="Precio Actual de GGAL (ARS)", value=f"{st.session_state.current_price:.2f}")
st.caption(f"Última actualización: {st.session_state.last_updated}")

# --- Sidebar for Inputs ---
st.sidebar.header("Configuración de Análisis")

expirations = sorted(list(set(o["expiration"] for o in ggal_options if o["expiration"] is not None)))
st.session_state.selected_exp = st.sidebar.selectbox(
    "Selecciona la fecha de vencimiento",
    expirations,
    format_func=lambda x: x.strftime("%Y-%m-%d")
)

st.session_state.num_contracts = st.sidebar.number_input("Número de contratos", min_value=1, value=1, step=1)
st.session_state.commission_rate = st.sidebar.number_input("Comisión (%)", min_value=0.0, value=0.5, step=0.1) / 100
st.session_state.iv = st.sidebar.number_input("Volatilidad Implícita (%)", min_value=0.0, value=utils.DEFAULT_IV * 100, step=1.0) / 100
strike_percentage = st.sidebar.slider("Rango de Strikes (% del precio actual)", 0.0, 100.0, 20.0) / 100
# ADD the following lines to the sidebar in 1_OpcionesGGAL.py, after the existing inputs in st.sidebar.header("Configuración de Análisis")

st.session_state.plot_range_pct = st.sidebar.slider("Rango de Precios en Gráficos 3D (% del precio actual)", 10.0, 200.0, 30.0) / 100
st.session_state.risk_free_rate = st.sidebar.number_input("Tasa Libre de Riesgo (%)", min_value=0.0, value=50.0, step=1.0) / 100
# --- Filter options based on inputs and store in session_state ---
min_strike = st.session_state.current_price * (1 - strike_percentage)
max_strike = st.session_state.current_price * (1 + strike_percentage)

st.session_state.filtered_calls = [o for o in ggal_options if o["type"] == "call" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike]
st.session_state.filtered_puts = [o for o in ggal_options if o["type"] == "put" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike]
st.session_state.expiration_days = max(1, (st.session_state.selected_exp - date.today()).days)

st.info("La configuración ha sido guardada. Por favor, seleccione una página de estrategia en la barra lateral.")