import streamlit as st
import utils
from datetime import datetime

st.set_page_config(page_title="Opciones GGAL", layout="wide")
st.title("Análisis de Opciones de GGAL")

# Data loading button
if st.button("Actualizar Datos"):
    with st.spinner("Cargando datos..."):
        data = utils.load_ggal_data()
        if data is not None:
            # Process data (e.g., filter calls and puts)
            st.session_state.filtered_calls = [opt for opt in data if opt["type"] == "call"]
            st.session_state.filtered_puts = [opt for opt in data if opt["type"] == "put"]
            st.session_state.current_price = data.get("current_price", 4500)  # Default or from data
            st.session_state.expiration_days = 30  # Adjust based on data or user input
            st.session_state.iv = data.get("implied_volatility", 0.2)  # Default or from data
            st.session_state.num_contracts = st.session_state.get("num_contracts", 1)
            st.session_state.commission_rate = st.session_state.get("commission_rate", 0.01)
            st.session_state.risk_free_rate = st.session_state.get("risk_free_rate", 0.50)
            st.session_state.plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
        else:
            st.error("Falló la actualización de datos. Verifique la conexión o intente de nuevo.")

# Display current data or cached data
if 'ggal_data' in st.session_state and st.session_state.ggal_data is not None:
    st.write("Datos actuales:", st.session_state.ggal_data)
else:
    st.warning("No hay datos disponibles. Actualice para cargar.")

# Rest of your app (e.g., tabs for strategies)

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