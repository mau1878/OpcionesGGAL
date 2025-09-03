import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import logging
from calc_utils import calculate_strategy_cost, black_scholes, intrinsic_value, _calculate_iv
from viz_utils import visualize_neutral_3d

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Scenario Analysis", layout="wide")
st.title("Análisis de Escenarios")

# Check for required session state
if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
current_price = st.session_state.current_price
expiration_days = st.session_state.expiration_days
risk_free_rate = st.session_state.risk_free_rate
commission_rate = st.session_state.commission_rate
plot_range_pct = st.session_state.plot_range_pct

# Log available options for debugging
logger.info(f"Available calls: {len(calls)}, puts: {len(puts)}")

# Sidebar inputs
st.sidebar.header("Parámetros de Análisis")
price_range_pct = st.sidebar.slider("Rango de Precio para Ganancia Potencial (%)", 5.0, 50.0, 20.0) / 100

# Initialize session state for selections
if "selected_options" not in st.session_state:
    st.session_state["selected_options"] = []
if "option_actions" not in st.session_state:
    st.session_state["option_actions"] = []
if "option_contracts" not in st.session_state:
    st.session_state["option_contracts"] = []
if "iv_failure_count" not in st.session_state:
    st.session_state["iv_failure_count"] = 0

# Function to calculate strategy value
def calculate_strategy_value(options, actions, contracts, price, T, sigma):
    total_value = 0.0
    use_intrinsic = T <= 1e-6
    for opt, action, num in zip(options, actions, contracts):
        strike = opt["strike"]
        opt_type = opt["type"]
        action_mult = 1 if action == "buy" else -1
        if use_intrinsic:
            value = intrinsic_value(price, strike, opt_type)
        else:
            value = black_scholes(price, strike, T, risk_free_rate, sigma, opt_type)
        total_value += value * action_mult * num * 100
    return total_value

# Custom strategy builder
st.subheader("Construir Estrategia Personalizada")
st.write("Selecciona hasta 4 opciones para analizar la estrategia personalizada.")
num_legs = st.number_input("Número de Patas", min_value=1, max_value=4, value=2, key="num_legs")
legs = []

# Create a list of all available options
all_options = calls + puts
option_symbols = sorted([o["symbol"] for o in all_options])

for i in range(num_legs):
    with st.expander(f"Pata {i+1}"):
        symbol = st.selectbox(f"Opción {i+1}", option_symbols, key=f"symbol_{i}")
        action = st.selectbox(f"Acción {i+1}", ["buy", "sell"], key=f"action_{i}")
        contracts = st.number_input(f"Contratos {i+1}", min_value=1, max_value=100, value=1, key=f"contracts_{i}")
        selected_option = next((o for o in all_options if o["symbol"] == symbol), None)
        if selected_option:
            legs.append({"option": selected_option, "action": action, "contracts": contracts})
        else:
            st.warning(f"Opción {symbol} no encontrada.")

# Calculate and visualize
if st.button("Analizar Estrategia"):
    if not legs:
        st.error("Por favor, seleccione al menos una opción válida.")
        logger.error("No valid legs selected")
        st.stop()

    options = [leg["option"] for leg in legs]
    actions = [leg["action"] for leg in legs]
    contracts = [leg["contracts"] for leg in legs]
    
    # Calculate strategy cost
    result = calculate_strategy_cost(options, actions, contracts, commission_rate)
    if not result:
        st.error("No se pudo calcular el costo de la estrategia. Verifique los datos de las opciones.")
        logger.error("Strategy cost calculation failed")
        st.stop()

    # Calculate IV
    def strategy_value_func(S, T, sigma):
        return calculate_strategy_value(options, actions, contracts, S, T, sigma)
    
    iv = _calculate_iv(
        result["raw_net"], current_price, expiration_days, strategy_value_func, options, actions, contracts
    )
    
    # Estimate max profit/loss (simplified; assumes wide price range)
    min_price = current_price * (1 - price_range_pct)
    max_price = current_price * (1 + price_range_pct)
    T = expiration_days / 365.0
    min_value = calculate_strategy_value(options, actions, contracts, min_price, T, iv)
    max_value = calculate_strategy_value(options, actions, contracts, max_price, T, iv)
    max_profit = max(max_value, min_value, 0) - result["net_cost"]
    max_loss = abs(min(max_value, min_value, 0)) + result["net_cost"] if min_value < 0 else result["net_cost"]
    
    # Estimate breakevens (simplified; assumes linear payoff for initial estimate)
    lower_breakeven = current_price - (result["net_cost"] / (100 * sum(contracts)))
    upper_breakeven = current_price + (result["net_cost"] / (100 * sum(contracts)))
    
    result.update({
        "max_profit": max_profit if max_profit > 0 else 0.01,
        "max_loss": max_loss,
        "lower_breakeven": lower_breakeven,
        "upper_breakeven": upper_breakeven,
        "strikes": [o["strike"] for o in options],
        "num_contracts": sum(contracts),
        "contract_ratios": contracts
    })

    # Visualize
    key = f"Custom Strategy {'/'.join(str(s) for s in result['strikes'])}"
    try:
        visualize_neutral_3d(result, current_price, expiration_days, iv, key, options, actions)
        st.success("Estrategia analizada y visualizada exitosamente.")
        logger.info(f"Custom strategy visualized: {key}")
    except Exception as e:
        st.error(f"Error al visualizar la estrategia: {e}")
        logger.error(f"Visualization error: {e}")

# Display results in a table
if st.session_state.get("selected_options"):
    options = st.session_state["selected_options"]
    actions = st.session_state["option_actions"]
    contracts = st.session_state["option_contracts"]
    result = calculate_strategy_cost(options, actions, contracts, commission_rate)
    if result:
        df = pd.DataFrame({
            "Symbol": [o["symbol"] for o in options],
            "Type": [o["type"] for o in options],
            "Strike": [o["strike"] for o in options],
            "Action": actions,
            "Contracts": contracts,
            "Price": [(o["px_ask"] if a == "buy" else o["px_bid"]) for o, a in zip(options, actions)]
        })
        st.subheader("Resumen de la Estrategia")
        st.dataframe(df)
        
        # Display key metrics
        st.metric("Costo Neto", f"{result['net_cost']:.2f} ARS")
        st.metric("Pérdida Máxima", f"{result.get('max_loss', 0):.2f} ARS")
        st.metric("Ganancia Máxima", f"{result.get('max_profit', 0):.2f} ARS")
        st.metric("Breakeven Inferior", f"{result.get('lower_breakeven', 0):.2f} ARS")
        st.metric("Breakeven Superior", f"{result.get('upper_breakeven', 0):.2f} ARS")

# Display IV failure warning
if st.session_state.get("iv_failure_count", 0) > 10:
    st.error("Multiple IV calibration failures detected. Check option prices or market data.")
    logger.warning(f"IV failure count: {st.session_state['iv_failure_count']}")