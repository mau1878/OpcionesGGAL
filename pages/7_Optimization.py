import streamlit as st
import pandas as pd
from calc_utils import calculate_strategy_cost
from viz_utils import visualize_bullish_3d, visualize_bearish_3d, visualize_neutral_3d, visualize_volatility_3d
import logging
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from itertools import product

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Strategy Optimizer", layout="wide")
st.title("Optimizador de Estrategias de Opciones")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
current_price = st.session_state.current_price
expiration_days = st.session_state.expiration_days
risk_free_rate = st.session_state.risk_free_rate
num_contracts = st.session_state.num_contracts
commission_rate = st.session_state.commission_rate
plot_range_pct = st.session_state.plot_range_pct

st.header("Buscar Estrategia Óptima")
st.write("Encuentra la estrategia de opciones que maximiza la probabilidad de estar por encima del nivel de breakeven al vencimiento, con pérdida limitada.")

# Sidebar inputs for optimization constraints
st.sidebar.header("Parámetros de Optimización")
max_options = st.sidebar.number_input("Número máximo de opciones en la estrategia", min_value=1, max_value=4, value=2, step=1)
max_contracts_per_option = st.sidebar.number_input("Contratos máximos por opción", min_value=1, max_value=5, value=3, step=1)
min_probability = st.sidebar.slider("Probabilidad mínima aceptable (%)", 0.0, 100.0, 50.0) / 100
limit_loss = st.sidebar.checkbox("Evitar pérdidas ilimitadas", value=True)
max_strategies = st.sidebar.number_input("Max Strategies to Evaluate", min_value=100, max_value=5000, value=500)  # Added for performance

# Initialize session state for results
if "optimal_strategy" not in st.session_state:
    st.session_state["optimal_strategy"] = None

# Track IV calibration failures
if "iv_failure_count" not in st.session_state:
    st.session_state["iv_failure_count"] = 0

# Function to check for limited loss
def has_limited_loss(options, actions, contracts):
    call_positions = {}
    put_positions = {}
    for opt, action, num in zip(options, actions, contracts):
        strike = opt["strike"]
        opt_type = opt["type"]
        multiplier = num if action == "buy" else -num
        if opt_type == "call":
            call_positions[strike] = call_positions.get(strike, 0) + multiplier
        else:
            put_positions[strike] = put_positions.get(strike, 0) + multiplier
    
    if call_positions:
        max_strike = max(call_positions.keys())
        net_call_position = sum(call_positions.values())
        if call_positions[max_strike] < 0 or (net_call_position < 0 and max(call_positions, key=lambda k: call_positions[k]) == max_strike):
            return False
    
    if put_positions:
        min_strike = min(put_positions.keys())
        net_put_position = sum(put_positions.values())
        if put_positions[min_strike] < 0 or (net_put_position < 0 and min(put_positions, key=lambda k: put_positions[k]) == min_strike):
            return False
    
    return True

# Function to calculate strategy P&L
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

# Function to estimate breakeven probability
def estimate_breakeven_probability(current_price, breakeven, expiration_days, iv, direction="above"):
    T = expiration_days / 365.0
    mu = risk_free_rate * T
    sigma = iv * np.sqrt(T)
    z = (np.log(breakeven / current_price) - mu) / sigma
    if direction == "above":
        return norm.cdf(z)
    else:
        return norm.cdf(-z)

# Optimization loop
all_options = calls + puts
nearest_options = sorted(all_options, key=lambda o: abs(o["strike"] - current_price))[:15]  # Limit for performance
optimal_prob = 0.0
optimal_strategy = None
count = 0
for num_opts in range(1, max_options + 1):
    for combo in product(nearest_options, repeat=num_opts):
        if count >= max_strategies:
            break
        count += 1
        # Generate possible actions (buy/sell) and contracts (1 to max_contracts_per_option)
        for action_combo in product(["buy", "sell"], repeat=num_opts):
            for contract_combo in product(range(1, max_contracts_per_option + 1), repeat=num_opts):
                options = list(combo)
                actions = list(action_combo)
                contracts = list(contract_combo)
                if limit_loss and not has_limited_loss(options, actions, contracts):
                    continue
                result = calculate_strategy_cost(options, actions, contracts, commission_rate)
                if result:
                    result["strikes"] = [o["strike"] for o in options]
                    result["max_profit"] = 10000  # Placeholder; calculate based on strategy
                    result["max_loss"] = result["net_cost"]  # Placeholder
                    result["breakeven"] = current_price + result["net_cost"] / 100  # Placeholder
                    prob = estimate_breakeven_probability(current_price, result["breakeven"], expiration_days, st.session_state.iv)
                    if prob > min_probability and prob > optimal_prob:
                        optimal_prob = prob
                        optimal_strategy = {
                            "options": options,
                            "actions": actions,
                            "contracts": contracts,
                            "result": result
                        }

if optimal_strategy:
    st.session_state["optimal_strategy"] = optimal_strategy
    st.write("Estrategia Óptima Encontrada")
    options, actions, contracts, result = optimal_strategy["options"], optimal_strategy["actions"], optimal_strategy["contracts"], optimal_strategy["result"]
    # Visualize (assume neutral for custom)
    result["num_contracts"] = sum(contracts)
    visualize_neutral_3d(result, current_price, expiration_days, st.session_state.iv, "Optimal Strategy", options, actions)
else:
    st.warning("No optimal strategy found.")

# Add IV failure warnings
if st.session_state["iv_failure_count"] > 10:
    st.error("Multiple IV calibration failures detected. Check option prices or market data.")