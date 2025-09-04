import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from calc_utils import calculate_strategy_cost
from viz_utils import visualize_bullish_3d, visualize_bearish_3d, visualize_neutral_3d, visualize_volatility_3d, \
    _calculate_iv, black_scholes, intrinsic_value
import logging
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
st.write(
    "Encuentra la estrategia de opciones que maximiza la probabilidad de estar por encima del nivel de breakeven al vencimiento, con pérdida limitada.")

# Sidebar inputs for optimization constraints
st.sidebar.header("Parámetros de Optimización")
max_options = st.sidebar.number_input("Número máximo de opciones en la estrategia", min_value=1, max_value=4, value=2,
                                      step=1)
max_contracts_per_option = st.sidebar.number_input("Contratos máximos por opción", min_value=1, max_value=5, value=3,
                                                   step=1)
min_probability = st.sidebar.slider("Probabilidad mínima aceptable (%)", 0.0, 100.0, 50.0) / 100
limit_loss = st.sidebar.checkbox("Evitar pérdidas ilimitadas", value=True)
max_strategies = st.sidebar.number_input("Max Strategies to Evaluate", min_value=100, max_value=5000, value=500)

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
        opt_type = opt.get("type", "call" if "call" in opt["symbol"].lower() else "put")  # Fallback to symbol inference
        multiplier = num if action == "buy" else -num
        if opt_type == "call":
            call_positions[strike] = call_positions.get(strike, 0) + multiplier
        else:
            put_positions[strike] = put_positions.get(strike, 0) + multiplier

    has_unlimited = False
    if call_positions:
        max_call_strike = max(call_positions.keys())
        net_call_position = sum(call_positions.values())
        if any(pos < 0 for pos in call_positions.values()) and (
                net_call_position > 0 or call_positions[max_call_strike] > 0):
            has_unlimited = True
    if put_positions:
        min_put_strike = min(put_positions.keys())
        net_put_position = sum(put_positions.values())
        if any(pos < 0 for pos in put_positions.values()) and (
                net_put_position > 0 or put_positions[min_put_strike] > 0):
            has_unlimited = True
    return not has_unlimited


# Function to calculate strategy P&L and key metrics
def calculate_strategy_metrics(options, actions, contracts, current_price, expiration_days, iv):
    net_entry = calculate_strategy_cost(options, actions, contracts, commission_rate)["net_cost"]
    strikes = [opt["strike"] for opt in options]
    num_contracts_total = sum(contracts)

    def strategy_value(price, T, sigma):
        total_value = 0.0
        use_intrinsic = T <= 1e-6
        for opt, action, num in zip(options, actions, contracts):
            strike = opt["strike"]
            opt_type = opt.get("type", "call" if "call" in opt["symbol"].lower() else "put")
            action_mult = 1 if action == "buy" else -1
            if use_intrinsic:
                value = intrinsic_value(price, strike, opt_type)
            else:
                value = black_scholes(price, strike, T, risk_free_rate, sigma, opt_type)
            total_value += value * action_mult * num * 100
        return total_value - net_entry

    # Calibrate IV
    iv_calibrated = _calculate_iv(net_entry, current_price, expiration_days, strategy_value, options, actions,
                                  contracts)
    iv_calibrated = max(iv_calibrated, 1e-9)
    if iv_calibrated < 1e-6:
        st.session_state["iv_failure_count"] += 1
        logger.warning(f"IV calibration failed for strategy; using fallback IV: {iv}")
        iv_calibrated = iv

    # Compute P&L at boundaries
    T = expiration_days / 365.0
    min_price = min(strikes) * 0.8
    max_price = max(strikes) * 1.2
    max_profit = max(strategy_value(max_price, T, iv_calibrated), strategy_value(min_price, T, iv_calibrated))
    max_loss = min(strategy_value(max_price, T, iv_calibrated), strategy_value(min_price, T, iv_calibrated))
    breakeven = current_price
    while strategy_value(breakeven, T, iv_calibrated) < 0:
        breakeven += 1
    while strategy_value(breakeven, T, iv_calibrated) > 0:
        breakeven -= 1

    prob = estimate_breakeven_probability(current_price, breakeven, expiration_days, iv_calibrated)
    return {
        "net_cost": net_entry,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "probability": prob,
        "strikes": strikes,
        "num_contracts": num_contracts_total,
        "iv": iv_calibrated
    }


# Function to estimate breakeven probability
def estimate_breakeven_probability(current_price, breakeven, expiration_days, iv):
    T = expiration_days / 365.0
    mu = risk_free_rate * T
    sigma = iv * np.sqrt(T)
    z = (np.log(breakeven / current_price) - mu) / sigma
    return norm.cdf(z)  # Probability of being above breakeven


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
        # Generate possible actions and contracts
        for action_combo in product(["buy", "sell"], repeat=num_opts):
            for contract_combo in product(range(1, max_contracts_per_option + 1), repeat=num_opts):
                options = list(combo)
                actions = list(action_combo)
                contracts = list(contract_combo)
                if limit_loss and not has_limited_loss(options, actions, contracts):
                    continue
                metrics = calculate_strategy_metrics(options, actions, contracts, current_price, expiration_days,
                                                     st.session_state.iv)
                if metrics["probability"] > min_probability and metrics["probability"] > optimal_prob:
                    optimal_prob = metrics["probability"]
                    optimal_strategy = {
                        "options": options,
                        "actions": actions,
                        "contracts": contracts,
                        "result": metrics
                    }

if optimal_strategy:
    st.session_state["optimal_strategy"] = optimal_strategy
    st.write("Estrategia Óptima Encontrada")
    options, actions, contracts, result = optimal_strategy["options"], optimal_strategy["actions"], optimal_strategy[
        "contracts"], optimal_strategy["result"]

    # Determine strategy type and visualize
    is_bullish = all(a == "buy" for a in actions[:2]) and len(set(o["strike"] for o in options)) == 2
    is_bearish = all(a == "sell" for a in actions[:2]) and len(set(o["strike"] for o in options)) == 2
    is_volatility = len(options) == 2 and options[0]["strike"] == options[1]["strike"] and actions[0] != actions[1]

    if is_bullish and len(options) == 2:
        visualize_bullish_3d(result, current_price, expiration_days, result["iv"], "Optimal Bullish Strategy", options,
                             actions)
    elif is_bearish and len(options) == 2:
        visualize_bearish_3d(result, current_price, expiration_days, result["iv"], "Optimal Bearish Strategy", options,
                             actions)
    elif is_volatility and len(options) == 2:
        visualize_volatility_3d(result, current_price, expiration_days, result["iv"], "Optimal Volatility Strategy",
                                options, actions)
    else:
        visualize_neutral_3d(result, current_price, expiration_days, result["iv"], "Optimal Neutral Strategy", options,
                             actions)
else:
    st.warning("No optimal strategy found.")

# Add IV failure warnings
if st.session_state["iv_failure_count"] > 10:
    st.error("Multiple IV calibration failures detected. Check option prices or market data.")

# Add debug logging
logger.debug(f"Processed {count} strategies. Optimal probability: {optimal_prob}")
logger.debug(f"Optimal strategy: {optimal_strategy}")