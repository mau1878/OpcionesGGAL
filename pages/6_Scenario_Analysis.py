import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Scenario Analysis", layout="wide")
st.title("Análisis de Escenarios")

# Check if required session state variables are available
if not all(key in st.session_state for key in ['filtered_calls', 'filtered_puts', 'current_price', 'expiration_days', 'risk_free_rate', 'commission_rate', 'plot_range_pct']):
    st.error("Datos de opciones no disponibles. Por favor, configure los parámetros en la página de Opciones GGAL.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
current_price = st.session_state.current_price
num_contracts = st.session_state.num_contracts
commission_rate = st.session_state.commission_rate
expiration_days = st.session_state.expiration_days

if not calls or not puts:
    st.warning("No hay datos de Calls o Puts disponibles para la fecha de vencimiento seleccionada.")
    st.stop()

# Sidebar for Inputs
# Sidebar for Inputs
st.sidebar.header("Parámetros de Análisis")
if 'plot_range_pct' not in st.session_state:
    st.session_state.plot_range_pct = 0.5  # Default to 50% in decimal
st.session_state.plot_range_pct = st.sidebar.slider(
    "Rango de Precio para Gráficos (% del precio actual)", 10.0, 200.0, st.session_state.get('plot_range_pct', 0.5) * 100
) / 100
filter_unlimited_loss = st.sidebar.checkbox("Filtrar Estrategias con Pérdidas Ilimitadas", value=True)

# Option Selection
st.header("Selección de Opciones")
num_legs = st.number_input("Número de patas", min_value=1, max_value=10, value=2, step=1)
option_data = []
for i in range(num_legs):
    col1, col2, col3 = st.columns(3)
    with col1:
        opt_type = st.selectbox(f"Tipo de Opción {i+1}", ["Call", "Put"], key=f"opt_type_{i}")
        options = calls if opt_type == "Call" else puts
    with col2:
        strike = st.selectbox(f"Strike {i+1}", [opt["strike"] for opt in options], key=f"strike_{i}")
    with col3:
        action = st.selectbox(f"Acción {i+1}", ["buy", "sell"], key=f"action_{i}")
    contracts = st.number_input(f"Contratos {i+1}", min_value=1, value=num_contracts, step=1, key=f"contracts_{i}")
    option_data.append({"type": opt_type.lower(), "strike": strike, "action": action, "contracts": contracts})

# Calculate Strategy
if st.button("Analizar Estrategia"):
    options = []
    actions = []
    contracts = []
    for data in option_data:
        opt = next((o for o in (calls if data["type"] == "call" else puts) if o["strike"] == data["strike"]), None)
        if opt:
            options.append(opt)
            actions.append(data["action"])
            contracts.append(data["contracts"])
        else:
            st.error(f"No se encontró la opción {data['type']} con strike {data['strike']}.")
            st.stop()
    
    # Limited Loss Check
    if filter_unlimited_loss and not utils.has_limited_loss(options, actions, contracts):
        st.warning("Estrategia descartada: contiene pérdidas potenciales ilimitadas.")
        st.stop()
    elif not utils.has_limited_loss(options, actions, contracts):
        st.warning("Advertencia: La estrategia seleccionada tiene pérdidas potenciales ilimitadas.")
    
    net_cost = utils.calculate_strategy_cost(options, actions, contracts)
    if net_cost is None:
        st.error("No se pudo calcular el costo neto de la estrategia. Verifique los datos de las opciones.")
        st.stop()
    
    T_years = expiration_days / 365.0
    iv_calibrated = utils._calibrate_iv(
        net_cost / (100 * sum(abs(c) for c in contracts)),
        current_price,
        expiration_days,
        lambda p, t, s: utils.calculate_strategy_value(options, actions, contracts, p, t, s) / (100 * sum(abs(c) for c in contracts)),
        options, actions, contracts
    )
    if iv_calibrated == utils.DEFAULT_IV:
        st.warning("No se pudo calibrar la volatilidad implícita. Usando valor predeterminado.")
    iv_calibrated = max(iv_calibrated, 1e-9)
    breakeven_prob = utils.estimate_breakeven_probability(options, actions, contracts, net_cost, T_years, iv_calibrated)
    
    # Strategy Description
    strategy_type = "Custom Strategy (" + ", ".join(f"{action} {opt['type']} {opt['strike']}" for opt, action in zip(options, actions)) + ")"
    
    # Metrics
    st.header("Métricas de la Estrategia")
    st.metric("Costo Neto (ARS)", f"{net_cost:.2f}")
    st.metric("Probabilidad de Breakeven", f"{breakeven_prob:.1%}")
    
    # Visualizations
    min_price = current_price * (1 - st.session_state.plot_range_pct)
    max_price = current_price * (1 + st.session_state.plot_range_pct)
    
    # 3D Plot
    X, Y, Z = utils._compute_payoff_grid(
        lambda p, t, s: utils.calculate_strategy_value(options, actions, contracts, p, t, s),
        current_price, expiration_days, iv_calibrated, net_cost
    )
    fig = utils._create_3d_figure(X, Y, Z, strategy_type, current_price)
    st.plotly_chart(fig, use_container_width=True, key="scenario_3d_plot")
    
    # 2D Plot
    st.subheader("Diagrama de P&L al Vencimiento")
    price_points = np.linspace(min_price, max_price, 100)
    payoff_at_expiration = [utils.calculate_strategy_value(options, actions, contracts, p, 0, iv_calibrated) - net_cost for p in price_points]
    fig_2d = go.Figure()
    fig_2d.add_trace(
        go.Scatter(
            x=price_points, y=payoff_at_expiration,
            mode="lines", name="P&L al Vencimiento", line=dict(color="blue")
        )
    )
    fig_2d.add_trace(
        go.Scatter(
            x=[current_price],
            y=[utils.calculate_strategy_value(options, actions, contracts, current_price, 0, iv_calibrated) - net_cost],
            mode="markers", name="Precio Actual", marker=dict(size=10, color="red")
        )
    )
    fig_2d.update_layout(
        title=f"P&L at Expiration for {strategy_type}",
        xaxis_title="Precio de GGAL (ARS)",
        yaxis_title="P&L (ARS)",
        xaxis=dict(tickvals=[min_price, current_price, max_price],
                   ticktext=[f"{min_price:.2f}", f"{current_price:.2f}", f"{max_price:.2f}"]),
        yaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=1),
        showlegend=True,
        height=400
    )
    def find_breakeven(price_range, tolerance=1e-3):
        breakevens = []
        for p in np.linspace(price_range[0], price_range[1], 1000):
            if abs(utils.calculate_strategy_value(options, actions, contracts, p, 0, iv_calibrated) - net_cost) < tolerance:
                breakevens.append(p)
        return breakevens
    breakevens = find_breakeven([min_price, max_price])
    for b in breakevens[:2]:
        fig_2d.add_vline(x=b, line_dash="dash", line_color="green", annotation_text=f"Breakeven: {b:.2f}")
    if breakevens:
        fig_2d.add_vrect(
            x0=min(breakevens), x1=max(breakevens),
            fillcolor="green", opacity=0.1, line_width=0,
            annotation_text="Rango de Ganancia", annotation_position="top"
        )
    st.plotly_chart(fig_2d, use_container_width=True, key="scenario_2d_plot")