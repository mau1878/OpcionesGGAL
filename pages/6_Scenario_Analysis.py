import streamlit as st
import pandas as pd
import utils
import logging
import plotly.graph_objects as go
import numpy as np

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Scenario Analysis", layout="wide")
st.title("Análisis de Escenarios Personalizados")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
current_price = st.session_state.current_price
expiration_days = st.session_state.expiration_days
iv = st.session_state.iv
risk_free_rate = st.session_state.risk_free_rate
num_contracts = st.session_state.num_contracts
commission_rate = st.session_state.commission_rate
plot_range_pct = st.session_state.plot_range_pct

st.header("Configura tu Estrategia Personalizada")
st.write("Selecciona hasta 4 opciones (call o put) para construir tu estrategia y ver el P&L estimado.")

# Initialize session state for scenario options
if "scenario_options" not in st.session_state:
    st.session_state["scenario_options"] = [
        {"type": "call", "strike": None, "action": "buy", "contracts": 1} for _ in range(4)
    ]

# Create a form for option selection
with st.form(key="scenario_form"):
    st.subheader("Selecciona Opciones")
    cols = st.columns(4)
    all_options = calls + puts
    all_strikes = sorted(list(set(o["strike"] for o in all_options)))
    option_types = ["call", "put"]
    actions = ["buy", "sell"]

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Opción {i+1}**")
            option = st.session_state["scenario_options"][i]
            option["type"] = st.selectbox(
                f"Tipo Opción {i+1}", option_types, index=0 if option["type"] == "call" else 1, key=f"opt_type_{i}"
            )
            option["strike"] = st.selectbox(
                f"Strike {i+1}", all_strikes, index=0 if option["strike"] is None else all_strikes.index(option["strike"]) if option["strike"] in all_strikes else 0,
                key=f"opt_strike_{i}"
            )
            option["action"] = st.selectbox(
                f"Acción {i+1}", actions, index=0 if option["action"] == "buy" else 1, key=f"opt_action_{i}"
            )
            option["contracts"] = st.number_input(
                f"Contratos {i+1}", min_value=0, value=option["contracts"], step=1, key=f"opt_contracts_{i}"
            )

    submit_button = st.form_submit_button("Calcular P&L")

# Calculate P&L when form is submitted
if submit_button:
    selected_options = []
    option_details = []
    total_cost = 0.0
    valid = True

    for i, opt in enumerate(st.session_state["scenario_options"]):
        if opt["contracts"] == 0:
            continue
        option = next((o for o in (calls if opt["type"] == "call" else puts) if o["strike"] == opt["strike"]), None)
        if not option:
            st.error(f"No se encontró la opción {opt['type']} con strike {opt['strike']}.")
            valid = False
            continue
        price = utils.get_strategy_price(option, opt["action"])
        if price is None:
            st.error(f"Precio inválido para la opción {opt['type']} con strike {opt['strike']}.")
            valid = False
            continue
        base_cost = price * 100 * opt["contracts"] * (1 if opt["action"] == "buy" else -1)
        fees = utils.calculate_fees(abs(base_cost), commission_rate)
        total_cost += base_cost + fees
        selected_options.append({
            "option": option,
            "action": opt["action"],
            "contracts": opt["contracts"]
        })
        option_details.append({
            "Type": opt["type"].capitalize(),
            "Strike": float(opt["strike"]),  # Ensure numeric
            "Action": opt["action"].capitalize(),
            "Contracts": int(opt["contracts"]),  # Ensure integer
            "Price": float(price),  # Store as float
            "Base Cost": float(base_cost),  # Store as float
            "Fees": float(fees)  # Store as float
        })

    if valid and selected_options:
        # Display selected options
        st.subheader("Opciones Seleccionadas")
        df_options = pd.DataFrame(option_details)
        st.dataframe(
            df_options.style.format({
                "Strike": "{:.2f}",
                "Price": "{:.2f}",
                "Base Cost": "{:.2f}",
                "Fees": "{:.2f}",
                "Contracts": "{:d}"  # Integer format for Contracts
            })
        )

        # Calculate and display P&L metrics
        st.subheader("Métricas de la Estrategia")
        net_cost = total_cost
        st.metric("Costo Neto (ARS)", f"{net_cost:.2f}")

        # Define strategy value function for visualization
        def strategy_value(price, T, sigma):
            total_value = 0.0
            use_intrinsic = T <= 1e-6
            for opt in selected_options:
                strike = opt["option"]["strike"]
                opt_type = opt["option"]["type"]
                action_mult = 1 if opt["action"] == "buy" else -1
                contracts = opt["contracts"]
                if use_intrinsic:
                    value = utils.intrinsic_value(price, strike, opt_type)
                else:
                    value = utils.black_scholes(price, strike, T, risk_free_rate, sigma, opt_type)
                total_value += value * action_mult * contracts * 100
            return total_value

        # Calibrate IV
        raw_net = sum(
            utils.get_strategy_price(opt["option"], opt["action"]) * opt["contracts"] * 100 * (1 if opt["action"] == "buy" else -1)
            for opt in selected_options
        )
        iv_calibrated = utils._calibrate_iv(raw_net, current_price, expiration_days, strategy_value, 
                                          [opt["option"] for opt in selected_options], 
                                          [opt["action"] for opt in selected_options])
        iv_calibrated = max(iv_calibrated, 1e-9)

        # Compute payoff grid
        min_price = current_price * (1 - plot_range_pct)
        max_price = current_price * (1 + plot_range_pct)
        price_points = np.linspace(min_price, max_price, 50)
        time_points = np.linspace(0, expiration_days / 365.0, 20)
        X, Y = np.meshgrid(price_points, time_points)
        Z = np.array([[strategy_value(p, t, iv_calibrated) - net_cost for p in price_points] for t in time_points])

        # Create 3D plot
        fig = go.Figure(data=[
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="P&L (ARS)"),
            )
        ])
        fig.update_layout(
            title=f"3D P&L para Estrategia Personalizada (IV: {iv_calibrated:.1%})",
            scene=dict(
                xaxis_title="Precio de GGAL (ARS)",
                yaxis_title="Tiempo hasta Vencimiento (Años)",
                zaxis_title="P&L (ARS)",
                xaxis=dict(tickvals=[min_price, current_price, max_price], ticktext=[f"{min_price:.2f}", f"{current_price:.2f}", f"{max_price:.2f}"]),
                yaxis=dict(tickvals=[0, expiration_days/365.0], ticktext=["Expiración", f"{expiration_days} días"]),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=600
        )
        fig.add_trace(go.Scatter3d(
            x=[current_price], y=[expiration_days/365.0], z=[0],
            mode="markers",
            marker=dict(size=5, color="red"),
            name="Precio Actual"
        ))

        st.plotly_chart(fig, use_container_width=True, key="scenario_3d_plot")
    elif not selected_options:
        st.warning("Por favor, seleccione al menos una opción con contratos mayores a 0.")
    else:
        st.error("No se pudo calcular el P&L debido a datos inválidos.")