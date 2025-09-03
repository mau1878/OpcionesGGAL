import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Optimization", layout="wide")
st.title("Optimización de Estrategias")

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
st.sidebar.header("Parámetros de Optimización")
st.session_state.plot_range_pct = st.sidebar.slider(
    "Rango de Precio para Gráficos (%)", 5.0, 50.0, st.session_state.get('plot_range_pct', 50.0)
) / 100
filter_unlimited_loss = st.sidebar.checkbox("Filtrar Estrategias con Pérdidas Ilimitadas", value=True)
max_legs = st.sidebar.number_input("Máximo número de patas", min_value=1, max_value=4, value=2, step=1)
objective = st.sidebar.selectbox("Objetivo de Optimización", ["Maximizar Probabilidad de Breakeven", "Minimizar Cost-to-Profit Ratio"])

# Initialize failure counter
if 'iv_failure_count' not in st.session_state:
    st.session_state.iv_failure_count = 0

# Strategy Optimization
results = []
min_price = current_price * (1 - st.session_state.plot_range_pct)
max_price = current_price * (1 + st.session_state.plot_range_pct)
for num_legs in range(1, max_legs + 1):
    for opt_combinations in combinations(calls + puts, num_legs):
        for actions in [[act for act in ['buy', 'sell']] for _ in range(num_legs)]:
            contracts = [num_contracts for _ in range(num_legs)]
            if filter_unlimited_loss and not utils.has_limited_loss(opt_combinations, actions, contracts):
                continue
            net_cost = utils.calculate_strategy_cost(opt_combinations, actions, contracts)
            if net_cost is None:
                continue
            T_years = expiration_days / 365.0
            iv_calibrated = utils._calibrate_iv(
                net_cost / (100 * sum(abs(c) for c in contracts)),
                current_price,
                expiration_days,
                lambda p, t, s: utils.calculate_strategy_value(opt_combinations, actions, contracts, p, t, s) / (100 * sum(abs(c) for c in contracts)),
                opt_combinations, actions, contracts
            )
            if iv_calibrated == utils.DEFAULT_IV:
                st.session_state.iv_failure_count += 1
                continue
            iv_calibrated = max(iv_calibrated, 1e-9)
            breakeven_prob = utils.estimate_breakeven_probability(opt_combinations, actions, contracts, net_cost, T_years, iv_calibrated)
            
            def find_breakeven(price_range, tolerance=1e-3):
                breakevens = []
                for p in np.linspace(price_range[0], price_range[1], 1000):
                    if abs(utils.calculate_strategy_value(opt_combinations, actions, contracts, p, 0, iv_calibrated) - net_cost) < tolerance:
                        breakevens.append(p)
                return breakevens
            
            breakevens = find_breakeven([min_price, max_price])
            profit_min_price = min(breakevens) if breakevens else min_price
            profit_max_price = max(breakevens) if breakevens else max_price
            potential_profit = utils.calculate_potential_metrics(opt_combinations, actions, contracts, profit_min_price, profit_max_price, net_cost, iv_calibrated)
            
            strategy_type = f"Strategy ({', '.join(f'{action} {opt['type']} {opt['strike']}' for opt, action in zip(opt_combinations, actions))})"
            results.append({
                "Strategy": strategy_type,
                "Net Cost": net_cost,
                "Potential Profit": potential_profit,
                "Cost-to-Profit Ratio": abs(net_cost / potential_profit) if potential_profit > 0 else float('inf'),
                "Breakeven Probability": breakeven_prob,
                "Breakevens": breakevens,
                "Options": list(opt_combinations),
                "Actions": actions,
                "Contracts": contracts,
                "IV": iv_calibrated,
                "Unlimited Loss": not utils.has_limited_loss(opt_combinations, actions, contracts)
            })

# Display IV calibration warning if failures occurred
if st.session_state.iv_failure_count > 0:
    st.warning(f"Se encontraron {st.session_state.iv_failure_count} estrategias con fallos en la calibración de volatilidad implícita. Estas fueron excluidas.")

# Display Results
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by="Breakeven Probability" if objective == "Maximizar Probabilidad de Breakeven" else "Cost-to-Profit Ratio", ascending=objective != "Maximizar Probabilidad de Breakeven")
    edited_df = df[["Strategy", "Net Cost", "Potential Profit", "Cost-to-Profit Ratio", "Breakeven Probability", "Unlimited Loss"]].copy()
    edited_df["Visualize"] = False
    for col in ["Net Cost", "Potential Profit", "Cost-to-Profit Ratio", "Breakeven Probability"]:
        edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}" if col != "Breakeven Probability" else f"{x:.1%}")
    edited_df["Unlimited Loss"] = edited_df["Unlimited Loss"].apply(lambda x: "Yes" if x else "No")
    
    def visualize_callback():
        edited = st.session_state.get("optimization_editor", {})
        edited_rows = edited.get('edited_rows', {})
        for idx in edited_rows:
            if isinstance(idx, int) and 0 <= idx < len(df):
                if edited_rows[idx].get('Visualize', False):
                    row = df.iloc[idx]
                    options = row["Options"]
                    actions = row["Actions"]
                    contracts = row["Contracts"]
                    net_cost = row["Net Cost"]
                    iv_calibrated = row["IV"]
                    breakevens = row["Breakevens"]
                    strategy_type = row["Strategy"]
                    min_price = current_price * (1 - st.session_state.plot_range_pct)
                    max_price = current_price * (1 + st.session_state.plot_range_pct)
                    
                    # 3D Plot
                    X, Y, Z = utils._compute_payoff_grid(
                        lambda p, t, s: utils.calculate_strategy_value(options, actions, contracts, p, t, s),
                        current_price, expiration_days, iv_calibrated, net_cost
                    )
                    fig = utils._create_3d_figure(X, Y, Z, strategy_type, current_price)
                    st.plotly_chart(fig, use_container_width=True, key=f"3d_plot_{idx}")
                    
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
                    for b in breakevens[:2]:
                        fig_2d.add_vline(x=b, line_dash="dash", line_color="green", annotation_text=f"Breakeven: {b:.2f}")
                    if breakevens:
                        fig_2d.add_vrect(
                            x0=min(breakevens), x1=max(breakevens),
                            fillcolor="green", opacity=0.1, line_width=0,
                            annotation_text="Rango de Ganancia", annotation_position="top"
                        )
                    st.plotly_chart(fig_2d, use_container_width=True, key=f"2d_plot_{idx}")
    
    edited_df = st.data_editor(
        edited_df,
        use_container_width=True,
        column_config={"Visualize": st.column_config.CheckboxColumn("Visualize", default=False)},
        disabled=["Strategy", "Net Cost", "Potential Profit", "Cost-to-Profit Ratio", "Breakeven Probability", "Unlimited Loss"],
        key="optimization_editor",
        on_change=visualize_callback
    )
else:
    st.warning("No se encontraron estrategias válidas con los filtros actuales.")