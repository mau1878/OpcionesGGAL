import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Neutral Strategies", layout="wide")
st.title("Estrategias Neutrales")

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
st.sidebar.header("Parámetros de Análisis")
strategy_type = st.sidebar.selectbox("Tipo de Estrategia", ["Call Butterfly", "Put Butterfly", "Call Condor", "Put Condor"])
default_ratios = {"Call Butterfly": "1,-2,1", "Put Butterfly": "1,-2,1", "Call Condor": "1,-1,-1,1", "Put Condor": "1,-1,-1,1"}
contract_ratios_input = st.sidebar.text_input(f"Ratios de Contratos para {strategy_type}", default_ratios[strategy_type])
try:
    contract_ratios = [int(x) for x in contract_ratios_input.split(",")]
    if strategy_type in ["Call Butterfly", "Put Butterfly"] and len(contract_ratios) != 3:
        raise ValueError("Butterfly requiere 3 ratios (e.g., 1,-2,1)")
    if strategy_type in ["Call Condor", "Put Condor"] and len(contract_ratios) != 4:
        raise ValueError("Condor requiere 4 ratios (e.g., 1,-1,-1,1)")
except ValueError as e:
    st.error(f"Error en ratios: {e}")
    contract_ratios = [int(x) for x in default_ratios[strategy_type].split(",")]
st.session_state.plot_range_pct = st.sidebar.slider(
    "Rango de Precio para Gráficos (% del precio actual)", 10.0, 200.0, st.session_state.get('plot_range_pct', 50.0)
) / 100
sort_by = st.sidebar.selectbox("Ordenar Tabla por", ["Cost-to-Profit Ratio", "Breakeven Probability"], key="sort_by")

# Initialize visualization state
if 'selected_visualizations_neutral' not in st.session_state:
    st.session_state.selected_visualizations_neutral = []

# Strategy Calculations
results = []
options_list = calls if "Call" in strategy_type else puts
sorted_strikes = sorted(set(opt["strike"] for opt in options_list))
num_options = 3 if "Butterfly" in strategy_type else 4
for strikes in combinations(sorted_strikes, num_options):
    if "Butterfly" in strategy_type and strikes[2] - strikes[1] != strikes[1] - strikes[0]:
        continue  # Ensure equal spacing for Butterfly
    if "Condor" in strategy_type and (strikes[3] - strikes[2] != strikes[1] - strikes[0] or strikes[2] - strikes[1] < 0):
        continue  # Ensure valid Condor structure
    options = [next((opt for opt in options_list if opt["strike"] == k), None) for k in strikes]
    if None in options:
        continue
    actions = ["buy", "sell", "buy"] if "Butterfly" in strategy_type else ["buy", "sell", "sell", "buy"]
    contracts = [num_contracts * r for r in contract_ratios]
    net_cost = utils.calculate_strategy_cost(options, actions, contracts)
    if net_cost is None:
        continue
    T_years = expiration_days / 365.0
    iv_calibrated = utils._calibrate_iv(
        net_cost / (100 * sum(abs(r) for r in contract_ratios)),
        current_price,
        expiration_days,
        lambda p, t, s: utils.calculate_strategy_value(options, actions, contracts, p, t, s) / (100 * sum(abs(r) for r in contract_ratios)),
        options, actions, contract_ratios
    )
    if iv_calibrated == utils.DEFAULT_IV:
        st.warning(f"No se pudo calibrar la volatilidad implícita para la estrategia con strikes {strikes}. Usando valor predeterminado.")
    iv_calibrated = max(iv_calibrated, 1e-9)
    breakeven_prob = utils.estimate_breakeven_probability(options, actions, contracts, net_cost, T_years, iv_calibrated)
    
    def find_breakeven(price_range, tolerance=1e-3):
        breakevens = []
        for p in np.linspace(price_range[0], price_range[1], 500):  # Reduced for performance
            if abs(utils.calculate_strategy_value(options, actions, contracts, p, 0, iv_calibrated) - net_cost) < tolerance:
                breakevens.append(p)
        return breakevens
    
    min_price = current_price * (1 - st.session_state.plot_range_pct)
    max_price = current_price * (1 + st.session_state.plot_range_pct)
    breakevens = find_breakeven([min_price, max_price])
    profit_min_price = min(breakevens) if breakevens else min_price
    profit_max_price = max(breakevens) if breakevens else max_price
    potential_profit = utils.calculate_potential_metrics(options, actions, contracts, profit_min_price, profit_max_price, net_cost, iv_calibrated)
    
    strikes_str = "-".join([f"{k:.2f}" for k in strikes])
    results.append({
        "Strikes": strikes_str,
        "Net Cost": net_cost,
        "Potential Profit": potential_profit,
        "Cost-to-Profit Ratio": abs(net_cost / potential_profit) if potential_profit > 0 else float('inf'),
        "Breakeven Probability": breakeven_prob,
        "Breakevens": breakevens,
        "Options": options,
        "Actions": actions,
        "Contracts": contracts,
        "IV": iv_calibrated
    })

# Display Results
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by=sort_by, ascending=True)
    edited_df = df[["Strikes", "Net Cost", "Potential Profit", "Cost-to-Profit Ratio", "Breakeven Probability"]].copy()
    edited_df["Visualize"] = False
    for col in ["Net Cost", "Potential Profit", "Cost-to-Profit Ratio", "Breakeven Probability"]:
        edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}" if col != "Breakeven Probability" else f"{x:.1%}")
    
    def visualize_callback():
        edited = st.session_state.get("neutral_strategies_editor", {})
        edited_rows = edited.get('edited_rows', {})
        for idx in edited_rows:
            if isinstance(idx, int) and 0 <= idx < len(df):
                visualize_state = edited_rows[idx].get('Visualize', False)
                if visualize_state and idx not in st.session_state["selected_visualizations_neutral"] and len(st.session_state["selected_visualizations_neutral"]) < 5:
                    st.session_state["selected_visualizations_neutral"].append(idx)
                elif not visualize_state and idx in st.session_state["selected_visualizations_neutral"]:
                    st.session_state["selected_visualizations_neutral"].remove(idx)
        
        for idx in st.session_state["selected_visualizations_neutral"]:
            row = df.iloc[idx]
            options = row["Options"]
            actions = row["Actions"]
            contracts = row["Contracts"]
            net_cost = row["Net Cost"]
            iv_calibrated = row["IV"]
            breakevens = row["Breakevens"]
            min_price = current_price * (1 - st.session_state.plot_range_pct)
            max_price = current_price * (1 + st.session_state.plot_range_pct)
            
            # 3D Plot
            X, Y, Z, min_price_3d, max_price_3d, times = utils._compute_payoff_grid(
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
        disabled=["Strikes", "Net Cost", "Potential Profit", "Cost-to-Profit Ratio", "Breakeven Probability"],
        key="neutral_strategies_editor",
        on_change=visualize_callback
    )
else:
    st.warning(f"No hay combinaciones válidas para {strategy_type} con los filtros actuales.")