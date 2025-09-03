import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import utils
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Bullish Spreads", layout="wide")
st.title("Estrategias Bullish Spreads")

# Check if required session state variables are available
if not all(key in st.session_state for key in ['filtered_calls', 'filtered_puts', 'current_price', 'expiration_days', 'risk_free_rate', 'commission_rate', 'plot_range_pct']):
    st.error("Datos de opciones no disponibles. Por favor, configure los parámetros en la página de Opciones GGAL.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
current_price = st.session_state.current_price
num_contracts = st.session_state.num_contracts
commission_rate = st.session_state.commission_rate

if not calls or not puts:
    st.warning("No hay datos de Calls o Puts disponibles para la fecha de vencimiento seleccionada.")
    st.stop()

# Initialize session state for visualization flags
# Initialize session state for visualization flags
if 'visualize_flags_call' not in st.session_state:
    st.session_state.visualize_flags_call = [False] * (len(calls) * (len(calls) - 1) // 2 if calls else 0)
if 'visualize_flags_put' not in st.session_state:
    st.session_state.visualize_flags_put = [False] * (len(puts) * (len(puts) - 1) // 2 if puts else 0)
if 'selected_visualizations_call' not in st.session_state:
    st.session_state.selected_visualizations_call = []
if 'selected_visualizations_put' not in st.session_state:
    st.session_state.selected_visualizations_put = []

# Sidebar for Inputs
# Sidebar for Inputs
st.sidebar.header("Parámetros de Análisis")
if 'plot_range_pct' not in st.session_state:
    st.session_state.plot_range_pct = 0.5  # Default to 50% in decimal
st.session_state.plot_range_pct = st.sidebar.slider(
    "Rango de Precio para Gráficos (% del precio actual)", 10.0, 200.0, st.session_state.get('plot_range_pct', 0.5) * 100
) / 100
st.session_state.include_breakeven_prob = st.sidebar.checkbox("Incluir Probabilidad de Breakeven", value=True)
show_2d_plot = st.sidebar.checkbox("Mostrar Diagrama 2D", value=True)
st.sidebar.button("Limpiar Visualizaciones", on_click=lambda: st.session_state.update({
    "selected_visualizations_call": [], "selected_visualizations_put": []
}))

# Bull Call Spread
st.header("Bull Call Spread (Debit Spread)")
min_strike = current_price * (1 - st.session_state.plot_range_pct)
max_strike = current_price * (1 + st.session_state.plot_range_pct)
filtered_calls = [opt for opt in calls if min_strike <= opt["strike"] <= max_strike]
detailed_df_call = utils.create_bullish_spread_table(filtered_calls, utils.calculate_bull_call_spread, num_contracts, commission_rate, is_debit=True)

def visualize_callback_call():
    logger.info("Visualize callback triggered for Bull Call Spread")
    edited = st.session_state.get("bull_call_spread_editor_unique", {})
    edited_rows = edited.get('edited_rows', {})
    edited_df_local = st.session_state["bull_call_df"]
    for idx in edited_rows:
        if isinstance(idx, int) and 0 <= idx < len(edited_df_local):
            visualize_state = edited_rows[idx].get('Visualize', False)
            if visualize_state and idx not in st.session_state["selected_visualizations_call"] and len(st.session_state["selected_visualizations_call"]) < 3:
                st.session_state["selected_visualizations_call"].append(idx)
            elif not visualize_state and idx in st.session_state["selected_visualizations_call"]:
                st.session_state["selected_visualizations_call"].remove(idx)
    
    for idx in st.session_state["selected_visualizations_call"]:
        row = edited_df_local.iloc[idx]
        result = {
            "net_cost": float(row["Net Cost"].replace(",", ".")),
            "max_profit": float(row["Max Profit"].replace(",", ".")),
            "max_loss": float(row["Max Loss"].replace(",", ".")),
            "breakeven": float(row["Breakeven"].replace(",", ".")),
            "strikes": [float(s) for s in row['Strikes'].split('-')],
            "num_contracts": num_contracts,
            "raw_net": float(row["Net Cost"].replace(",", "."))
        }
        long_opt = next((opt for opt in filtered_calls if opt["strike"] == result["strikes"][0]), None)
        short_opt = next((opt for opt in filtered_calls if opt["strike"] == result["strikes"][1]), None)
        if long_opt and short_opt:
            options = [long_opt, short_opt]
            actions = ["buy", "sell"]
            contracts = [num_contracts, num_contracts]
            net_cost = result["net_cost"]
            iv_calibrated = utils._calibrate_iv(
                result["raw_net"] / (100 * num_contracts),
                current_price,
                st.session_state.expiration_days,
                lambda p, t, s: (utils.black_scholes(p, long_opt["strike"], t, st.session_state.risk_free_rate, s, "call") -
                                 utils.black_scholes(p, short_opt["strike"], t, st.session_state.risk_free_rate, s, "call")),
                options, actions
            )
            if iv_calibrated == utils.DEFAULT_IV:
                st.warning(f"No se pudo calibrar la volatilidad implícita para {long_opt['symbol']}/{short_opt['symbol']}. Usando valor predeterminado.")
            iv_calibrated = max(iv_calibrated, 1e-9)

            # 3D Plot
            utils.visualize_bullish_3d(
                result, current_price, st.session_state.expiration_days,
                iv_calibrated, "Bull Call Spread", options, actions
            )

            # 2D Plot
            if show_2d_plot:
                st.subheader("Diagrama de P&L al Vencimiento")
                min_price = current_price * (1 - st.session_state.plot_range_pct)
                max_price = current_price * (1 + st.session_state.plot_range_pct)
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
                    title="P&L at Expiration for Bull Call Spread",
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
                    for p in np.linspace(min_price, max_price, 1000):
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
                st.plotly_chart(fig_2d, use_container_width=True, key=f"bull_call_2d_plot_{idx}_{id(options)}")
        else:
            st.warning("Datos de opciones no disponibles para esta combinación.")
    st.rerun()

if not detailed_df_call.empty:
    st.session_state["bull_call_df"] = detailed_df_call
    edited_df = detailed_df_call.copy()
    for col in ["Net Cost", "Max Profit", "Max Loss", "Breakeven", "Breakeven Probability"]:
        if col in edited_df.columns:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}" if col != "Breakeven Probability" else f"{x:.1%}" if x is not None else "-")
    edited_df["Visualize"] = False
    edited_df = st.data_editor(
        edited_df,
        use_container_width=True,
        column_config={"Visualize": st.column_config.CheckboxColumn("Visualize", default=False)},
        disabled=["Long Strike", "Short Strike", "Net Cost", "Max Profit", "Max Loss", "Breakeven", "Breakeven Probability", "Cost-to-Profit Ratio"],
        key="bull_call_spread_editor_unique",
        on_change=visualize_callback_call
    )
else:
    st.warning("No hay combinaciones válidas para Bull Call Spread con los filtros actuales.")

# Bull Put Spread
st.header("Bull Put Spread (Credit Spread)")
filtered_puts = [opt for opt in puts if min_strike <= opt["strike"] <= max_strike]
detailed_df_put = utils.create_bullish_spread_table(filtered_puts, utils.calculate_bull_put_spread, num_contracts, commission_rate, is_debit=False)

def visualize_callback_put():
    logger.info("Visualize callback triggered for Bull Put Spread")
    edited = st.session_state.get("bull_put_spread_editor_unique", {})
    edited_rows = edited.get('edited_rows', {})
    edited_df_local = st.session_state["bull_put_df"]
    for idx in edited_rows:
        if isinstance(idx, int) and 0 <= idx < len(edited_df_local):
            visualize_state = edited_rows[idx].get('Visualize', False)
            if visualize_state and idx not in st.session_state["selected_visualizations_put"] and len(st.session_state["selected_visualizations_put"]) < 3:
                st.session_state["selected_visualizations_put"].append(idx)
            elif not visualize_state and idx in st.session_state["selected_visualizations_put"]:
                st.session_state["selected_visualizations_put"].remove(idx)
    
    for idx in st.session_state["selected_visualizations_put"]:
        row = edited_df_local.iloc[idx]
        result = {
            "net_credit": float(row["Net Credit"].replace(",", ".")),
            "max_profit": float(row["Max Profit"].replace(",", ".")),
            "max_loss": float(row["Max Loss"].replace(",", ".")),
            "breakeven": float(row["Breakeven"].replace(",", ".")),
            "strikes": [float(s) for s in row['Strikes'].split('-')],
            "num_contracts": num_contracts,
            "raw_net": float(row["Net Credit"].replace(",", "."))
        }
        long_opt = next((opt for opt in filtered_puts if opt["strike"] == result["strikes"][0]), None)
        short_opt = next((opt for opt in filtered_puts if opt["strike"] == result["strikes"][1]), None)
        if long_opt and short_opt:
            options = [long_opt, short_opt]
            actions = ["buy", "sell"]
            contracts = [num_contracts, num_contracts]
            net_credit = result["net_credit"]
            iv_calibrated = utils._calibrate_iv(
                result["raw_net"] / (100 * num_contracts),
                current_price,
                st.session_state.expiration_days,
                lambda p, t, s: (utils.black_scholes(p, short_opt["strike"], t, st.session_state.risk_free_rate, s, "put") -
                                 utils.black_scholes(p, long_opt["strike"], t, st.session_state.risk_free_rate, s, "put")),
                options, actions
            )
            if iv_calibrated == utils.DEFAULT_IV:
                st.warning(f"No se pudo calibrar la volatilidad implícita para {long_opt['symbol']}/{short_opt['symbol']}. Usando valor predeterminado.")
            iv_calibrated = max(iv_calibrated, 1e-9)

            # 3D Plot
            utils.visualize_bullish_3d(
                result, current_price, st.session_state.expiration_days,
                iv_calibrated, "Bull Put Spread", options, actions
            )

            # 2D Plot
            if show_2d_plot:
                st.subheader("Diagrama de P&L al Vencimiento")
                min_price = current_price * (1 - st.session_state.plot_range_pct)
                max_price = current_price * (1 + st.session_state.plot_range_pct)
                price_points = np.linspace(min_price, max_price, 100)
                payoff_at_expiration = [utils.calculate_strategy_value(options, actions, contracts, p, 0, iv_calibrated) + net_credit for p in price_points]
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
                        y=[utils.calculate_strategy_value(options, actions, contracts, current_price, 0, iv_calibrated) + net_credit],
                        mode="markers", name="Precio Actual", marker=dict(size=10, color="red")
                    )
                )
                fig_2d.update_layout(
                    title="P&L at Expiration for Bull Put Spread",
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
                    for p in np.linspace(min_price, max_price, 1000):
                        if abs(utils.calculate_strategy_value(options, actions, contracts, p, 0, iv_calibrated) + net_credit) < tolerance:
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
                st.plotly_chart(fig_2d, use_container_width=True, key=f"bull_put_2d_plot_{idx}_{id(options)}")
        else:
            st.warning("Datos de opciones no disponibles para esta combinación.")
    st.rerun()

if not detailed_df_put.empty:
    st.session_state["bull_put_df"] = detailed_df_put
    edited_df = detailed_df_put.copy()
    for col in ["Net Credit", "Max Profit", "Max Loss", "Breakeven", "Breakeven Probability"]:
        if col in edited_df.columns:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}" if col != "Breakeven Probability" else f"{x:.1%}" if x is not None else "-")
    edited_df["Visualize"] = False
    edited_df = st.data_editor(
        edited_df,
        use_container_width=True,
        column_config={"Visualize": st.column_config.CheckboxColumn("Visualize", default=False)},
        disabled=["Long Strike", "Short Strike", "Net Credit", "Max Profit", "Max Loss", "Breakeven", "Breakeven Probability", "Cost-to-Profit Ratio"],
        key="bull_put_spread_editor_unique",
        on_change=visualize_callback_put
    )
else:
    st.warning("No hay combinaciones válidas para Bull Put Spread con los filtros actuales.")