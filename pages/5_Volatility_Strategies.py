import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils
import logging
from itertools import product

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Volatility Strategies", layout="wide")
st.title("Estrategias de Volatilidad")

if not all(key in st.session_state for key in ['filtered_calls', 'filtered_puts', 'current_price', 'expiration_days', 'risk_free_rate', 'commission_rate', 'plot_range_pct']):
    st.error("Datos de opciones no disponibles. Por favor, configure los parámetros en la página de Opciones GGAL.")
    st.stop()

calls = sorted(st.session_state.filtered_calls, key=lambda x: x['strike'])
puts = sorted(st.session_state.filtered_puts, key=lambda x: x['strike'])
current_price = st.session_state.current_price
num_contracts = st.session_state.num_contracts
commission_rate = st.session_state.commission_rate
expiration_days = st.session_state.expiration_days

if not calls or not puts:
    st.warning("No hay datos de Calls o Puts disponibles para la fecha de vencimiento seleccionada.")
    st.stop()

st.sidebar.header("Parámetros de Análisis")
strategy_type = st.sidebar.selectbox("Tipo de Estrategia", ["Straddle", "Strangle"])
st.session_state.plot_range_pct = st.sidebar.slider(
    "Rango de Precio para Gráficos (% del precio actual)", 10.0, 200.0, st.session_state.get('plot_range_pct', 50.0)
) / 100
sort_by = st.sidebar.selectbox("Ordenar Tabla por", ["Cost-to-Profit Ratio", "Breakeven Probability"], key="sort_by")

if 'selected_visualizations_vol' not in st.session_state:
    st.session_state.selected_visualizations_vol = []

results = []
with st.spinner("Calculando estrategias..."):
    if strategy_type == "Straddle":
        common_strikes = sorted(set(c['strike'] for c in calls) & set(p['strike'] for p in puts))
        for strike in common_strikes:
            call = next(c for c in calls if c['strike'] == strike)
            put = next(p for p in puts if p['strike'] == strike)
            options = [call, put]
            actions = ["buy", "buy"]
            contracts = [num_contracts, num_contracts]
            net_cost = utils.calculate_strategy_cost(options, actions, contracts)
            if net_cost is None:
                continue
            T_years = expiration_days / 365.0
            iv_calibrated = utils._calibrate_iv(
                net_cost / (100 * num_contracts * 2),
                current_price,
                expiration_days,
                lambda p, t, s: utils.calculate_strategy_value(options, actions, contracts, p, t, s) / (100 * num_contracts * 2),
                options, actions, [1, 1]
            )
            if iv_calibrated == utils.DEFAULT_IV:
                st.warning(f"No se pudo calibrar la volatilidad implícita para la estrategia con strike {call['strike']}. Usando valor predeterminado.")
            iv_calibrated = max(iv_calibrated, 1e-9)
            breakeven_prob = utils.estimate_breakeven_probability(options, actions, contracts, net_cost, T_years, iv_calibrated)
            
            def find_breakeven(price_range, tolerance=1e-3):
                breakevens = []
                for p in np.linspace(price_range[0], price_range[1], 1000):
                    if abs(utils.calculate_strategy_value(options, actions, contracts, p, 0, iv_calibrated) - net_cost) < tolerance:
                        breakevens.append(p)
                return breakevens
            
            min_price = current_price * (1 - st.session_state.plot_range_pct)
            max_price = current_price * (1 + st.session_state.plot_range_pct)
            breakevens = find_breakeven([min_price, max_price])
            profit_min_price = min(breakevens) if breakevens else min_price
            profit_max_price = max(breakevens) if breakevens else max_price
            potential_profit = utils.calculate_potential_metrics(options, actions, contracts, profit_min_price, profit_max_price, net_cost, iv_calibrated)
            
            results.append({
                "Strike": call["strike"],
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
    elif strategy_type == "Strangle":
        for put, call in product(puts, calls):
            if call['strike'] <= put['strike']:
                continue
            options = [call, put]
            actions = ["buy", "buy"]
            contracts = [num_contracts, num_contracts]
            net_cost = utils.calculate_strategy_cost(options, actions, contracts)
            if net_cost is None:
                continue
            T_years = expiration_days / 365.0
            iv_calibrated = utils._calibrate_iv(
                net_cost / (100 * num_contracts * 2),
                current_price,
                expiration_days,
                lambda p, t, s: utils.calculate_strategy_value(options, actions, contracts, p, t, s) / (100 * num_contracts * 2),
                options, actions, [1, 1]
            )
            if iv_calibrated == utils.DEFAULT_IV:
                st.warning(f"No se pudo calibrar la volatilidad implícita para la estrategia con strikes {call['strike']}-{put['strike']}. Usando valor predeterminado.")
            iv_calibrated = max(iv_calibrated, 1e-9)
            breakeven_prob = utils.estimate_breakeven_probability(options, actions, contracts, net_cost, T_years, iv_calibrated)
            
            def find_breakeven(price_range, tolerance=1e-3):
                breakevens = []
                for p in np.linspace(price_range[0], price_range[1], 1000):
                    if abs(utils.calculate_strategy_value(options, actions, contracts, p, 0, iv_calibrated) - net_cost) < tolerance:
                        breakevens.append(p)
                return breakevens
            
            min_price = current_price * (1 - st.session_state.plot_range_pct)
            max_price = current_price * (1 + st.session_state.plot_range_pct)
            breakevens = find_breakeven([min_price, max_price])
            profit_min_price = min(breakevens) if breakevens else min_price
            profit_max_price = max(breakevens) if breakevens else max_price
            potential_profit = utils.calculate_potential_metrics(options, actions, contracts, profit_min_price, profit_max_price, net_cost, iv_calibrated)
            
            results.append({
                "Strikes": f"{call['strike']}-{put['strike']}",
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

if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by=sort_by, ascending=True)
    edited_df = df[["Strike" if strategy_type == "Straddle" else "Strikes", "Net Cost", "Potential Profit", "Cost-to-Profit Ratio", "Breakeven Probability"]].copy()
    edited_df["Visualize"] = False
    for col in ["Net Cost", "Potential Profit", "Cost-to-Profit Ratio", "Breakeven Probability"]:
        edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}" if col != "Breakeven Probability" else f"{x:.1%}")
    
    def visualize_callback():
        edited = st.session_state.get("volatility_strategies_editor", {})
        edited_rows = edited.get('edited_rows', {})
        for idx in edited_rows:
            if isinstance(idx, int) and 0 <= idx < len(df):
                visualize_state = edited_rows[idx].get('Visualize', False)
                if visualize_state and idx not in st.session_state["selected_visualizations_vol"] and len(st.session_state["selected_visualizations_vol"]) < 5:
                    st.session_state["selected_visualizations_vol"].append(idx)
                elif not visualize_state and idx in st.session_state["selected_visualizations_vol"]:
                    st.session_state["selected_visualizations_vol"].remove(idx)
        
        for idx in st.session_state["selected_visualizations_vol"]:
            row = df.iloc[idx]
            options = row["Options"]
            actions = row["Actions"]
            contracts = row["Contracts"]
            net_cost = row["Net Cost"]
            iv_calibrated = row["IV"]
            breakevens = row["Breakevens"]
            min_price = current_price * (1 - st.session_state.plot_range_pct)
            max_price = current_price * (1 + st.session_state.plot_range_pct)
            
            utils.visualize_volatility_3d(
                {"raw_net": net_cost / (100 * num_contracts * 2), "net_cost": net_cost, "num_contracts": num_contracts, "strikes": [o["strike"] for o in options]},
                current_price, expiration_days, iv_calibrated, strategy_type, options, actions
            )
            
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
        disabled=["Strike" if strategy_type == "Straddle" else "Strikes", "Net Cost", "Potential Profit", "Cost-to-Profit Ratio", "Breakeven Probability"],
        key="volatility_strategies_editor",
        on_change=visualize_callback
    )
else:
    st.warning(f"No hay combinaciones válidas para {strategy_type} con los filtros actuales.")