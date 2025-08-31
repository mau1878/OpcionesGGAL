import streamlit as st
import numpy as np
import utils

st.set_page_config(page_title="Bullish Spreads", layout="wide")
st.title("Estrategias Alcistas: Spreads")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página 'Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts

tab1, tab2 = st.tabs(["Bull Call Spread", "Bull Put Spread"])

with tab1:
    st.header("Bull Call Spread (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    detailed_df_call = utils.create_bullish_spread_table(calls, utils.calculate_bull_call_spread, st.session_state.num_contracts, st.session_state.commission_rate, is_debit=True)
    if not detailed_df_call.empty:
        st.dataframe(detailed_df_call.style.format({
            "Net Cost": "{:.2f}", "Max Profit": "{:.2f}", "Max Loss": "{:.2f}", "Cost-to-Profit Ratio": "{:.2%}", "Breakeven": "{:.2f}"
        }))
    else:
        st.dataframe(detailed_df_call)
        st.warning("No hay datos disponibles para Bull Call Spread. Asegúrese de que hay suficientes opciones call.")
    st.subheader("Matriz de Costo Neto (Compra en Fila, Venta en Columna)")
    profit_df, _, _, _ = utils.create_spread_matrix(calls, utils.calculate_bull_call_spread, st.session_state.num_contracts, st.session_state.commission_rate, True)
    if not profit_df.empty:
        st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis_r'))
    else:
        st.warning("No hay datos disponibles para la matriz de costos.")
    st.subheader("Visualización 3D")
    if not detailed_df_call.empty:
        options = detailed_df_call.index
        selected = st.selectbox("Selecciona una combinación para visualizar", options, key="bcs_select")
        row = detailed_df_call.loc[selected]
        result = row.to_dict()
        result["strikes"] = list(selected) if isinstance(selected, tuple) else [selected]
        result["num_contracts"] = st.session_state.num_contracts
        result["raw_net"] = result["Net Cost"]  # Add for calibration
        result["net_cost"] = result["Net Cost"]
        if result:
            utils.visualize_bullish_3d(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Bull Call Spread")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")

with tab2:
    st.header("Bull Put Spread (Crédito)")
    st.subheader("Análisis Detallado por Ratio")
    detailed_df_put = utils.create_bullish_spread_table(puts, utils.calculate_bull_put_spread, st.session_state.num_contracts, st.session_state.commission_rate, is_debit=False)
    if not detailed_df_put.empty:
        st.dataframe(detailed_df_put.style.format({
            "Net Credit": "{:.2f}", "Max Profit": "{:.2f}", "Max Loss": "{:.2f}", "Cost-to-Profit Ratio": "{:.2%}", "Breakeven": "{:.2f}"
        }))
    else:
        st.dataframe(detailed_df_put)
        st.warning("No hay datos disponibles para Bull Put Spread. Asegúrese de que hay suficientes opciones put.")
    st.subheader("Matriz de Crédito Neto (Venta en Fila, Compra en Columna)")
    profit_df, _, _, _ = utils.create_spread_matrix(puts, utils.calculate_bull_put_spread, st.session_state.num_contracts, st.session_state.commission_rate, False)
    if not profit_df.empty:
        st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis'))
    else:
        st.warning("No hay datos disponibles para la matriz de créditos.")
    st.subheader("Visualización 3D")
    if not detailed_df_put.empty:
        options = detailed_df_put.index
        selected = st.selectbox("Selecciona una combinación para visualizar", options, key="bps_select")
        row = detailed_df_put.loc[selected]
        result = row.to_dict()
        result["strikes"] = list(selected) if isinstance(selected, tuple) else [selected]
        result["num_contracts"] = st.session_state.num_contracts
        result["raw_net"] = -result.get("Net Credit", 0)  # Credit is negative net_cost
        result["net_cost"] = -result.get("Net Credit", 0)
        if result:
            utils.visualize_bullish_3d(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Bull Put Spread")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")