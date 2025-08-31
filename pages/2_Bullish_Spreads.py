# REPLACE the entire contents of this file

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
    detailed_df_call = utils.create_spread_table(calls, utils.calculate_bull_call_spread,
                                                 st.session_state.num_contracts, st.session_state.commission_rate, True)
    st.dataframe(detailed_df_call.style.format({
        "Net Cost": "{:.2f}", "Max Profit": "{:.2f}", "Max Loss": "{:.2f}", "Cost-to-Profit Ratio": "{:.2%}"
    }))

    st.subheader("Matriz de Costo Neto (Compra en Fila, Venta en Columna)")
    profit_df, _, _, _ = utils.create_spread_matrix(calls, utils.calculate_bull_call_spread,
                                                    st.session_state.num_contracts, st.session_state.commission_rate,
                                                    True)
    st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis_r'))

    st.subheader("Visualización 3D")
    strike_prices = sorted([opt['strike'] for opt in calls])
    col1, col2 = st.columns(2)
    long_strike = col1.selectbox("Strike de Compra (Long)", strike_prices, key="bcs_long")
    short_strike = col2.selectbox("Strike de Venta (Short)", strike_prices, key="bcs_short")

    if long_strike and short_strike and long_strike < short_strike:
        long_opt = next((opt for opt in calls if opt['strike'] == long_strike), None)
        short_opt = next((opt for opt in calls if opt['strike'] == short_strike), None)
        if long_opt and short_opt:
            result = utils.calculate_bull_call_spread(long_opt, short_opt, st.session_state.num_contracts,
                                                      st.session_state.commission_rate)
            if result:
                utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days,
                                          st.session_state.iv, key="Bull Call Spread")

with tab2:
    st.header("Bull Put Spread (Crédito)")

    st.subheader("Análisis Detallado por Ratio")
    detailed_df_put = utils.create_spread_table(puts, utils.calculate_bull_put_spread, st.session_state.num_contracts,
                                                st.session_state.commission_rate, False)
    st.dataframe(detailed_df_put.style.format({
        "Net Credit": "{:.2f}", "Max Profit": "{:.2f}", "Max Loss": "{:.2f}", "Cost-to-Profit Ratio": "{:.2%}"
    }))

    st.subheader("Matriz de Crédito Neto (Venta en Fila, Compra en Columna)")
    profit_df, _, _, _ = utils.create_spread_matrix(puts, utils.calculate_bull_put_spread,
                                                    st.session_state.num_contracts, st.session_state.commission_rate,
                                                    False)
    st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis'))

    st.subheader("Visualización 3D")
    strike_prices = sorted([opt['strike'] for opt in puts])
    col1, col2 = st.columns(2)
    long_strike = col1.selectbox("Strike de Compra (Long)", strike_prices, key="bps_long")
    short_strike = col2.selectbox("Strike de Venta (Short)", strike_prices, key="bps_short")

    if long_strike and short_strike and long_strike < short_strike:
        long_opt = next((opt for opt in puts if opt['strike'] == long_strike), None)
        short_opt = next((opt for opt in puts if opt['strike'] == short_strike), None)
        if long_opt and short_opt:
            result = utils.calculate_bull_put_spread(long_opt, short_opt, st.session_state.num_contracts,
                                                     st.session_state.commission_rate)
            if result:
                utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days,
                                          st.session_state.iv, key="Bull Put Spread")