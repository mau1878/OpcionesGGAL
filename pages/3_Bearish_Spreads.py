import streamlit as st
import numpy as np
import utils

st.set_page_config(page_title="Bearish Spreads", layout="wide")
st.title("Estrategias Bajistas: Spreads")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts

tab1, tab2 = st.tabs(["Bear Call Spread", "Bear Put Spread"])

with tab1:
    st.subheader("Bear Call Spread (Crédito)")
    profit_df, _, _, _ = utils.create_spread_matrix(calls, utils.calculate_bear_call_spread, st.session_state.num_contracts, st.session_state.commission_rate, False)
    st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn'))

    st.subheader("Visualización 3D")
    strike_prices = sorted([opt['strike'] for opt in calls])
    col1, col2 = st.columns(2)
    long_strike = col1.selectbox("Strike de Compra (Long)", strike_prices, key="becs_long")
    short_strike = col2.selectbox("Strike de Venta (Short)", strike_prices, key="becs_short")

    if long_strike and short_strike:
        long_opt = next((opt for opt in calls if opt['strike'] == long_strike), None)
        short_opt = next((opt for opt in calls if opt['strike'] == short_strike), None)
        if long_opt and short_opt:
            result = utils.calculate_bear_call_spread(long_opt, short_opt, st.session_state.num_contracts, st.session_state.commission_rate)
            if result:
                utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Bear Call Spread")
            else:
                st.warning("Combinación de strikes no válida.")

with tab2:
    st.subheader("Bear Put Spread (Débito)")
    profit_df, _, _, _ = utils.create_spread_matrix(puts, utils.calculate_bear_put_spread, st.session_state.num_contracts, st.session_state.commission_rate, True)
    st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn'))

    st.subheader("Visualización 3D")
    strike_prices = sorted([opt['strike'] for opt in puts])
    col1, col2 = st.columns(2)
    long_strike = col1.selectbox("Strike de Compra (Long)", strike_prices, key="beps_long")
    short_strike = col2.selectbox("Strike de Venta (Short)", strike_prices, key="beps_short")

    if long_strike and short_strike:
        long_opt = next((opt for opt in puts if opt['strike'] == long_strike), None)
        short_opt = next((opt for opt in puts if opt['strike'] == short_strike), None)
        if long_opt and short_opt:
            result = utils.calculate_bear_put_spread(long_opt, short_opt, st.session_state.num_contracts, st.session_state.commission_rate)
            if result:
                utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Bear Put Spread")
            else:
                st.warning("Combinación de strikes no válida.")