import streamlit as st
import numpy as np
import utils

st.set_page_config(page_title="Bullish Spreads", layout="wide")
st.title("Estrategias Alcistas: Spreads")

# Check if data is loaded
if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts

tab1, tab2 = st.tabs(["Bull Call Spread", "Bull Put Spread"])

with tab1:
    st.subheader("Bull Call Spread (Débito)")
    profit_df, _, _, _ = utils.create_spread_matrix(calls, utils.calculate_bull_call_spread,
                                                    st.session_state.num_contracts, st.session_state.commission_rate,
                                                    True)
    st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn'))

    st.subheader("Visualización 3D")
    strike_prices = sorted([opt['strike'] for opt in calls])
    col1, col2 = st.columns(2)
    long_strike = col1.selectbox("Strike de Compra (Long)", strike_prices, key="bcs_long")
    short_strike = col2.selectbox("Strike de Venta (Short)", strike_prices, key="bcs_short")

    if long_strike and short_strike:
        long_opt = next((opt for opt in calls if opt['strike'] == long_strike), None)
        short_opt = next((opt for opt in calls if opt['strike'] == short_strike), None)
        if long_opt and short_opt:
            result = utils.calculate_bull_call_spread(long_opt, short_opt, st.session_state.num_contracts,
                                                      st.session_state.commission_rate)
            if result:
                utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days,
                                          st.session_state.iv, key="Bull Call Spread")
            else:
                st.warning("Combinación de strikes no válida.")

with tab2:
    st.subheader("Bull Put Spread (Crédito)")
    profit_df, _, _, _ = utils.create_spread_matrix(puts, utils.calculate_bull_put_spread,
                                                    st.session_state.num_contracts, st.session_state.commission_rate,
                                                    False)
    st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn'))

    st.subheader("Visualización 3D")
    strike_prices = sorted([opt['strike'] for opt in puts])
    col1, col2 = st.columns(2)
    long_strike = col1.selectbox("Strike de Compra (Long)", strike_prices, key="bps_long")
    short_strike = col2.selectbox("Strike de Venta (Short)", strike_prices, key="bps_short")

    if long_strike and short_strike:
        long_opt = next((opt for opt in puts if opt['strike'] == long_strike), None)
        short_opt = next((opt for opt in puts if opt['strike'] == short_strike), None)
        if long_opt and short_opt:
            result = utils.calculate_bull_put_spread(long_opt, short_opt, st.session_state.num_contracts,
                                                     st.session_state.commission_rate)
            if result:
                utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days,
                                          st.session_state.iv, key="Bull Put Spread")
            else:
                st.warning("Combinación de strikes no válida.")