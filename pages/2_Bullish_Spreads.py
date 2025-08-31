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
    detailed_df_call = utils.create_spread_table(calls, utils.calculate_bull_call_spread, st.session_state.num_contracts, st.session_state.commission_rate, True)
    st.dataframe(detailed_df_call.style.format({
        "Net Cost": "{:.2f}", "Max Profit": "{:.2f}", "Max Loss": "{:.2f}", "Cost-to-Profit Ratio": "{:.2%}"
    }))
    st.subheader("Visualización 3D")
    if not detailed_df_call.empty:
        selected = st.selectbox("Selecciona una combinación para visualizar", detailed_df_call.index, key="bcs_select")
        if isinstance(selected, tuple) and len(selected) == 2:
            long_strike, short_strike = selected
            row = detailed_df_call.loc[selected]
            result = {
                "net_cost": row["Net Cost"],
                "max_profit": row["Max Profit"],
                "max_loss": row["Max Loss"],
                "strikes": [long_strike, short_strike],
                "num_contracts": st.session_state.num_contracts
            }
            if result:
                utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Bull Call Spread")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")


with tab2:
    st.header("Bull Put Spread (Crédito)")
    st.subheader("Análisis Detallado por Ratio")
    # --- FIX: Changed the last parameter from True to False ---
    detailed_df_put = utils.create_spread_table(puts, utils.calculate_bull_put_spread, st.session_state.num_contracts, st.session_state.commission_rate, False)
    st.dataframe(detailed_df_put.style.format({
        "Net Credit": "{:.2f}", "Max Profit": "{:.2f}", "Max Loss": "{:.2f}", "Cost-to-Profit Ratio": "{:.2%}"
    }))
    st.subheader("Visualización 3D")
    if not detailed_df_put.empty:
        selected = st.selectbox("Selecciona una combinación para visualizar", detailed_df_put.index, key="bps_select")
        if isinstance(selected, tuple) and len(selected) == 2:
            long_strike, short_strike = selected
            row = detailed_df_put.loc[selected]
            result = {
                "net_cost": -row["Net Credit"],
                "max_profit": row["Max Profit"],
                "max_loss": row["Max Loss"],
                "strikes": [long_strike, short_strike],
                "num_contracts": st.session_state.num_contracts
            }
            if result:
                utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Bull Put Spread")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")