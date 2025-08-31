import streamlit as st
import utils

st.set_page_config(page_title="Neutral Strategies", layout="wide")
st.title("Estrategias Neutrales: Butterfly & Condor")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts

tab1, tab2, tab3, tab4 = st.tabs(["Call Butterfly", "Put Butterfly", "Call Condor", "Put Condor"])

with tab1:
    st.subheader("Call Butterfly")
    df = utils.create_complex_strategy_table(calls, utils.calculate_call_butterfly, st.session_state.num_contracts, st.session_state.commission_rate, 3)
    if not df.empty:
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_profit": "{:.2f}", "max_loss": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No data available for Call Butterfly.")
    if not df.empty:
        selected = st.selectbox("Selecciona una combinación para visualizar", df.index, key="cb_select")
        result = df.loc[selected].to_dict()
        result["strikes"] = list(selected)
        result["num_contracts"] = st.session_state.num_contracts
        utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Call Butterfly")

with tab2:
    st.subheader("Put Butterfly")
    df = utils.create_complex_strategy_table(puts, utils.calculate_put_butterfly, st.session_state.num_contracts, st.session_state.commission_rate, 3)
    if not df.empty:
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_profit": "{:.2f}", "max_loss": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No data available for Put Butterfly.")
    if not df.empty:
        selected = st.selectbox("Selecciona una combinación para visualizar", df.index, key="pb_select")
        result = df.loc[selected].to_dict()
        result["strikes"] = list(selected)
        result["num_contracts"] = st.session_state.num_contracts
        utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Put Butterfly")

with tab3:
    st.subheader("Call Condor")
    df = utils.create_complex_strategy_table(calls, utils.calculate_call_condor, st.session_state.num_contracts, st.session_state.commission_rate, 4)
    if not df.empty:
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_profit": "{:.2f}", "max_loss": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No data available for Call Condor.")
    if not df.empty:
        selected = st.selectbox("Selecciona una combinación para visualizar", df.index, key="cc_select")
        result = df.loc[selected].to_dict()
        result["strikes"] = list(selected)
        result["num_contracts"] = st.session_state.num_contracts
        utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Call Condor")

with tab4:
    st.subheader("Put Condor")
    df = utils.create_complex_strategy_table(puts, utils.calculate_put_condor, st.session_state.num_contracts, st.session_state.commission_rate, 4)
    if not df.empty:
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_profit": "{:.2f}", "max_loss": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No data available for Put Condor.")
    if not df.empty:
        selected = st.selectbox("Selecciona una combinación para visualizar", df.index, key="pc_select")
        result = df.loc[selected].to_dict()
        result["strikes"] = list(selected)
        result["num_contracts"] = st.session_state.num_contracts
        utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Put Condor")