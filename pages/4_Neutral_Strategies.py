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
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_profit": "{:.2f}", "max_loss": "{:.2f}", "lower_breakeven": "{:.2f}", "upper_breakeven": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No hay datos disponibles para Call Butterfly. Asegúrese de que hay suficientes opciones disponibles.")
    if not df.empty:
        options = df.index
        selected = st.selectbox("Selecciona una combinación para visualizar", options, key="cb_select")
        row = df.loc[selected]
        result = row.to_dict()
        result["strikes"] = list(selected) if isinstance(selected, tuple) else [selected]
        result["num_contracts"] = st.session_state.num_contracts
        result["contract_ratios"] = [1, -2, 1]
        if result:
            utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Call Butterfly")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")

with tab2:
    st.subheader("Put Butterfly")
    df = utils.create_complex_strategy_table(puts, utils.calculate_put_butterfly, st.session_state.num_contracts, st.session_state.commission_rate, 3)
    if not df.empty:
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_profit": "{:.2f}", "max_loss": "{:.2f}", "lower_breakeven": "{:.2f}", "upper_breakeven": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No hay datos disponibles para Put Butterfly. Asegúrese de que hay suficientes opciones disponibles.")
    if not df.empty:
        options = df.index
        selected = st.selectbox("Selecciona una combinación para visualizar", options, key="pb_select")
        row = df.loc[selected]
        result = row.to_dict()
        result["strikes"] = list(selected) if isinstance(selected, tuple) else [selected]
        result["num_contracts"] = st.session_state.num_contracts
        result["contract_ratios"] = [1, -2, 1]
        if result:
            utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Put Butterfly")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")

with tab3:
    st.subheader("Call Condor")
    df = utils.create_complex_strategy_table(calls, utils.calculate_call_condor, st.session_state.num_contracts, st.session_state.commission_rate, 4)
    if not df.empty:
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_profit": "{:.2f}", "max_loss": "{:.2f}", "lower_breakeven": "{:.2f}", "upper_breakeven": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No hay datos disponibles para Call Condor. Asegúrese de que hay suficientes opciones disponibles.")
    if not df.empty:
        options = df.index
        selected = st.selectbox("Selecciona una combinación para visualizar", options, key="cc_select")
        row = df.loc[selected]
        result = row.to_dict()
        result["strikes"] = list(selected) if isinstance(selected, tuple) else [selected]
        result["num_contracts"] = st.session_state.num_contracts
        result["contract_ratios"] = [1, -1, -1, 1]
        if result:
            utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Call Condor")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")

with tab4:
    st.subheader("Put Condor")
    df = utils.create_complex_strategy_table(puts, utils.calculate_put_condor, st.session_state.num_contracts, st.session_state.commission_rate, 4)
    if not df.empty:
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_profit": "{:.2f}", "max_loss": "{:.2f}", "lower_breakeven": "{:.2f}", "upper_breakeven": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No hay datos disponibles para Put Condor. Asegúrese de que hay suficientes opciones disponibles.")
    if not df.empty:
        options = df.index
        selected = st.selectbox("Selecciona una combinación para visualizar", options, key="pc_select")
        row = df.loc[selected]
        result = row.to_dict()
        result["strikes"] = list(selected) if isinstance(selected, tuple) else [selected]
        result["num_contracts"] = st.session_state.num_contracts
        result["contract_ratios"] = [1, -1, -1, 1]
        if result:
            utils.visualize_3d_payoff(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Put Condor")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")