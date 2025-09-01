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
    df = utils.create_neutral_table(calls, utils.calculate_call_butterfly, st.session_state.num_contracts, st.session_state.commission_rate, 3)
    if not df.empty:
        styled_df = df.style.format({
            "net_cost": "{:.2f}",
            "max_profit": "{:.2f}",
            "max_loss": "{:.2f}",
            "lower_breakeven": "{:.2f}",
            "upper_breakeven": "{:.2f}",
            "Cost-to-Profit Ratio": "{:.2%}"
        })
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
        result["raw_net"] = result["net_cost"]
        # Find options corresponding to selected strikes
        long_low = next((opt for opt in calls if opt["strike"] == selected[0]), None)
        short_mid = next((opt for opt in calls if opt["strike"] == selected[1]), None)
        long_high = next((opt for opt in calls if opt["strike"] == selected[2]), None)
        if result and long_low and short_mid and long_high:
            utils.visualize_neutral_3d(
                result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                "Call Butterfly", options=[long_low, short_mid, long_high], option_actions=["buy", "sell", "buy"]
            )
        else:
            st.warning("Selección inválida o datos de opciones no disponibles.")

with tab2:
    st.subheader("Put Butterfly")
    df = utils.create_neutral_table(puts, utils.calculate_put_butterfly, st.session_state.num_contracts, st.session_state.commission_rate, 3)
    if not df.empty:
        styled_df = df.style.format({
            "net_cost": "{:.2f}",
            "max_profit": "{:.2f}",
            "max_loss": "{:.2f}",
            "lower_breakeven": "{:.2f}",
            "upper_breakeven": "{:.2f}",
            "Cost-to-Profit Ratio": "{:.2%}"
        })
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
        result["raw_net"] = result["net_cost"]
        # Find options corresponding to selected strikes
        long_low = next((opt for opt in puts if opt["strike"] == selected[0]), None)
        short_mid = next((opt for opt in puts if opt["strike"] == selected[1]), None)
        long_high = next((opt for opt in puts if opt["strike"] == selected[2]), None)
        if result and long_low and short_mid and long_high:
            utils.visualize_neutral_3d(
                result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                "Put Butterfly", options=[long_high, short_mid, long_low], option_actions=["buy", "sell", "buy"]
            )
        else:
            st.warning("Selección inválida o datos de opciones no disponibles.")

with tab3:
    st.subheader("Call Condor")
    df = utils.create_neutral_table(calls, utils.calculate_call_condor, st.session_state.num_contracts, st.session_state.commission_rate, 4)
    if not df.empty:
        styled_df = df.style.format({
            "net_cost": "{:.2f}",
            "max_profit": "{:.2f}",
            "max_loss": "{:.2f}",
            "lower_breakeven": "{:.2f}",
            "upper_breakeven": "{:.2f}",
            "Cost-to-Profit Ratio": "{:.2%}"
        })
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
        result["raw_net"] = result["net_cost"]
        # Find options corresponding to selected strikes
        long_low = next((opt for opt in calls if opt["strike"] == selected[0]), None)
        short_mid_low = next((opt for opt in calls if opt["strike"] == selected[1]), None)
        short_mid_high = next((opt for opt in calls if opt["strike"] == selected[2]), None)
        long_high = next((opt for opt in calls if opt["strike"] == selected[3]), None)
        if result and long_low and short_mid_low and short_mid_high and long_high:
            utils.visualize_neutral_3d(
                result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                "Call Condor", options=[long_low, short_mid_low, short_mid_high, long_high],
                option_actions=["buy", "sell", "sell", "buy"]
            )
        else:
            st.warning("Selección inválida o datos de opciones no disponibles.")

with tab4:
    st.subheader("Put Condor")
    df = utils.create_neutral_table(puts, utils.calculate_put_condor, st.session_state.num_contracts, st.session_state.commission_rate, 4)
    if not df.empty:
        styled_df = df.style.format({
            "net_cost": "{:.2f}",
            "max_profit": "{:.2f}",
            "max_loss": "{:.2f}",
            "lower_breakeven": "{:.2f}",
            "upper_breakeven": "{:.2f}",
            "Cost-to-Profit Ratio": "{:.2%}"
        })
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
        result["raw_net"] = result["net_cost"]
        # Find options corresponding to selected strikes
        long_low = next((opt for opt in puts if opt["strike"] == selected[0]), None)
        short_mid_low = next((opt for opt in puts if opt["strike"] == selected[1]), None)
        short_mid_high = next((opt for opt in puts if opt["strike"] == selected[2]), None)
        long_high = next((opt for opt in puts if opt["strike"] == selected[3]), None)
        if result and long_low and short_mid_low and short_mid_high and long_high:
            utils.visualize_neutral_3d(
                result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                "Put Condor", options=[long_high, short_mid_high, short_mid_low, long_low],
                option_actions=["buy", "sell", "sell", "buy"]
            )
        else:
            st.warning("Selección inválida o datos de opciones no disponibles.")