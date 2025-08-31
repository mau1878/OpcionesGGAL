import streamlit as st
import utils

st.set_page_config(page_title="Volatility Strategies", layout="wide")
st.title("Estrategias de Volatilidad: Straddle & Strangle")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts

tab1, tab2 = st.tabs(["Straddle", "Strangle"])

with tab1:
    st.subheader("Long Straddle")
    df = utils.create_volatility_table(calls, puts, utils.calculate_straddle, st.session_state.num_contracts, st.session_state.commission_rate)
    if not df.empty:
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_loss": "{:.2f}", "lower_breakeven": "{:.2f}", "upper_breakeven": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No hay datos disponibles para Long Straddle. Verifique si hay strikes coincidentes para calls y puts en el rango seleccionado.")
    if not df.empty:
        options = df.index
        selected = st.selectbox("Selecciona una combinación para visualizar", options, key="straddle_select")
        row = df.loc[selected]
        result = row.to_dict()
        result["strikes"] = [selected] if isinstance(selected, (int, float)) else list(selected)
        result["num_contracts"] = st.session_state.num_contracts
        result["raw_net"] = result["net_cost"]
        if result:
            utils.visualize_volatility_3d(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Straddle")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")

with tab2:
    st.subheader("Long Strangle")
    df = utils.create_volatility_table(puts, calls, utils.calculate_strangle, st.session_state.num_contracts, st.session_state.commission_rate)
    if not df.empty:
        styled_df = df.style.format({"net_cost": "{:.2f}", "max_loss": "{:.2f}", "lower_breakeven": "{:.2f}", "upper_breakeven": "{:.2f}"})
        st.dataframe(styled_df)
    else:
        st.dataframe(df)
        st.warning("No hay datos disponibles para Long Strangle. Verifique si hay puts con strikes inferiores a calls en el rango seleccionado.")
    if not df.empty:
        options = df.index
        selected = st.selectbox("Selecciona una combinación para visualizar", options, key="strangle_select")
        row = df.loc[selected]
        result = row.to_dict()
        result["strikes"] = list(selected) if isinstance(selected, tuple) else [selected]
        result["num_contracts"] = st.session_state.num_contracts
        result["raw_net"] = result["net_cost"]
        if result:
            utils.visualize_volatility_3d(result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv, key="Strangle")
        else:
            st.warning("Selección inválida. Por favor, seleccione una combinación válida.")