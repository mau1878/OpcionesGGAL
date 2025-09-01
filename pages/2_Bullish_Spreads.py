import streamlit as st
import pandas as pd
import utils
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Bullish Spreads", layout="wide")
st.title("Estrategias Alcistas: Spreads")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página 'Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
min_strike = st.session_state.current_price * (1 - st.session_state.plot_range_pct)
max_strike = st.session_state.current_price * (1 + st.session_state.plot_range_pct)

tab1, tab2 = st.tabs(["Bull Call Spread", "Bull Put Spread"])

with tab1:
    st.header("Bull Call Spread (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_call = utils.create_bullish_spread_table(
            calls, utils.calculate_bull_call_spread, st.session_state.num_contracts,
            st.session_state.commission_rate, is_debit=True
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Bull Call Spread: {e}")
        logger.error(f"Error in create_bullish_spread_table: {e}")
        detailed_df_call = pd.DataFrame()

    if not detailed_df_call.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_call.copy()
        for col in ["Net Cost", "Max Profit", "Max Loss", "Breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2%}")

        # Add a visualization column
        edited_df['Visualize'] = False

        # Define callback function
        def visualize_callback():
            edited = st.session_state.get("bull_call_spread_editor", {})
            for idx, row in edited_df.iterrows():
                if edited.get(str(idx), {}).get('Visualize', False):
                    result = {
                        "net_cost": float(row["Net Cost"].replace(",", ".")),
                        "max_profit": float(row["Max Profit"].replace(",", ".")),
                        "max_loss": float(row["Max Loss"].replace(",", ".")),
                        "breakeven": float(row["Breakeven"].replace(",", ".")),
                        "strikes": list(idx),  # Use index as strikes
                        "num_contracts": st.session_state.num_contracts,
                        "raw_net": float(row["Net Cost"].replace(",", "."))
                    }
                    long_opt = next((opt for opt in calls if opt["strike"] == idx[0]), None)
                    short_opt = next((opt for opt in calls if opt["strike"] == idx[1]), None)
                    if long_opt and short_opt:
                        utils.visualize_bullish_3d(
                            result, st.session_state.current_price, st.session_state.expiration_days,
                            st.session_state.iv, "Bull Call Spread",
                            options=[long_opt, short_opt], option_actions=["buy", "sell"]
                        )
                    else:
                        st.warning("Datos de opciones no disponibles para esta combinación.")
                    # Reset the checkbox
                    edited_df.at[idx, 'Visualize'] = False
                    st.experimental_rerun()  # Rerun to update the editor state

        # Use data_editor with checkbox column
        edited_df = st.data_editor(
            edited_df,
            column_config={
                "Visualize": st.column_config.CheckboxColumn(
                    "Visualize 3D",
                    help="Check to generate 3D plot",
                    disabled=False
                )
            },
            key="bull_call_spread_editor",
            on_change=visualize_callback,
            use_container_width=True
        )
    else:
        st.warning("No hay datos disponibles para Bull Call Spread. Asegúrese de que hay suficientes opciones call en el rango seleccionado o intente actualizar los datos.")
        logger.warning(f"No data for Bull Call Spread. Filtered calls: {len(calls)}, Strikes: {min_strike:.2f}-{max_strike:.2f}, Expiration: {st.session_state.selected_exp}")

    st.subheader("Matriz de Costo Neto (Compra en Fila, Venta en Columna)")
    try:
        profit_df, _, _, _ = utils.create_spread_matrix(
            calls, utils.calculate_bull_call_spread, st.session_state.num_contracts,
            st.session_state.commission_rate, True
        )
        if not profit_df.empty:
            st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis_r'))
        else:
            st.warning("No hay datos disponibles para la matriz de costos.")
            logger.warning("Empty profit_df for Bull Call Spread matrix.")
    except Exception as e:
        st.error(f"Error al crear la matriz de costos: {e}")
        logger.error(f"Error in create_spread_matrix: {e}")

with tab2:
    st.header("Bull Put Spread (Crédito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_put = utils.create_bullish_spread_table(
            puts, utils.calculate_bull_put_spread, st.session_state.num_contracts,
            st.session_state.commission_rate, is_debit=False
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Bull Put Spread: {e}")
        logger.error(f"Error in create_bullish_spread_table: {e}")
        detailed_df_put = pd.DataFrame()

    if not detailed_df_put.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_put.copy()
        for col in ["Net Credit", "Max Profit", "Max Loss", "Breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2%}")

        # Add a visualization column
        edited_df['Visualize'] = False

        # Define callback function
        def visualize_callback_put():
            edited = st.session_state.get("bull_put_spread_editor", {})
            for idx, row in edited_df.iterrows():
                if edited.get(str(idx), {}).get('Visualize', False):
                    result = {
                        "net_cost": -float(row["Net Credit"].replace(",", ".")),
                        "max_profit": float(row["Max Profit"].replace(",", ".")),
                        "max_loss": float(row["Max Loss"].replace(",", ".")),
                        "breakeven": float(row["Breakeven"].replace(",", ".")),
                        "strikes": list(idx),
                        "num_contracts": st.session_state.num_contracts,
                        "raw_net": -float(row["Net Credit"].replace(",", "."))
                    }
                    short_opt = next((opt for opt in puts if opt["strike"] == idx[1]), None)
                    long_opt = next((opt for opt in puts if opt["strike"] == idx[0]), None)
                    if short_opt and long_opt:
                        utils.visualize_bullish_3d(
                            result, st.session_state.current_price, st.session_state.expiration_days,
                            st.session_state.iv, "Bull Put Spread",
                            options=[short_opt, long_opt], option_actions=["sell", "buy"]
                        )
                    else:
                        st.warning("Datos de opciones no disponibles para esta combinación.")
                    # Reset the checkbox
                    edited_df.at[idx, 'Visualize'] = False
                    st.experimental_rerun()  # Rerun to update the editor state

        # Use data_editor with checkbox column
        edited_df = st.data_editor(
            edited_df,
            column_config={
                "Visualize": st.column_config.CheckboxColumn(
                    "Visualize 3D",
                    help="Check to generate 3D plot",
                    disabled=False
                )
            },
            key="bull_put_spread_editor",
            on_change=visualize_callback_put,
            use_container_width=True
        )
    else:
        st.warning("No hay datos disponibles para Bull Put Spread. Asegúrese de que hay suficientes opciones put en el rango seleccionado o intente actualizar los datos.")
        logger.warning(f"No data for Bull Put Spread. Filtered puts: {len(puts)}, Strikes: {min_strike:.2f}-{max_strike:.2f}, Expiration: {st.session_state.selected_exp}")

    st.subheader("Matriz de Crédito Neto (Venta en Fila, Compra en Columna)")
    try:
        profit_df, _, _, _ = utils.create_spread_matrix(
            puts, utils.calculate_bull_put_spread, st.session_state.num_contracts,
            st.session_state.commission_rate, False
        )
        if not profit_df.empty:
            st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis'))
        else:
            st.warning("No hay datos disponibles para la matriz de créditos.")
            logger.warning("Empty profit_df for Bull Put Spread matrix.")
    except Exception as e:
        st.error(f"Error al crear la matriz de créditos: {e}")
        logger.error(f"Error in create_spread_matrix: {e}")