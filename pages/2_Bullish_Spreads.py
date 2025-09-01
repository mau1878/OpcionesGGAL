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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)  # Raw number

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            # Join the MultiIndex levels into a Strikes column
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}" if 'level_1' in row else str(row['level_0']),
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1'])

        # Add a visualization column
        edited_df['Visualize'] = False

        # Define callback function for Bull Call Spread with debug
        def visualize_callback_call():
            logger.info("Visualize callback triggered for Bull Call Spread")
            edited = st.session_state.get("bull_call_spread_editor_unique", {})
            logger.info(f"Edited state: {edited}")
            edited_rows = edited.get('edited_rows', {})
            logger.info(f"Edited rows: {edited_rows}")
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df):
                    row = edited_df.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    logger.info(f"Row idx: {idx}, Visualize state: {visualize_state}, Columns: {edited_df.columns.tolist()}")
                    if visualize_state:
                        logger.info(f"Visualizing row {idx}: {row}")
                        if "Net Cost" not in row:
                            logger.error(f"Missing 'Net Cost' column for row {idx}")
                            st.error(f"Data mismatch: 'Net Cost' not found in row {idx}")
                            continue
                        result = {
                            "net_cost": float(row["Net Cost"].replace(",", ".")),
                            "max_profit": float(row["Max Profit"].replace(",", ".")),
                            "max_loss": float(row["Max Loss"].replace(",", ".")),
                            "breakeven": float(row["Breakeven"].replace(",", ".")),
                            "strikes": [float(s) for s in row['Strikes'].split('-')],
                            "num_contracts": st.session_state.num_contracts,
                            "raw_net": float(row["Net Cost"].replace(",", "."))
                        }
                        long_opt = next((opt for opt in calls if opt["strike"] == result["strikes"][0]), None)
                        short_opt = next((opt for opt in calls if opt["strike"] == result["strikes"][1]), None)
                        logger.info(f"Options found: long={long_opt}, short={short_opt}")
                        if long_opt and short_opt:
                            try:
                                utils.visualize_bullish_3d(
                                    result, st.session_state.current_price, st.session_state.expiration_days,
                                    st.session_state.iv, "Bull Call Spread",
                                    options=[long_opt, short_opt], option_actions=["buy", "sell"]
                                )
                                logger.info("3D plot triggered successfully for Bull Call Spread")
                            except Exception as e:
                                logger.error(f"Error in visualize_bullish_3d: {e}")
                                st.error(f"Failed to generate 3D plot: {e}")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Datos de opciones no disponibles para esta combinación.")
                        # Reset the checkbox
                        edited_df.at[idx, 'Visualize'] = False
                        st.session_state["bull_call_spread_editor_unique"] = edited_df.to_dict()

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
            key="bull_call_spread_editor_unique",
            on_change=visualize_callback_call,
            width='stretch'
        )
        logger.info(f"Initial editor state for Bull Call Spread: {st.session_state.get('bull_call_spread_editor_unique', {})}")
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
            st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis_r', subset=profit_df.columns[profit_df.notna().any()]))
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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)  # Raw number

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            # Join the MultiIndex levels into a Strikes column
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}" if 'level_1' in row else str(row['level_0']),
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1'])

        # Add a visualization column
        edited_df['Visualize'] = False

        # Define callback function for Bull Put Spread with debug
        def visualize_callback_put():
            logger.info("Visualize callback triggered for Bull Put Spread")
            edited = st.session_state.get("bull_put_spread_editor_unique", {})
            logger.info(f"Edited state: {edited}")
            edited_rows = edited.get('edited_rows', {})
            logger.info(f"Edited rows: {edited_rows}")
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df):
                    row = edited_df.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    logger.info(f"Row idx: {idx}, Visualize state: {visualize_state}, Columns: {edited_df.columns.tolist()}")
                    if visualize_state:
                        logger.info(f"Visualizing row {idx}: {row}")
                        if "Net Credit" not in row:
                            logger.error(f"Missing 'Net Credit' column for row {idx}")
                            st.error(f"Data mismatch: 'Net Credit' not found in row {idx}")
                            continue
                        result = {
                            "net_cost": -float(row["Net Credit"].replace(",", ".")),
                            "max_profit": float(row["Max Profit"].replace(",", ".")),
                            "max_loss": float(row["Max Loss"].replace(",", ".")),
                            "breakeven": float(row["Breakeven"].replace(",", ".")),
                            "strikes": [float(s) for s in row['Strikes'].split('-')],
                            "num_contracts": st.session_state.num_contracts,
                            "raw_net": -float(row["Net Credit"].replace(",", "."))
                        }
                        short_opt = next((opt for opt in puts if opt["strike"] == result["strikes"][1]), None)
                        long_opt = next((opt for opt in puts if opt["strike"] == result["strikes"][0]), None)
                        logger.info(f"Options found: short={short_opt}, long={long_opt}")
                        if short_opt and long_opt:
                            try:
                                utils.visualize_bullish_3d(
                                    result, st.session_state.current_price, st.session_state.expiration_days,
                                    st.session_state.iv, "Bull Put Spread",
                                    options=[short_opt, long_opt], option_actions=["sell", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Bull Put Spread")
                            except Exception as e:
                                logger.error(f"Error in visualize_bullish_3d for Put Spread: {e}")
                                st.error(f"Failed to generate 3D plot for Put Spread: {e}")
                        else:
                            logger.warning("Options not found for this Put Spread combination")
                            st.warning("Datos de opciones no disponibles para esta combinación.")
                        # Reset the checkbox
                        edited_df.at[idx, 'Visualize'] = False
                        st.session_state["bull_put_spread_editor_unique"] = edited_df.to_dict()

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
            key="bull_put_spread_editor_unique",
            on_change=visualize_callback_put,
            width='stretch'
        )
        logger.info(f"Initial editor state for Bull Put Spread: {st.session_state.get('bull_put_spread_editor_unique', {})}")
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
            st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis', subset=profit_df.columns[profit_df.notna().any()]))
        else:
            st.warning("No hay datos disponibles para la matriz de créditos.")
            logger.warning("Empty profit_df for Bull Put Spread matrix.")
    except Exception as e:
        st.error(f"Error al crear la matriz de créditos: {e}")
        logger.error(f"Error in create_spread_matrix: {e}")