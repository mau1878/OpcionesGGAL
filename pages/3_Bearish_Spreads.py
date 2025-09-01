import streamlit as st
import pandas as pd
import utils
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Bearish Spreads", layout="wide")
st.title("Estrategias Bajistas: Spreads")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página 'Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
min_strike = st.session_state.current_price * (1 - st.session_state.plot_range_pct)
max_strike = st.session_state.current_price * (1 + st.session_state.plot_range_pct)

tab1, tab2 = st.tabs(["Bear Call Spread", "Bear Put Spread"])

with tab1:
    st.header("Bear Call Spread (Crédito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_call = utils.create_bearish_spread_table(
            calls, utils.calculate_bear_call_spread, st.session_state.num_contracts,
            st.session_state.commission_rate, is_debit=False
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Bear Call Spread: {e}")
        logger.error(f"Error in create_bearish_spread_table: {e}")
        detailed_df_call = pd.DataFrame()

    if not detailed_df_call.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_call.copy()
        for col in ["Net Credit", "Max Profit", "Max Loss", "Breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)  # Raw number

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}" if 'level_1' in row else str(row['level_0']),
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1'])

        # Add a visualization column
        edited_df['Visualize'] = False

        # Initialize separate state for visualize flags
        if "visualize_flags_bear_call" not in st.session_state:
            st.session_state["visualize_flags_bear_call"] = [False] * len(edited_df)

        # Define callback function for Bear Call Spread
        def visualize_callback_bear_call():
            logger.info("Visualize callback triggered for Bear Call Spread")
            edited = st.session_state.get("bear_call_spread_editor", {})
            logger.info(f"Edited state: {edited}")
            edited_rows = edited.get('edited_rows', {})
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df):
                    row = edited_df.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    logger.info(f"Row idx: {idx}, Visualize state: {visualize_state}, row.name={row.name}, Strikes={row['Strikes']}")
                    if visualize_state:
                        logger.info(f"Visualizing row {idx}: {row}")
                        result = row.to_dict()
                        # Use Strikes column instead of row.name
                        strikes = [float(s) for s in row['Strikes'].split('-')]
                        if len(strikes) != 2:
                            logger.error(f"Invalid strike pair: {row['Strikes']}")
                            st.error(f"Invalid strike pair: {row['Strikes']}")
                            continue
                        result["strikes"] = strikes
                        result["num_contracts"] = st.session_state.num_contracts
                        result["raw_net"] = -float(result.get("Net Credit", 0))
                        result["net_cost"] = -float(result.get("Net Credit", 0))
                        short_opt = next((opt for opt in calls if opt["strike"] == strikes[0]), None)
                        long_opt = next((opt for opt in calls if opt["strike"] == strikes[1]), None)
                        logger.info(f"Options found: short={short_opt}, long={long_opt}")
                        if result and short_opt and long_opt:
                            try:
                                utils.visualize_bearish_3d(
                                    result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                                    "Bear Call Spread", options=[short_opt, long_opt], option_actions=["sell", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Bear Call Spread")
                            except Exception as e:
                                logger.error(f"Error in visualize_bearish_3d: {e}")
                                st.error(f"Failed to generate 3D plot: {e}")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Selección inválida o datos de opciones no disponibles.")
                        # Reset the checkbox via separate state
                        st.session_state["visualize_flags_bear_call"][idx] = False
                        st.rerun()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_bear_call"][idx]

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
            key="bear_call_spread_editor",
            on_change=visualize_callback_bear_call,
            width='stretch'
        )
        # Update flags after edit
        for idx in range(len(edited_df)):
            st.session_state["visualize_flags_bear_call"][idx] = edited_df.at[idx, 'Visualize']
        logger.info(f"Initial editor state for Bear Call Spread: {st.session_state.get('bear_call_spread_editor', {})}")
        logger.info(f"Edited DataFrame for Bear Call Spread: {edited_df.head().to_dict()}")
    else:
        st.warning("No hay datos disponibles para Bear Call Spread. Asegúrese de que hay suficientes opciones call.")
        logger.warning(f"No data for Bear Call Spread. Filtered calls: {len(calls)}, Strikes: {min_strike:.2f}-{max_strike:.2f}, Expiration: {st.session_state.selected_exp}")

    st.subheader("Matriz de Crédito Neto (Venta en Fila, Compra en Columna)")
    try:
        profit_df, _, _, _ = utils.create_spread_matrix(
            calls, utils.calculate_bear_call_spread, st.session_state.num_contracts,
            st.session_state.commission_rate, False
        )
        if not profit_df.empty:
            st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis'))
        else:
            st.warning("No hay datos disponibles para la matriz de créditos.")
            logger.warning("Empty profit_df for Bear Call Spread matrix.")
    except Exception as e:
        st.error(f"Error al crear la matriz de créditos: {e}")
        logger.error(f"Error in create_spread_matrix: {e}")

with tab2:
    st.header("Bear Put Spread (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_put = utils.create_bearish_spread_table(
            puts, utils.calculate_bear_put_spread, st.session_state.num_contracts,
            st.session_state.commission_rate, is_debit=True
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Bear Put Spread: {e}")
        logger.error(f"Error in create_bearish_spread_table: {e}")
        detailed_df_put = pd.DataFrame()

    if not detailed_df_put.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_put.copy()
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

        # Initialize separate state for visualize flags
        if "visualize_flags_bear_put" not in st.session_state:
            st.session_state["visualize_flags_bear_put"] = [False] * len(edited_df)

        # Define callback function for Bear Put Spread
        def visualize_callback_bear_put():
            logger.info("Visualize callback triggered for Bear Put Spread")
            edited = st.session_state.get("bear_put_spread_editor", {})
            logger.info(f"Edited state: {edited}")
            edited_rows = edited.get('edited_rows', {})
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df):
                    row = edited_df.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    logger.info(f"Row idx: {idx}, Visualize state: {visualize_state}, row.name={row.name}, Strikes={row['Strikes']}")
                    if visualize_state:
                        logger.info(f"Visualizing row {idx}: {row}")
                        result = row.to_dict()
                        # Use Strikes column instead of row.name
                        strikes = [float(s) for s in row['Strikes'].split('-')]
                        if len(strikes) != 2:
                            logger.error(f"Invalid strike pair: {row['Strikes']}")
                            st.error(f"Invalid strike pair: {row['Strikes']}")
                            continue
                        result["strikes"] = strikes
                        result["num_contracts"] = st.session_state.num_contracts
                        result["raw_net"] = float(result.get("Net Cost", 0))
                        result["net_cost"] = float(result.get("Net Cost", 0))
                        long_opt = next((opt for opt in puts if opt["strike"] == strikes[1]), None)
                        short_opt = next((opt for opt in puts if opt["strike"] == strikes[0]), None)
                        logger.info(f"Options found: long={long_opt}, short={short_opt}")
                        if result and long_opt and short_opt:
                            try:
                                utils.visualize_bearish_3d(
                                    result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                                    "Bear Put Spread", options=[long_opt, short_opt], option_actions=["buy", "sell"]
                                )
                                logger.info("3D plot triggered successfully for Bear Put Spread")
                            except Exception as e:
                                logger.error(f"Error in visualize_bearish_3d: {e}")
                                st.error(f"Failed to generate 3D plot: {e}")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Selección inválida o datos de opciones no disponibles.")
                        # Reset the checkbox via separate state
                        st.session_state["visualize_flags_bear_put"][idx] = False
                        st.rerun()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_bear_put"][idx]

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
            key="bear_put_spread_editor",
            on_change=visualize_callback_bear_put,
            width='stretch'
        )
        # Update flags after edit
        for idx in range(len(edited_df)):
            st.session_state["visualize_flags_bear_put"][idx] = edited_df.at[idx, 'Visualize']
        logger.info(f"Initial editor state for Bear Put Spread: {st.session_state.get('bear_put_spread_editor', {})}")
        logger.info(f"Edited DataFrame for Bear Put Spread: {edited_df.head().to_dict()}")
    else:
        st.warning("No hay datos disponibles para Bear Put Spread. Asegúrese de que hay suficientes opciones put.")
        logger.warning(f"No data for Bear Put Spread. Filtered puts: {len(puts)}, Strikes: {min_strike:.2f}-{max_strike:.2f}, Expiration: {st.session_state.selected_exp}")

    st.subheader("Matriz de Costo Neto (Compra en Fila, Venta en Columna)")
    try:
        profit_df, _, _, _ = utils.create_spread_matrix(
            puts, utils.calculate_bear_put_spread, st.session_state.num_contracts,
            st.session_state.commission_rate, True
        )
        if not profit_df.empty:
            st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis_r'))
        else:
            st.warning("No hay datos disponibles para la matriz de costos.")
            logger.warning("Empty profit_df for Bear Put Spread matrix.")
    except Exception as e:
        st.error(f"Error al crear la matriz de costos: {e}")
        logger.error(f"Error in create_spread_matrix: {e}")