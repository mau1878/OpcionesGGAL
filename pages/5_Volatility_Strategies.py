import streamlit as st
import utils
import logging
import pandas as pd

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Volatility Strategies", layout="wide")
st.title("Estrategias de Volatilidad: Straddle & Strangle")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts

# Log available strikes for debugging
call_strikes = {opt["strike"] for opt in calls}
put_strikes = {opt["strike"] for opt in puts}
logger.info(f"Available call strikes: {call_strikes}")
logger.info(f"Available put strikes: {put_strikes}")

tab1, tab2 = st.tabs(["Straddle", "Strangle"])

with tab1:
    st.subheader("Long Straddle")
    try:
        df = utils.create_volatility_table(calls, puts, utils.calculate_straddle, st.session_state.num_contracts, st.session_state.commission_rate)
    except Exception as e:
        st.error(f"Error al crear la tabla de Long Straddle: {e}")
        logger.error(f"Error in create_volatility_table: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # Apply formatting to the DataFrame
        edited_df = df.copy()
        for col in ["net_cost", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)  # Raw number

        # Preserve the original index as strike, only reset if MultiIndex and extract strike
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df = edited_df.rename(columns={'level_0': 'Strike'})
            # Set the first level of MultiIndex as the index if it represents the strike
            edited_df.set_index('Strike', inplace=True)
        else:
            # Ensure the index remains the strike value
            pass  # No reset needed if already a single strike index

        # Add a visualization column
        edited_df['Visualize'] = False

        # Initialize or resize state for visualize flags
        if "visualize_flags_straddle" not in st.session_state or len(st.session_state["visualize_flags_straddle"]) != len(edited_df):
            st.session_state["visualize_flags_straddle"] = [False] * len(edited_df)
        logger.info(f"Resized visualize_flags_straddle to length: {len(st.session_state['visualize_flags_straddle'])}, edited_df length: {len(edited_df)}")

        # Define callback function for Long Straddle
        def visualize_callback_straddle():
            logger.info("Visualize callback triggered for Long Straddle")
            edited = st.session_state.get("straddle_editor", {})
            logger.info(f"Edited state: {edited}")
            edited_rows = edited.get('edited_rows', {})
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df):
                    row = edited_df.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    logger.info(f"Row idx: {idx}, Visualize state: {visualize_state}, row.name={row.name}, Columns={row.index.tolist()}")
                    if visualize_state:
                        logger.info(f"Visualizing row {idx}: {row}")
                        result = row.to_dict()
                        # Use original DataFrame index value as strike for Straddle
                        original_strike = df.index[idx] if pd.notna(df.index[idx]) else None
                        result["strikes"] = [float(original_strike)] if original_strike is not None else []
                        if not result["strikes"]:
                            logger.error(f"Invalid strike value: {original_strike}")
                            st.error(f"Invalid strike value: {original_strike}")
                            continue
                        strike = result["strikes"][0]
                        logger.info(f"Validating strike: {strike}")
                        if strike not in call_strikes or strike not in put_strikes:
                            logger.warning(f"Strike {strike} not found in calls or puts")
                            st.warning(f"Strike {strike} not available in option data.")
                            continue
                        result["num_contracts"] = st.session_state.num_contracts
                        result["raw_net"] = float(result["net_cost"])
                        result["net_cost"] = float(result["net_cost"])
                        call_opt = next((opt for opt in calls if abs(opt["strike"] - strike) < 0.01), None)
                        put_opt = next((opt for opt in puts if abs(opt["strike"] - strike) < 0.01), None)
                        logger.info(f"Options found: call={call_opt}, put={put_opt}")
                        if result and call_opt and put_opt:
                            try:
                                utils.visualize_volatility_3d(
                                    result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                                    "Straddle", options=[call_opt, put_opt], option_actions=["buy", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Long Straddle")
                            except Exception as e:
                                logger.error(f"Error in visualize_volatility_3d: {e}")
                                st.error(f"Failed to generate 3D plot: {e}")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Selección inválida o datos de opciones no disponibles.")
                        # Reset the checkbox via separate state
                        st.session_state["visualize_flags_straddle"][idx] = False
                        st.rerun()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_straddle"][idx]

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
            key="straddle_editor",
            on_change=visualize_callback_straddle,
            width='stretch'
        )
        # Update flags after edit, ensuring safe indexing
        for idx in range(len(edited_df)):
            if idx < len(st.session_state["visualize_flags_straddle"]):
                st.session_state["visualize_flags_straddle"][idx] = edited_df.at[idx, 'Visualize']
        logger.info(f"Initial editor state for Straddle: {st.session_state.get('straddle_editor', {})}")
        logger.info(f"Edited DataFrame for Straddle: {edited_df.head().to_dict()}")
    else:
        st.warning("No hay datos disponibles para Long Straddle. Verifique si hay strikes coincidentes para calls y puts en el rango seleccionado.")
        logger.warning("No data for Long Straddle. Filtered calls: {len(calls)}, puts: {len(puts)}")

with tab2:
    st.subheader("Long Strangle")
    try:
        df = utils.create_volatility_table(puts, calls, utils.calculate_strangle, st.session_state.num_contracts, st.session_state.commission_rate)
    except Exception as e:
        st.error(f"Error al crear la tabla de Long Strangle: {e}")
        logger.error(f"Error in create_volatility_table: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # Apply formatting to the DataFrame
        edited_df = df.copy()
        for col in ["net_cost", "max_loss", "lower_breakeven", "upper_breakeven"]:
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
        if "visualize_flags_strangle" not in st.session_state or len(st.session_state["visualize_flags_strangle"]) != len(edited_df):
            st.session_state["visualize_flags_strangle"] = [False] * len(edited_df)
        logger.info(f"Resized visualize_flags_strangle to length: {len(st.session_state['visualize_flags_strangle'])}, edited_df length: {len(edited_df)}")

        # Define callback function for Long Strangle
        def visualize_callback_strangle():
            logger.info("Visualize callback triggered for Long Strangle")
            edited = st.session_state.get("strangle_editor", {})
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
                        # Use Strikes column for Strangle (two strikes)
                        strikes = [float(s) for s in row['Strikes'].split('-')]
                        if len(strikes) != 2:
                            logger.error(f"Invalid strike pair: {row['Strikes']}")
                            st.error(f"Invalid strike pair: {row['Strikes']}")
                            continue
                        result["strikes"] = strikes
                        logger.info(f"Validating strikes: {strikes}")
                        if strikes[0] not in put_strikes or strikes[1] not in call_strikes:
                            logger.warning(f"Strikes {strikes} not found in puts or calls")
                            st.warning(f"Strikes {strikes} not available in option data.")
                            continue
                        result["num_contracts"] = st.session_state.num_contracts
                        result["raw_net"] = float(result["net_cost"])
                        result["net_cost"] = float(result["net_cost"])
                        put_opt = next((opt for opt in puts if abs(opt["strike"] - strikes[0]) < 0.01), None)
                        call_opt = next((opt for opt in calls if abs(opt["strike"] - strikes[1]) < 0.01), None)
                        logger.info(f"Options found: put={put_opt}, call={call_opt}")
                        if result and put_opt and call_opt:
                            try:
                                utils.visualize_volatility_3d(
                                    result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                                    "Strangle", options=[put_opt, call_opt], option_actions=["buy", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Long Strangle")
                            except Exception as e:
                                logger.error(f"Error in visualize_volatility_3d: {e}")
                                st.error(f"Failed to generate 3D plot: {e}")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Selección inválida o datos de opciones no disponibles.")
                        # Reset the checkbox via separate state
                        st.session_state["visualize_flags_strangle"][idx] = False
                        st.rerun()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_strangle"][idx]

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
            key="strangle_editor",
            on_change=visualize_callback_strangle,
            width='stretch'
        )
        # Update flags after edit, ensuring safe indexing
        for idx in range(len(edited_df)):
            if idx < len(st.session_state["visualize_flags_strangle"]):
                st.session_state["visualize_flags_strangle"][idx] = edited_df.at[idx, 'Visualize']
        logger.info(f"Initial editor state for Strangle: {st.session_state.get('strangle_editor', {})}")
        logger.info(f"Edited DataFrame for Strangle: {edited_df.head().to_dict()}")
    else:
        st.warning("No hay datos disponibles para Long Strangle. Verifique si hay puts con strikes inferiores a calls en el rango seleccionado.")
        logger.warning("No data for Long Strangle. Filtered puts: {len(puts)}, calls: {len(calls)}")