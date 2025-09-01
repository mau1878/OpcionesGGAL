import streamlit as st
import utils
import logging
import pandas as pd

logger = logging.getLogger(__name__)

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
    try:
        df = utils.create_neutral_table(calls, utils.calculate_call_butterfly, st.session_state.num_contracts, st.session_state.commission_rate, 3)
    except Exception as e:
        st.error(f"Error al crear la tabla de Call Butterfly: {e}")
        logger.error(f"Error in create_neutral_table: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # Apply formatting to the DataFrame
        edited_df = df.copy()
        for col in ["net_cost", "max_profit", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)  # Raw number

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}" if 'level_2' in row else str(row['level_0']),
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1', 'level_2'])

        # Add a visualization column
        edited_df['Visualize'] = False

        # Initialize separate state for visualize flags
        if "visualize_flags_call_butterfly" not in st.session_state:
            st.session_state["visualize_flags_call_butterfly"] = [False] * len(edited_df)

        # Define callback function for Call Butterfly
        def visualize_callback_call_butterfly():
            logger.info("Visualize callback triggered for Call Butterfly")
            edited = st.session_state.get("call_butterfly_editor", {})
            logger.info(f"Edited state: {edited}")
            edited_rows = edited.get('edited_rows', {})
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df):
                    row = edited_df.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        logger.info(f"Visualizing row {idx}: {row}")
                        result = row.to_dict()
                        result["strikes"] = [float(s) for s in row['Strikes'].split('-')]
                        result["num_contracts"] = st.session_state.num_contracts
                        result["contract_ratios"] = [1, -2, 1]
                        result["raw_net"] = float(result["net_cost"])
                        result["net_cost"] = float(result["net_cost"])
                        long_low = next((opt for opt in calls if opt["strike"] == result["strikes"][0]), None)
                        short_mid = next((opt for opt in calls if opt["strike"] == result["strikes"][1]), None)
                        long_high = next((opt for opt in calls if opt["strike"] == result["strikes"][2]), None)
                        logger.info(f"Options found: long_low={long_low}, short_mid={short_mid}, long_high={long_high}")
                        if result and long_low and short_mid and long_high:
                            utils.visualize_neutral_3d(
                                result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                                "Call Butterfly", options=[long_low, short_mid, long_high], option_actions=["buy", "sell", "buy"]
                            )
                            logger.info("3D plot triggered successfully for Call Butterfly")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Selección inválida o datos de opciones no disponibles.")
                        # Reset the checkbox via separate state
                        st.session_state["visualize_flags_call_butterfly"][idx] = False
                        st.rerun()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_call_butterfly"][idx]

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
            key="call_butterfly_editor",
            on_change=visualize_callback_call_butterfly,
            width='stretch'
        )
        # Update flags after edit
        for idx in range(len(edited_df)):
            st.session_state["visualize_flags_call_butterfly"][idx] = edited_df.at[idx, 'Visualize']
        logger.info(f"Initial editor state for Call Butterfly: {st.session_state.get('call_butterfly_editor', {})}")
        logger.info(f"Edited DataFrame for Call Butterfly: {edited_df.head().to_dict()}")
    else:
        st.warning("No hay datos disponibles para Call Butterfly. Asegúrese de que hay suficientes opciones disponibles.")
        logger.warning(f"No data for Call Butterfly. Filtered calls: {len(calls)}")

# Similar pattern for tab2 (Put Butterfly), tab3 (Call Condor), tab4 (Put Condor)
# For brevity, apply the same logic, changing keys and flags (e.g., "put_butterfly_editor", "visualize_flags_put_butterfly", etc.), and adjusting the result["contract_ratios"] and options/action for condors ([1, -1, -1, 1]) and appropriate option finding/order.

with tab2:
    st.subheader("Put Butterfly")
    try:
        df = utils.create_neutral_table(puts, utils.calculate_put_butterfly, st.session_state.num_contracts, st.session_state.commission_rate, 3)
    except Exception as e:
        st.error(f"Error al crear la tabla de Put Butterfly: {e}")
        logger.error(f"Error in create_neutral_table: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # Apply formatting to the DataFrame
        edited_df = df.copy()
        for col in ["net_cost", "max_profit", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)  # Raw number

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}" if 'level_2' in row else str(row['level_0']),
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1', 'level_2'])

        # Add a visualization column
        edited_df['Visualize'] = False

        # Initialize separate state for visualize flags
        if "visualize_flags_put_butterfly" not in st.session_state:
            st.session_state["visualize_flags_put_butterfly"] = [False] * len(edited_df)

        # Define callback function for Put Butterfly
        def visualize_callback_put_butterfly():
            logger.info("Visualize callback triggered for Put Butterfly")
            edited = st.session_state.get("put_butterfly_editor", {})
            logger.info(f"Edited state: {edited}")
            edited_rows = edited.get('edited_rows', {})
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df):
                    row = edited_df.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        logger.info(f"Visualizing row {idx}: {row}")
                        result = row.to_dict()
                        result["strikes"] = [float(s) for s in row['Strikes'].split('-')]
                        result["num_contracts"] = st.session_state.num_contracts
                        result["contract_ratios"] = [1, -2, 1]
                        result["raw_net"] = float(result["net_cost"])
                        result["net_cost"] = float(result["net_cost"])
                        long_low = next((opt for opt in puts if opt["strike"] == result["strikes"][0]), None)
                        short_mid = next((opt for opt in puts if opt["strike"] == result["strikes"][1]), None)
                        long_high = next((opt for opt in puts if opt["strike"] == result["strikes"][2]), None)
                        logger.info(f"Options found: long_low={long_low}, short_mid={short_mid}, long_high={long_high}")
                        if result and long_low and short_mid and long_high:
                            utils.visualize_neutral_3d(
                                result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                                "Put Butterfly", options=[long_low, short_mid, long_high], option_actions=["buy", "sell", "buy"]
                            )
                            logger.info("3D plot triggered successfully for Put Butterfly")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Selección inválida o datos de opciones no disponibles.")
                        # Reset the checkbox via separate state
                        st.session_state["visualize_flags_put_butterfly"][idx] = False
                        st.rerun()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_put_butterfly"][idx]

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
            key="put_butterfly_editor",
            on_change=visualize_callback_put_butterfly,
            width='stretch'
        )
        # Update flags after edit
        for idx in range(len(edited_df)):
            st.session_state["visualize_flags_put_butterfly"][idx] = edited_df.at[idx, 'Visualize']
        logger.info(f"Initial editor state for Put Butterfly: {st.session_state.get('put_butterfly_editor', {})}")
        logger.info(f"Edited DataFrame for Put Butterfly: {edited_df.head().to_dict()}")
    else:
        st.warning("No hay datos disponibles para Put Butterfly. Asegúrese de que hay suficientes opciones disponibles.")
        logger.warning(f"No data for Put Butterfly. Filtered puts: {len(puts)}")

# Apply similar pattern for tab3 (Call Condor) and tab4 (Put Condor), using unique keys like "call_condor_editor", "visualize_flags_call_condor", contract_ratios = [1, -1, -1, 1], and appropriate option finding.

with tab3:
    st.subheader("Call Condor")
    try:
        df = utils.create_neutral_table(calls, utils.calculate_call_condor, st.session_state.num_contracts, st.session_state.commission_rate, 4)
    except Exception as e:
        st.error(f"Error al crear la tabla de Call Condor: {e}")
        logger.error(f"Error in create_neutral_table: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # Apply formatting to the DataFrame
        edited_df = df.copy()
        for col in ["net_cost", "max_profit", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)  # Raw number

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}-{row['level_3']}" if 'level_3' in row else str(row['level_0']),
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1', 'level_2', 'level_3'])

        # Add a visualization column
        edited_df['Visualize'] = False

        # Initialize separate state for visualize flags
        if "visualize_flags_call_condor" not in st.session_state:
            st.session_state["visualize_flags_call_condor"] = [False] * len(edited_df)

        # Define callback function for Call Condor
        def visualize_callback_call_condor():
            logger.info("Visualize callback triggered for Call Condor")
            edited = st.session_state.get("call_condor_editor", {})
            logger.info(f"Edited state: {edited}")
            edited_rows = edited.get('edited_rows', {})
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df):
                    row = edited_df.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        logger.info(f"Visualizing row {idx}: {row}")
                        result = row.to_dict()
                        result["strikes"] = [float(s) for s in row['Strikes'].split('-')]
                        result["num_contracts"] = st.session_state.num_contracts
                        result["contract_ratios"] = [1, -1, -1, 1]
                        result["raw_net"] = float(result["net_cost"])
                        result["net_cost"] = float(result["net_cost"])
                        long_low = next((opt for opt in calls if opt["strike"] == result["strikes"][0]), None)
                        short_mid_low = next((opt for opt in calls if opt["strike"] == result["strikes"][1]), None)
                        short_mid_high = next((opt for opt in calls if opt["strike"] == result["strikes"][2]), None)
                        long_high = next((opt for opt in calls if opt["strike"] == result["strikes"][3]), None)
                        logger.info(f"Options found: long_low={long_low}, short_mid_low={short_mid_low}, short_mid_high={short_mid_high}, long_high={long_high}")
                        if result and long_low and short_mid_low and short_mid_high and long_high:
                            utils.visualize_neutral_3d(
                                result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                                "Call Condor", options=[long_low, short_mid_low, short_mid_high, long_high],
                                option_actions=["buy", "sell", "sell", "buy"]
                            )
                            logger.info("3D plot triggered successfully for Call Condor")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Selección inválida o datos de opciones no disponibles.")
                        # Reset the checkbox via separate state
                        st.session_state["visualize_flags_call_condor"][idx] = False
                        st.rerun()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_call_condor"][idx]

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
            key="call_condor_editor",
            on_change=visualize_callback_call_condor,
            width='stretch'
        )
        # Update flags after edit
        for idx in range(len(edited_df)):
            st.session_state["visualize_flags_call_condor"][idx] = edited_df.at[idx, 'Visualize']
        logger.info(f"Initial editor state for Call Condor: {st.session_state.get('call_condor_editor', {})}")
        logger.info(f"Edited DataFrame for Call Condor: {edited_df.head().to_dict()}")
    else:
        st.warning("No hay datos disponibles para Call Condor. Asegúrese de que hay suficientes opciones disponibles.")
        logger.warning(f"No data for Call Condor. Filtered calls: {len(calls)}")

# Apply similar pattern for tab4 (Put Condor), adjusting options/action as in the original code.

with tab4:
    st.subheader("Put Condor")
    try:
        df = utils.create_neutral_table(puts, utils.calculate_put_condor, st.session_state.num_contracts, st.session_state.commission_rate, 4)
    except Exception as e:
        st.error(f"Error al crear la tabla de Put Condor: {e}")
        logger.error(f"Error in create_neutral_table: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # Apply formatting to the DataFrame
        edited_df = df.copy()
        for col in ["net_cost", "max_profit", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)  # Raw number

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}-{row['level_3']}" if 'level_3' in row else str(row['level_0']),
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1', 'level_2', 'level_3'])

        # Add a visualization column
        edited_df['Visualize'] = False

        # Initialize separate state for visualize flags
        if "visualize_flags_put_condor" not in st.session_state:
            st.session_state["visualize_flags_put_condor"] = [False] * len(edited_df)

        # Define callback function for Put Condor
        def visualize_callback_put_condor():
            logger.info("Visualize callback triggered for Put Condor")
            edited = st.session_state.get("put_condor_editor", {})
            logger.info(f"Edited state: {edited}")
            edited_rows = edited.get('edited_rows', {})
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df):
                    row = edited_df.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        logger.info(f"Visualizing row {idx}: {row}")
                        result = row.to_dict()
                        result["strikes"] = [float(s) for s in row['Strikes'].split('-')]
                        result["num_contracts"] = st.session_state.num_contracts
                        result["contract_ratios"] = [1, -1, -1, 1]
                        result["raw_net"] = float(result["net_cost"])
                        result["net_cost"] = float(result["net_cost"])
                        long_low = next((opt for opt in puts if opt["strike"] == result["strikes"][0]), None)
                        short_mid_low = next((opt for opt in puts if opt["strike"] == result["strikes"][1]), None)
                        short_mid_high = next((opt for opt in puts if opt["strike"] == result["strikes"][2]), None)
                        long_high = next((opt for opt in puts if opt["strike"] == result["strikes"][3]), None)
                        logger.info(f"Options found: long_low={long_low}, short_mid_low={short_mid_low}, short_mid_high={short_mid_high}, long_high={long_high}")
                        if result and long_low and short_mid_low and short_mid_high and long_high:
                            utils.visualize_neutral_3d(
                                result, st.session_state.current_price, st.session_state.expiration_days, st.session_state.iv,
                                "Put Condor", options=[long_high, short_mid_high, short_mid_low, long_low],
                                option_actions=["buy", "sell", "sell", "buy"]
                            )
                            logger.info("3D plot triggered successfully for Put Condor")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Selección inválida o datos de opciones no disponibles.")
                        # Reset the checkbox via separate state
                        st.session_state["visualize_flags_put_condor"][idx] = False
                        st.rerun()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_put_condor"][idx]

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
            key="put_condor_editor",
            on_change=visualize_callback_put_condor,
            width='stretch'
        )
        # Update flags after edit
        for idx in range(len(edited_df)):
            st.session_state["visualize_flags_put_condor"][idx] = edited_df.at[idx, 'Visualize']
        logger.info(f"Initial editor state for Put Condor: {st.session_state.get('put_condor_editor', {})}")
        logger.info(f"Edited DataFrame for Put Condor: {edited_df.head().to_dict()}")
    else:
        st.warning("No hay datos disponibles para Put Condor. Asegúrese de que hay suficientes opciones disponibles.")
        logger.warning(f"No data for Put Condor. Filtered puts: {len(puts)}")