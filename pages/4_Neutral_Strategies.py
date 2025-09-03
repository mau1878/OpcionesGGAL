import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from itertools import combinations
import logging
from calc_utils import calculate_call_butterfly, calculate_put_butterfly, calculate_call_condor, calculate_put_condor, black_scholes, intrinsic_value
from viz_utils import create_neutral_table, visualize_neutral_3d

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Neutral Strategies", layout="wide")
st.title("Estrategias Neutrales: Butterfly & Condor")

# Check for required session state
if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
current_price = st.session_state.current_price
expiration_days = st.session_state.expiration_days
risk_free_rate = st.session_state.risk_free_rate
commission_rate = st.session_state.commission_rate
num_contracts = st.session_state.num_contracts

# Log available strikes for debugging
call_strikes_set = {opt["strike"] for opt in calls}
put_strikes_set = {opt["strike"] for opt in puts}
call_strikes = sorted(call_strikes_set)
put_strikes = sorted(put_strikes_set)
logger.info(f"Available call strikes: {call_strikes}")
logger.info(f"Available put strikes: {put_strikes}")

# Sidebar inputs
st.sidebar.header("Parámetros de Análisis")
price_range_pct = st.sidebar.slider("Rango de Precio para Ganancia Potencial (%)", 5.0, 50.0, 20.0) / 100
sort_by = st.sidebar.selectbox("Ordenar Tabla por", ["Cost-to-Profit Ratio", "Breakeven Probability"], key="sort_by")
max_strikes = st.sidebar.number_input("Max Strikes to Evaluate", min_value=5, max_value=50, value=15)  # Added for performance

# Log price range for debugging
logger.info(f"Sidebar price_range_pct: {price_range_pct}, min_price: {current_price * (1 - price_range_pct):.2f}, max_price: {current_price * (1 + price_range_pct):.2f}")

# Track IV calibration failures
if "iv_failure_count" not in st.session_state:
    st.session_state["iv_failure_count"] = 0

# Initialize session state for visualization
if "visualize_strategy" not in st.session_state:
    st.session_state["visualize_strategy"] = None

# Filter to nearest strikes for performance
nearest_calls = sorted(calls, key=lambda o: abs(o["strike"] - current_price))[:max_strikes]
nearest_puts = sorted(puts, key=lambda o: abs(o["strike"] - current_price))[:max_strikes]

# Tabs for different strategies
tab1, tab2, tab3, tab4 = st.tabs(["Call Butterfly", "Put Butterfly", "Call Condor", "Put Condor"])

# Call Butterfly
with tab1:
    st.header("Call Butterfly (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_call = create_neutral_table(
            nearest_calls, calculate_call_butterfly, num_contracts, commission_rate, num_legs=3
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Call Butterfly: {e}")
        logger.error(f"Error in create_neutral_table: {e}")
        detailed_df_call = pd.DataFrame()

    if not detailed_df_call.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_call.copy()
        for col in ["net_cost", "max_profit", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}",
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
            edited_rows = edited.get('edited_rows', {})
            edited_df_local = st.session_state.get("call_butterfly_df", edited_df)
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df_local):
                    row = edited_df_local.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        strikes = [float(s) for s in row['Strikes'].split('-')]
                        if len(strikes) != 3:
                            logger.error(f"Invalid strikes format: {row['Strikes']}")
                            st.error("Invalid strikes format.")
                            continue
                        long_low = next((opt for opt in nearest_calls if opt["strike"] == strikes[0]), None)
                        short_mid = next((opt for opt in nearest_calls if opt["strike"] == strikes[1]), None)
                        long_high = next((opt for opt in nearest_calls if opt["strike"] == strikes[2]), None)
                        if long_low and short_mid and long_high:
                            result = calculate_call_butterfly(long_low, short_mid, long_high, num_contracts, commission_rate)
                            if result:
                                result["contract_ratios"] = [1, -2, 1]
                                visualize_neutral_3d(
                                    result, current_price, expiration_days, st.session_state.iv,
                                    f"Call Butterfly {row['Strikes']}", 
                                    [long_low, short_mid, long_high], ["buy", "sell", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Call Butterfly")
                            else:
                                st.warning("Invalid calculation for this combination.")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Datos de opciones no disponibles para esta combinación.")
                        # Reset checkbox
                        st.session_state["visualize_flags_call_butterfly"][idx] = False
                        st.rerun()

        # Store DataFrame in session state
        st.session_state["call_butterfly_df"] = edited_df.copy()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_call_butterfly"][idx]

        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(row['upper_breakeven'] / current_price) - risk_free_rate * expiration_days / 365.0) / 
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

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
    else:
        st.warning("No hay datos disponibles para Call Butterfly. Asegúrese de que hay suficientes opciones call.")
        logger.warning(f"No data for Call Butterfly. Filtered calls: {len(nearest_calls)}, Expiration: {st.session_state.selected_exp}")

# Put Butterfly
with tab2:
    st.header("Put Butterfly (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_put = create_neutral_table(
            nearest_puts, calculate_put_butterfly, num_contracts, commission_rate, num_legs=3
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Put Butterfly: {e}")
        logger.error(f"Error in create_neutral_table: {e}")
        detailed_df_put = pd.DataFrame()

    if not detailed_df_put.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_put.copy()
        for col in ["net_cost", "max_profit", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}",
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
            edited_rows = edited.get('edited_rows', {})
            edited_df_local = st.session_state.get("put_butterfly_df", edited_df)
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df_local):
                    row = edited_df_local.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        strikes = [float(s) for s in row['Strikes'].split('-')]
                        if len(strikes) != 3:
                            logger.error(f"Invalid strikes format: {row['Strikes']}")
                            st.error("Invalid strikes format.")
                            continue
                        long_low = next((opt for opt in nearest_puts if opt["strike"] == strikes[0]), None)
                        short_mid = next((opt for opt in nearest_puts if opt["strike"] == strikes[1]), None)
                        long_high = next((opt for opt in nearest_puts if opt["strike"] == strikes[2]), None)
                        if long_low and short_mid and long_high:
                            result = calculate_put_butterfly(long_high, short_mid, long_low, num_contracts, commission_rate)
                            if result:
                                result["contract_ratios"] = [1, -2, 1]
                                visualize_neutral_3d(
                                    result, current_price, expiration_days, st.session_state.iv,
                                    f"Put Butterfly {row['Strikes']}", 
                                    [long_high, short_mid, long_low], ["buy", "sell", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Put Butterfly")
                            else:
                                st.warning("Invalid calculation for this combination.")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Datos de opciones no disponibles para esta combinación.")
                        # Reset checkbox
                        st.session_state["visualize_flags_put_butterfly"][idx] = False
                        st.rerun()

        # Store DataFrame in session state
        st.session_state["put_butterfly_df"] = edited_df.copy()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_put_butterfly"][idx]

        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(row['upper_breakeven'] / current_price) - risk_free_rate * expiration_days / 365.0) / 
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

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
    else:
        st.warning("No hay datos disponibles para Put Butterfly. Asegúrese de que hay suficientes opciones put.")
        logger.warning(f"No data for Put Butterfly. Filtered puts: {len(nearest_puts)}, Expiration: {st.session_state.selected_exp}")

# Call Condor
with tab3:
    st.header("Call Condor (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_call_condor = create_neutral_table(
            nearest_calls, calculate_call_condor, num_contracts, commission_rate, num_legs=4
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Call Condor: {e}")
        logger.error(f"Error in create_neutral_table: {e}")
        detailed_df_call_condor = pd.DataFrame()

    if not detailed_df_call_condor.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_call_condor.copy()
        for col in ["net_cost", "max_profit", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}-{row['level_3']}",
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
            edited_rows = edited.get('edited_rows', {})
            edited_df_local = st.session_state.get("call_condor_df", edited_df)
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df_local):
                    row = edited_df_local.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        strikes = [float(s) for s in row['Strikes'].split('-')]
                        if len(strikes) != 4:
                            logger.error(f"Invalid strikes format: {row['Strikes']}")
                            st.error("Invalid strikes format.")
                            continue
                        long_low = next((opt for opt in nearest_calls if opt["strike"] == strikes[0]), None)
                        short_mid_low = next((opt for opt in nearest_calls if opt["strike"] == strikes[1]), None)
                        short_mid_high = next((opt for opt in nearest_calls if opt["strike"] == strikes[2]), None)
                        long_high = next((opt for opt in nearest_calls if opt["strike"] == strikes[3]), None)
                        if long_low and short_mid_low and short_mid_high and long_high:
                            result = calculate_call_condor(long_low, short_mid_low, short_mid_high, long_high, num_contracts, commission_rate)
                            if result:
                                result["contract_ratios"] = [1, -1, -1, 1]
                                visualize_neutral_3d(
                                    result, current_price, expiration_days, st.session_state.iv,
                                    f"Call Condor {row['Strikes']}", 
                                    [long_low, short_mid_low, short_mid_high, long_high], ["buy", "sell", "sell", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Call Condor")
                            else:
                                st.warning("Invalid calculation for this combination.")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Datos de opciones no disponibles para esta combinación.")
                        # Reset checkbox
                        st.session_state["visualize_flags_call_condor"][idx] = False
                        st.rerun()

        # Store DataFrame in session state
        st.session_state["call_condor_df"] = edited_df.copy()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_call_condor"][idx]

        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(row['upper_breakeven'] / current_price) - risk_free_rate * expiration_days / 365.0) / 
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

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
    else:
        st.warning("No hay datos disponibles para Call Condor. Asegúrese de que hay suficientes opciones call.")
        logger.warning(f"No data for Call Condor. Filtered calls: {len(nearest_calls)}, Expiration: {st.session_state.selected_exp}")

# Put Condor
with tab4:
    st.header("Put Condor (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_put_condor = create_neutral_table(
            nearest_puts, calculate_put_condor, num_contracts, commission_rate, num_legs=4
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Put Condor: {e}")
        logger.error(f"Error in create_neutral_table: {e}")
        detailed_df_put_condor = pd.DataFrame()

    if not detailed_df_put_condor.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_put_condor.copy()
        for col in ["net_cost", "max_profit", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}-{row['level_3']}",
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
            edited_rows = edited.get('edited_rows', {})
            edited_df_local = st.session_state.get("put_condor_df", edited_df)
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df_local):
                    row = edited_df_local.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        strikes = [float(s) for s in row['Strikes'].split('-')]
                        if len(strikes) != 4:
                            logger.error(f"Invalid strikes format: {row['Strikes']}")
                            st.error("Invalid strikes format.")
                            continue
                        long_low = next((opt for opt in nearest_puts if opt["strike"] == strikes[0]), None)
                        short_mid_low = next((opt for opt in nearest_puts if opt["strike"] == strikes[1]), None)
                        short_mid_high = next((opt for opt in nearest_puts if opt["strike"] == strikes[2]), None)
                        long_high = next((opt for opt in nearest_puts if opt["strike"] == strikes[3]), None)
                        if long_low and short_mid_low and short_mid_high and long_high:
                            result = calculate_put_condor(long_high, short_mid_high, short_mid_low, long_low, num_contracts, commission_rate)
                            if result:
                                result["contract_ratios"] = [1, -1, -1, 1]
                                visualize_neutral_3d(
                                    result, current_price, expiration_days, st.session_state.iv,
                                    f"Put Condor {row['Strikes']}", 
                                    [long_high, short_mid_high, short_mid_low, long_low], ["buy", "sell", "sell", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Put Condor")
                            else:
                                st.warning("Invalid calculation for this combination.")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Datos de opciones no disponibles para esta combinación.")
                        # Reset checkbox
                        st.session_state["visualize_flags_put_condor"][idx] = False
                        st.rerun()

        # Store DataFrame in session state
        st.session_state["put_condor_df"] = edited_df.copy()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_put_condor"][idx]

        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(row['upper_breakeven'] / current_price) - risk_free_rate * expiration_days / 365.0) / 
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

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
    else:
        st.warning("No hay datos disponibles para Put Condor. Asegúrese de que hay suficientes opciones put.")
        logger.warning(f"No data for Put Condor. Filtered puts: {len(nearest_puts)}, Expiration: {st.session_state.selected_exp}")

# Display IV failure warning
if st.session_state["iv_failure_count"] > 10:
    st.error("Multiple IV calibration failures detected. Check option prices or market data.")