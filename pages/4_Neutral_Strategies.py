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
max_strikes = st.sidebar.number_input("Max Strikes to Evaluate", min_value=5, max_value=50, value=15)

# Log price range for debugging
logger.info(f"Sidebar price_range_pct: {price_range_pct}, min_price: {current_price * (1 - price_range_pct):.2f}, max_price: {current_price * (1 + price_range_pct):.2f}")

# Track IV calibration failures
if "iv_failure_count" not in st.session_state:
    st.session_state["iv_failure_count"] = 0

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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2f}")

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}",
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1', 'level_2'])
        else:
            edited_df['Strikes'] = edited_df.index.astype(str)
            edited_df = edited_df.reset_index(drop=True)

        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(float(row['upper_breakeven']) / current_price) - risk_free_rate * expiration_days / 365.0) /
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

        # Calculate Probability of Profit (PoP: prob outside breakevens)
        T = expiration_days / 365.0
        edited_df['Prob of Profit'] = edited_df.apply(
            lambda row: (
                    norm.cdf(  # Prob S_T < lower_breakeven
                        (np.log(float(row['lower_breakeven']) / current_price) + (
                                    -risk_free_rate + 0.5 * st.session_state.iv ** 2) * T) /
                        (st.session_state.iv * np.sqrt(T))
                    ) +
                    (1 - norm.cdf(  # Prob S_T > upper_breakeven
                        (np.log(current_price / float(row['upper_breakeven'])) + (
                                    risk_free_rate - 0.5 * st.session_state.iv ** 2) * T) /
                        (st.session_state.iv * np.sqrt(T))
                    ))
            ), axis=1
        )
        edited_df['Prob of Profit'] = edited_df['Prob of Profit'].apply(lambda x: f"{x * 100:.1f}%")
        # Store DataFrame in session state
        st.session_state["call_butterfly_df"] = edited_df.copy()

        # Display DataFrame with index
        st.dataframe(edited_df, use_container_width=True, hide_index=False)

        # Create selectbox options with row index
        strike_options = [f"{idx:02d}: {row['Strikes']}" for idx, row in edited_df.iterrows()]
        selected_option = st.selectbox(
            "Select Strikes for 3D Visualization",
            [""] + strike_options,
            key="call_butterfly_visualize_select"
        )
        if selected_option:
            try:
                # Extract index and strikes from selected option
                idx_str, strikes_str = selected_option.split(": ", 1)
                idx = int(idx_str)
                strikes = [float(s) for s in strikes_str.split('-')]
                if len(strikes) != 3:
                    logger.error(f"Invalid strikes format: {strikes_str}")
                    st.error("Formato de strikes inválido.")
                else:
                    long_low, short_mid, long_high = strikes
                    logger.debug(f"Visualizing Call Butterfly for index: {idx}, strikes: {strikes}")
                    long_low_opt = next((opt for opt in nearest_calls if opt["strike"] == long_low), None)
                    short_mid_opt = next((opt for opt in nearest_calls if opt["strike"] == short_mid), None)
                    long_high_opt = next((opt for opt in nearest_calls if opt["strike"] == long_high), None)
                    if long_low_opt and short_mid_opt and long_high_opt:
                        logger.debug(f"Found options: long_low={long_low_opt['strike']}, short_mid={short_mid_opt['strike']}, long_high={long_high_opt['strike']}")
                        result = calculate_call_butterfly(long_low_opt, short_mid_opt, long_high_opt, num_contracts, commission_rate)
                        if result:
                            result["contract_ratios"] = [1, -2, 1]
                            visualize_neutral_3d(
                                result, current_price, expiration_days, st.session_state.iv,
                                f"Call Butterfly {strikes_str}",
                                [long_low_opt, short_mid_opt, long_high_opt], ["buy", "sell", "buy"]
                            )
                            logger.info(f"3D plot generated for Call Butterfly {strikes_str}")
                        else:
                            st.error("Cálculo inválido para esta combinación.")
                            logger.error("Invalid calculation for Call Butterfly")
                    else:
                        st.error("Datos de opciones no disponibles para esta combinación.")
                        logger.error(f"Options not found: long_low={long_low_opt}, short_mid={short_mid_opt}, long_high={long_high_opt}")
            except ValueError as e:
                st.error(f"Error al procesar los strikes: {e}")
                logger.error(f"Error parsing strikes {selected_option}: {e}")

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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2f}")

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}",
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1', 'level_2'])
        else:
            edited_df['Strikes'] = edited_df.index.astype(str)
            edited_df = edited_df.reset_index(drop=True)

        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(float(row['upper_breakeven']) / current_price) - risk_free_rate * expiration_days / 365.0) /
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

        # Calculate Probability of Profit (PoP: prob outside breakevens)
        T = expiration_days / 365.0
        edited_df['Prob of Profit'] = edited_df.apply(
            lambda row: (
                    norm.cdf(  # Prob S_T < lower_breakeven
                        (np.log(float(row['lower_breakeven']) / current_price) + (
                                    -risk_free_rate + 0.5 * st.session_state.iv ** 2) * T) /
                        (st.session_state.iv * np.sqrt(T))
                    ) +
                    (1 - norm.cdf(  # Prob S_T > upper_breakeven
                        (np.log(current_price / float(row['upper_breakeven'])) + (
                                    risk_free_rate - 0.5 * st.session_state.iv ** 2) * T) /
                        (st.session_state.iv * np.sqrt(T))
                    ))
            ), axis=1
        )
        edited_df['Prob of Profit'] = edited_df['Prob of Profit'].apply(lambda x: f"{x * 100:.1f}%")
        # Store DataFrame in session state
        st.session_state["put_butterfly_df"] = edited_df.copy()

        # Display DataFrame with index
        st.dataframe(edited_df, use_container_width=True, hide_index=False)

        # Create selectbox options with row index
        strike_options = [f"{idx:02d}: {row['Strikes']}" for idx, row in edited_df.iterrows()]
        selected_option = st.selectbox(
            "Select Strikes for 3D Visualization",
            [""] + strike_options,
            key="put_butterfly_visualize_select"
        )
        if selected_option:
            try:
                # Extract index and strikes from selected option
                idx_str, strikes_str = selected_option.split(": ", 1)
                idx = int(idx_str)
                strikes = [float(s) for s in strikes_str.split('-')]
                if len(strikes) != 3:
                    logger.error(f"Invalid strikes format: {strikes_str}")
                    st.error("Formato de strikes inválido.")
                else:
                    long_low, short_mid, long_high = strikes
                    logger.debug(f"Visualizing Put Butterfly for index: {idx}, strikes: {strikes}")
                    long_low_opt = next((opt for opt in nearest_puts if opt["strike"] == long_low), None)
                    short_mid_opt = next((opt for opt in nearest_puts if opt["strike"] == short_mid), None)
                    long_high_opt = next((opt for opt in nearest_puts if opt["strike"] == long_high), None)
                    if long_low_opt and short_mid_opt and long_high_opt:
                        logger.debug(f"Found options: long_low={long_low_opt['strike']}, short_mid={short_mid_opt['strike']}, long_high={long_high_opt['strike']}")
                        result = calculate_put_butterfly(long_high_opt, short_mid_opt, long_low_opt, num_contracts, commission_rate)
                        if result:
                            result["contract_ratios"] = [1, -2, 1]
                            visualize_neutral_3d(
                                result, current_price, expiration_days, st.session_state.iv,
                                f"Put Butterfly {strikes_str}",
                                [long_high_opt, short_mid_opt, long_low_opt], ["buy", "sell", "buy"]
                            )
                            logger.info(f"3D plot generated for Put Butterfly {strikes_str}")
                        else:
                            st.error("Cálculo inválido para esta combinación.")
                            logger.error("Invalid calculation for Put Butterfly")
                    else:
                        st.error("Datos de opciones no disponibles para esta combinación.")
                        logger.error(f"Options not found: long_low={long_low_opt}, short_mid={short_mid_opt}, long_high={long_high_opt}")
            except ValueError as e:
                st.error(f"Error al procesar los strikes: {e}")
                logger.error(f"Error parsing strikes {selected_option}: {e}")

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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2f}")

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}-{row['level_3']}",
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1', 'level_2', 'level_3'])
        else:
            edited_df['Strikes'] = edited_df.index.astype(str)
            edited_df = edited_df.reset_index(drop=True)

        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(float(row['upper_breakeven']) / current_price) - risk_free_rate * expiration_days / 365.0) /
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

        # Calculate Probability of Profit (PoP: prob outside breakevens)
        T = expiration_days / 365.0
        edited_df['Prob of Profit'] = edited_df.apply(
            lambda row: (
                    norm.cdf(  # Prob S_T < lower_breakeven
                        (np.log(float(row['lower_breakeven']) / current_price) + (
                                    -risk_free_rate + 0.5 * st.session_state.iv ** 2) * T) /
                        (st.session_state.iv * np.sqrt(T))
                    ) +
                    (1 - norm.cdf(  # Prob S_T > upper_breakeven
                        (np.log(current_price / float(row['upper_breakeven'])) + (
                                    risk_free_rate - 0.5 * st.session_state.iv ** 2) * T) /
                        (st.session_state.iv * np.sqrt(T))
                    ))
            ), axis=1
        )
        edited_df['Prob of Profit'] = edited_df['Prob of Profit'].apply(lambda x: f"{x * 100:.1f}%")
        # Store DataFrame in session state
        st.session_state["call_condor_df"] = edited_df.copy()

        # Display DataFrame with index
        st.dataframe(edited_df, use_container_width=True, hide_index=False)

        # Create selectbox options with row index
        strike_options = [f"{idx:02d}: {row['Strikes']}" for idx, row in edited_df.iterrows()]
        selected_option = st.selectbox(
            "Select Strikes for 3D Visualization",
            [""] + strike_options,
            key="call_condor_visualize_select"
        )
        if selected_option:
            try:
                # Extract index and strikes from selected option
                idx_str, strikes_str = selected_option.split(": ", 1)
                idx = int(idx_str)
                strikes = [float(s) for s in strikes_str.split('-')]
                if len(strikes) != 4:
                    logger.error(f"Invalid strikes format: {strikes_str}")
                    st.error("Formato de strikes inválido.")
                else:
                    long_low, short_mid_low, short_mid_high, long_high = strikes
                    logger.debug(f"Visualizing Call Condor for index: {idx}, strikes: {strikes}")
                    long_low_opt = next((opt for opt in nearest_calls if opt["strike"] == long_low), None)
                    short_mid_low_opt = next((opt for opt in nearest_calls if opt["strike"] == short_mid_low), None)
                    short_mid_high_opt = next((opt for opt in nearest_calls if opt["strike"] == short_mid_high), None)
                    long_high_opt = next((opt for opt in nearest_calls if opt["strike"] == long_high), None)
                    if long_low_opt and short_mid_low_opt and short_mid_high_opt and long_high_opt:
                        logger.debug(f"Found options: long_low={long_low_opt['strike']}, short_mid_low={short_mid_low_opt['strike']}, short_mid_high={short_mid_high_opt['strike']}, long_high={long_high_opt['strike']}")
                        result = calculate_call_condor(long_low_opt, short_mid_low_opt, short_mid_high_opt, long_high_opt, num_contracts, commission_rate)
                        if result:
                            result["contract_ratios"] = [1, -1, -1, 1]
                            visualize_neutral_3d(
                                result, current_price, expiration_days, st.session_state.iv,
                                f"Call Condor {strikes_str}",
                                [long_low_opt, short_mid_low_opt, short_mid_high_opt, long_high_opt], ["buy", "sell", "sell", "buy"]
                            )
                            logger.info(f"3D plot generated for Call Condor {strikes_str}")
                        else:
                            st.error("Cálculo inválido para esta combinación.")
                            logger.error("Invalid calculation for Call Condor")
                    else:
                        st.error("Datos de opciones no disponibles para esta combinación.")
                        logger.error(f"Options not found: long_low={long_low_opt}, short_mid_low={short_mid_low_opt}, short_mid_high={short_mid_high_opt}, long_high={long_high_opt}")
            except ValueError as e:
                st.error(f"Error al procesar los strikes: {e}")
                logger.error(f"Error parsing strikes {selected_option}: {e}")

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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2f}")

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}-{row['level_2']}-{row['level_3']}",
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1', 'level_2', 'level_3'])
        else:
            edited_df['Strikes'] = edited_df.index.astype(str)
            edited_df = edited_df.reset_index(drop=True)

        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(float(row['upper_breakeven']) / current_price) - risk_free_rate * expiration_days / 365.0) /
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

        # Calculate Probability of Profit (PoP: prob outside breakevens)
        T = expiration_days / 365.0
        edited_df['Prob of Profit'] = edited_df.apply(
            lambda row: (
                    norm.cdf(  # Prob S_T < lower_breakeven
                        (np.log(float(row['lower_breakeven']) / current_price) + (
                                    -risk_free_rate + 0.5 * st.session_state.iv ** 2) * T) /
                        (st.session_state.iv * np.sqrt(T))
                    ) +
                    (1 - norm.cdf(  # Prob S_T > upper_breakeven
                        (np.log(current_price / float(row['upper_breakeven'])) + (
                                    risk_free_rate - 0.5 * st.session_state.iv ** 2) * T) /
                        (st.session_state.iv * np.sqrt(T))
                    ))
            ), axis=1
        )
        edited_df['Prob of Profit'] = edited_df['Prob of Profit'].apply(lambda x: f"{x * 100:.1f}%")
        # Store DataFrame in session state
        st.session_state["put_condor_df"] = edited_df.copy()

        # Display DataFrame with index
        st.dataframe(edited_df, use_container_width=True, hide_index=False)

        # Create selectbox options with row index
        strike_options = [f"{idx:02d}: {row['Strikes']}" for idx, row in edited_df.iterrows()]
        selected_option = st.selectbox(
            "Select Strikes for 3D Visualization",
            [""] + strike_options,
            key="put_condor_visualize_select"
        )
        if selected_option:
            try:
                # Extract index and strikes from selected option
                idx_str, strikes_str = selected_option.split(": ", 1)
                idx = int(idx_str)
                strikes = [float(s) for s in strikes_str.split('-')]
                if len(strikes) != 4:
                    logger.error(f"Invalid strikes format: {strikes_str}")
                    st.error("Formato de strikes inválido.")
                else:
                    long_low, short_mid_low, short_mid_high, long_high = strikes
                    logger.debug(f"Visualizing Put Condor for index: {idx}, strikes: {strikes}")
                    long_low_opt = next((opt for opt in nearest_puts if opt["strike"] == long_low), None)
                    short_mid_low_opt = next((opt for opt in nearest_puts if opt["strike"] == short_mid_low), None)
                    short_mid_high_opt = next((opt for opt in nearest_puts if opt["strike"] == short_mid_high), None)
                    long_high_opt = next((opt for opt in nearest_puts if opt["strike"] == long_high), None)
                    if long_low_opt and short_mid_low_opt and short_mid_high_opt and long_high_opt:
                        logger.debug(f"Found options: long_low={long_low_opt['strike']}, short_mid_low={short_mid_low_opt['strike']}, short_mid_high={short_mid_high_opt['strike']}, long_high={long_high_opt['strike']}")
                        result = calculate_put_condor(long_high_opt, short_mid_high_opt, short_mid_low_opt, long_low_opt, num_contracts, commission_rate)
                        if result:
                            result["contract_ratios"] = [1, -1, -1, 1]
                            visualize_neutral_3d(
                                result, current_price, expiration_days, st.session_state.iv,
                                f"Put Condor {strikes_str}",
                                [long_high_opt, short_mid_high_opt, short_mid_low_opt, long_low_opt], ["buy", "sell", "sell", "buy"]
                            )
                            logger.info(f"3D plot generated for Put Condor {strikes_str}")
                        else:
                            st.error("Cálculo inválido para esta combinación.")
                            logger.error("Invalid calculation for Put Condor")
                    else:
                        st.error("Datos de opciones no disponibles para esta combinación.")
                        logger.error(f"Options not found: long_low={long_low_opt}, short_mid_low={short_mid_low_opt}, short_mid_high={short_mid_high_opt}, long_high={long_high_opt}")
            except ValueError as e:
                st.error(f"Error al procesar los strikes: {e}")
                logger.error(f"Error parsing strikes {selected_option}: {e}")

    else:
        st.warning("No hay datos disponibles para Put Condor. Asegúrese de que hay suficientes opciones put.")
        logger.warning(f"No data for Put Condor. Filtered puts: {len(nearest_puts)}, Expiration: {st.session_state.selected_exp}")

# Display IV failure warning
if st.session_state["iv_failure_count"] > 10:
    st.error("Multiple IV calibration failures detected. Check option prices or market data.")