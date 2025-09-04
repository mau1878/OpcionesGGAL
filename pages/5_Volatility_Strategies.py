import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from itertools import product
import logging
from calc_utils import calculate_straddle, calculate_strangle, calculate_strategy_cost, black_scholes, intrinsic_value
from viz_utils import create_volatility_table, visualize_volatility_3d

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Volatility Strategies", layout="wide")
st.title("Estrategias de Volatilidad: Straddle & Strangle")

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
tab1, tab2 = st.tabs(["Straddle", "Strangle"])

# Straddle
with tab1:
    st.header("Straddle (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_straddle = create_volatility_table(
            nearest_calls, nearest_puts, calculate_straddle, num_contracts, commission_rate
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Straddle: {e}")
        logger.error(f"Error in create_volatility_table: {e}")
        detailed_df_straddle = pd.DataFrame()

    if not detailed_df_straddle.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_straddle.copy()
        for col in ["net_cost", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2f}")

        # Ensure index is integer-based
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df['level_0'].astype(str)
            edited_df = edited_df.drop(columns=['level_0'])
        else:
            edited_df['Strikes'] = edited_df.index.astype(str)
            edited_df = edited_df.reset_index(drop=True)

        # Calculate Probability of Profit (PoP: prob outside breakevens for volatility strategies)
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
        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(float(row['upper_breakeven']) / current_price) - risk_free_rate * expiration_days / 365.0) /
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

        # Store DataFrame in session state
        st.session_state["straddle_df"] = edited_df.copy()

        # Display DataFrame with index
        st.dataframe(edited_df, use_container_width=True, hide_index=False)

        # Create selectbox options with row index
        strike_options = [f"{idx:02d}: {row['Strikes']}" for idx, row in edited_df.iterrows()]
        selected_option = st.selectbox(
            "Select Strike for 3D Visualization",
            [""] + strike_options,
            key="straddle_visualize_select"
        )
        if selected_option:
            try:
                # Extract index and strike from selected option
                idx_str, strike_str = selected_option.split(": ", 1)
                idx = int(idx_str)
                strike = float(strike_str)
                logger.debug(f"Visualizing Straddle for index: {idx}, strike: {strike}")
                call_opt = next((opt for opt in nearest_calls if opt["strike"] == strike), None)
                put_opt = next((opt for opt in nearest_puts if opt["strike"] == strike), None)
                if call_opt and put_opt:
                    logger.debug(f"Found call: {call_opt['strike']}, put: {put_opt['strike']}")
                    result = calculate_straddle(call_opt, put_opt, num_contracts, commission_rate)
                    if result:
                        result["contract_ratios"] = [1, 1]
                        visualize_volatility_3d(
                            result, current_price, expiration_days, st.session_state.iv,
                            f"Straddle {strike}",
                            [call_opt, put_opt], ["buy", "buy"]
                        )
                        logger.info(f"3D plot generated for Straddle {strike}")
                    else:
                        st.error("Cálculo inválido para esta combinación.")
                        logger.error("Invalid calculation for Straddle")
                else:
                    st.error("Datos de opciones no disponibles para esta combinación.")
                    logger.error(f"Options not found: call={call_opt}, put={put_opt}")
            except ValueError as e:
                st.error(f"Error al procesar el strike: {e}")
                logger.error(f"Error parsing strike {selected_option}: {e}")

    else:
        st.warning("No hay datos disponibles para Straddle. Asegúrese de que hay suficientes opciones call y put.")
        logger.warning(f"No data for Straddle. Filtered calls: {len(nearest_calls)}, puts: {len(nearest_puts)}, Expiration: {st.session_state.selected_exp}")

# Strangle
with tab2:
    st.header("Strangle (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_strangle = create_volatility_table(
            nearest_puts, nearest_calls, calculate_strangle, num_contracts, commission_rate
        )
    except Exception as e:
        st.error(f"Error al crear la tabla de Strangle: {e}")
        logger.error(f"Error in create_volatility_table: {e}")
        detailed_df_strangle = pd.DataFrame()

    if not detailed_df_strangle.empty:
        # Apply formatting to the DataFrame
        edited_df = detailed_df_strangle.copy()
        for col in ["net_cost", "max_loss", "lower_breakeven", "upper_breakeven"]:
            edited_df[col] = edited_df[col].apply(lambda x: f"{x:.2f}")
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2f}")

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}",
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1'])
        else:
            edited_df['Strikes'] = edited_df.index.astype(str)
            edited_df = edited_df.reset_index(drop=True)

        # Calculate Probability of Profit (PoP: prob outside breakevens for volatility strategies)
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
        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(float(row['upper_breakeven']) / current_price) - risk_free_rate * expiration_days / 365.0) /
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

        # Store DataFrame in session state
        st.session_state["strangle_df"] = edited_df.copy()

        # Display DataFrame with index
        st.dataframe(edited_df, use_container_width=True, hide_index=False)

        # Create selectbox options with row index
        strike_options = [f"{idx:02d}: {row['Strikes']}" for idx, row in edited_df.iterrows()]
        selected_option = st.selectbox(
            "Select Strikes for 3D Visualization",
            [""] + strike_options,
            key="strangle_visualize_select"
        )
        if selected_option:
            try:
                # Extract index and strikes from selected option
                idx_str, strikes_str = selected_option.split(": ", 1)
                idx = int(idx_str)
                strikes = [float(s) for s in strikes_str.split('-')]
                if len(strikes) != 2:
                    logger.error(f"Invalid strikes format: {strikes_str}")
                    st.error("Formato de strikes inválido.")
                else:
                    put_strike, call_strike = strikes
                    logger.debug(f"Visualizing Strangle for index: {idx}, strikes: put={put_strike}, call={call_strike}")
                    put_opt = next((opt for opt in nearest_puts if opt["strike"] == put_strike), None)
                    call_opt = next((opt for opt in nearest_calls if opt["strike"] == call_strike), None)
                    if put_opt and call_opt:
                        logger.debug(f"Found put: {put_opt['strike']}, call: {call_opt['strike']}")
                        result = calculate_strangle(put_opt, call_opt, num_contracts, commission_rate)
                        if result:
                            result["contract_ratios"] = [1, 1]
                            visualize_volatility_3d(
                                result, current_price, expiration_days, st.session_state.iv,
                                f"Strangle {strikes_str}",
                                [put_opt, call_opt], ["buy", "buy"]
                            )
                            logger.info(f"3D plot generated for Strangle {strikes_str}")
                        else:
                            st.error("Cálculo inválido para esta combinación.")
                            logger.error("Invalid calculation for Strangle")
                    else:
                        st.error("Datos de opciones no disponibles para esta combinación.")
                        logger.error(f"Options not found: put={put_opt}, call={call_opt}")
            except ValueError as e:
                st.error(f"Error al procesar los strikes: {e}")
                logger.error(f"Error parsing strikes {selected_option}: {e}")

    else:
        st.warning("No hay datos disponibles para Strangle. Asegúrese de que hay suficientes opciones call y put.")
        logger.warning(f"No data for Strangle. Filtered calls: {len(nearest_calls)}, puts: {len(nearest_puts)}, Expiration: {st.session_state.selected_exp}")

# Display IV failure warning
if st.session_state["iv_failure_count"] > 10:
    st.error("Multiple IV calibration failures detected. Check option prices or market data.")