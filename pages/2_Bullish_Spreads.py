import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from calc_utils import calculate_bull_call_spread, calculate_bull_put_spread
from viz_utils import create_bullish_spread_table, create_spread_matrix, visualize_bullish_3d
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Bullish Spreads", layout="wide")
st.title("Estrategias Alcistas: Spreads")

# Check for required session state
if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página 'Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
current_price = st.session_state.current_price
expiration_days = st.session_state.expiration_days
risk_free_rate = st.session_state.risk_free_rate
commission_rate = st.session_state.commission_rate
num_contracts = st.session_state.num_contracts
min_strike = current_price * (1 - st.session_state.plot_range_pct)
max_strike = current_price * (1 + st.session_state.plot_range_pct)

# Log available strikes and option data for debugging
call_strikes = sorted({opt["strike"] for opt in calls})
put_strikes = sorted({opt["strike"] for opt in puts})
logger.info(f"Available call strikes: {call_strikes}")
logger.info(f"Available put strikes: {put_strikes}")
logger.info(f"Strike range: {min_strike:.2f}-{max_strike:.2f}")
logger.info(f"Calls data: {[{k: v for k, v in opt.items()} for opt in calls]}")
logger.info(f"Puts data: {[{k: v for k, v in opt.items()} for opt in puts]}")

# Tabs for different strategies
tab1, tab2 = st.tabs(["Bull Call Spread", "Bull Put Spread"])

# Bull Call Spread
with tab1:
    st.header("Bull Call Spread (Débito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_call = create_bullish_spread_table(
            calls, calculate_bull_call_spread, num_contracts, commission_rate, is_debit=True
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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2f}")

        # Ensure simple index
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

        # Sort DataFrame
        if "Breakeven Probability" in st.session_state and st.session_state.get("sort_by") == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(float(row['Breakeven']) / current_price) - risk_free_rate * expiration_days / 365.0) /
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

        # Store DataFrame in session state
        st.session_state["bull_call_df"] = edited_df.copy()

        # Display DataFrame with index
        st.write("Select a row by its index to visualize the 3D payoff plot.")
        st.dataframe(edited_df, use_container_width=True, hide_index=False)

        # Create selectbox options with row index
        strike_options = [f"{idx:02d}: {row['Strikes']}" for idx, row in edited_df.iterrows()]
        selected_option = st.selectbox(
            "Select Strikes for 3D Visualization",
            [""] + strike_options,
            key="bull_call_spread_visualize_select"
        )
        # Replace the 'if selected_option:' block in each tab
        # Bull Call Spread tab
        if selected_option:
            try:
                logger.debug(f"Selected option: {selected_option}")
                idx_str, strikes_str = selected_option.split(": ", 1)
                idx = int(idx_str)
                # Parse strikes from the Strikes column value
                if idx < len(edited_df):
                    strikes = [float(s.strip()) for s in strikes_str.split('-')]
                    if len(strikes) != 2:
                        logger.error(f"Invalid strikes format: {strikes_str}")
                        st.error("Formato de strikes inválido. Se esperaban 2 strikes (strike1-strike2).")
                    else:
                        long_strike, short_strike = strikes  # For Bull Call Spread, long is lower, short is higher
                        logger.debug(f"Parsed strikes: long={long_strike}, short={short_strike}")
                        long_opt = next((opt for opt in calls if opt["strike"] == long_strike), None)
                        short_opt = next((opt for opt in calls if opt["strike"] == short_strike), None)
                        if long_opt and short_opt:
                            logger.debug(f"Found options: long={long_opt['strike']}, short={short_opt['strike']}")
                            result = calculate_bull_call_spread(long_opt, short_opt, num_contracts, commission_rate)
                            if result:
                                result["contract_ratios"] = [1, -1]
                                visualize_bullish_3d(
                                    result, current_price, expiration_days, st.session_state.iv,
                                    f"Bull Call Spread {long_strike:.1f}-{short_strike:.1f}",
                                    [long_opt, short_opt], ["buy", "sell"]
                                )
                                logger.info(
                                    f"3D plot generated for Bull Call Spread {long_strike:.1f}-{short_strike:.1f}")
                            else:
                                st.error("Cálculo inválido para esta combinación.")
                                logger.error("Invalid calculation for Bull Call Spread")
                        else:
                            st.error("Datos de opciones no disponibles para esta combinación.")
                            logger.error(f"Options not found: long={long_opt}, short={short_opt}")
                else:
                    logger.error(f"Index {idx} out of range for DataFrame length {len(edited_df)}")
                    st.error("Índice fuera de rango.")
            except ValueError as e:
                st.error(f"Error al procesar los strikes: {e}")
                logger.error(f"Error parsing strikes {selected_option}: {e}")
            except Exception as e:
                st.error(f"Error inesperado: {e}")
                logger.error(f"Unexpected error in selectbox handling: {e}")




    else:
        st.warning("No hay datos disponibles para Bull Call Spread. Asegúrese de que hay suficientes opciones call en el rango seleccionado o intente actualizar los datos.")
        logger.warning(f"No data for Bull Call Spread. Filtered calls: {len(calls)}, Strikes: {min_strike:.2f}-{max_strike:.2f}, Expiration: {st.session_state.selected_exp}")

    st.subheader("Matriz de Costo Neto (Compra en Fila, Venta en Columna)")
    try:
        profit_df, _, _, _ = create_spread_matrix(
            calls, calculate_bull_call_spread, num_contracts, commission_rate, is_debit=True
        )
        if not profit_df.empty:
            st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis'))
        else:
            st.warning("No hay datos disponibles para la matriz de costos.")
            logger.warning("Empty profit_df for Bull Call Spread matrix.")
    except Exception as e:
        st.error(f"Error al crear la matriz de costos: {e}")
        logger.error(f"Error in create_spread_matrix: {e}")

# Bull Put Spread
with tab2:
    st.header("Bull Put Spread (Crédito)")
    st.subheader("Análisis Detallado por Ratio")
    try:
        detailed_df_put = create_bullish_spread_table(
            puts, calculate_bull_put_spread, num_contracts, commission_rate, is_debit=False
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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: f"{x:.2f}")

        # Ensure simple index
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

        # Sort DataFrame
        if "Breakeven Probability" in st.session_state and st.session_state.get("sort_by") == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(float(row['Breakeven']) / current_price) - risk_free_rate * expiration_days / 365.0) /
                                     (st.session_state.iv * np.sqrt(expiration_days / 365.0))), axis=1
            )
            edited_df = edited_df.sort_values(by="Breakeven Probability", ascending=False)
        else:
            edited_df = edited_df.sort_values(by="Cost-to-Profit Ratio")

        # Store DataFrame in session state
        st.session_state["bull_put_df"] = edited_df.copy()

        # Display DataFrame with index
        st.write("Select a row by its index to visualize the 3D payoff plot.")
        st.dataframe(edited_df, use_container_width=True, hide_index=False)

        # Create selectbox options with row index
        strike_options = [f"{idx:02d}: {row['Strikes']}" for idx, row in edited_df.iterrows()]
        selected_option = st.selectbox(
            "Select Strikes for 3D Visualization",
            [""] + strike_options,
            key="bull_put_spread_visualize_select"
        )
        # Replace the 'if selected_option:' block in each tab
        # Bull Put Spread tab
        if selected_option:
            try:
                logger.debug(f"Selected option: {selected_option}")
                idx_str, strikes_str = selected_option.split(": ", 1)
                idx = int(idx_str)
                # Parse strikes from the Strikes column value
                if idx < len(edited_df):
                    strikes = [float(s.strip()) for s in strikes_str.split('-')]
                    if len(strikes) != 2:
                        logger.error(f"Invalid strikes format: {strikes_str}")
                        st.error("Formato de strikes inválido. Se esperaban 2 strikes (strike1-strike2).")
                    else:
                        short_strike, long_strike = strikes  # For Bull Put Spread, short is higher, long is lower
                        logger.debug(f"Parsed strikes: short={short_strike}, long={long_strike}")
                        short_opt = next((opt for opt in puts if opt["strike"] == short_strike), None)
                        long_opt = next((opt for opt in puts if opt["strike"] == long_strike), None)
                        if short_opt and long_opt:
                            logger.debug(f"Found options: short={short_opt['strike']}, long={long_opt['strike']}")
                            result = calculate_bull_put_spread(short_opt, long_opt, num_contracts, commission_rate)
                            if result:
                                result["contract_ratios"] = [-1, 1]
                                visualize_bullish_3d(
                                    result, current_price, expiration_days, st.session_state.iv,
                                    f"Bull Put Spread {short_strike:.1f}-{long_strike:.1f}",
                                    [short_opt, long_opt], ["sell", "buy"]
                                )
                                logger.info(
                                    f"3D plot generated for Bull Put Spread {short_strike:.1f}-{long_strike:.1f}")
                            else:
                                st.error("Cálculo inválido para esta combinación.")
                                logger.error("Invalid calculation for Bull Put Spread")
                        else:
                            st.error("Datos de opciones no disponibles para esta combinación.")
                            logger.error(f"Options not found: short={short_opt}, long={long_opt}")
                else:
                    logger.error(f"Index {idx} out of range for DataFrame length {len(edited_df)}")
                    st.error("Índice fuera de rango.")
            except ValueError as e:
                st.error(f"Error al procesar los strikes: {e}")
                logger.error(f"Error parsing strikes {selected_option}: {e}")
            except Exception as e:
                st.error(f"Error inesperado: {e}")
                logger.error(f"Unexpected error in selectbox handling: {e}")


        st.warning("No hay datos disponibles para Bull Put Spread. Asegúrese de que hay suficientes opciones put en el rango seleccionado o intente actualizar los datos.")
        logger.warning(f"No data for Bull Put Spread. Filtered puts: {len(puts)}, Strikes: {min_strike:.2f}-{max_strike:.2f}, Expiration: {st.session_state.selected_exp}")

    st.subheader("Matriz de Crédito Neto (Venta en Fila, Compra en Columna)")
    try:
        profit_df, _, _, _ = create_spread_matrix(
            puts, calculate_bull_put_spread, num_contracts, commission_rate, is_debit=False
        )
        if not profit_df.empty:
            st.dataframe(profit_df.style.format("{:.2f}").background_gradient(cmap='viridis_r'))
        else:
            st.warning("No hay datos disponibles para la matriz de créditos.")
            logger.warning("Empty profit_df for Bull Put Spread matrix.")
    except Exception as e:
        st.error(f"Error al crear la matriz de créditos: {e}")
        logger.error(f"Error in create_spread_matrix: {e}")