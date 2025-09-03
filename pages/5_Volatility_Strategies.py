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
tab1, tab2 = st.tabs(["Straddle", "Strangle"])

# Straddle
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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)

        # Convert index to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df['level_0'].astype(str)  # Convert strike to string
            edited_df = edited_df.drop(columns=['level_0'])
            edited_df = edited_df.reset_index(drop=True)  # Ensure integer indices

        # Add a visualization column
        edited_df['Visualize'] = False

        # Initialize separate state for visualize flags
        if "visualize_flags_straddle" not in st.session_state:
            st.session_state["visualize_flags_straddle"] = [False] * len(edited_df)

        # Define callback function for Straddle
        def visualize_callback_straddle():
            logger.info("Visualize callback triggered for Straddle")
            edited = st.session_state.get("straddle_editor", {})
            edited_rows = edited.get('edited_rows', {})
            edited_df_local = st.session_state.get("straddle_df", edited_df)
            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df_local):
                    row = edited_df_local.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        strike = float(row['Strikes'])
                        call_opt = next((opt for opt in nearest_calls if opt["strike"] == strike), None)
                        put_opt = next((opt for opt in nearest_puts if opt["strike"] == strike), None)
                        if call_opt and put_opt:
                            result = calculate_straddle(call_opt, put_opt, num_contracts, commission_rate)
                            if result:
                                result["contract_ratios"] = [1, 1]
                                visualize_volatility_3d(
                                    result, current_price, expiration_days, st.session_state.iv,
                                    f"Straddle {strike}",
                                    [call_opt, put_opt], ["buy", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Straddle")
                            else:
                                st.warning("Invalid calculation for this combination.")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Datos de opciones no disponibles para esta combinación.")
                        # Reset checkbox
                        st.session_state["visualize_flags_straddle"][idx] = False
                        st.rerun()

        # Store DataFrame in session state
        st.session_state["straddle_df"] = edited_df.copy()

        # Sync visualize flags to DataFrame
        for idx in range(len(edited_df)):
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_straddle"][idx]

        # Sort DataFrame
        if sort_by == "Breakeven Probability":
            edited_df['Breakeven Probability'] = edited_df.apply(
                lambda row: norm.cdf((np.log(float(row['upper_breakeven']) / current_price) - risk_free_rate * expiration_days / 365.0) /
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
            key="straddle_editor",
            on_change=visualize_callback_straddle,
            width='stretch'
        )
        # Update flags after edit
        for idx in range(len(edited_df)):
            st.session_state["visualize_flags_straddle"][idx] = edited_df.at[idx, 'Visualize']
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
        edited_df["Cost-to-Profit Ratio"] = edited_df["Cost-to-Profit Ratio"].apply(lambda x: x)

        # Convert MultiIndex to a simple index
        if isinstance(edited_df.index, pd.MultiIndex):
            edited_df = edited_df.reset_index()
            edited_df['Strikes'] = edited_df.apply(
                lambda row: f"{row['level_0']}-{row['level_1']}",
                axis=1
            )
            edited_df = edited_df.drop(columns=['level_0', 'level_1'])

        # Add a visualization column
        edited_df['Visualize'] = False

        # Initialize separate state for visualize flags
        if "visualize_flags_strangle" not in st.session_state:
            st.session_state["visualize_flags_strangle"] = [False] * len(edited_df)

        # Define callback function for Strangle
        def visualize_callback_strangle():
            logger.info("Visualize callback triggered for Strangle")
            edited = st.session_state.get("strangle_editor", {})
            edited_rows = edited.get('edited_rows', {})
            edited_df_local = st.session_state.get("strangle_df", edited_df)

            for idx in edited_rows:
                if isinstance(idx, int) and 0 <= idx < len(edited_df_local):
                    row = edited_df_local.iloc[idx]
                    visualize_state = edited_rows[idx].get('Visualize', False)
                    if visualize_state:
                        strikes = [float(s) for s in row['Strikes'].split('-')]
                        if len(strikes) != 2:
                            logger.error(f"Invalid strikes format: {row['Strikes']}")
                            st.error("Invalid strikes format.")
                            continue
                        put_opt = next((opt for opt in nearest_puts if opt["strike"] == strikes[0]), None)
                        call_opt = next((opt for opt in nearest_calls if opt["strike"] == strikes[1]), None)
                        if put_opt and call_opt:
                            result = calculate_strangle(put_opt, call_opt, num_contracts, commission_rate)
                            if result:
                                result["contract_ratios"] = [1, 1]
                                visualize_volatility_3d(
                                    result, current_price, expiration_days, st.session_state.iv,
                                    f"Strangle {row['Strikes']}",
                                    [put_opt, call_opt], ["buy", "buy"]
                                )
                                logger.info("3D plot triggered successfully for Strangle")
                            else:
                                st.warning("Invalid calculation for this combination.")
                        else:
                            logger.warning("Options not found for this combination")
                            st.warning("Datos de opciones no disponibles para esta combinación.")
                        # Reset checkbox
                        st.session_state["visualize_flags_strangle"][idx] = False
                        st.rerun()

        # Store DataFrame in session state
        st.session_state["strangle_df"] = edited_df.copy()

        # Sync visualize flags to DataFrame
        for idx in edited_df.index:
            edited_df.at[idx, 'Visualize'] = st.session_state["visualize_flags_strangle"][idx]

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
            key="strangle_editor",
            on_change=visualize_callback_strangle,
            width='stretch'
        )
        # Update flags after edit
        for idx in edited_df.index:
            st.session_state["visualize_flags_strangle"][idx] = edited_df.at[idx, 'Visualize']
    else:
        st.warning("No hay datos disponibles para Strangle. Asegúrese de que hay suficientes opciones call y put.")
        logger.warning(f"No data for Strangle. Filtered calls: {len(nearest_calls)}, puts: {len(nearest_puts)}, Expiration: {st.session_state.selected_exp}")

# Display IV failure warning
if st.session_state["iv_failure_count"] > 10:
    st.error("Multiple IV calibration failures detected. Check option prices or market data.")