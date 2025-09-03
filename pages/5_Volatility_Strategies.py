import streamlit as st
import utils
import logging
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from itertools import product

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Volatility Strategies", layout="wide")
st.title("Estrategias de Volatilidad: Straddle & Strangle")

if 'filtered_calls' not in st.session_state or not st.session_state.filtered_calls:
    st.warning("Por favor, cargue los datos y seleccione los parámetros en la página '1_Home' primero.")
    st.stop()

calls = st.session_state.filtered_calls
puts = st.session_state.filtered_puts
current_price = st.session_state.current_price
expiration_days = st.session_state.expiration_days
risk_free_rate = st.session_state.risk_free_rate
commission_rate = st.session_state.commission_rate

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

# Log price range for debugging
logger.info(f"Sidebar price_range_pct: {price_range_pct}, min_price: {current_price * (1 - price_range_pct):.2f}, max_price: {current_price * (1 + price_range_pct):.2f}")

# Track IV calibration failures
if "iv_failure_count" not in st.session_state:
    st.session_state["iv_failure_count"] = 0

# Initialize session state for visualization
if "visualize_strategy" not in st.session_state:
    st.session_state["visualize_strategy"] = None

# Function to calculate strategy P&L
def calculate_strategy_value(options, actions, contracts, price, T, sigma):
    total_value = 0.0
    use_intrinsic = T <= 1e-6
    for opt, action, num in zip(options, actions, contracts):
        strike = opt["strike"]
        opt_type = opt["type"]
        action_mult = 1 if action == "buy" else -1
        if use_intrinsic:
            value = utils.intrinsic_value(price, strike, opt_type)
        else:
            value = utils.black_scholes(price, strike, T, risk_free_rate, sigma, opt_type)
        total_value += value * action_mult * num * 100
    return total_value

# Function to calculate net cost of strategy
def calculate_strategy_cost(options, actions, contracts):
    total_cost = 0.0
    for opt, action, num in zip(options, actions, contracts):
        price = utils.get_strategy_price(opt, action)
        px_bid = opt.get("px_bid", 0)
        px_ask = opt.get("px_ask", 0)
        if price is None or price <= 0 or px_bid <= 0 or px_ask <= 0 or px_bid > px_ask:
            logger.debug(f"Invalid price for option {opt['symbol']}: price={price}, bid={px_bid}, ask={px_ask}, action={action}")
            return None
        if abs(px_ask - px_bid) / max(px_ask, px_bid) > 0.5:
            logger.debug(f"Wide bid-ask spread for option {opt['symbol']}: bid={px_bid}, ask={px_ask}")
            return None
        base_cost = price * 100 * num * (1 if action == "buy" else -1)
        if abs(base_cost) < 1e-6:
            logger.debug(f"Near-zero base cost for option {opt['symbol']}: base_cost={base_cost}")
            return None
        fees = utils.calculate_fees(abs(base_cost), commission_rate)
        total_cost += base_cost + fees
    if abs(total_cost) < 1e-6:
        logger.debug(f"Total cost too small: {total_cost}")
        return None
    return total_cost

# Function to estimate probability of P&L >= 0
def estimate_breakeven_probability(options, actions, contracts, net_cost, T_years, sigma):
    def payoff_at_expiration(price):
        return calculate_strategy_value(options, actions, contracts, price, 0, sigma) - net_cost

    price_range = np.linspace(current_price * (1 - price_range_pct), current_price * (1 + price_range_pct), 1000)
    payoffs = [payoff_at_expiration(p) for p in price_range]
    breakeven_points = []
    for i in range(1, len(payoffs)):
        if payoffs[i-1] * payoffs[i] <= 0:
            breakeven = price_range[i-1] + (price_range[i] - price_range[i-1]) * (-payoffs[i-1]) / (payoffs[i] - payoffs[i-1])
            breakeven_points.append(breakeven)

    if not breakeven_points:
        return 0.0

    mu = np.log(current_price) + (risk_free_rate - 0.5 * sigma**2) * T_years
    sigma_t = sigma * np.sqrt(T_years)
    prob = 0.0
    sorted_breakevens = sorted(breakeven_points)
    for i in range(0, len(sorted_breakevens) + 1):
        if i == 0:
            lower = current_price * (1 - price_range_pct)
            upper = sorted_breakevens[0] if sorted_breakevens else current_price * (1 + price_range_pct)
        elif i == len(sorted_breakevens):
            lower = sorted_breakevens[-1]
            upper = current_price * (1 + price_range_pct)
        else:
            lower = sorted_breakevens[i-1]
            upper = sorted_breakevens[i]
        mid_point = (lower + upper) / 2
        if payoff_at_expiration(mid_point) >= 0:
            lower_log = np.log(lower) if lower > 0 else -np.inf
            upper_log = np.log(upper) if upper > 0 else np.inf
            prob += norm.cdf((upper_log - mu) / sigma_t) - norm.cdf((lower_log - mu) / sigma_t)
    return prob

# Function to calculate potential profit and loss
def calculate_potential_metrics(options, actions, contracts, net_cost, price_range_pct, sigma):
    min_price = current_price * (1 - price_range_pct)
    max_price = current_price * (1 + price_range_pct)
    price_points = np.linspace(min_price, max_price, 100)
    payoffs = [calculate_strategy_value(options, actions, contracts, p, 0, sigma) - net_cost for p in price_points]
    max_profit = max(payoffs) if payoffs else 0.0
    max_loss = min(payoffs) if payoffs else 0.0
    return max_profit, max_loss

# Function to find breakeven points
def find_breakeven(options, actions, contracts, net_cost, iv_calibrated, min_price, max_price, tolerance=1e-3):
    breakevens = []
    price_range = np.linspace(min_price, max_price, 1000)
    for p in price_range:
        if abs(calculate_strategy_value(options, actions, contracts, p, 0, iv_calibrated) - net_cost) < tolerance:
            breakevens.append(p)
    return breakevens[:2]  # Return up to two breakevens

# Generate all straddle and strangle combinations
def generate_strategy_combinations():
    strategies = []
    common_strikes = sorted(call_strikes_set & put_strikes_set)  # Use sets for intersection
    num_contracts = st.session_state.num_contracts
    T_years = expiration_days / 365.0

    # Straddles (same strike for call and put)
    for strike in common_strikes:
        for strategy_type in ["Long Straddle", "Short Straddle"]:
            call_opt = next((opt for opt in calls if abs(opt["strike"] - strike) < utils.MIN_GAP), None)
            put_opt = next((opt for opt in puts if abs(opt["strike"] - strike) < utils.MIN_GAP), None)
            if not (call_opt and put_opt):
                logger.debug(f"No call/put for strike {strike} in {strategy_type}")
                continue
            options = [call_opt, put_opt]
            actions = ["buy", "buy"] if strategy_type == "Long Straddle" else ["sell", "sell"]
            contracts_list = [num_contracts, num_contracts]
            
            net_cost = calculate_strategy_cost(options, actions, contracts_list)
            if net_cost is None:
                logger.debug(f"Invalid cost for {strategy_type}, strike {strike}")
                continue
                
            raw_net = sum(
                utils.get_strategy_price(opt, action) * num * 100 * (1 if action == "buy" else -1)
                for opt, action, num in zip(options, actions, contracts_list)
            )
            iv_calibrated = utils._calibrate_iv(
                raw_net, current_price, expiration_days,
                lambda p, t, s: calculate_strategy_value(options, actions, contracts_list, p, t, s),
                options, actions
            )
            if iv_calibrated < 0.01 or iv_calibrated > 5.0:
                st.session_state["iv_failure_count"] += 1
                logger.debug(f"Unrealistic IV ({iv_calibrated}) for {strategy_type}, strike {strike}, using fallback IV=0.30")
                iv_calibrated = utils.DEFAULT_IV

            prob = estimate_breakeven_probability(options, actions, contracts_list, net_cost, T_years, iv_calibrated)
            max_profit, max_loss = calculate_potential_metrics(options, actions, contracts_list, net_cost, price_range_pct, iv_calibrated)
            profit_ratio = abs(net_cost / max_profit) if max_profit > 0 else float('inf')
            breakevens = find_breakeven(options, actions, contracts_list, net_cost, iv_calibrated, 
                                     current_price * (1 - price_range_pct), current_price * (1 + price_range_pct))

            strategies.append({
                "Strategy": strategy_type,
                "Strikes": f"{strike:.2f}",
                "Net Cost": net_cost,
                "Max Profit": max_profit,
                "Max Loss": max_loss,
                "Cost-to-Profit Ratio": profit_ratio,
                "Breakeven Probability": prob,
                "Lower Breakeven": breakevens[0] if breakevens else None,
                "Upper Breakeven": breakevens[1] if len(breakevens) > 1 else None,
                "Options": options,
                "Actions": actions,
                "Contracts": contracts_list,
                "IV": iv_calibrated
            })

    # Strangles (put strike < call strike)
    for put_strike, call_strike in product(put_strikes, call_strikes):
        if put_strike >= call_strike:
            continue
        for strategy_type in ["Long Strangle", "Short Strangle"]:
            put_opt = next((opt for opt in puts if abs(opt["strike"] - put_strike) < utils.MIN_GAP), None)
            call_opt = next((opt for opt in calls if abs(opt["strike"] - call_strike) < utils.MIN_GAP), None)
            if not (put_opt and call_opt):
                logger.debug(f"No call/put for put_strike {put_strike}, call_strike {call_strike} in {strategy_type}")
                continue
            options = [put_opt, call_opt]
            actions = ["buy", "buy"] if strategy_type == "Long Strangle" else ["sell", "sell"]
            contracts_list = [num_contracts, num_contracts]

            net_cost = calculate_strategy_cost(options, actions, contracts_list)
            if net_cost is None:
                logger.debug(f"Invalid cost for {strategy_type}, strikes {put_strike}-{call_strike}")
                continue

            raw_net = sum(
                utils.get_strategy_price(opt, action) * num * 100 * (1 if action == "buy" else -1)
                for opt, action, num in zip(options, actions, contracts_list)
            )
            iv_calibrated = utils._calibrate_iv(
                raw_net, current_price, expiration_days,
                lambda p, t, s: calculate_strategy_value(options, actions, contracts_list, p, t, s),
                options, actions
            )
            if iv_calibrated < 0.01 or iv_calibrated > 5.0:
                st.session_state["iv_failure_count"] += 1
                logger.debug(f"Unrealistic IV ({iv_calibrated}) for {strategy_type}, strikes {put_strike}-{call_strike}, using fallback IV=0.30")
                iv_calibrated = utils.DEFAULT_IV

            prob = estimate_breakeven_probability(options, actions, contracts_list, net_cost, T_years, iv_calibrated)
            max_profit, max_loss = calculate_potential_metrics(options, actions, contracts_list, net_cost, price_range_pct, iv_calibrated)
            profit_ratio = abs(net_cost / max_profit) if max_profit > 0 else float('inf')
            breakevens = find_breakeven(options, actions, contracts_list, net_cost, iv_calibrated, 
                                     current_price * (1 - price_range_pct), current_price * (1 + price_range_pct))

            strategies.append({
                "Strategy": strategy_type,
                "Strikes": f"{put_strike:.2f}-{call_strike:.2f}",
                "Net Cost": net_cost,
                "Max Profit": max_profit,
                "Max Loss": max_loss,
                "Cost-to-Profit Ratio": profit_ratio,
                "Breakeven Probability": prob,
                "Lower Breakeven": breakevens[0] if breakevens else None,
                "Upper Breakeven": breakevens[1] if len(breakevens) > 1 else None,
                "Options": options,
                "Actions": actions,
                "Contracts": contracts_list,
                "IV": iv_calibrated
            })

    return strategies

# Create and display the strategy table
st.header("Todas las Combinaciones de Straddle y Strangle")
strategies = generate_strategy_combinations()
if not strategies:
    st.warning("No se encontraron combinaciones válidas de straddle o strangle. Verifique los datos de opciones.")
    st.stop()

df_strategies = pd.DataFrame(strategies)
df_strategies["Visualize"] = False

# Sort the table
sort_column = "Cost-to-Profit Ratio" if sort_by == "Cost-to-Profit Ratio" else "Breakeven Probability"
df_strategies = df_strategies.sort_values(by=sort_column, ascending=(sort_by == "Cost-to-Profit Ratio"))

# Initialize visualize flags
if "visualize_flags" not in st.session_state or len(st.session_state["visualize_flags"]) != len(df_strategies):
    st.session_state["visualize_flags"] = [False] * len(df_strategies)

# Callback for visualization
def visualize_callback():
    edited = st.session_state.get("strategy_editor", {})
    edited_rows = edited.get('edited_rows', {})
    for idx in edited_rows:
        if isinstance(idx, int) and 0 <= idx < len(df_strategies):
            if edited_rows[idx].get('Visualize', False):
                st.session_state["visualize_strategy"] = df_strategies.iloc[idx].to_dict()
                st.session_state["visualize_flags"][idx] = False
                st.rerun()

# Sync visualize flags to DataFrame
for idx in range(len(df_strategies)):
    df_strategies.at[idx, "Visualize"] = st.session_state["visualize_flags"][idx]

# Display the table
st.data_editor(
    df_strategies.style.format({
        "Net Cost": "{:.2f}",
        "Max Profit": "{:.2f}",
        "Max Loss": "{:.2f}",
        "Cost-to-Profit Ratio": "{:.2f}" if df_strategies["Cost-to-Profit Ratio"].ne(float('inf')).any() else "N/A",
        "Breakeven Probability": "{:.1%}",
        "Lower Breakeven": "{:.2f}" if df_strategies["Lower Breakeven"].notna().any() else "N/A",
        "Upper Breakeven": "{:.2f}" if df_strategies["Upper Breakeven"].notna().any() else "N/A",
        "IV": "{:.1%}"
    }),
    column_config={
        "Visualize": st.column_config.CheckboxColumn(
            "Visualize 3D",
            help="Check to generate 3D and 2D P&L plots",
            disabled=False
        ),
        "Options": None,
        "Actions": None,
        "Contracts": None
    },
    key="strategy_editor",
    on_change=visualize_callback,
    hide_index=True
)

if st.session_state["iv_failure_count"] > 0:
    st.warning(f"Se detectaron {st.session_state['iv_failure_count']} problemas de calibración de volatilidad. Se usó una volatilidad por defecto (30%) en algunos casos.")

# Display selected strategy
if st.session_state["visualize_strategy"]:
    strategy = st.session_state["visualize_strategy"]
    options = strategy["Options"]
    actions = strategy["Actions"]
    contracts = strategy["Contracts"]
    net_cost = strategy["Net Cost"]
    iv_calibrated = strategy["IV"]
    strategy_type = strategy["Strategy"]

    st.subheader(f"Detalles de la Estrategia: {strategy_type}")
    option_details = [
        {
            "Symbol": opt["symbol"],
            "Type": opt["type"].capitalize(),
            "Strike": float(opt["strike"]),
            "Action": action.capitalize(),
            "Contracts": int(num),
            "Price": float(utils.get_strategy_price(opt, action)),
            "Base Cost": float(utils.get_strategy_price(opt, action) * 100 * num * (1 if action == "buy" else -1)),
            "Fees": float(utils.calculate_fees(abs(utils.get_strategy_price(opt, action) * 100 * num), commission_rate))
        }
        for opt, action, num in zip(options, actions, contracts)
    ]
    df_options = pd.DataFrame(option_details)
    st.dataframe(
        df_options.style.format({
            "Strike": "{:.2f}",
            "Price": "{:.2f}",
            "Base Cost": "{:.2f}",
            "Fees": "{:.2f}",
            "Contracts": "{:d}"
        })
    )

    st.subheader("Métricas de la Estrategia")
    st.metric("Costo Neto (ARS)", f"{strategy['Net Cost']:.2f}")
    st.metric("Probabilidad de P&L ≥ 0 (%)", f"{strategy['Breakeven Probability']:.1%}")
    st.metric("Volatilidad Implícita Calibrada (%)", f"{strategy['IV']:.1%}")
    st.metric("Ganancia Máxima en Rango (ARS)", f"{strategy['Max Profit']:.2f}")
    st.metric("Pérdida Máxima en Rango (ARS)", f"{strategy['Max Loss']:.2f}")
    st.metric("Ratio Costo Neto / Ganancia Máxima", 
              f"{strategy['Cost-to-Profit Ratio']:.2f}" if strategy['Cost-to-Profit Ratio'] != float('inf') else "N/A")

    # 3D P&L Plot
    st.subheader("P&L 3D")
    min_price = current_price * (1 - price_range_pct)  # Use sidebar price_range_pct
    max_price = current_price * (1 + price_range_pct)  # Use sidebar price_range_pct
    logger.info(f"3D Plot price range: min={min_price:.2f}, max={max_price:.2f}")
    price_points = np.linspace(min_price, max_price, 50)
    time_points_days = np.linspace(0, expiration_days, 20)
    time_points_years = time_points_days / 365.0
    X, Y = np.meshgrid(price_points, time_points_days)
    Z = np.array([
        [calculate_strategy_value(options, actions, contracts, p, t, iv_calibrated) - net_cost for p in price_points]
        for t in time_points_years
    ])

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[
                [0, 'rgb(255, 0, 0)'],
                [0.5, 'rgb(255, 255, 255)'],
                [1, 'rgb(0, 128, 0)']
            ],
            showscale=True,
            colorbar=dict(title="P&L (ARS)"),
        )
    ])

    y_range = [0, expiration_days]
    z_min, z_max = np.min(Z), np.max(Z)
    X_plane = np.array([[current_price, current_price], [current_price, current_price]])
    Y_plane = np.array([[y_range[0], y_range[0]], [y_range[1], y_range[1]]])
    Z_plane = np.array([[z_min, z_max], [z_min, z_max]])
    fig.add_trace(
        go.Surface(
            x=X_plane,
            y=Y_plane,
            z=Z_plane,
            colorscale=[[0, 'rgba(255, 0, 0, 0.3)'], [1, 'rgba(255, 0, 0, 0.3)']],
            showscale=False,
            name="Precio Actual",
            opacity=0.5
        )
    )

    x_range = [min_price, max_price]
    X_plane_horizontal = np.array([[x_range[0], x_range[1]], [x_range[0], x_range[1]]])
    Y_plane_horizontal = np.array([[y_range[0], y_range[0]], [y_range[1], y_range[1]]])
    Z_plane_horizontal = np.array([[0, 0], [0, 0]])
    fig.add_trace(
        go.Surface(
            x=X_plane_horizontal,
            y=Y_plane_horizontal,
            z=Z_plane_horizontal,
            colorscale=[[0, 'rgba(0, 0, 255, 0.3)'], [1, 'rgba(0, 0, 255, 0.3)']],
            showscale=False,
            name="Punto de Equilibrio",
            opacity=0.5
        )
    )

    num_time_ticks = 5
    tick_vals_time = np.linspace(0, expiration_days, num_time_ticks)
    tick_text_time = [f"{int(t)} días" for t in tick_vals_time]
    tick_text_time[0] = "Expiración"

    num_price_ticks = 7
    tick_vals_price = np.linspace(min_price, max_price, num_price_ticks)
    tick_text_price = [f"{p:.2f}" for p in tick_vals_price]

    fig.update_layout(
        title=f"3D P&L para {strategy_type} (IV: {iv_calibrated:.1%})",
        scene=dict(
            xaxis_title="Precio de GGAL (ARS)",
            yaxis_title="Tiempo hasta Vencimiento (Días)",
            zaxis_title="P&L (ARS)",
            xaxis=dict(tickvals=tick_vals_price, ticktext=tick_text_price),
            yaxis=dict(tickvals=tick_vals_time, ticktext=tick_text_time),
            zaxis=dict(range=[z_min, z_max]),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=1000
    )
    fig.add_trace(go.Scatter3d(
        x=[current_price], y=[expiration_days], z=[0],
        mode="markers",
        marker=dict(size=5, color="red"),
        name="Precio Actual"
    ))
    st.plotly_chart(fig, use_container_width=True, key="strategy_3d_plot")

    # 2D Payoff Diagram at Expiration
    st.subheader("Diagrama de P&L al Vencimiento")
    price_points = np.linspace(min_price, max_price, 100)
    payoff_at_expiration = [calculate_strategy_value(options, actions, contracts, p, 0, iv_calibrated) - net_cost for p in price_points]

    fig_2d = go.Figure()
    fig_2d.add_trace(
        go.Scatter(
            x=price_points,
            y=payoff_at_expiration,
            mode="lines",
            name="P&L al Vencimiento",
            line=dict(color="blue")
        )
    )
    fig_2d.add_trace(
        go.Scatter(
            x=[current_price],
            y=[calculate_strategy_value(options, actions, contracts, current_price, 0, iv_calibrated) - net_cost],
            mode="markers",
            name="Precio Actual",
            marker=dict(size=10, color="red")
        )
    )
    fig_2d.update_layout(
        title=f"P&L al Vencimiento para {strategy_type}",
        xaxis_title="Precio de GGAL (ARS)",
        yaxis_title="P&L (ARS)",
        xaxis=dict(tickvals=[min_price, current_price, max_price], ticktext=[f"{min_price:.2f}", f"{current_price:.2f}", f"{max_price:.2f}"]),
        yaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=1),
        showlegend=True,
        height=400
    )

    breakevens = find_breakeven(options, actions, contracts, net_cost, iv_calibrated, min_price, max_price)
    for b in breakevens:
        fig_2d.add_vline(x=b, line_dash="dash", line_color="green", annotation_text=f"Breakeven: {b:.2f}")

    profit_min_price = current_price * (1 - price_range_pct)
    profit_max_price = current_price * (1 + price_range_pct)
    fig_2d.add_vrect(
        x0=profit_min_price, x1=profit_max_price,
        fillcolor="green", opacity=0.1, line_width=0,
        annotation_text="Rango de Ganancia",
        annotation_position="top"
    )

    st.plotly_chart(fig_2d, use_container_width=True, key="strategy_2d_plot")