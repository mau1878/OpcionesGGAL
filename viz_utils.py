import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from itertools import combinations, product
from calc_utils import black_scholes, intrinsic_value, _calculate_iv, calculate_fees

logger = logging.getLogger(__name__)

def _compute_payoff_grid(strategy_value_func, current_price, expiration_days, iv, net_entry, strikes):
    plot_range_pct = st.session_state.get("plot_range_pct", 0.3)
    min_strike = min(strikes)
    max_strike = max(strikes)
    min_price = min(min_strike * 0.8, current_price * (1 - plot_range_pct))
    max_price = max(max_strike * 1.2, current_price * (1 + plot_range_pct))
    
    prices = np.linspace(min_price, max_price, 50)
    times = np.linspace(expiration_days, 0, 20)  # Reversed: today to expiration
    X, Y = np.meshgrid(prices, times)
    Z = np.zeros_like(X)

    for i in range(len(times)):
        for j in range(len(prices)):
            price = X[i, j]
            T = max(Y[i, j] / 365.0, 1e-9)  # Reversed time
            try:
                Z[i, j] = strategy_value_func(price, T, iv) - net_entry
            except Exception as e:
                logger.warning(f"Error computing payoff at price={price}, T={T}: {e}")
                Z[i, j] = 0
    return X, Y, Z, min_price, max_price, times

def _create_3d_figure(X, Y, Z, title, current_price):
    z_min = np.min(Z) * 1.1 if np.min(Z) < 0 else -np.max(Z) * 0.1
    z_max = np.max(Z) * 1.1 if np.max(Z) > 0 else -np.min(Z) * 0.1
    norm_Z = Z / max(abs(z_min), abs(z_max), 1e-6)  # Normalize for color contrast
    
    z_zero = np.zeros_like(X)
    blue_plane = go.Surface(
        x=X, y=Y, z=z_zero,
        colorscale=[[0, 'rgba(0, 0, 255, 0.2)'], [1, 'rgba(0, 0, 255, 0.2)']],
        showscale=False,
        name="Zero Profit/Loss"
    )
    
    y_min, y_max = np.min(Y), np.max(Y)
    vertices = [
        [current_price, y_min, z_min],
        [current_price, y_max, z_min],
        [current_price, y_min, z_max],
        [current_price, y_max, z_max]
    ]
    i, j, k = [0, 0], [1, 2], [2, 3]
    red_plane = go.Mesh3d(
        x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
        i=i, j=j, k=k, opacity=0.2, color='red', flatshading=True, name="Current GGAL Price"
    )
    
    payoff_surface = go.Surface(
        x=X, y=Y, z=Z, surfacecolor=norm_Z, colorscale='RdYlGn', showscale=True, colorbar=dict(title="P&L (ARS)")
    )
    
    fig = go.Figure(data=[payoff_surface, blue_plane, red_plane])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Underlying Price",
            yaxis_title="Days to Expiration",
            zaxis_title="Profit / Loss",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            yaxis=dict(autorange="reversed")  # Today at front
        ),
        width=800, height=600, margin=dict(l=65, r=50, b=65, t=90)
    )
    return fig

def visualize_bullish_3d(result, current_price, expiration_days, iv, key, options, option_actions):
    raw_net = result["raw_net"]
    net_entry = result["net_cost"]
    num_contracts = result["num_contracts"]
    strikes = result["strikes"]
    r = st.session_state.get("risk_free_rate", 0.50)
    
    def strategy_value(price, T, sigma):
        k1, k2 = min(strikes), max(strikes)
        use_intrinsic = T <= 1e-6
        if "call" in key.lower():
            if use_intrinsic:
                return (intrinsic_value(price, k1, "call") - intrinsic_value(price, k2, "call"))
            else:
                return (black_scholes(price, k1, T, r, sigma, "call") - 
                        black_scholes(price, k2, T, r, sigma, "call"))
        elif "put" in key.lower():
            if use_intrinsic:
                return (-intrinsic_value(price, k2, "put") + intrinsic_value(price, k1, "put"))
            else:
                return (-black_scholes(price, k2, T, r, sigma, "put") + 
                        black_scholes(price, k1, T, r, sigma, "put"))
        return 0.0
    
    iv = _calculate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions)
    iv = max(iv, 1e-9)
    if iv < 1e-6:
        st.warning(f"IV calibration failed for {key}; using fallback value. Plots may be inaccurate.")
    if st.session_state.get("iv_failure_count", 0) > 10:
        st.error("Multiple IV calibration failures detected. Check option prices or market data.")
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        lambda p, t, s: strategy_value(p, t, s) * 100 * num_contracts, 
        current_price, expiration_days, iv, net_entry, strikes
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)

def visualize_bearish_3d(result, current_price, expiration_days, iv, key, options, option_actions):
    raw_net = result["raw_net"]
    net_entry = result["net_cost"]
    num_contracts = result["num_contracts"]
    strikes = result["strikes"]
    r = st.session_state.get("risk_free_rate", 0.50)
    
    def strategy_value(price, T, sigma):
        k1, k2 = min(strikes), max(strikes)
        use_intrinsic = T <= 1e-6
        if "call" in key.lower():
            if use_intrinsic:
                return (-intrinsic_value(price, k1, "call") + intrinsic_value(price, k2, "call"))
            else:
                return (-black_scholes(price, k1, T, r, sigma, "call") + 
                        black_scholes(price, k2, T, r, sigma, "call"))
        elif "put" in key.lower():
            if use_intrinsic:
                return (intrinsic_value(price, k2, "put") - intrinsic_value(price, k1, "put"))
            else:
                return (black_scholes(price, k2, T, r, sigma, "put") - 
                        black_scholes(price, k1, T, r, sigma, "put"))
        return 0.0
    
    iv = _calculate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions)
    iv = max(iv, 1e-9)
    if iv < 1e-6:
        st.warning(f"IV calibration failed for {key}; using fallback value. Plots may be inaccurate.")
    if st.session_state.get("iv_failure_count", 0) > 10:
        st.error("Multiple IV calibration failures detected. Check option prices or market data.")
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        lambda p, t, s: strategy_value(p, t, s) * 100 * num_contracts, 
        current_price, expiration_days, iv, net_entry, strikes
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)

def visualize_neutral_3d(result, current_price, expiration_days, iv, key, options, option_actions):
    raw_net = result["raw_net"]
    net_entry = result["net_cost"]
    num_contracts = result["num_contracts"]
    ratios = result.get("contract_ratios", [])
    strikes = result["strikes"]
    opt_type = "call" if "call" in key.lower() else "put"
    r = st.session_state.get("risk_free_rate", 0.50)
    
    def strategy_value(price, T, sigma):
        use_intrinsic = T <= 1e-6
        if use_intrinsic:
            vals = [intrinsic_value(price, k, opt_type) for k in strikes]
        else:
            vals = [black_scholes(price, k, T, r, sigma, opt_type) for k in strikes]
        return sum(r * v for r, v in zip(ratios, vals)) * 100 * num_contracts
    
    iv = _calculate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions, contract_ratios=ratios)
    iv = max(iv, 1e-9)
    if iv < 1e-6:
        st.warning(f"IV calibration failed for {key}; using fallback value. Plots may be inaccurate.")
    if st.session_state.get("iv_failure_count", 0) > 10:
        st.error("Multiple IV calibration failures detected. Check option prices or market data.")
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        strategy_value, current_price, expiration_days, iv, net_entry, strikes
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)

def visualize_volatility_3d(result, current_price, expiration_days, iv, key, options, option_actions):
    raw_net = result["raw_net"]
    net_entry = result["net_cost"]
    num_contracts = result["num_contracts"]
    strikes = result["strikes"]
    r = st.session_state.get("risk_free_rate", 0.50)
    
    def strategy_value(price, T, sigma):
        use_intrinsic = T <= 1e-6
        if len(strikes) == 1:
            k = strikes[0]
            if use_intrinsic:
                call_val = intrinsic_value(price, k, "call")
                put_val = intrinsic_value(price, k, "put")
            else:
                call_val = black_scholes(price, k, T, r, sigma, "call")
                put_val = black_scholes(price, k, T, r, sigma, "put")
            return (call_val + put_val) * 100 * num_contracts
        else:
            put_k, call_k = min(strikes), max(strikes)
            if use_intrinsic:
                call_val = intrinsic_value(price, call_k, "call")
                put_val = intrinsic_value(price, put_k, "put")
            else:
                call_val = black_scholes(price, call_k, T, r, sigma, "call")
                put_val = black_scholes(price, put_k, T, r, sigma, "put")
            return (call_val + put_val) * 100 * num_contracts
        return 0.0
    
    iv = _calculate_iv(raw_net, current_price, expiration_days, strategy_value, options, option_actions)
    iv = max(iv, 1e-9)
    if iv < 1e-6:
        st.warning(f"IV calibration failed for {key}; using fallback value. Plots may be inaccurate.")
    if st.session_state.get("iv_failure_count", 0) > 10:
        st.error("Multiple IV calibration failures detected. Check option prices or market data.")
    
    X, Y, Z, min_price, max_price, times = _compute_payoff_grid(
        strategy_value, current_price, expiration_days, iv, net_entry, strikes
    )
    
    fig = _create_3d_figure(X, Y, Z, f"3D Payoff for {key} (IV: {iv:.1%})", current_price)
    st.plotly_chart(fig, use_container_width=True, key=key)


logger = logging.getLogger(__name__)

def create_bullish_spread_table(options, calc_func, num_contracts, commission_rate, is_debit=True):
    data = []
    min_strike = st.session_state.current_price * (1 - st.session_state.plot_range_pct)
    max_strike = st.session_state.current_price * (1 + st.session_state.plot_range_pct)
    filtered_options = [
        opt for opt in options
        if min_strike <= opt["strike"] <= max_strike and
           opt.get("px_bid", 0) > 0 and
           opt.get("px_ask", 0) > 0 and
           opt.get("px_ask", 0) >= opt.get("px_bid", 0)
    ]
    logger.info(f"Filtered options for bullish spread: {[opt['strike'] for opt in filtered_options]}")
    if len(filtered_options) < 2:
        logger.warning(f"Insufficient valid options: {len(filtered_options)} options after filtering, min_strike={min_strike:.2f}, max_strike={max_strike:.2f}")
        return pd.DataFrame()

    valid_combinations = 0
    for opt1, opt2 in combinations(filtered_options, 2):
        if calc_func.__name__ == "calculate_bull_call_spread":
            long_opt, short_opt = (opt1, opt2) if opt1["strike"] < opt2["strike"] else (opt2, opt1)
        else:  # calculate_bull_put_spread
            short_opt, long_opt = (opt1, opt2) if opt1["strike"] > opt2["strike"] else (opt2, opt1)
        result = calc_func(long_opt, short_opt, num_contracts, commission_rate)
        if result and isinstance(result, dict) and "lower_breakeven" in result:
            row = {
                "Net Cost" if is_debit else "Net Credit": result["net_cost"] if is_debit else -result["net_cost"],
                "Max Profit": result["max_profit"],
                "Max Loss": result["max_loss"],
                "Breakeven": result["lower_breakeven"],
                "Cost-to-Profit Ratio": result["max_loss"] / result["max_profit"] if result["max_profit"] > 0 else float('inf')
            }
            data.append(((long_opt["strike"], short_opt["strike"]), row))
            valid_combinations += 1
        else:
            logger.debug(f"Skipping invalid result for strikes {long_opt['strike']}-{short_opt['strike']}: {result}")
    if not data:
        logger.warning(f"No valid bullish spreads generated after processing. Valid combinations: {valid_combinations}")
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(dict(data), orient='index')
    logger.debug(f"Initial DataFrame index: {df.index.tolist()}")
    df = df.reset_index()
    logger.debug(f"After reset_index, columns: {df.columns.tolist()}")
    df.columns = ['level_0', 'level_1', 'Net Cost' if is_debit else 'Net Credit', 'Max Profit', 'Max Loss', 'Breakeven', 'Cost-to-Profit Ratio']
    df['Strikes'] = df.apply(lambda row: f"{row['level_0']:.1f}-{row['level_1']:.1f}" if pd.notna(row['level_0']) and pd.notna(row['level_1']) else str(row.name), axis=1)
    df = df.drop(columns=['level_0', 'level_1'])
    logger.debug(f"Final Strikes column: {df['Strikes'].tolist()}")
    return df

def create_bearish_spread_table(options, calc_func, num_contracts, commission_rate, is_debit=True):
    data = []
    min_strike = st.session_state.current_price * (1 - st.session_state.plot_range_pct)
    max_strike = st.session_state.current_price * (1 + st.session_state.plot_range_pct)
    filtered_options = [
        opt for opt in options
        if min_strike <= opt["strike"] <= max_strike and
           opt.get("px_bid", 0) > 0 and
           opt.get("px_ask", 0) > 0 and
           opt.get("px_ask", 0) >= opt.get("px_bid", 0)
    ]
    logger.info(f"Filtered options for bearish spread: {[opt['strike'] for opt in filtered_options]}")
    if len(filtered_options) < 2:
        logger.warning(f"Insufficient valid options: {len(filtered_options)} options after filtering, min_strike={min_strike:.2f}, max_strike={max_strike:.2f}")
        return pd.DataFrame()

    valid_combinations = 0
    for opt1, opt2 in combinations(filtered_options, 2):
        if calc_func.__name__ == "calculate_bear_call_spread":
            short_opt, long_opt = (opt1, opt2) if opt1["strike"] < opt2["strike"] else (opt2, opt1)
        else:  # calculate_bear_put_spread
            long_opt, short_opt = (opt1, opt2) if opt1["strike"] > opt2["strike"] else (opt2, opt1)
        result = calc_func(short_opt, long_opt, num_contracts, commission_rate) if calc_func.__name__ == "calculate_bear_call_spread" else calc_func(long_opt, short_opt, num_contracts, commission_rate)
        if result and isinstance(result, dict) and "lower_breakeven" in result:
            row = {
                "Net Cost" if is_debit else "Net Credit": result["net_cost"] if is_debit else -result["net_cost"],
                "Max Profit": result["max_profit"],
                "Max Loss": result["max_loss"],
                "Breakeven": result["lower_breakeven"],
                "Cost-to-Profit Ratio": result["max_loss"] / result["max_profit"] if result["max_profit"] > 0 else float('inf')
            }
            data.append(((short_opt["strike"], long_opt["strike"]) if calc_func.__name__ == "calculate_bear_call_spread" else (long_opt["strike"], short_opt["strike"]), row))
            valid_combinations += 1
        else:
            logger.debug(f"Skipping invalid result for strikes {short_opt['strike']}-{long_opt['strike']}: {result}")
    if not data:
        logger.warning(f"No valid bearish spreads generated after processing. Valid combinations: {valid_combinations}")
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(dict(data), orient='index')
    logger.debug(f"Initial DataFrame index: {df.index.tolist()}")
    df = df.reset_index()
    logger.debug(f"After reset_index, columns: {df.columns.tolist()}")
    df.columns = ['level_0', 'level_1', 'Net Cost' if is_debit else 'Net Credit', 'Max Profit', 'Max Loss', 'Breakeven', 'Cost-to-Profit Ratio']
    df['Strikes'] = df.apply(lambda row: f"{row['level_0']:.1f}-{row['level_1']:.1f}" if pd.notna(row['level_0']) and pd.notna(row['level_1']) else str(row.name), axis=1)
    df = df.drop(columns=['level_0', 'level_1'])
    logger.debug(f"Final Strikes column: {df['Strikes'].tolist()}")
    return df

def create_neutral_table(options, calc_func, num_contracts, commission_rate, num_legs):
    data = []
    options_sorted = sorted(options, key=lambda o: o["strike"])
    for combo in combinations(options_sorted, num_legs):
        if all(combo[i]["strike"] < combo[i+1]["strike"] for i in range(num_legs-1)):
            if num_legs == 3:
                result = calc_func(combo[0], combo[1], combo[2], num_contracts, commission_rate)
            elif num_legs == 4:
                result = calc_func(combo[0], combo[1], combo[2], combo[3], num_contracts, commission_rate)
            if result and result["max_profit"] > 0:  # Skip unprofitable
                row = {
                    "net_cost": result["net_cost"],
                    "max_profit": result["max_profit"],
                    "max_loss": result["max_loss"],
                    "lower_breakeven": result["lower_breakeven"],
                    "upper_breakeven": result["upper_breakeven"],
                    "Cost-to-Profit Ratio": result["max_loss"] / result["max_profit"] if result["max_profit"] > 0 else float('inf')
                }
                data.append((tuple(c["strike"] for c in combo), row))
    df = pd.DataFrame.from_dict(dict(data), orient='index') if data else pd.DataFrame()
    return df

def create_volatility_table(leg1_options, leg2_options, calc_func, num_contracts, commission_rate):
    data = []
    for opt1, opt2 in product(leg1_options, leg2_options):
        result = calc_func(opt1, opt2, num_contracts, commission_rate)
        if result and result["max_profit"] > 0:  # Skip unprofitable
            row = {
                "net_cost": result["net_cost"],
                "max_loss": result["max_loss"],
                "lower_breakeven": result["lower_breakeven"],
                "upper_breakeven": result["upper_breakeven"],
                "Cost-to-Profit Ratio": result["max_loss"] / result.get("max_profit", 0.01) if result.get("max_profit", 0.01) > 0 else float('inf')
            }
            strikes = (opt1["strike"], opt2["strike"]) if opt1["strike"] != opt2["strike"] else opt1["strike"]
            data.append((strikes, row))
    df = pd.DataFrame.from_dict(dict(data), orient='index') if data else pd.DataFrame()
    return df

def create_spread_matrix(options, calc_func, num_contracts, commission_rate, is_debit=True):
    options_sorted = sorted(options, key=lambda o: o["strike"])
    strikes = [o["strike"] for o in options_sorted]
    profit_df = pd.DataFrame(index=strikes, columns=strikes, dtype=float)
    
    for i in range(len(strikes)):
        for j in range(len(strikes)):
            if i == j:
                continue
            opt1 = options_sorted[i]
            opt2 = options_sorted[j]
            result = calc_func(opt1, opt2, num_contracts, commission_rate)
            if result and result["max_profit"] > 0:  # Skip unprofitable
                if is_debit:
                    val = result["net_cost"]
                else:
                    val = -result["net_cost"]  # Make credit positive
                profit_df.at[opt1["strike"], opt2["strike"]] = val
    
    return profit_df, None, None, None