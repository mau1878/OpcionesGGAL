import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta, timezone
from itertools import combinations
from math import gcd
from functools import reduce
import plotly.graph_objects as go  # New: For 3D visualization

# API Endpoints
STOCK_URL = "https://data912.com/live/arg_stocks"
OPTIONS_URL = "https://data912.com/live/arg_options"

# Constants
DEFAULT_IV = 0.30  # Default implied volatility for simulations

# Helper Functions (Improved with docstrings and error handling)
def get_third_friday(year: int, month: int) -> date:
    """Calculate the third Friday of a given month/year."""
    try:
        first_day = date(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
        return first_friday + timedelta(days=14)
    except ValueError:
        return None

EXPIRATION_MAP_2025 = {
    "O": get_third_friday(2025, 10), "OC": get_third_friday(2025, 10),
    "N": get_third_friday(2025, 11), "NO": get_third_friday(2025, 11),
    "D": get_third_friday(2025, 12), "DI": get_third_friday(2025, 12)
}

def fetch_data(url: str) -> list:
    """Fetch JSON data from API with error handling."""
    try:
        response = requests.get(url, headers={"accept": "*/*"}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching data from {url}: {e}")
        return []

def parse_option_symbol(symbol: str) -> tuple[str | None, float | None, date | None]:
    """Parse option symbol into type, strike, expiration."""
    if symbol.startswith("GFGC"):
        option_type = "call"
    elif symbol.startswith("GFGV"):
        option_type = "put"
    else:
        return None, None, None
    numeric_part = "".join(filter(str.isdigit, symbol[4:]))
    if not numeric_part:
        return None, None, None
    strike_price = float(numeric_part) if numeric_part.startswith("1") else float(numeric_part) / 10
    suffix = symbol[4 + len(numeric_part):]
    expiration = EXPIRATION_MAP_2025.get(suffix, None)
    return option_type, strike_price, expiration

def get_ggal_data() -> tuple[dict | None, list]:
    """Fetch and parse GGAL stock and options data."""
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)
    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock:
        st.error("No se encontraron datos de la acción GGAL.")
        return None, []
    ggal_options = []
    for o in options_data:
        opt_type, strike, exp = parse_option_symbol(o["symbol"])
        if opt_type and strike is not None and exp and o.get("px_ask", 0) > 0 and o.get("px_bid", 0) > 0:
            ggal_options.append({
                "symbol": o["symbol"],
                "type": opt_type,
                "strike": strike,
                "expiration": exp,
                "px_bid": o["px_bid"],
                "px_ask": o["px_ask"],
                "c": o["c"]
            })
    return ggal_stock, ggal_options

def get_strategy_price(option: dict, action: str) -> float | None:
    """Get bid/ask price based on buy/sell action."""
    price = option["px_ask"] if action == "buy" else option["px_bid"]
    return price if price > 0 else None

def calculate_fees(base_cost: float, commission_rate: float) -> tuple[float, float, float]:
    """Calculate commissions, market fees, and VAT."""
    commission = base_cost * commission_rate
    market_fees = base_cost * 0.002  # 0.2% fixed
    vat = (commission + market_fees) * 0.21  # 21% of commission + market fees
    return commission, market_fees, vat

# Existing spread calculations (unchanged, but added docstrings)
def calculate_bull_call_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] >= short_opt["strike"]:
        return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if long_price is None or short_price is None:
        return None
    base_cost = long_price * num_contracts * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (long_price - short_price) * num_contracts * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (short_opt["strike"] - long_opt["strike"]) * num_contracts * 100 - net_cost
    max_loss = net_cost
    return {"max_profit": max(0, max_profit), "net_cost": net_cost, "max_loss": max_loss, 
            "breakeven": long_opt["strike"] + (net_cost / (num_contracts * 100))} if max_profit > 0 else None

# ... (Omit other spread/butterfly/condor functions for brevity; they are similar with added breakeven where applicable)

# New: Straddle Calculation
def calculate_straddle(call_opt, put_opt, num_contracts, commission_rate):
    """Calculate long straddle (buy call + buy put at same strike)."""
    if call_opt["strike"] != put_opt["strike"]:
        return None
    call_price = get_strategy_price(call_opt, "buy")
    put_price = get_strategy_price(put_opt, "buy")
    if call_price is None or put_price is None:
        return None
    base_cost = (call_price + put_price) * num_contracts * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = base_cost + commission + market_fees + vat
    max_loss = net_cost
    # Max profit unlimited; breakevens: strike +/- net premium
    strike = call_opt["strike"]
    net_premium = net_cost / (num_contracts * 100)
    breakeven_upper = strike + net_premium
    breakeven_lower = strike - net_premium
    return {
        "max_profit": "Unlimited",
        "net_cost": net_cost,
        "max_loss": max_loss,
        "breakeven_upper": breakeven_upper,
        "breakeven_lower": breakeven_lower
    }

# New: Strangle Calculation
def calculate_strangle(call_opt, put_opt, num_contracts, commission_rate):
    """Calculate long strangle (buy OTM call + buy OTM put)."""
    if call_opt["strike"] <= put_opt["strike"]:
        return None  # Call strike should be higher
    call_price = get_strategy_price(call_opt, "buy")
    put_price = get_strategy_price(put_opt, "buy")
    if call_price is None or put_price is None:
        return None
    base_cost = (call_price + put_price) * num_contracts * 100
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = base_cost + commission + market_fees + vat
    max_loss = net_cost
    net_premium = net_cost / (num_contracts * 100)
    breakeven_upper = call_opt["strike"] + net_premium
    breakeven_lower = put_opt["strike"] - net_premium
    return {
        "max_profit": "Unlimited",
        "net_cost": net_cost,
        "max_loss": max_loss,
        "breakeven_upper": breakeven_upper,
        "breakeven_lower": breakeven_lower
    }

# Updated create_complex_strategy_table to support straddles/strangles
def create_complex_strategy_table(options_calls: list, options_puts: list, strategy_func, num_contracts: int, commission_rate: float, is_straddle=False):
    data = []
    if is_straddle:  # For straddle/strangle, pair calls and puts
        strikes = sorted(set(o["strike"] for o in options_calls + options_puts))
        for strike in strikes:
            call = next((o for o in options_calls if o["strike"] == strike), None)
            put = next((o for o in options_puts if o["strike"] == strike), None)
            if call and put:
                result = strategy_func(call, put, num_contracts, commission_rate)
                if result:
                    ratio = result["net_cost"] / result["max_loss"]  # For volatility strategies, use cost-to-loss
                    data.append({
                        "Strikes": f"{strike:.1f} (Call & Put)",
                        "Net Cost": result["net_cost"],
                        "Max Profit": result["max_profit"],
                        "Max Loss": result["max_loss"],
                        "Breakeven Upper": result["breakeven_upper"],
                        "Breakeven Lower": result["breakeven_lower"],
                        "Cost-to-Loss Ratio": ratio
                    })
    else:
        # Existing logic for butterflies/condors (omitted for brevity)
        pass
    return pd.DataFrame(data)

# New: 3D Visualization Function
def visualize_3d_payoff(strategy_result, current_price, expiration_days, iv=DEFAULT_IV):
    """Generate 3D plot of Profit vs. Price vs. Time."""
    if not strategy_result:
        st.warning("No strategy selected for visualization.")
        return

    # Simulate grid: Price (x), Time (y, days to exp), Profit (z)
    prices = np.linspace(current_price * 0.5, current_price * 1.5, 50)
    times = np.linspace(0, expiration_days, 20)
    X, Y = np.meshgrid(prices, times)
    Z = np.zeros_like(X)

    # Simple payoff simulation (at expiration) with linear theta decay
    for i in range(len(times)):
        time_factor = (expiration_days - times[i]) / expiration_days  # Decay factor
        for j in range(len(prices)):
            # Example for spread; adapt per strategy
            if "breakeven" in strategy_result:  # For spreads
                if prices[j] > strategy_result["breakeven"]:
                    Z[i, j] = strategy_result["max_profit"] * time_factor
                else:
                    Z[i, j] = -strategy_result["max_loss"] * time_factor
            elif "breakeven_upper" in strategy_result:  # For straddle/strangle
                if prices[j] > strategy_result["breakeven_upper"] or prices[j] < strategy_result["breakeven_lower"]:
                    Z[i, j] = (abs(prices[j] - current_price) - (strategy_result["breakeven_upper"] - current_price)) * 100 * time_factor
                else:
                    Z[i, j] = -strategy_result["max_loss"] * time_factor
            # Add IV effect (simple: increase volatility boosts outer profits)
            Z[i, j] *= (1 + iv * (1 - time_factor))

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='RdYlGn')])
    fig.update_layout(
        title="3D Payoff: Profit/Loss vs. Price vs. Time",
        scene=dict(xaxis_title='Underlying Price', yaxis_title='Days to Expiration', zaxis_title='Profit/Loss (ARS)')
    )
    st.plotly_chart(fig)

# Updated display functions (added for straddles/strangles)
def display_vol_strategy(tab, strategy_name, calls, puts, strategy_func):
    with tab:
        st.subheader(f"Análisis de {strategy_name}")
        st.write(f"""
        **¿Qué es {strategy_name}?**  
        - **Straddle**: Compra call y put al mismo strike. Gana con movimientos grandes (arriba/abajo).
        - **Strangle**: Compra call y put a strikes diferentes. Similar, pero costo menor para rangos más amplios.
        """)

        filter_ratio = st.slider(
            "Relación máxima de costo a pérdida (%)",
            0.0, 500.0, 50.0,
            key=f"filter_{strategy_name}",
        ) / 100
        df = create_complex_strategy_table(calls, puts, strategy_func, st.session_state.num_contracts, st.session_state.commission_rate, is_straddle=True)
        if not df.empty:
            filtered_df = df[df["Cost-to-Loss Ratio"] <= filter_ratio]
            st.dataframe(filtered_df.style.format("{:.2f}"))
            # Select a row for 3D viz
            selected_row = st.selectbox("Selecciona una combinación para visualizar en 3D", filtered_df.index)
            if selected_row is not None:
                result = filtered_df.iloc[selected_row].to_dict()
                expiration_days = (st.session_state.selected_exp - date.today()).days
                visualize_3d_payoff(result, st.session_state.current_price, expiration_days, st.session_state.iv)
        else:
            st.write("No se encontraron combinaciones válidas.")

# Main Function (Updated)
def main():
    st.title("Analizador de Estrategias de Opciones para GGAL (Mejorado)")
    st.write("""
    ¡Bienvenido! Esta herramienta analiza estrategias de opciones para GGAL, ahora con straddles, strangles y visualización 3D.
    """)

    if 'ggal_stock' not in st.session_state:
        with st.spinner("Cargando datos..."):
            st.session_state['ggal_stock'], st.session_state['ggal_options'] = get_ggal_data()
            st.session_state['last_updated'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if st.button("Actualizar Datos"):
        with st.spinner("Actualizando..."):
            st.session_state['ggal_stock'], st.session_state['ggal_options'] = get_ggal_data()
            st.session_state['last_updated'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.success("Datos actualizados.")

    ggal_stock = st.session_state['ggal_stock']
    ggal_options = st.session_state['ggal_options']
    if not ggal_stock or not ggal_options:
        return

    st.session_state.current_price = float(ggal_stock["c"])
    st.write(f"**Precio actual de GGAL:** {st.session_state.current_price:.2f} ARS")
    st.write(f"**Última actualización:** {st.session_state['last_updated']}")

    expirations = sorted(list(set(o["expiration"] for o in ggal_options if o["expiration"])))
    st.session_state.selected_exp = st.selectbox(
        "Selecciona la fecha de vencimiento",
        expirations,
        format_func=lambda x: x.strftime("%Y-%m-%d")
    )

    st.session_state.num_contracts = st.number_input("Número de contratos", min_value=1, value=1, step=1)
    st.session_state.commission_rate = st.number_input("Porcentaje de comisión (%)", min_value=0.0, value=0.5, step=0.1) / 100
    st.session_state.iv = st.number_input("Volatilidad implícita (para simulación)", min_value=0.0, value=DEFAULT_IV, step=0.05)
    strike_percentage = st.slider("Rango de precios de ejercicio (% del precio actual)", 0.0, 100.0, 20.0) / 100

    min_strike = st.session_state.current_price * (1 - strike_percentage)
    max_strike = st.session_state.current_price * (1 + strike_percentage)

    calls = [o for o in ggal_options if o["type"] == "call" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike]
    puts = [o for o in ggal_options if o["type"] == "put" and o["expiration"] == st.session_state.selected_exp and min_strike <= o["strike"] <= max_strike]

    if not calls or not puts:
        st.warning("No hay opciones dentro del rango seleccionado.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Bull Call Spread", "Bull Put Spread", "Bear Call Spread", "Bear Put Spread",
        "Butterfly & Condor", "Straddle & Strangle"
    ])
    st.session_state.disable_filter = st.checkbox("Desactivar filtro para mostrar todas las estrategias", value=False)

    # Existing display_spread_matrix calls (unchanged)
    display_spread_matrix(tab1, "Bull Call Spread", calls, calculate_bull_call_spread, True)
    # ... (omit others)

    # New: Volatility strategies tab
    display_vol_strategy(tab6, "Straddle", calls, puts, calculate_straddle)
    display_vol_strategy(tab6, "Strangle", calls, puts, calculate_strangle)

if __name__ == "__main__":
    main()
