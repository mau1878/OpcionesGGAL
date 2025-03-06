import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta, timezone
from itertools import combinations
from math import gcd
from functools import reduce

# API Endpoints
STOCK_URL = "https://data912.com/live/arg_stocks"
OPTIONS_URL = "https://data912.com/live/arg_options"


# Helper Functions
def get_third_friday(year: int, month: int) -> date:
    first_day = date(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
    return first_friday + timedelta(days=14)


EXPIRATION_MAP_2025 = {
    "M": get_third_friday(2025, 3), "MA": get_third_friday(2025, 3),
    "A": get_third_friday(2025, 4), "AB": get_third_friday(2025, 4),
    "J": get_third_friday(2025, 6), "JU": get_third_friday(2025, 6)
}


@st.cache_data
def fetch_data(url: str) -> list:
    response = requests.get(url, headers={"accept": "*/*"}, timeout=10)
    return response.json() if response.status_code == 200 else []


def parse_option_symbol(symbol: str) -> tuple[str | None, float | None, date | None]:
    if symbol.startswith("GFGC"):
        option_type = "call"
    elif symbol.startswith("GFGV"):
        option_type = "put"
    else:
        return None, None, None
    numeric_part = "".join(filter(str.isdigit, symbol[4:]))
    strike_price = float(numeric_part) if numeric_part.startswith("1") else float(numeric_part) / 10
    suffix = symbol[4 + len(numeric_part):]
    expiration = EXPIRATION_MAP_2025.get(suffix, None)
    return option_type, strike_price, expiration


def get_ggal_data() -> tuple[dict | None, list]:
    stock_data = fetch_data(STOCK_URL)
    options_data = fetch_data(OPTIONS_URL)
    ggal_stock = next((s for s in stock_data if s["symbol"] == "GGAL"), None)
    if not ggal_stock:
        st.error("No se encontraron datos de la acción GGAL.")
        return None, []
    ggal_options = []
    for o in options_data:
        opt_type, strike, exp = parse_option_symbol(o["symbol"])
        if opt_type and strike is not None and exp and o["px_ask"] > 0 and o["px_bid"] > 0:
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
    price = option["px_ask"] if action == "buy" else option["px_bid"]
    return price if price is not None and price > 0 else None


def calculate_fees(base_cost: float, commission_rate: float) -> tuple[float, float, float]:
    """
    Calculate commission, market fees, and VAT based on the base cost of contracts.
    - base_cost: Total cost of contracts before fees (in ARS, per 100 options).
    - commission_rate: User-defined percentage (e.g., 0.5% = 0.005).
    Returns: (commission, market_fees, vat)
    """
    commission = base_cost * commission_rate
    market_fees = base_cost * 0.002  # 0.2% fixed
    vat = (commission + market_fees) * 0.21  # 21% of commission + market fees
    return commission, market_fees, vat


# Spread Calculation Functions with Fees
def calculate_bull_call_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] >= short_opt["strike"]:
        return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if long_price is None or short_price is None:
        return None
    base_cost = long_price * num_contracts * 100  # Cost of buying long leg
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (long_price - short_price) * num_contracts * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (short_opt["strike"] - long_opt["strike"]) * num_contracts * 100 - net_cost
    max_loss = net_cost
    return {"max_profit": max(0, max_profit), "net_cost": net_cost, "max_loss": max_loss} if max_profit > 0 else None


def calculate_bull_put_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] >= short_opt["strike"]:
        return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if long_price is None or short_price is None:
        return None
    base_cost = long_price * num_contracts * 100  # Cost of buying long leg
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (long_price - short_price) * num_contracts * 100 + commission + market_fees + vat
    if net_cost >= 0:
        return None
    max_profit = -net_cost
    max_loss = (short_opt["strike"] - long_opt["strike"]) * num_contracts * 100 + net_cost
    return {"max_profit": max_profit, "net_cost": net_cost, "max_loss": max_loss}


def calculate_bear_call_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] <= short_opt["strike"]:
        return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if long_price is None or short_price is None:
        return None
    base_cost = long_price * num_contracts * 100  # Cost of buying long leg
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (long_price - short_price) * num_contracts * 100 + commission + market_fees + vat
    if net_cost >= 0:
        return None
    max_profit = -net_cost
    max_loss = (long_opt["strike"] - short_opt["strike"]) * num_contracts * 100 + net_cost
    return {"max_profit": max_profit, "net_cost": net_cost,
            "max_loss": max(0, max_loss)} if max_profit > 0 and max_loss > 0 else None


def calculate_bear_put_spread(long_opt, short_opt, num_contracts, commission_rate):
    if long_opt["strike"] <= short_opt["strike"]:
        return None
    long_price = get_strategy_price(long_opt, "buy")
    short_price = get_strategy_price(short_opt, "sell")
    if long_price is None or short_price is None:
        return None
    base_cost = long_price * num_contracts * 100  # Cost of buying long leg
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (long_price - short_price) * num_contracts * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (long_opt["strike"] - short_opt["strike"]) * num_contracts * 100 - net_cost
    max_loss = net_cost
    return {"max_profit": max(0, max_profit), "net_cost": net_cost, "max_loss": max_loss}


def calculate_call_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_opt["strike"] < high_opt["strike"]):
        return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_price = get_strategy_price(mid_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if any(p is None for p in [low_price, mid_price, high_price]):
        return None

    gap1 = mid_opt["strike"] - low_opt["strike"]
    gap2 = high_opt["strike"] - mid_opt["strike"]
    g = gcd(int(gap1), int(gap2))
    gap1_units = int(gap1 / g)
    gap2_units = int(gap2 / g)

    if gap1 == gap2:
        low_contracts = num_contracts
        mid_contracts = -2 * num_contracts
        high_contracts = num_contracts
        min_contracts = 1
    else:
        if gap1 < gap2:
            low_contracts = num_contracts * gap2_units
            mid_contracts = -2 * num_contracts * gap2_units
            high_contracts = num_contracts * gap1_units
        else:
            low_contracts = num_contracts * gap2_units
            mid_contracts = -2 * num_contracts * gap2_units
            high_contracts = num_contracts * gap1_units
        min_contracts = max(gap1_units, gap2_units)

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100  # Only buying legs
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (
                           low_price * low_contracts + mid_price * mid_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (mid_opt["strike"] - low_opt["strike"]) * abs(low_contracts) * 100 - net_cost
    max_loss = net_cost
    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts} : {mid_contracts} : {high_contracts}",
        "min_contracts": min_contracts
    }
    return result if max_profit > 0 else None


def calculate_put_butterfly(low_opt, mid_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_opt["strike"] < high_opt["strike"]):
        return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_price = get_strategy_price(mid_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if any(p is None for p in [low_price, mid_price, high_price]):
        return None

    gap1 = mid_opt["strike"] - low_opt["strike"]
    gap2 = high_opt["strike"] - mid_opt["strike"]
    g = gcd(int(gap1), int(gap2))
    gap1_units = int(gap1 / g)
    gap2_units = int(gap2 / g)

    if gap1 == gap2:
        low_contracts = num_contracts
        mid_contracts = -2 * num_contracts
        high_contracts = num_contracts
        min_contracts = 1
    else:
        if gap1 < gap2:
            low_contracts = num_contracts * gap2_units
            mid_contracts = -2 * num_contracts * gap2_units
            high_contracts = num_contracts * gap1_units
        else:
            low_contracts = num_contracts * gap2_units
            mid_contracts = -2 * num_contracts * gap2_units
            high_contracts = num_contracts * gap1_units
        min_contracts = max(gap1_units, gap2_units)

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100  # Only buying legs
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (
                           low_price * low_contracts + mid_price * mid_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (high_opt["strike"] - mid_opt["strike"]) * abs(high_contracts) * 100 - net_cost
    max_loss = net_cost
    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts} : {mid_contracts} : {high_contracts}",
        "min_contracts": min_contracts
    }
    return result if max_profit > 0 else None


def lcm(a, b):
    return abs(a * b) // gcd(int(a), int(b))


def calculate_call_condor(low_opt, mid_low_opt, mid_high_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_low_opt["strike"] < mid_high_opt["strike"] < high_opt["strike"]):
        return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_low_price = get_strategy_price(mid_low_opt, "sell")
    mid_high_price = get_strategy_price(mid_high_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if any(p is None for p in [low_price, mid_low_price, mid_high_price, high_price]):
        return None

    gap1 = mid_low_opt["strike"] - low_opt["strike"]
    gap2 = mid_high_opt["strike"] - mid_low_opt["strike"]
    gap3 = high_opt["strike"] - mid_high_opt["strike"]
    gaps = [gap1, gap2, gap3]
    if gap1 == gap2 == gap3:
        low_contracts = num_contracts
        mid_low_contracts = -num_contracts
        mid_high_contracts = -num_contracts
        high_contracts = num_contracts
        min_contracts = 1
    else:
        lcm_gaps = reduce(lcm, gaps)
        low_contracts = num_contracts * int(lcm_gaps / gap1)
        mid_low_contracts = -num_contracts * int(lcm_gaps / gap1)
        mid_high_contracts = -num_contracts * int(lcm_gaps / gap3)
        high_contracts = num_contracts * int(lcm_gaps / gap3)
        min_contracts = max(1, int(lcm_gaps / min(gaps)))

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100  # Only buying legs
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (low_price * low_contracts + mid_low_price * mid_low_contracts +
                mid_high_price * mid_high_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (mid_high_opt["strike"] - mid_low_opt["strike"]) * abs(mid_low_contracts) * 100 - net_cost
    max_loss = net_cost
    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts} : {mid_low_contracts} : {mid_high_contracts} : {high_contracts}",
        "min_contracts": min_contracts
    }
    return result if max_profit > 0 else None


def calculate_put_condor(low_opt, mid_low_opt, mid_high_opt, high_opt, num_contracts, commission_rate):
    if not (low_opt["strike"] < mid_low_opt["strike"] < mid_high_opt["strike"] < high_opt["strike"]):
        return None
    low_price = get_strategy_price(low_opt, "buy")
    mid_low_price = get_strategy_price(mid_low_opt, "sell")
    mid_high_price = get_strategy_price(mid_high_opt, "sell")
    high_price = get_strategy_price(high_opt, "buy")
    if any(p is None for p in [low_price, mid_low_price, mid_high_price, high_price]):
        return None

    gap1 = mid_low_opt["strike"] - low_opt["strike"]
    gap2 = mid_high_opt["strike"] - mid_low_opt["strike"]
    gap3 = high_opt["strike"] - mid_high_opt["strike"]
    gaps = [gap1, gap2, gap3]
    if gap1 == gap2 == gap3:
        low_contracts = num_contracts
        mid_low_contracts = -num_contracts
        mid_high_contracts = -num_contracts
        high_contracts = num_contracts
        min_contracts = 1
    else:
        lcm_gaps = reduce(lcm, gaps)
        low_contracts = num_contracts * int(lcm_gaps / gap1)
        mid_low_contracts = -num_contracts * int(lcm_gaps / gap1)
        mid_high_contracts = -num_contracts * int(lcm_gaps / gap3)
        high_contracts = num_contracts * int(lcm_gaps / gap3)
        min_contracts = max(1, int(lcm_gaps / min(gaps)))

    base_cost = (low_price * abs(low_contracts) + high_price * abs(high_contracts)) * 100  # Only buying legs
    commission, market_fees, vat = calculate_fees(base_cost, commission_rate)
    net_cost = (low_price * low_contracts + mid_low_price * mid_low_contracts +
                mid_high_price * mid_high_contracts + high_price * high_contracts) * 100 + commission + market_fees + vat
    if net_cost <= 0:
        return None
    max_profit = (mid_high_opt["strike"] - mid_low_opt["strike"]) * abs(mid_low_contracts) * 100 - net_cost
    max_loss = net_cost
    result = {
        "max_profit": max(0, max_profit),
        "net_cost": net_cost,
        "max_loss": max_loss,
        "contracts": f"{low_contracts} : {mid_low_contracts} : {mid_high_contracts} : {high_contracts}",
        "min_contracts": min_contracts
    }
    return result if max_profit > 0 else None


# Matrix and Table Creation
def create_spread_matrix(options: list, strategy_func, num_contracts: int, commission_rate: float, is_bullish: bool):
    strikes = sorted(options, key=lambda x: x["strike"])
    profit_matrix, cost_matrix, ratio_matrix = [], [], []
    for long_opt in strikes:
        profit_row, cost_row, ratio_row = [], [], []
        for short_opt in strikes:
            if strategy_func.__name__ == "calculate_bull_call_spread" and long_opt["strike"] < short_opt["strike"]:
                result = strategy_func(long_opt, short_opt, num_contracts, commission_rate)
            elif strategy_func.__name__ == "calculate_bull_put_spread" and long_opt["strike"] < short_opt["strike"]:
                result = strategy_func(long_opt, short_opt, num_contracts, commission_rate)
            elif strategy_func.__name__ == "calculate_bear_call_spread" and long_opt["strike"] > short_opt["strike"]:
                result = strategy_func(long_opt, short_opt, num_contracts, commission_rate)
            elif strategy_func.__name__ == "calculate_bear_put_spread" and long_opt["strike"] > short_opt["strike"]:
                result = strategy_func(long_opt, short_opt, num_contracts, commission_rate)
            else:
                result = None
            if result:
                profit_row.append(result["max_profit"])
                cost_row.append(result["net_cost"])
                ratio = (result["net_cost"] / result["max_profit"] if is_bullish else
                         -result["net_cost"] / result["max_loss"] if result["net_cost"] < 0 else float('inf')) if \
                result["max_profit"] > 0 and result["max_loss"] > 0 else float('inf')
                ratio_row.append(ratio)
            else:
                profit_row.append(np.nan)
                cost_row.append(np.nan)
                ratio_row.append(np.nan)
        profit_matrix.append(profit_row)
        cost_matrix.append(cost_row)
        ratio_matrix.append(ratio_row)
    strike_labels = [f"{s['strike']:.1f}" for s in strikes]
    return (pd.DataFrame(profit_matrix, columns=strike_labels, index=strike_labels),
            pd.DataFrame(cost_matrix, columns=strike_labels, index=strike_labels),
            pd.DataFrame(ratio_matrix, columns=strike_labels, index=strike_labels))


def create_complex_strategy_table(options: list, strategy_func, num_contracts: int, commission_rate: float,
                                  combo_size: int) -> pd.DataFrame:
    strikes = sorted(options, key=lambda x: x["strike"])
    combos = list(combinations(strikes, combo_size))
    data = []
    for combo in combos:
        if all(combo[i]["strike"] < combo[i + 1]["strike"] for i in range(len(combo) - 1)):
            result = strategy_func(*combo, num_contracts, commission_rate)
            if result and result["max_profit"] > 0:
                if "min_contracts" in result and result["min_contracts"] > 1:
                    if num_contracts >= result["min_contracts"]:
                        ratio = result["net_cost"] / result["max_profit"]
                        data.append({
                            "Strikes": " - ".join(f"{opt['strike']:.1f}" for opt in combo),
                            "Net Cost": result["net_cost"],
                            "Max Profit": result["max_profit"],
                            "Max Loss": result["max_loss"],
                            "Cost-to-Profit Ratio": ratio,
                            "Contracts": result["contracts"]
                        })
                else:
                    ratio = result["net_cost"] / result["max_profit"]
                    data.append({
                        "Strikes": " - ".join(f"{opt['strike']:.1f}" for opt in combo),
                        "Net Cost": result["net_cost"],
                        "Max Profit": result["max_profit"],
                        "Max Loss": result["max_loss"],
                        "Cost-to-Profit Ratio": ratio,
                        "Contracts": result["contracts"]
                    })
    return pd.DataFrame(data)


# Main Function
def main():
    st.title("Analizador de Estrategias de Opciones para GGAL")
    st.write("""
    ¡Bienvenido! Esta herramienta te ayuda a analizar estrategias de opciones basadas en la acción GGAL. 
    Aquí puedes explorar diferentes combinaciones de opciones "call" y "put" para ver cuánto podrías ganar o perder. 
    Todo está en pesos argentinos (ARS), refleja datos en tiempo real e incluye costos como comisiones, derechos de mercado e IVA.
    """)

    ggal_stock, ggal_options = get_ggal_data()
    if not ggal_stock or not ggal_options:
        return

    current_price = float(ggal_stock["c"])
    st.write(f"**Precio actual de GGAL:** {current_price:.2f} ARS")
    st.write("Este es el precio más reciente de la acción GGAL en el mercado.")

    st.write("### Configuración inicial")
    st.write("""
    A continuación, puedes ajustar las opciones para personalizar tu análisis. Estos ajustes determinan qué opciones se mostrarán y cómo se calcularán las estrategias, incluyendo los costos asociados.
    """)

    expirations = sorted(list(set(o["expiration"] for o in ggal_options)))
    selected_exp = st.selectbox(
        "Selecciona la fecha de vencimiento",
        expirations,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        help="Elige la fecha en la que las opciones vencerán. Esto limita las opciones disponibles a esa fecha específica."
    )

    num_contracts = st.number_input(
        "Número de contratos",
        min_value=1,
        value=1,
        step=1,
        help="Indica cuántos contratos quieres usar en cada estrategia. Un contrato equivale a 100 opciones. Si hay diferencias en los saltos entre precios de ejercicio, algunas estrategias avanzadas (como butterflies o condors) podrían requerir un número mínimo mayor."
    )

    commission_rate = st.number_input(
        "Porcentaje de comisión (%)",
        min_value=0.0,
        value=0.5,
        step=0.1,
        help="Define el porcentaje de comisión que cobra tu broker por cada contrato comprado. Por defecto es 0.5%, pero puedes poner 0 si tu broker es 'libre de comisiones'. Esto afecta el costo neto de las estrategias."
    ) / 100

    # Corrected string formatting
    st.write(f"""
    **Costos incluidos:**  
    - **Comisión**: {commission_rate * 100:.2f}% del costo de los contratos comprados (ajustable arriba).  
    - **Derechos de mercado**: 0.2% del costo de los contratos comprados (fijo).  
    - **IVA**: 21% de la suma de comisión y derechos de mercado (fijo).  
    Estos costos se suman al precio de compra de las opciones, aumentando el 'Net Cost' en las estrategias.
    """)

    strike_percentage = st.slider(
        "Rango de precios de ejercicio (% del precio actual)",
        0.0, 100.0, 20.0,
        help="Define un rango alrededor del precio actual de GGAL para filtrar los precios de ejercicio. Por ejemplo, un 20% significa mostrar opciones desde un 20% por debajo hasta un 20% por encima del precio actual."
    ) / 100
    min_strike = current_price * (1 - strike_percentage)
    max_strike = current_price * (1 + strike_percentage)

    calls = [o for o in ggal_options if
             o["type"] == "call" and o["expiration"] == selected_exp and min_strike <= o["strike"] <= max_strike]
    puts = [o for o in ggal_options if
            o["type"] == "put" and o["expiration"] == selected_exp and min_strike <= o["strike"] <= max_strike]

    st.write(f"**Número de opciones call disponibles:** {len(calls)}")
    st.write(f"**Precios de ejercicio de las opciones call:** {[c['strike'] for c in calls]}")
    st.write(f"**Número de opciones put disponibles:** {len(puts)}")
    st.write(f"**Precios de ejercicio de las opciones put:** {[p['strike'] for p in puts]}")
    st.write("""
    Estas son las opciones disponibles según tus ajustes. 'Call' son opciones de compra (esperas que el precio suba), y 'put' son opciones de venta (esperas que el precio baje).
    """)

    if not calls or not puts:
        st.warning("No hay opciones dentro del rango seleccionado.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Bull Call Spread",
        "Bull Put Spread",
        "Bear Call Spread",
        "Bear Put Spread",
        "Butterfly & Condor"
    ])
    disable_filter = st.checkbox(
        "Desactivar filtro para mostrar todas las estrategias",
        value=False,
        help="Si marcas esta casilla, se mostrarán todas las combinaciones posibles sin aplicar filtros adicionales."
    )

    def display_spread_matrix(tab, strategy_name, options, strategy_func, is_bullish):
        with tab:
            st.subheader(f"Matriz de {strategy_name}")
            st.write(f"""
            **¿Qué es {strategy_name}?**  
            - **Bull Call Spread**: Una estrategia optimista con opciones "call". Compras una call a un precio más bajo y vendes otra a un precio más alto. Ganas si el precio sube, pero limitas tus ganancias y pérdidas. Incluye costos de comisión, derechos de mercado e IVA.
            - **Bull Put Spread**: Una estrategia optimista con opciones "put". Vendes una put a un precio más alto y compras otra a un precio más bajo. Ganas un crédito inicial (menos costos) y esperas que el precio suba o se mantenga.
            - **Bear Call Spread**: Una estrategia pesimista con opciones "call". Vendes una call a un precio más bajo y compras otra a un precio más alto. Ganas un crédito (menos costos) y esperas que el precio baje o se mantenga.
            - **Bear Put Spread**: Una estrategia pesimista con opciones "put". Compras una put a un precio más alto y vendes otra a un precio más bajo. Ganas si el precio baja, con ganancias y pérdidas limitadas, ajustadas por costos.
            """)

            if is_bullish:
                filter_ratio = st.slider(
                    "Relación máxima de costo a ganancia (%)",
                    0.0, 500.0, 50.0,
                    key=f"filter_{strategy_name}",
                    help="Filtra las combinaciones para mostrar solo aquellas donde el costo (incluyendo comisiones) dividido por la ganancia máxima es menor a este porcentaje. Ajusta este valor para ver más o menos opciones."
                ) / 100
                label = "Relación costo a ganancia"
            else:
                filter_ratio = st.slider(
                    "Relación mínima de crédito a pérdida (%)",
                    0.0, 100.0, 50.0,
                    key=f"filter_{strategy_name}",
                    help="Filtra las combinaciones para mostrar solo aquellas donde el crédito recibido (después de costos) dividido por la pérdida máxima es mayor a este porcentaje. Ajusta este valor para refinar los resultados."
                ) / 100
                label = "Relación crédito a pérdida"

            profit_df, cost_df, ratio_df = create_spread_matrix(options, strategy_func, num_contracts, commission_rate,
                                                                is_bullish)
            if disable_filter:
                filtered_profit_df = profit_df
            else:
                filtered_profit_df = profit_df.where(
                    ratio_df <= filter_ratio if is_bullish else ratio_df >= filter_ratio, np.nan)

            st.write("**Matriz de ganancia máxima (ARS)**")
            st.write("""
            Esta tabla muestra la ganancia máxima que podrías obtener con cada combinación de precios de ejercicio, después de restar todos los costos (comisión, derechos de mercado e IVA). 
            Los números en verde indican ganancias más altas, mientras que los rojos muestran ganancias menores o pérdidas.
            """)
            st.dataframe(filtered_profit_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn'))

            st.write("**Matriz de costo neto (ARS)**")
            st.write("""
            Aquí ves el costo neto (o crédito recibido) de cada estrategia, incluyendo comisiones, derechos de mercado e IVA. 
            Un número positivo significa que pagas para entrar en la estrategia, mientras que un número negativo indica que recibes dinero al iniciarla. 
            Los colores van de rojo (costos altos) a verde (créditos altos).
            """)
            st.dataframe(cost_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn_r'))

            st.write(f"**Matriz de {label}**")
            st.write("""
            Esta matriz muestra la relación entre costo y ganancia (o crédito y pérdida), considerando todos los costos adicionales. 
            Valores más bajos son mejores para estrategias de costo, y valores más altos son mejores para estrategias de crédito. Usa el filtro arriba para ajustar lo que ves.
            """)
            st.dataframe(ratio_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn'))

    def display_complex_strategy(tab, strategy_name, options, strategy_func, combo_size):
        with tab:
            st.subheader(f"Análisis de {strategy_name}")
            st.write(f"""
            **¿Qué es {strategy_name}?**  
            - **Call Butterfly**: Una estrategia con tres opciones "call" que combina comprar y vender a diferentes precios de ejercicio. 
              Ideal para cuando esperas que el precio se mantenga cerca del precio medio. Puede tener "piernas desiguales" si los saltos entre precios varían.
            - **Put Butterfly**: Similar, pero con opciones "put". Ganas si el precio se estabiliza cerca del medio.
            - **Call Condor**: Usa cuatro opciones "call" para crear un rango más amplio donde esperas que el precio se mantenga. 
              Perfecto para mercados estables con poca volatilidad.
            - **Put Condor**: Igual que el Call Condor, pero con opciones "put". También busca estabilidad.

            **Costos incluidos:** Todos los cálculos reflejan comisión ({commission_rate * 100:.2f}%), derechos de mercado (0.2%) e IVA (21% de comisión + derechos). 
            Si los saltos entre precios de ejercicio no son iguales, ajustamos el número de contratos para balancear la estrategia. 
            Estas combinaciones solo aparecen si eliges suficientes contratos (mínimo indicado en la tabla).
            """)

            filter_ratio = st.slider(
                "Relación máxima de costo a ganancia (%)",
                0.0, 500.0, 50.0,
                key=f"filter_{strategy_name}",
                help="Filtra las combinaciones para mostrar solo aquellas donde el costo (incluyendo comisiones) dividido por la ganancia máxima es menor a este porcentaje."
            ) / 100
            df = create_complex_strategy_table(options, strategy_func, num_contracts, commission_rate, combo_size)
            if not df.empty:
                filtered_df = df[df["Cost-to-Profit Ratio"] <= filter_ratio]
                st.write("""
                **Tabla de resultados:**  
                - **Strikes**: Los precios de ejercicio usados en la estrategia.  
                - **Net Cost**: Costo neto (positivo = pagas, negativo = recibes), incluyendo todos los costos.  
                - **Max Profit**: Ganancia máxima posible en ARS, después de costos.  
                - **Max Loss**: Pérdida máxima posible en ARS, incluyendo costos.  
                - **Cost-to-Profit Ratio**: Relación entre costo y ganancia (menor es mejor).  
                - **Contracts**: Número de contratos por cada pierna (positivo = comprar, negativo = vender).  
                """)
                st.dataframe(filtered_df.style.format({
                    "Net Cost": "{:.2f}", "Max Profit": "{:.2f}", "Max Loss": "{:.2f}", "Cost-to-Profit Ratio": "{:.2f}"
                }))
            else:
                st.write("No se encontraron combinaciones válidas con los ajustes actuales.")

    display_spread_matrix(tab1, "Bull Call Spread", calls, calculate_bull_call_spread, True)
    display_spread_matrix(tab2, "Bull Put Spread", puts, calculate_bull_put_spread, False)
    display_spread_matrix(tab3, "Bear Call Spread", calls, calculate_bear_call_spread, False)
    display_spread_matrix(tab4, "Bear Put Spread", puts, calculate_bear_put_spread, True)
    display_complex_strategy(tab5, "Call Butterfly", calls, calculate_call_butterfly, 3)
    display_complex_strategy(tab5, "Put Butterfly", puts, calculate_put_butterfly, 3)
    display_complex_strategy(tab5, "Call Condor", calls, calculate_call_condor, 4)
    display_complex_strategy(tab5, "Put Condor", puts, calculate_put_condor, 4)


if __name__ == "__main__":
    main()
