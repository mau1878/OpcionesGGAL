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

# [Other helper functions like get_strategy_price, calculate_fees, etc., remain unchanged]

# Main Function with "Actualizar" Button
def main():
    st.title("Analizador de Estrategias de Opciones para GGAL")
    st.write("""
    ¡Bienvenido! Esta herramienta te ayuda a analizar estrategias de opciones basadas en la acción GGAL. 
    Aquí puedes explorar diferentes combinaciones de opciones "call" y "put" para ver cuánto podrías ganar o perder. 
    Todo está en pesos argentinos (ARS), refleja datos en tiempo real e incluye costos como comisiones, derechos de mercado e IVA.
    """)

    # Initialize session state for data
    if 'ggal_stock' not in st.session_state or 'ggal_options' not in st.session_state:
        st.session_state['ggal_stock'], st.session_state['ggal_options'] = get_ggal_data()
        st.session_state['last_updated'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Add "Actualizar" button
    if st.button("Actualizar"):
        st.session_state['ggal_stock'], st.session_state['ggal_options'] = get_ggal_data()
        st.session_state['last_updated'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.success("Datos actualizados exitosamente.")

    ggal_stock = st.session_state['ggal_stock']
    ggal_options = st.session_state['ggal_options']
    last_updated = st.session_state['last_updated']

    if not ggal_stock or not ggal_options:
        return

    current_price = float(ggal_stock["c"])
    st.write(f"**Precio actual de GGAL:** {current_price:.2f} ARS")
    st.write(f"**Última actualización:** {last_updated}")
    st.write("Este es el precio más reciente de la acción GGAL en el mercado. Haz clic en 'Actualizar' para obtener los datos más recientes.")

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
        help="Indica cuántos contratos quieres usar en cada estrategia. Un contrato equivale a 100 opciones."
    )

    commission_rate = st.number_input(
        "Porcentaje de comisión (%)",
        min_value=0.0,
        value=0.5,
        step=0.1,
        help="Define el porcentaje de comisión que cobra tu broker por cada contrato comprado."
    ) / 100

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
        help="Define un rango alrededor del precio actual de GGAL para filtrar los precios de ejercicio."
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
    Estas son las opciones disponibles según tus ajustes. 'Call' son opciones de compra, y 'put' son opciones de venta.
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

    # [Rest of the display_spread_matrix and display_complex_strategy functions remain unchanged]

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
