"""
app.py
------
Dashboard Streamlit: Modelo de Frontera Eficiente de Markowitz aplicado a
los segmentos de Tarjeta de Crédito (TDC) del banco, usando como "activos"
los 7 segmentos de producto (Clasica, Oro, Platino, Black/Infinite,
Empresarial, Recompensas, Estudiantil) y como "rendimientos" los
rendimientos netos mensuales por segmento.

Input:
    - data/BD_Clientes_TDC_Markowitz_complementaria.xlsx (incluido en el repo,
      usado por defecto) o un archivo equivalente subido por el usuario.
      Hojas requeridas:
        * 'Rendimientos_Mensuales_Segmento' -> serie de tiempo de rendimientos
        * 'Clientes' -> base de clientes (opcional, para comparar cartera actual)

Output:
    - Frontera eficiente interactiva, portafolio óptimo (mín. varianza y
      máximo Sharpe), matriz de correlación, y comparación contra la
      cartera actual del banco.

Despliegue: GitHub + Streamlit Community Cloud.
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from markowitz_engine import (
    anualizar_covarianza,
    anualizar_rendimientos,
    frontera_eficiente,
    nube_portafolios_aleatorios,
    pesos_cartera_actual,
    portafolio_maximo_sharpe,
    portafolio_minima_varianza,
)

# ------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Frontera Eficiente Markowitz | TDC",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

AZUL = "#0038A8"
AZUL_CLARO = "#3D8BFF"
NEGRO = "#0A0A0A"
BLANCO = "#FFFFFF"
GRIS = "#8A8F98"
VERDE = "#00C48C"

DEFAULT_DATA_PATH = "data/BD_Clientes_TDC_Markowitz_complementaria.xlsx"

# ------------------------------------------------------------------------
# ESTILO FINTECH: azul + negro, Arial blanco sobre negro, Arial negro sobre blanco
# ------------------------------------------------------------------------
st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: Arial, Helvetica, sans-serif !important;
}}

/* Fondo principal: blanco con texto negro */
[data-testid="stAppViewContainer"] {{
    background-color: {BLANCO};
    color: {NEGRO};
}}
[data-testid="stAppViewContainer"] * {{
    color: {NEGRO};
}}

/* Sidebar: fondo negro con texto blanco */
[data-testid="stSidebar"] {{
    background-color: {NEGRO};
}}
[data-testid="stSidebar"] * {{
    color: {BLANCO} !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: #333333;
}}

/* Header superior */
[data-testid="stHeader"] {{
    background-color: {BLANCO};
}}

/* Títulos con acento azul */
h1, h2, h3 {{
    color: {NEGRO} !important;
    font-weight: 700 !important;
}}
h1 {{
    border-bottom: 3px solid {AZUL};
    padding-bottom: 0.4rem;
}}

/* Banner superior estilo "ticker" bursátil */
.ticker-banner {{
    background-color: {NEGRO};
    color: {BLANCO} !important;
    padding: 10px 18px;
    border-radius: 6px;
    border-left: 6px solid {AZUL_CLARO};
    font-family: Arial, sans-serif;
    margin-bottom: 1.2rem;
}}
.ticker-banner * {{ color: {BLANCO} !important; }}

/* Tarjetas de métricas */
div[data-testid="stMetric"] {{
    background-color: {NEGRO};
    border: 1px solid {AZUL};
    border-radius: 10px;
    padding: 14px 16px;
}}
div[data-testid="stMetric"] * {{
    color: {BLANCO} !important;
}}
div[data-testid="stMetricLabel"] * {{
    color: {AZUL_CLARO} !important;
}}

/* Botones y controles */
.stButton>button, .stDownloadButton>button {{
    background-color: {AZUL};
    color: {BLANCO} !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    background-color: {AZUL_CLARO};
}}

/* Tabs */
button[data-baseweb="tab"] {{
    color: {NEGRO} !important;
    font-weight: 600;
}}
button[aria-selected="true"] {{
    color: {AZUL} !important;
    border-bottom: 3px solid {AZUL} !important;
}}

/* DataFrames */
[data-testid="stDataFrame"] {{
    border: 1px solid {AZUL};
}}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=BLANCO,
    plot_bgcolor=BLANCO,
    font=dict(family="Arial, sans-serif", color=NEGRO),
    xaxis=dict(gridcolor="#E5E5E5", zerolinecolor="#CCCCCC"),
    yaxis=dict(gridcolor="#E5E5E5", zerolinecolor="#CCCCCC"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ------------------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos(archivo) -> dict:
    rendimientos = pd.read_excel(archivo, sheet_name="Rendimientos_Mensuales_Segmento")
    rendimientos["Fecha"] = pd.to_datetime(rendimientos["Fecha"], format="%Y-%m")
    rendimientos = rendimientos.set_index("Fecha").sort_index()

    try:
        clientes = pd.read_excel(archivo, sheet_name="Clientes")
    except Exception:
        clientes = None

    return {"rendimientos": rendimientos, "clientes": clientes}


# ------------------------------------------------------------------------
# SIDEBAR: CONTROLES
# ------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Parámetros del modelo")

    archivo_subido = st.file_uploader(
        "Base de datos (.xlsx)", type=["xlsx"],
        help="Debe contener las hojas 'Rendimientos_Mensuales_Segmento' y 'Clientes'."
    )

    st.markdown("---")
    tasa_libre_riesgo = st.slider(
        "Tasa libre de riesgo anual (%)", 0.0, 15.0, 7.0, 0.25
    ) / 100.0

    permitir_corto = st.checkbox("Permitir ventas en corto (pesos negativos)", value=False)

    n_puntos_frontera = st.slider("Puntos en la frontera eficiente", 20, 150, 60, 5)
    n_simulaciones = st.slider("Portafolios simulados (nube Monte Carlo)", 500, 10000, 4000, 500)

    st.markdown("---")
    segmentos_todos_placeholder = st.empty()

    st.markdown("---")
    st.caption("📊 Modelo de Markowitz · Frontera Eficiente")
    st.caption("Activos = segmentos de tarjeta de crédito")

fuente_datos = archivo_subido if archivo_subido is not None else DEFAULT_DATA_PATH

try:
    datos = cargar_datos(fuente_datos)
except Exception as e:
    st.error(f"No fue posible leer el archivo: {e}")
    st.stop()

rendimientos_df = datos["rendimientos"]
clientes_df = datos["clientes"]

with segmentos_todos_placeholder.container():
    segmentos_seleccionados = st.multiselect(
        "Segmentos incluidos en el modelo",
        options=list(rendimientos_df.columns),
        default=list(rendimientos_df.columns),
    )

if len(segmentos_seleccionados) < 2:
    st.warning("Selecciona al menos 2 segmentos para construir la frontera eficiente.")
    st.stop()

rendimientos_df = rendimientos_df[segmentos_seleccionados]

# ------------------------------------------------------------------------
# ENCABEZADO / TICKER
# ------------------------------------------------------------------------
st.markdown(f"""
<div class="ticker-banner">
    📈 &nbsp;<b>MODELO DE OPTIMIZACIÓN DE PORTAFOLIOS — MARKOWITZ</b>
    &nbsp;|&nbsp; Activos: segmentos de Tarjeta de Crédito
    &nbsp;|&nbsp; Periodo: {rendimientos_df.index.min().strftime('%Y-%m')} a {rendimientos_df.index.max().strftime('%Y-%m')}
    &nbsp;|&nbsp; n = {len(rendimientos_df)} meses
</div>
""", unsafe_allow_html=True)

st.title("Frontera Eficiente de Markowitz · Segmentos TDC")
st.caption("Optimización riesgo-rendimiento sobre los segmentos de tarjeta de crédito como clases de activo.")

# ------------------------------------------------------------------------
# CÁLCULOS
# ------------------------------------------------------------------------
mu = anualizar_rendimientos(rendimientos_df)
cov = anualizar_covarianza(rendimientos_df)

with st.spinner("Calculando frontera eficiente..."):
    frontera_df = frontera_eficiente(mu, cov, n_puntos=n_puntos_frontera,
                                      permitir_ventas_en_corto=permitir_corto)
    nube_df = nube_portafolios_aleatorios(mu, cov, tasa_libre_riesgo,
                                           n_simulaciones=n_simulaciones,
                                           permitir_ventas_en_corto=permitir_corto)
    port_min_var = portafolio_minima_varianza(mu, cov, permitir_ventas_en_corto=permitir_corto)
    port_max_sharpe = portafolio_maximo_sharpe(mu, cov, tasa_libre_riesgo,
                                                permitir_ventas_en_corto=permitir_corto)

cartera_actual = None
if clientes_df is not None and "Segmento_Tarjeta" in clientes_df.columns:
    cartera_actual = pesos_cartera_actual(
        clientes_df, activos_ordenados=list(mu.index)
    )
    cartera_actual = cartera_actual.dropna()
    ret_actual = float(np.dot(cartera_actual.values, mu.reindex(cartera_actual.index).values))
    vol_actual = float(np.sqrt(np.dot(
        cartera_actual.values.T,
        np.dot(cov.loc[cartera_actual.index, cartera_actual.index].values, cartera_actual.values)
    )))

# ------------------------------------------------------------------------
# MÉTRICAS PRINCIPALES
# ------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rendimiento — Mín. Varianza", f"{port_min_var['rendimiento']*100:.2f}%")
c2.metric("Volatilidad — Mín. Varianza", f"{port_min_var['volatilidad']*100:.2f}%")
c3.metric("Rendimiento — Máx. Sharpe", f"{port_max_sharpe['rendimiento']*100:.2f}%")
c4.metric("Sharpe Ratio óptimo", f"{port_max_sharpe['sharpe']:.2f}")

st.markdown("")

# ------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------
tab_frontera, tab_correlacion, tab_cartera, tab_datos = st.tabs(
    ["📈 Frontera Eficiente", "🔗 Correlación / Covarianza", "🏦 Cartera Actual vs. Óptima", "📄 Datos"]
)

# --- TAB 1: FRONTERA EFICIENTE -------------------------------------------
with tab_frontera:
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=nube_df["volatilidad"] * 100, y=nube_df["rendimiento"] * 100,
        mode="markers",
        marker=dict(size=4, color=nube_df["sharpe"], colorscale="Blues",
                    showscale=True, colorbar=dict(title="Sharpe")),
        name="Portafolios simulados",
        opacity=0.55,
    ))

    fig.add_trace(go.Scatter(
        x=frontera_df["volatilidad"] * 100, y=frontera_df["rendimiento"] * 100,
        mode="lines", line=dict(color=NEGRO, width=3),
        name="Frontera eficiente",
    ))

    for activo in mu.index:
        fig.add_trace(go.Scatter(
            x=[np.sqrt(cov.loc[activo, activo]) * 100], y=[mu[activo] * 100],
            mode="markers+text", text=[activo], textposition="top center",
            marker=dict(size=11, color=AZUL, line=dict(color=NEGRO, width=1)),
            name=activo, showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[port_min_var["volatilidad"] * 100], y=[port_min_var["rendimiento"] * 100],
        mode="markers", marker=dict(size=16, color=VERDE, symbol="diamond",
                                     line=dict(color=NEGRO, width=1)),
        name="Mínima varianza",
    ))

    fig.add_trace(go.Scatter(
        x=[port_max_sharpe["volatilidad"] * 100], y=[port_max_sharpe["rendimiento"] * 100],
        mode="markers", marker=dict(size=16, color=AZUL_CLARO, symbol="star",
                                     line=dict(color=NEGRO, width=1)),
        name="Máximo Sharpe (tangente)",
    ))

    if cartera_actual is not None:
        fig.add_trace(go.Scatter(
            x=[vol_actual * 100], y=[ret_actual * 100],
            mode="markers", marker=dict(size=16, color="#E63946", symbol="x",
                                         line=dict(color=NEGRO, width=1)),
            name="Cartera actual del banco",
        ))

    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="Frontera Eficiente de Markowitz",
        xaxis_title="Volatilidad anualizada (%)",
        yaxis_title="Rendimiento esperado anualizado (%)",
        height=600,
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Pesos — Portafolio Mínima Varianza")
        st.dataframe(
            (port_min_var["pesos"] * 100).round(2).rename("Peso (%)").to_frame(),
            use_container_width=True,
        )
    with col_b:
        st.subheader("Pesos — Portafolio Máximo Sharpe")
        st.dataframe(
            (port_max_sharpe["pesos"] * 100).round(2).rename("Peso (%)").to_frame(),
            use_container_width=True,
        )

# --- TAB 2: CORRELACIÓN --------------------------------------------------
with tab_correlacion:
    st.subheader("Matriz de correlación entre segmentos")
    corr = rendimientos_df.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="Blues", zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
    ))
    fig_corr.update_layout(**PLOTLY_TEMPLATE, height=520, margin=dict(t=30))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Rendimiento y riesgo anualizado por segmento")
    resumen = pd.DataFrame({
        "Rendimiento esperado anual (%)": mu * 100,
        "Volatilidad anual (%)": np.sqrt(np.diag(cov)) * 100,
    }).round(2)
    st.dataframe(resumen, use_container_width=True)

# --- TAB 3: CARTERA ACTUAL VS ÓPTIMA -------------------------------------
with tab_cartera:
    if cartera_actual is None:
        st.info("No se encontró la hoja 'Clientes' con la columna 'Segmento_Tarjeta' "
                 "en el archivo cargado, por lo que no es posible comparar contra la cartera actual.")
    else:
        colx, coly, colz = st.columns(3)
        colx.metric("Rendimiento — Cartera actual", f"{ret_actual*100:.2f}%")
        coly.metric("Volatilidad — Cartera actual", f"{vol_actual*100:.2f}%")
        sharpe_actual = (ret_actual - tasa_libre_riesgo) / vol_actual if vol_actual > 0 else 0
        colz.metric("Sharpe — Cartera actual", f"{sharpe_actual:.2f}")

        comparativo = pd.DataFrame({
            "Cartera actual (%)": (cartera_actual * 100).round(2),
            "Mínima varianza (%)": (port_min_var["pesos"].reindex(cartera_actual.index) * 100).round(2),
            "Máximo Sharpe (%)": (port_max_sharpe["pesos"].reindex(cartera_actual.index) * 100).round(2),
        })

        fig_bar = go.Figure()
        for col, color in zip(comparativo.columns, [NEGRO, VERDE, AZUL_CLARO]):
            fig_bar.add_trace(go.Bar(x=comparativo.index, y=comparativo[col], name=col, marker_color=color))
        fig_bar.update_layout(**PLOTLY_TEMPLATE, barmode="group", height=450,
                               title="Composición de cartera: actual vs. óptima", margin=dict(t=50))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(comparativo, use_container_width=True)

        mejora_sharpe = port_max_sharpe["sharpe"] - sharpe_actual
        st.markdown(
            f"**Brecha de eficiencia:** la cartera actual tiene un Sharpe de "
            f"`{sharpe_actual:.2f}` frente a `{port_max_sharpe['sharpe']:.2f}` del portafolio óptimo "
            f"(diferencia de `{mejora_sharpe:.2f}` puntos)."
        )

# --- TAB 4: DATOS ----------------------------------------------------------
with tab_datos:
    st.subheader("Rendimientos mensuales por segmento")
    st.dataframe((rendimientos_df * 100).round(2), use_container_width=True)
    st.download_button(
        "⬇️ Descargar rendimientos (CSV)",
        rendimientos_df.to_csv().encode("utf-8"),
        file_name="rendimientos_mensuales_segmento.csv",
        mime="text/csv",
    )

    if clientes_df is not None:
        st.subheader("Muestra de la base de clientes")
        st.dataframe(clientes_df.head(50), use_container_width=True)

st.markdown("---")
st.caption(
    "Nota metodológica: el modelo asume que los segmentos de TDC se comportan como clases de "
    "activo cuyo 'rendimiento' es el rendimiento neto mensual histórico. Rendimiento y volatilidad "
    "se anualizan multiplicando/escalando por 12 a partir de datos mensuales. Este dashboard es una "
    "herramienta de apoyo a la decisión y no constituye una recomendación de inversión."
)
