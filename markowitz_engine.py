"""
markowitz_engine.py
--------------------
Motor de cálculo del modelo de optimización de portafolios de Markowitz,
aplicado a los segmentos de tarjeta de crédito (TDC) como "activos".

Cada segmento (Clasica, Oro, Platino, Black/Infinite, Empresarial,
Recompensas, Estudiantil) se trata como un activo cuyo "rendimiento"
es el rendimiento neto mensual observado en la hoja
'Rendimientos_Mensuales_Segmento' del archivo fuente.

Este módulo NO depende de Streamlit: contiene únicamente lógica numérica,
para que pueda probarse de forma aislada (unit tests) o reutilizarse
en otros contextos (notebooks, batch jobs, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_PERIODS_PER_YEAR = 12  # los datos son mensuales


def anualizar_rendimientos(rendimientos_mensuales: pd.DataFrame) -> pd.Series:
    """Rendimiento esperado anualizado por activo (media simple * 12)."""
    return rendimientos_mensuales.mean() * TRADING_PERIODS_PER_YEAR


def anualizar_covarianza(rendimientos_mensuales: pd.DataFrame) -> pd.DataFrame:
    """Matriz de covarianza anualizada (covarianza mensual * 12)."""
    return rendimientos_mensuales.cov() * TRADING_PERIODS_PER_YEAR


def rendimiento_portafolio(pesos: np.ndarray, mu: pd.Series) -> float:
    return float(np.dot(pesos, mu.values))


def volatilidad_portafolio(pesos: np.ndarray, cov: pd.DataFrame) -> float:
    var = float(np.dot(pesos.T, np.dot(cov.values, pesos)))
    return float(np.sqrt(max(var, 0.0)))


def sharpe_ratio(pesos: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float) -> float:
    vol = volatilidad_portafolio(pesos, cov)
    if vol == 0:
        return 0.0
    return (rendimiento_portafolio(pesos, mu) - rf) / vol


def _bounds(n_activos: int, permitir_ventas_en_corto: bool):
    if permitir_ventas_en_corto:
        return tuple((-1.0, 1.0) for _ in range(n_activos))
    return tuple((0.0, 1.0) for _ in range(n_activos))


def _pesos_iniciales(n_activos: int) -> np.ndarray:
    return np.repeat(1.0 / n_activos, n_activos)


def portafolio_minima_varianza(mu: pd.Series, cov: pd.DataFrame,
                                permitir_ventas_en_corto: bool = False) -> dict:
    """Portafolio de mínima varianza global, sujeto a suma de pesos = 1."""
    n = len(mu)
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    resultado = minimize(
        lambda w: volatilidad_portafolio(w, cov) ** 2,
        _pesos_iniciales(n),
        method="SLSQP",
        bounds=_bounds(n, permitir_ventas_en_corto),
        constraints=restricciones,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    pesos = resultado.x
    return {
        "pesos": pd.Series(pesos, index=mu.index),
        "rendimiento": rendimiento_portafolio(pesos, mu),
        "volatilidad": volatilidad_portafolio(pesos, cov),
        "exito": resultado.success,
    }


def portafolio_maximo_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float,
                              permitir_ventas_en_corto: bool = False) -> dict:
    """Portafolio tangente: máximo Sharpe ratio."""
    n = len(mu)
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    resultado = minimize(
        lambda w: -sharpe_ratio(w, mu, cov, rf),
        _pesos_iniciales(n),
        method="SLSQP",
        bounds=_bounds(n, permitir_ventas_en_corto),
        constraints=restricciones,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    pesos = resultado.x
    return {
        "pesos": pd.Series(pesos, index=mu.index),
        "rendimiento": rendimiento_portafolio(pesos, mu),
        "volatilidad": volatilidad_portafolio(pesos, cov),
        "sharpe": sharpe_ratio(pesos, mu, cov, rf),
        "exito": resultado.success,
    }


def frontera_eficiente(mu: pd.Series, cov: pd.DataFrame, n_puntos: int = 60,
                        permitir_ventas_en_corto: bool = False) -> pd.DataFrame:
    """
    Calcula la frontera eficiente: para cada rendimiento objetivo dentro del
    rango observado, encuentra el portafolio de mínima varianza que lo logra.
    """
    n = len(mu)
    objetivos = np.linspace(mu.min(), mu.max(), n_puntos)
    filas = []
    pesos_previos = _pesos_iniciales(n)

    for objetivo in objetivos:
        restricciones = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, obj=objetivo: rendimiento_portafolio(w, mu) - obj},
        ]
        resultado = minimize(
            lambda w: volatilidad_portafolio(w, cov) ** 2,
            pesos_previos,
            method="SLSQP",
            bounds=_bounds(n, permitir_ventas_en_corto),
            constraints=restricciones,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if resultado.success:
            pesos_previos = resultado.x
            fila = {"rendimiento": objetivo, "volatilidad": volatilidad_portafolio(resultado.x, cov)}
            for activo, peso in zip(mu.index, resultado.x):
                fila[f"peso_{activo}"] = peso
            filas.append(fila)

    return pd.DataFrame(filas)


def nube_portafolios_aleatorios(mu: pd.Series, cov: pd.DataFrame, rf: float,
                                 n_simulaciones: int = 4000,
                                 permitir_ventas_en_corto: bool = False,
                                 semilla: int = 42) -> pd.DataFrame:
    """
    Genera portafolios aleatorios (pesos vía Dirichlet) para visualizar la
    nube de combinaciones riesgo-rendimiento posibles alrededor de la
    frontera eficiente (estética tipo Monte Carlo, común en dashboards
    financieros/fintech).
    """
    rng = np.random.default_rng(semilla)
    n = len(mu)
    resultados = []
    for _ in range(n_simulaciones):
        if permitir_ventas_en_corto:
            pesos = rng.normal(size=n)
            pesos = pesos / np.sum(np.abs(pesos))
        else:
            pesos = rng.dirichlet(np.ones(n))
        ret = rendimiento_portafolio(pesos, mu)
        vol = volatilidad_portafolio(pesos, cov)
        resultados.append({
            "rendimiento": ret,
            "volatilidad": vol,
            "sharpe": (ret - rf) / vol if vol > 0 else 0.0,
        })
    return pd.DataFrame(resultados)


def pesos_cartera_actual(clientes: pd.DataFrame,
                          columna_segmento: str = "Segmento_Tarjeta",
                          columna_saldo: str = "Saldo_Actual_MXN",
                          activos_ordenados: list | None = None) -> pd.Series:
    """
    Calcula los pesos de la cartera ACTUAL del banco, agregando el saldo
    vigente por segmento de tarjeta y expresándolo como proporción del
    saldo total. Sirve para ubicar la cartera real dentro (o fuera) de la
    frontera eficiente y cuantificar el margen de mejora.
    """
    saldos = clientes.groupby(columna_segmento)[columna_saldo].sum()
    pesos = saldos / saldos.sum()
    if activos_ordenados is not None:
        pesos = pesos.reindex(activos_ordenados).fillna(0.0)
    return pesos
