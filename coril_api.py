"""
coril_api.py — Conexión a la API de market data de Grupo Coril.

Reemplaza a yfinance como fuente de datos. Cubre mercado peruano (PER) e
internacional (USA). Toda la autenticación se lee de st.secrets, nunca del código.

Endpoints usados:
  - /api/v1/bff/stocks/{country}/{symbol}/detail  → histórico (chartData) + precio actual
  - /api/v1/marketdata/{country}/quotes           → cotización puntual (respaldo)

Configuración requerida en Streamlit Secrets (Settings → Secrets):

    CORIL_API_BASE = "https://sab-dma-marketdata-services-qa.grupocoril.pe"
    CORIL_API_AUTH = "Basic XXXXXXXX"      # header Authorization completo
    CORIL_USER_ID  = "12345"               # X-User-ID

Nunca escribas estas credenciales en el código ni las subas a GitHub.
"""
from __future__ import annotations
import uuid
import datetime as _dt
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    import streamlit as st
except Exception:
    st = None


# ─────────────────────────── Configuración ────────────────────────────────
def _secret(key: str, default: str = "") -> str:
    """Lee un secreto de Streamlit; si no existe, devuelve el default."""
    try:
        if st is not None and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


def _base_url() -> str:
    return _secret("CORIL_API_BASE",
                   "https://sab-dma-marketdata-services-qa.grupocoril.pe").rstrip("/")


def _headers() -> dict:
    """Construye los headers de autenticación desde Secrets."""
    auth = _secret("CORIL_API_AUTH")   # ej: "Basic Ym9sc..."
    user = _secret("CORIL_USER_ID", "12345")
    h = {
        "accept": "application/json",
        "X-User-ID": user,
        "X-Trace-ID": str(uuid.uuid4()),
    }
    if auth:
        h["Authorization"] = auth
    return h


def api_configurada() -> bool:
    """¿Están las credenciales mínimas presentes?"""
    return bool(_secret("CORIL_API_AUTH"))


# ─────────────────────── Detección de país ────────────────────────────────
# Heurística: los símbolos peruanos de la BVL siguen un patrón (letras + C1/I1/etc).
# Las acciones de EE.UU. son tickers cortos alfabéticos (AAPL, MSFT).
# Ajustable según lo que confirme el equipo de Coril.
def detectar_pais(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    # Símbolos peruanos típicos terminan en C1, I1, B1, etc. o son largos
    if len(s) >= 7 and s[-2:] in {"C1", "I1", "B1", "A1", "P1", "T1"}:
        return "PER"
    if s.startswith("^"):
        return "USA"   # índices se tratan aparte
    # Por defecto, tickers cortos alfabéticos → EE.UU.
    return "USA"


# ─────────────────────── Descarga de histórico ────────────────────────────
def _get_detail(symbol: str, country: str, interval: str = "1d",
                timeout: int = 15) -> Optional[dict]:
    """Llama al endpoint /detail y devuelve el JSON 'data', o None si falla."""
    url = f"{_base_url()}/api/v1/bff/stocks/{country}/{symbol}/detail"
    try:
        r = requests.get(url, params={"interval": interval},
                         headers=_headers(), timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        if not j.get("success"):
            return None
        return j.get("data")
    except Exception:
        return None


def descargar_historico(symbol: str, country: Optional[str] = None,
                        interval: str = "1d") -> Optional[pd.Series]:
    """
    Descarga la serie histórica de cierres de un símbolo, con forward-fill
    de los días sin negociación.

    Devuelve una pd.Series indexada por fecha (diaria) con el precio de cierre,
    o None si no hay datos suficientes.
    """
    country = country or detectar_pais(symbol)
    data = _get_detail(symbol, country, interval)
    if not data:
        return None
    chart = data.get("chartData") or []
    if len(chart) < 2:
        return None   # sin historia utilizable (p.ej. solo el punto de hoy)

    # Construir serie de cierres
    fechas, cierres = [], []
    for punto in chart:
        t = punto.get("time")
        c = punto.get("close", punto.get("price"))
        if t is None or c is None:
            continue
        try:
            fechas.append(pd.Timestamp(t).normalize().tz_localize(None))
            cierres.append(float(c))
        except Exception:
            continue
    if len(fechas) < 2:
        return None

    serie = pd.Series(cierres, index=fechas).sort_index()
    serie = serie[~serie.index.duplicated(keep="last")]

    # Forward-fill sobre un calendario diario de días hábiles
    cal = pd.date_range(serie.index.min(), serie.index.max(), freq="B")
    serie = serie.reindex(cal).ffill()
    serie.name = symbol
    return serie


def descargar_precios_historicos(symbols, interval: str = "1d") -> pd.DataFrame:
    """
    Descarga el histórico de varios símbolos y los alinea en un DataFrame
    (columnas = símbolos, índice = fechas diarias con forward-fill).
    """
    series = {}
    for s in symbols:
        serie = descargar_historico(s, interval=interval)
        if serie is not None and len(serie) > 1:
            series[s] = serie
    if not series:
        return pd.DataFrame()
    df = pd.DataFrame(series)
    # Forward-fill final para alinear calendarios distintos entre activos
    df = df.sort_index().ffill()
    return df


def log_retornos(symbols, interval: str = "1d") -> pd.DataFrame:
    """Descarga histórico y devuelve log-retornos diarios (para la optimización)."""
    px = descargar_precios_historicos(symbols, interval)
    if px.empty:
        return pd.DataFrame()
    lr = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return lr


# ─────────────────────── Precios actuales ─────────────────────────────────
def precio_actual(symbol: str, country: Optional[str] = None) -> Optional[float]:
    """Último precio de un símbolo (para el plan de compra)."""
    country = country or detectar_pais(symbol)
    # Primero intentar el /detail (trae lastPrice)
    data = _get_detail(symbol, country, "1d")
    if data and data.get("lastPrice"):
        try:
            return float(data["lastPrice"])
        except Exception:
            pass
    # Respaldo: endpoint /quotes
    url = f"{_base_url()}/api/v1/marketdata/{country}/quotes"
    try:
        r = requests.get(url, params={"symbol": symbol},
                         headers=_headers(), timeout=10)
        if r.status_code == 200:
            j = r.json()
            d = j.get("data") or {}
            p = d.get("lastPrice") or d.get("closePrice")
            if p:
                return float(p)
    except Exception:
        pass
    return None


def precios_actuales(symbols) -> dict:
    """Precios actuales de varios símbolos → dict {symbol: precio}."""
    out = {}
    for s in symbols:
        if not s or s.startswith("^"):
            continue
        p = precio_actual(s)
        if p is not None:
            out[s] = p
    return out


def nombre_valor(symbol: str, country: Optional[str] = None) -> Optional[str]:
    """Nombre del instrumento (securityName) si la API lo provee."""
    country = country or detectar_pais(symbol)
    data = _get_detail(symbol, country, "1d")
    if data:
        return data.get("securityName") or None
    return None
