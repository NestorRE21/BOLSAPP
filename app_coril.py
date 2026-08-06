# -*- coding: utf-8 -*-
"""Coril SAB · Optimizador BL v7 · Compacto"""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from optimizer import RiskProfile, ForcedAsset, View, BLConfig, run_profile, GKConfig, generate_gk_views, estimate_covariance, inject_forced_assets
from projections import monte_carlo, stress_test, CRISIS_PERIODS

# Módulo de la API de Coril (BVL). Import protegido: si falta, la app sigue con yfinance.
try:
    import importlib
    import coril_api
    importlib.reload(coril_api)   # fuerza recarga fresca (evita caché de módulo viejo)
    CORIL_OK = True
    _CORIL_IMPORT_ERR = None
except Exception as _e:
    coril_api = None
    CORIL_OK = False
    _CORIL_IMPORT_ERR = f"{type(_e).__name__}: {_e}"


st.set_page_config(page_title="Simulador de inversiones · Coril SAB", page_icon="📈", layout="wide")

# ═══════════════════ ESTILO VISUAL PERSONALIZADO ══════════════════════════════
st.markdown("""
<style>
/* Importar tipografía moderna */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton, .stMetric {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Fondo general con degradado muy sutil */
.stApp {
    background: linear-gradient(180deg, #f7f9fc 0%, #eef2f8 100%);
}

/* Títulos principales con color de marca */
h1 {
    color: #1a3a5c !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
h2, h3 { color: #23405f !important; font-weight: 700 !important; }
h4, h5 { color: #2e5e8c !important; font-weight: 600 !important; }

/* Botones: gradiente de marca, redondeados, con elevación */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.18s ease !important;
    padding: 0.55rem 1rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2e5e8c 0%, #3d7ab8 100%) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(46,94,140,0.30) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(46,94,140,0.42) !important;
}
.stButton > button[kind="secondary"] {
    background: white !important;
    color: #2e5e8c !important;
    border: 1.5px solid #d4deea !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #2e5e8c !important;
    background: #f4f8fc !important;
}

/* Tarjetas de métricas con fondo blanco, sombra y borde suave */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e6edf5;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(30,60,90,0.05);
    transition: box-shadow 0.18s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 16px rgba(30,60,90,0.10);
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: #1a3a5c !important; font-weight: 700 !important; }

/* Cajas de info / success / warning más suaves y redondeadas */
[data-testid="stNotification"], .stAlert {
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(30,60,90,0.06);
}

/* Sidebar con fondo blanco limpio */
[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #e6edf5;
}

/* Inputs y selectores redondeados */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    border-radius: 10px !important;
}

/* Expanders como tarjetas */
[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e6edf5 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(30,60,90,0.04);
}

/* Divisores más sutiles */
hr { border-color: #e6edf5 !important; margin: 1.1rem 0 !important; }

/* Radio buttons y checkboxes con acento de marca */
.stRadio [data-baseweb="radio"] div[aria-checked="true"] { border-color: #2e5e8c !important; }

/* ══ MÉTRICAS MÁS GRANDES Y LEGIBLES ══ */
[data-testid="stMetricValue"] {
    font-size: 2.1rem !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.95rem !important; }

/* Más aire entre secciones del contenido principal */
.block-container { padding-top: 2.2rem !important; padding-bottom: 3rem !important; }
.main .block-container { max-width: 1200px; }

/* Títulos de sección (h5) con un poco más de presencia y espacio arriba */
h5 { margin-top: 1.4rem !important; font-size: 1.15rem !important; }
h4 { font-size: 1.35rem !important; }

/* ══ SIDEBAR MEJORADO ══ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafd 100%);
    border-right: 1px solid #e6edf5;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }
/* Etiquetas de widgets del sidebar más marcadas */
[data-testid="stSidebar"] label { font-weight: 600 !important; color: #34506e !important; }
/* Number input del sidebar más grande */
[data-testid="stSidebar"] .stNumberInput input { font-size: 1.15rem !important; font-weight: 700 !important; text-align: center; }
/* Slider del sidebar con acento */
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] { background: #2e5e8c !important; }

/* Botones de la barra de pasos más altos */
.stButton > button { min-height: 44px; }

/* Toast/notificaciones */
[data-testid="stToast"] { border-radius: 12px !important; }

/* Cajas de éxito con verde de marca */
[data-testid="stNotificationContentSuccess"] { font-size: 0.98rem; }
</style>
""", unsafe_allow_html=True)
RF,PPY = 0.02,252   # PPY=252 días hábiles al año (datos diarios)
FICO_TK = "FICCMP13"
FICO_DISPLAY = "Fondo de inversión"
FICO = ForcedAsset(ret_annual=0.0625,vol_annual=0.010,beta=0.30,sector="Factoring",region="Perú",moneda="USD",instrumento="Fondo")

def disp(ticker):
    """Nombre visible de un ticker (FICCMP13 → Fondo de inversión). Para uso en la interfaz."""
    return FICO_DISPLAY if ticker == FICO_TK else ticker
PERFILES = {"Conservador (30/70)":(0.30,0.70),"Moderado-bajo (40/60)":(0.40,0.60),
            "Moderado (50/50)":(0.50,0.50),"Crecimiento (60/40)":(0.60,0.40),"Agresivo (70/30)":(0.70,0.30)}
P_DESC = {"Conservador (30/70)":"Preservar capital.","Moderado-bajo (40/60)":"Leve crecimiento.",
          "Moderado (50/50)":"Balance.","Crecimiento (60/40)":"Mayor exposición.","Agresivo (70/30)":"Máxima RV."}

def detectar_perfil(rv_pct):
    """Detecta el nombre del perfil según el % de renta variable (0-100)."""
    if rv_pct <= 20:   return "Muy conservador", "Máxima preservación de capital."
    if rv_pct <= 35:   return "Conservador", "Preservar capital."
    if rv_pct <= 45:   return "Moderado-bajo", "Leve crecimiento."
    if rv_pct <= 55:   return "Moderado", "Balance entre riesgo y retorno."
    if rv_pct <= 65:   return "Crecimiento", "Mayor exposición a renta variable."
    if rv_pct <= 80:   return "Agresivo", "Alta exposición a renta variable."
    return "Muy agresivo", "Máxima exposición a renta variable."
EJ = ["AAPL","MSFT","NVDA","JNJ","KO","QQQ"]

# Catálogo de inversiones populares con nombres amigables (para quien no conoce tickers)
POPULARES_RV = [
    ("AAPL","Apple","📱"),("MSFT","Microsoft","💻"),("GOOGL","Google","🔎"),
    ("AMZN","Amazon","📦"),("NVDA","Nvidia","🎮"),("META","Meta (Facebook)","👥"),
    ("TSLA","Tesla","🚗"),("KO","Coca-Cola","🥤"),("MCD","McDonald's","🍔"),
    ("DIS","Disney","🎬"),("NFLX","Netflix","🎥"),("V","Visa","💳"),
    ("JNJ","Johnson & Johnson","💊"),("JPM","JPMorgan","🏦"),
    ("QQQ","Tecnológicas EE.UU. (QQQ)","📈"),("SPY","S&P 500 (SPY)","🇺🇸"),
    ("VTI","Todo el mercado EE.UU. (VTI)","🌐"),
]
# Acciones populares de la Bolsa de Valores de Lima (BVL) — vía API de Coril
POPULARES_BVL = [
    ("BUENAVC1","Buenaventura (minera)","⛏️"),("MINSURI1","Minsur (estaño)","🪨"),
    ("CREDITC1","Credicorp (banca)","🏦"),("ALICORC1","Alicorp (consumo)","🛒"),
    ("BAP","Credicorp (ADR)","🏦"),("VOLCABC1","Volcan (minera)","⛏️"),
    ("FERREYC1","Ferreycorp (maquinaria)","🚜"),("CVERDEC1","Cerro Verde (cobre)","🟠"),
    ("INRETC1","InRetail (retail)","🏬"),("LUSURC1","Luz del Sur (energía)","💡"),
]
POPULARES_RF = [
    ("AGG","Bonos EE.UU. amplio (AGG)","🏛️"),("BND","Bonos totales (BND)","🏦"),
    ("TLT","Bonos largo plazo (TLT)","📉"),("SHY","Bonos corto plazo (SHY)","🛡️"),
    ("LQD","Bonos corporativos (LQD)","🏢"),("TIP","Bonos anti-inflación (TIP)","📊"),
]
POPULARES_BK = [
    ("^GSPC","S&P 500 (EE.UU.)","🇺🇸"),("^IXIC","Nasdaq (tecnología)","💻"),
    ("^DJI","Dow Jones","🏭"),("ACWI","Mundo (ACWI)","🌍"),
]

# Mapa ticker → nombre amigable (a partir de los catálogos populares)
_NOMBRES_CONOCIDOS = {}
for _cat in (POPULARES_RV, POPULARES_BVL, POPULARES_RF, POPULARES_BK):
    for _tk,_nm,_emo in _cat:
        _NOMBRES_CONOCIDOS[_tk] = _nm

# Nombres de índices comunes (para mostrar en todas las pantallas)
_NOMBRES_CONOCIDOS.update({
    "^GSPC":"S&P 500","^IXIC":"Nasdaq Composite","^DJI":"Dow Jones","^RUT":"Russell 2000",
    "^VIX":"Índice de volatilidad (VIX)","^FTSE":"FTSE 100 (Reino Unido)","^N225":"Nikkei 225 (Japón)",
    "^GDAXI":"DAX (Alemania)","^FCHI":"CAC 40 (Francia)","^STOXX50E":"Euro Stoxx 50",
    "^HSI":"Hang Seng (Hong Kong)","^BVSP":"Bovespa (Brasil)","^MXX":"IPC (México)",
    "^TNX":"Bono EE.UU. 10 años","^GSPTSE":"S&P/TSX (Canadá)",
})

def nombre_activo(tk):
    """Nombre amigable de un ticker: catálogo conocido, nombre buscado, o el ticker."""
    if tk == FICO_TK: return FICO_DISPLAY
    if tk in _NOMBRES_CONOCIDOS: return _NOMBRES_CONOCIDOS[tk]
    try:
        nm = st.session_state.get("asset_names",{}).get(tk)
        if nm: return nm
    except Exception:
        pass
    return tk
C_RV,C_RF,C_OPT = "#2E5E8C","#2CA02C","#D6604D"
BC = ["#888","#E377C2","#FF7F0E","#9467BD","#17BECF"]

def usd(x):
    """Formatea monto en USD con el signo $ escapado para markdown de Streamlit."""
    return f"\\${x:,.0f}"

for k,v in {"tickers":[],"rf_tickers":[],"include_fico":True,"benchmarks":["^GSPC"],"views":[],"optimized":False,"result":None,
            "manual_weights":None,"returns":None,"bench_rets":None,"betas":None,"sectors":None,
            "returns_full":None,"bench_full":None,"last_period":None,"data_range":"",
            "mode":None,"gk_ic":0.05,"step":0,"_w_ver":0,"asset_names":{}}.items():
    st.session_state.setdefault(k,v)

# ═══════════════════ BACKEND ══════════════════════════════════════════════════
def _yf_period(period):
    """Convierte períodos custom (como 15y) a parámetros de yfinance."""
    if period in ["1y","2y","3y","5y","10y","max","ytd"]:
        return {"period": period}
    # Períodos custom: extraer años y calcular fecha de inicio
    if period.endswith("y"):
        years = int(period.replace("y",""))
        start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        return {"start": start}
    return {"period": period}

@st.cache_data(show_spinner=False,ttl=600)
def dl_eq(tickers,period="15y"):
    import yfinance as yf
    tickers=[t.strip().upper() for t in tickers if t and t.strip()]
    # Separar BVL (Coril) de extranjeras (yfinance)
    bvl = [t for t in tickers if CORIL_OK and coril_api.es_bvl(t)] if CORIL_OK else []
    ext = [t for t in tickers if t not in bvl]

    frames=[]
    # Extranjeras por yfinance (DIARIO, con forward-fill de días sin precio)
    if ext:
        params = _yf_period(period)
        raw=yf.download(ext,**params,interval="1d",auto_adjust=True,progress=False)
        if raw is not None and not raw.empty:
            px=raw["Close"].copy() if isinstance(raw.columns,pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":ext[0]})
            px=px.dropna(how="all"); px.index=pd.to_datetime(px.index).tz_localize(None)
            # Forward-fill sobre calendario diario hábil (arrastra precio del día anterior)
            cal=pd.date_range(px.index.min(),px.index.max(),freq="B")
            px=px.reindex(cal).ffill()
            lr_ext=np.log(px/px.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(how="all")
            frames.append(lr_ext)
    # BVL por API de Coril (DIARIO, con forward-fill)
    if bvl and hasattr(coril_api, "precios_diarios"):
        try:
            px_bvl=coril_api.precios_diarios(bvl)
            if px_bvl is not None and not px_bvl.empty:
                lr_bvl=np.log(px_bvl/px_bvl.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(how="all")
                frames.append(lr_bvl)
                st.session_state["_bvl_diag"]=f"BVL OK: {list(px_bvl.columns)} · {len(px_bvl)} días"
            else:
                st.session_state["_bvl_diag"]=f"BVL vacío para {bvl}: precios_diarios devolvió sin datos."
        except Exception as e:
            st.session_state["_bvl_diag"]=f"BVL error para {bvl}: {type(e).__name__}: {e}"
    elif bvl:
        st.session_state["_bvl_diag"]=f"BVL {bvl} detectado pero coril_api sin precios_diarios (módulo viejo)."

    if not frames: return None
    # Combinar por fecha (outer join) y rellenar huecos de alineación
    combinado=pd.concat(frames,axis=1).sort_index()
    return combinado

@st.cache_data(show_spinner=False,ttl=600)
def dl_bk(tks,period="15y"):
    import yfinance as yf
    out={}
    for b in tks:
        b=b.strip().upper()
        if not b: continue
        # Benchmark BVL por Coril
        if CORIL_OK and coril_api.es_bvl(b) and hasattr(coril_api,"precios_diarios"):
            try:
                pxb=coril_api.precios_diarios([b])
                if pxb is not None and not pxb.empty:
                    p=pxb.iloc[:,0]
                    lr=np.log(p/p.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(); lr.name=b; out[b]=lr
                    continue
            except Exception: pass
        # Benchmark internacional por yfinance (DIARIO con forward-fill)
        try:
            params = _yf_period(period)
            raw=yf.download(b,**params,interval="1d",auto_adjust=True,progress=False)
            if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
            p=raw["Close"]
            if isinstance(p,pd.DataFrame): p=p.iloc[:,0]
            p.index=pd.to_datetime(p.index).tz_localize(None)
            cal=pd.date_range(p.index.min(),p.index.max(),freq="B")
            p=p.reindex(cal).ffill()
            lr=np.log(p/p.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(); lr.name=b; out[b]=lr
        except: pass
    return out

def calc_betas(r,b):
    c=r.index.intersection(b.index); bv=b.loc[c].values; bvar=np.var(bv,ddof=1)
    out={}
    for t in r.columns:
        tv=r.loc[c,t].values; m=np.isfinite(tv)&np.isfinite(bv)
        out[t]=round(float(np.cov(tv[m],bv[m],ddof=1)[0,1]/bvar),3) if m.sum()>10 and bvar>1e-12 else 1.0
    return pd.Series(out)

# Sectores conocidos de tickers comunes (respaldo si yfinance falla o limita)
_KNOWN_SECTORS = {
    # Tecnología
    "AAPL":"Tecnología","MSFT":"Tecnología","NVDA":"Tecnología","GOOGL":"Tecnología",
    "GOOG":"Tecnología","META":"Tecnología","AVGO":"Tecnología","ORCL":"Tecnología",
    "CRM":"Tecnología","ADBE":"Tecnología","AMD":"Tecnología","INTC":"Tecnología",
    "CSCO":"Tecnología","QCOM":"Tecnología","TXN":"Tecnología","IBM":"Tecnología",
    "NOW":"Tecnología","INTU":"Tecnología","AMAT":"Tecnología","MU":"Tecnología",
    # Consumo discrecional
    "AMZN":"Consumo discrecional","TSLA":"Consumo discrecional","HD":"Consumo discrecional",
    "MCD":"Consumo discrecional","NKE":"Consumo discrecional","SBUX":"Consumo discrecional",
    "LOW":"Consumo discrecional","BKNG":"Consumo discrecional","TJX":"Consumo discrecional",
    # Consumo básico
    "KO":"Consumo básico","PEP":"Consumo básico","PG":"Consumo básico","COST":"Consumo básico",
    "WMT":"Consumo básico","MDLZ":"Consumo básico","CL":"Consumo básico","MO":"Consumo básico",
    # Salud
    "JNJ":"Salud","UNH":"Salud","LLY":"Salud","PFE":"Salud","ABBV":"Salud","MRK":"Salud",
    "TMO":"Salud","ABT":"Salud","DHR":"Salud","BMY":"Salud","AMGN":"Salud","CVS":"Salud",
    # Financiero
    "JPM":"Financiero","BAC":"Financiero","WFC":"Financiero","GS":"Financiero","MS":"Financiero",
    "V":"Financiero","MA":"Financiero","AXP":"Financiero","BLK":"Financiero","C":"Financiero",
    "SCHW":"Financiero","BRK-B":"Financiero","SPGI":"Financiero",
    # Energía / Industrial / Comunicaciones
    "XOM":"Energía","CVX":"Energía","COP":"Energía","SLB":"Energía",
    "BA":"Industrial","CAT":"Industrial","GE":"Industrial","HON":"Industrial","UPS":"Industrial",
    "DIS":"Comunicaciones","NFLX":"Comunicaciones","CMCSA":"Comunicaciones","T":"Comunicaciones",
    "VZ":"Comunicaciones",
    # ETFs de equity (índices amplios / sectoriales)
    "QQQ":"ETF · Tecnología (Nasdaq 100)","SPY":"ETF · Mercado amplio (S&P 500)",
    "VOO":"ETF · Mercado amplio (S&P 500)","VTI":"ETF · Mercado total EE.UU.",
    "IWM":"ETF · Small caps","DIA":"ETF · Dow Jones","VUG":"ETF · Crecimiento",
    "VTV":"ETF · Valor","VYM":"ETF · Dividendos","SCHD":"ETF · Dividendos",
    "XLK":"ETF · Tecnología","XLF":"ETF · Financiero","XLE":"ETF · Energía",
    "XLV":"ETF · Salud","XLY":"ETF · Consumo discrecional","XLP":"ETF · Consumo básico",
    "SOXX":"ETF · Semiconductores","SMH":"ETF · Semiconductores","ARKK":"ETF · Innovación",
    "EFA":"ETF · Internacional desarrollado","EEM":"ETF · Emergentes","VEA":"ETF · Internacional",
}

@st.cache_data(show_spinner=False,ttl=600)
def fetch_sec(tickers):
    import yfinance as yf
    out={}
    for t in tickers:
        # Respaldo conocido primero (siempre disponible, aunque yfinance falle)
        if t in _KNOWN_SECTORS:
            out[t]=_KNOWN_SECTORS[t]; continue
        try:
            i=yf.Ticker(t).info or {}; s=i.get("sector","")
            out[t]=s if s else (f"ETF · {i.get('category','')[:25]}" if i.get("quoteType")=="ETF" else i.get("industry","") or "Otros")
        except: out[t]="Otros"
    return pd.Series(out)

@st.cache_data(show_spinner=False,ttl=300)
def fetch_precios(tickers):
    """Precio actual de cada ticker. BVL vía Coril, extranjeras vía yfinance."""
    import yfinance as yf
    out={}
    tickers=[t.strip().upper() for t in tickers if t and not t.startswith("^")]
    if not tickers: return out
    # BVL por API de Coril
    bvl=[t for t in tickers if CORIL_OK and coril_api.es_bvl(t)] if CORIL_OK else []
    ext=[t for t in tickers if t not in bvl]
    if bvl:
        try:
            out.update(coril_api.precios_actuales(bvl))
        except Exception: pass
    # Extranjeras por yfinance
    if ext:
        try:
            data=yf.download(list(ext),period="5d",interval="1d",
                             auto_adjust=True,progress=False)
            close=data["Close"] if "Close" in data else data
            if isinstance(close,pd.Series):
                v=close.dropna()
                if len(v)>0: out[ext[0]]=float(v.iloc[-1])
            else:
                for t in ext:
                    if t in close.columns:
                        v=close[t].dropna()
                        if len(v)>0: out[t]=float(v.iloc[-1])
        except Exception:
            pass
    return out

def puede_agregar(nuevo_tk, capital):
    """
    ¿Agregar `nuevo_tk` a RV mantiene la suma de precios <= capital?
    Devuelve (ok: bool, precio_nuevo: float|None, suma_actual: float, precio_total: float).
    Los índices y activos sin precio no cuentan.
    """
    actuales = list(st.session_state.tickers)
    todos = actuales + ([nuevo_tk] if nuevo_tk not in actuales else [])
    precios = fetch_precios(tuple(todos))
    suma_actual = sum(precios.get(t,0) for t in actuales)
    precio_nuevo = precios.get(nuevo_tk)
    if precio_nuevo is None:
        # Sin precio disponible → permitir (no podemos verificar)
        return True, None, suma_actual, suma_actual
    total = suma_actual + precio_nuevo
    return (total <= capital), precio_nuevo, suma_actual, total

def acciones_enteras(pesos, precios, capital, fraccionables):
    """
    Convierte pesos objetivo (%) en número de acciones ENTERAS a comprar
    (algoritmo greedy). Los activos 'fraccionables' (Fondo, ETFs de RF)
    aceptan monto libre y absorben el sobrante.

    Devuelve: (unidades{tk:n}, monto_fracc{tk:$}, efectivo_sobrante, gastado_acciones)
    """
    enteros = [a for a in pesos if a not in fraccionables and precios.get(a,0)>0]
    fracc   = [a for a in pesos if a in fraccionables]
    objetivo = {a: pesos[a]*capital for a in pesos}

    # Paso 1: compra base (floor)
    unidades={}; gastado=0.0
    for a in enteros:
        p=precios[a]; u=int(objetivo[a]//p); unidades[a]=u; gastado+=u*p
    restante=capital-gastado

    # Paso 2: greedy — comprar de a una la que más acerca al objetivo
    mejora=True
    while mejora and restante>0:
        mejora=False; mejor_a=None; mejor_g=0.0
        for a in enteros:
            p=precios[a]
            if p>restante: continue
            actual=unidades[a]*p
            g=abs(objetivo[a]-actual)-abs(objetivo[a]-(actual+p))
            if g>mejor_g: mejor_g=g; mejor_a=a
        if mejor_a is not None:
            p=precios[mejor_a]; unidades[mejor_a]+=1; restante-=p; gastado+=p; mejora=True

    # Paso 3: el restante va a fraccionables (por peso) o queda como efectivo
    monto_fracc={}; obj_fracc=sum(objetivo[a] for a in fracc)
    if fracc and obj_fracc>0:
        for a in fracc: monto_fracc[a]=restante*(objetivo[a]/obj_fracc)
        efectivo=0.0
    else:
        efectivo=restante
    return unidades, monto_fracc, efectivo, gastado

@st.cache_data(show_spinner=False,ttl=300)
def search_yf(q):
    import requests
    resultados=[]
    # Si el texto parece un nemónico BVL, validarlo contra la API de Coril
    q_up=q.strip().upper()
    if CORIL_OK and coril_api.es_bvl(q_up) and coril_api.api_configurada():
        try:
            _nombre=coril_api.nombre_valor(q_up)
            if _nombre:
                resultados.append({"tk":q_up,"nm":_nombre,"tp":"EQUITY","ex":"BVL"})
        except Exception: pass
    # Búsqueda normal por Yahoo (internacionales)
    try:
        r=requests.get("https://query2.finance.yahoo.com/v1/finance/search",params={"q":q,"quotesCount":12,"newsCount":0},
                       headers={"User-Agent":"Mozilla/5.0"},timeout=5)
        resultados+=[{"tk":x["symbol"],"nm":x.get("shortname") or x.get("longname",""),
                 "tp":x.get("quoteType",""),"ex":x.get("exchange","")}
                for x in r.json().get("quotes",[]) if x.get("symbol")]
    except: pass
    return resultados

# Keywords en nombre que indican renta fija
_RF_KW = {"bond","treasury","income","fixed","aggregate","debt","govt","municipal",
          "corporate bond","tips","tbill","bill","note","yield","interest rate",
          "money market","short term","intermediate","long term","investment grade",
          "high yield","credit","inflation","sovereign",
          "bono","renta fija","deuda","tesor","mortgage","securit"}

# Tickers conocidos de ETFs de renta fija (para catch rápido)
_RF_TICKERS = {"TLT","SHY","IEF","AGG","BND","LQD","HYG","TIP","BIL","SHV",
               "GOVT","MBB","VCSH","VCIT","VCLT","BIV","BSV","BLV","VGSH","VGIT",
               "VGLT","SCHO","SCHR","SCHZ","SPAB","USIG","IGSB","IGIB","FLOT",
               "STIP","LTPZ","ZROZ","EDV","SPLB","SPTL","SPTS","BNDX","IAGG",
               "EMB","PCY","BWX","IGOV","VTIP","SCHP","JPST","MINT","GSY","NEAR",
               "FLRN","SRLN","BKLN","HYDB","ANGL","SJNK","JNK","SHYG","USHY",
               "LMBS","VMBS","GNMA","FBND","NUBD","TOTL","GTO","FIXD","BOND"}

# Keywords de exclusión DURA — cripto, commodities (bloqueados en todos lados)
_EXCLUDE_KW = {"bitcoin","ethereum","crypto","blockchain","digital asset",
               "gold","silver","platinum","palladium","oil","crude","natural gas",
               "commodity","commodit","agricult","wheat","corn","soybean",
               "cannabis","marijuana","weed"}

# Keywords y tickers de productos APALANCADOS / INVERSOS.
# Permitidos en Renta Variable y Benchmark; NO en Renta Fija.
_LEV_KW = {"leverag","2x","3x","-2x","-3x","ultra","direxion","proshares ultra",
           "inverse","bull 2x","bull 3x","bear 2x","bear 3x"}
_LEV_TICKERS = {"TQQQ","SQQQ","UPRO","SPXU","QLD","QID","SSO","SDS","UDOW","SDOW",
                "TNA","TZA","LABU","LABD","SOXL","SOXS","FNGU","FNGD","JNUG","JDST",
                "NUGT","DUST","UVXY","SVXY","VIXY","TMF","TMV","TYD","TYO"}

def _is_excluded(r):
    """Exclusión dura: cripto, commodities. Bloqueado en todas las categorías."""
    nm = (r.get("nm","") or "").lower()
    return any(kw in nm for kw in _EXCLUDE_KW)

def _is_leveraged(r):
    """¿Es un producto apalancado o inverso?"""
    nm = (r.get("nm","") or "").lower()
    tk = (r.get("tk","") or "").upper()
    if tk in _LEV_TICKERS: return True
    if any(kw in nm for kw in _LEV_KW): return True
    return False

def _is_index(r):
    """¿Es un índice de mercado? (solo permitido en benchmark)"""
    if r.get("tp","")=="INDEX": return True
    if (r.get("tk","") or "").startswith("^"): return True
    return False

def _is_rf_candidate(r):
    """Heurística: ¿el resultado parece renta fija? (nunca apalancados ni índices)"""
    if _is_excluded(r) or _is_leveraged(r) or _is_index(r): return False
    nm = (r.get("nm","") or "").lower()
    tk = (r.get("tk","") or "").upper()
    # Ticker conocido de RF
    if tk in _RF_TICKERS: return True
    # Keywords de RF en nombre
    if any(kw in nm for kw in _RF_KW): return True
    return False

def _is_rv_candidate(r):
    """Heurística: ¿el resultado parece renta variable? (apalancados sí, índices no)"""
    if _is_excluded(r) or _is_index(r): return False   # cripto/commodities/índices no
    tp = r.get("tp","")
    # Apalancados de equity → sí van en RV
    if _is_leveraged(r): return True
    # Acciones individuales → siempre RV
    if tp == "EQUITY": return True
    # ETFs/fondos → solo si NO parece RF
    if tp in ("ETF","MUTUALFUND"):
        return not _is_rf_candidate(r)
    return True  # por defecto permitir

def filter_search(results, category):
    """Filtra resultados de búsqueda según la categoría seleccionada."""
    if category == "🔵 Renta variable":
        # RV: acciones, ETFs de equity y apalancados. No cripto/commodities/índices.
        return [r for r in results if _is_rv_candidate(r)]
    elif category == "🟢 Renta fija":
        # RF: solo lo que positivamente parece RF, nunca apalancados ni índices.
        return [r for r in results if _is_rf_candidate(r)]
    else:  # Benchmark: permite índices y apalancados; solo bloquea cripto/commodities.
        return [r for r in results if not _is_excluded(r)]

def do_opt(eq_tickers, rf_tickers, include_fico, views_cfg, eq_t, fi_t, pb, auto=False):
    betas=st.session_state.betas.copy()
    forced = {FICO_TK: FICO} if include_fico else {}
    if include_fico: betas[FICO_TK]=FICO.beta
    all_assets = set(eq_tickers) | set(rf_tickers) | (set(forced.keys()) if forced else set())

    if auto:
        # ── MODO AUTOMÁTICO: views generadas por Grinold-Kahn (α = vol·IC·score) ──
        gk_cfg = GKConfig(ic=st.session_state.get("gk_ic",0.05),
                          lookback_months=12, skip_months=1, confidence=0.5)
        # Reconstruir el universo tal como lo verá el motor (equity + RF mercado + FICO)
        rf_market = [a for a in rf_tickers if a in st.session_state.returns.columns]
        data_cols = [a for a in eq_tickers if a in st.session_state.returns.columns] + rf_market
        full = inject_forced_assets(st.session_state.returns[data_cols], forced, PPY)
        cov_full = estimate_covariance(full, PPY)
        views = generate_gk_views(full, list(full.columns), cov_full, gk_cfg,
                                  forced_tickers=list(forced.keys()),
                                  periods_per_year=PPY, rf_annual=RF)
        return run_profile(returns=st.session_state.returns,equity_assets=eq_tickers,
                           forced_assets=forced, rf_assets=rf_tickers,
                           profile=RiskProfile.for_split(eq_t,fi_t),views=views,
                           config=BLConfig(rf_annual=RF,periods_per_year=PPY,tau=0.05,max_weight_equity=0.25,gamma_beta=5.0),
                           benchmark_returns=pb,betas=betas,views_as_alpha=True)

    # ── MODO MANUAL: views del usuario ──
    views=[View(kind="absolute",asset=v["asset"],q=v["q"],confidence=v["confidence"]) if v["type"]=="absolute"
           else View(kind="relative",long=v["long"],short=v["short"],q=v["q"],confidence=v["confidence"])
           for v in views_cfg if (v["type"]=="absolute" and v.get("asset") in all_assets) or
                                  (v["type"]!="absolute" and v.get("long") in all_assets and v.get("short") in all_assets)]
    return run_profile(returns=st.session_state.returns,equity_assets=eq_tickers,
                       forced_assets=forced, rf_assets=rf_tickers,
                       profile=RiskProfile.for_split(eq_t,fi_t),views=views,
                       config=BLConfig(rf_annual=RF,periods_per_year=PPY,tau=0.05,max_weight_equity=0.25,gamma_beta=5.0),
                       benchmark_returns=pb,betas=betas)

def wdd(w,rets,bd,cap):
    if not isinstance(bd,dict): bd={}
    eq=[a for a in w.index if a in rets.columns and a!=FICO_TK]
    pr=sum(w.get(c,0)*rets[c].fillna(0) for c in eq) if eq else pd.Series(0,index=rets.index)
    if FICO_TK in w.index and w[FICO_TK]>1e-8: pr=pr+w[FICO_TK]*(np.log(1+FICO.ret_annual)/PPY)
    pr=pr.fillna(0); common=pr.index
    for v in bd.values(): common=common.intersection(v.index)
    pr=pr.loc[common]; wl=np.exp(pr.cumsum())*cap; dd=wl/wl.cummax()-1
    bw,bdd={},{}
    for n,v in bd.items():
        br=v.loc[common].fillna(0); bw[n]=np.exp(br.cumsum())*cap; bdd[n]=bw[n]/bw[n].cummax()-1
    return pr,wl,dd,bw,bdd

def run_dl(period):
    tks=st.session_state.tickers; bks=st.session_state.benchmarks
    rf_tks=st.session_state.rf_tickers
    # Se necesita al menos un activo (RV o RF de mercado) y un benchmark.
    all_market = list(set(tks + rf_tks))  # sin duplicados
    if not all_market or not bks:
        st.session_state["_dl_error"]="Faltan activos o benchmark."
        return False
    lr=dl_eq(tuple(all_market),period)
    if lr is None or lr.empty:
        st.session_state["_dl_error"]=f"No se obtuvieron datos de los activos: {', '.join(all_market)}."
        return False
    bd=dl_bk(tuple(bks),period)
    if not bd:
        st.session_state["_dl_error"]=f"No se obtuvieron datos del benchmark: {', '.join(bks)}."
        return False
    # Alinear por unión de fechas + forward-fill (tolerante a calendarios distintos
    # entre EE.UU. y BVL). Antes se usaba intersección estricta, que podía vaciar todo.
    idx_union = lr.index
    for v in bd.values():
        idx_union = idx_union.union(v.index)
    idx_union = idx_union.sort_values()
    lr = lr.reindex(idx_union).ffill().dropna(how="all")
    bd = {k: v.reindex(lr.index).ffill() for k, v in bd.items()}
    # Quitar filas iniciales donde aún no hay ningún dato
    lr = lr.dropna(how="all")
    if lr.empty:
        st.session_state["_dl_error"]="Los datos quedaron vacíos tras alinear fechas."
        return False
    st.session_state.returns=lr; st.session_state.bench_rets={k:v.loc[lr.index] for k,v in bd.items()}
    st.session_state.returns_full=lr; st.session_state.bench_full=bd
    b=calc_betas(lr,list(bd.values())[0].loc[lr.index]); b[FICO_TK]=FICO.beta; st.session_state.betas=b
    ok=[t for t in tks if t in lr.columns]
    s=fetch_sec(tuple(ok)); s[FICO_TK]=FICO.sector; st.session_state.sectors=s
    st.session_state.last_period=period
    st.session_state.data_range=f"{lr.index.min().strftime('%Y-%m-%d')} → {lr.index.max().strftime('%Y-%m-%d')}"
    st.session_state["_dl_error"]=None
    for k in list(st.session_state.keys()):
        if k.startswith("s_"): del st.session_state[k]
    st.session_state.optimized=False; st.session_state.result=None; st.session_state.manual_weights=None
    for x in ["mc","stress"]:
        if x in st.session_state: del st.session_state[x]
    return True

# ═══════════════════ SIDEBAR ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; padding:6px 0 10px 0;">
        <span style="font-size:1.6rem;">📈</span>
        <div>
            <div style="font-size:1.3rem; font-weight:800; color:#1a3a5c; line-height:1;">Coril</div>
            <div style="font-size:0.72rem; color:#8595a8; font-weight:500;">Simulador de inversiones</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.mode is None:
        # Pantalla de bienvenida: sidebar mínimo, sin configuración todavía.
        eq_t,fi_t,capital = 0.5,0.5,100_000   # defaults (no se usan hasta elegir modo)
    else:
        _mode_label = "🤖 Automático" if st.session_state.mode=="auto" else "🎛️ Manual"
        mc1,mc2=st.columns([1.4,1])
        mc1.markdown(f"<div style='font-size:0.8rem;color:#7a8ba0;'>Modo actual</div>"
                     f"<div style='font-weight:700;color:#2e5e8c;'>{_mode_label}</div>",unsafe_allow_html=True)
        with mc2:
            if st.button("Cambiar", use_container_width=True):
                st.session_state.mode=None; st.rerun()
        st.divider()

        # ── Perfil de riesgo ──
        st.markdown("### ⚖️ Perfil de riesgo")
        st.caption("¿Cuánto riesgo aceptas? Más renta variable = más ganancia potencial "
                   "pero más riesgo.")
        rv_pct=st.number_input("Renta variable (%)",0,100,
                                int(st.session_state.get("_rv_pct",50)),5,key="rv_pct_input",
                                help="Porcentaje de tu dinero en inversiones de mayor riesgo y "
                                     "mayor ganancia potencial (acciones, ETFs de equity).")
        rf_pct=100-rv_pct
        st.session_state._rv_pct=rv_pct
        eq_t,fi_t=rv_pct/100.0, rf_pct/100.0
        # Barra visual de la división RV/RF
        st.markdown(f"""
        <div style="display:flex; height:26px; border-radius:8px; overflow:hidden; margin:6px 0 4px 0;
                    box-shadow:0 1px 3px rgba(0,0,0,0.08);">
            <div style="width:{rv_pct}%; background:linear-gradient(90deg,#2e5e8c,#3d7ab8);
                        color:white; font-size:0.72rem; font-weight:700; display:flex;
                        align-items:center; justify-content:center;">{rv_pct}%</div>
            <div style="width:{rf_pct}%; background:linear-gradient(90deg,#2ca02c,#4cb84c);
                        color:white; font-size:0.72rem; font-weight:700; display:flex;
                        align-items:center; justify-content:center;">{rf_pct}%</div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#7a8ba0; margin-bottom:8px;">
            <span>🔵 Renta variable</span><span>Renta fija 🟢</span>
        </div>
        """, unsafe_allow_html=True)

        _perfil_nombre,_perfil_desc=detectar_perfil(rv_pct)
        _emoji_perfil = "🛡️" if rv_pct<=35 else ("⚖️" if rv_pct<=65 else "🚀")
        _color_perfil = "#2ca02c" if rv_pct<=35 else ("#2e5e8c" if rv_pct<=65 else "#d6604d")
        st.markdown(f"""
        <div style="background:white; border-left:4px solid {_color_perfil}; border-radius:10px;
                    padding:12px 14px; box-shadow:0 2px 8px rgba(30,60,90,0.06); margin-bottom:4px;">
            <div style="font-size:0.75rem; color:#7a8ba0; font-weight:600;">TU PERFIL</div>
            <div style="font-size:1.25rem; font-weight:800; color:{_color_perfil};">{_emoji_perfil} {_perfil_nombre}</div>
            <div style="font-size:0.82rem; color:#64748b; margin-top:2px;">{_perfil_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── Inversión ──
        st.markdown("### 💵 Monto a invertir")
        capital=st.number_input("¿Cuánto quieres invertir? (USD)",min_value=100,max_value=100_000_000,
                          value=int(st.session_state.get("_capital",100_000)),step=1_000,
                          label_visibility="collapsed",
                          help="Escribe el monto que quieres simular, en dólares.")
        st.session_state._capital=capital
        st.markdown(f"<div style='text-align:center; font-size:1.5rem; font-weight:800; color:#1a3a5c; "
                    f"margin-top:2px;'>${capital:,.0f}</div>",unsafe_allow_html=True)
        st.divider()
        with st.expander("⚙️ Detalles avanzados"):
            _p=RiskProfile.for_split(eq_t,fi_t)
            st.caption(f"Fondo de inversión: {FICO.ret_annual:.2%} anual")
            st.caption(f"Beta objetivo: {_p.beta_min:.2f} a {_p.beta_max:.2f}")
            st.caption(f"Caída máxima tolerada: {_p.max_drawdown:.0%}")
            st.caption("Los cálculos usan datos diarios (hasta 15 años en instrumentos "
                       "internacionales; 5 años en acciones de la BVL).")

        with st.expander("🔧 Probar conexión Coril (BVL)"):
            if _CORIL_IMPORT_ERR:
                st.error(f"Error al importar coril_api: {_CORIL_IMPORT_ERR}")
            if CORIL_OK:
                import os as _os
                _ruta=getattr(coril_api,"__file__","(desconocida)")
                _tiene_pd=hasattr(coril_api,"precios_diarios")
                st.caption(f"Módulo: `{_os.path.basename(_ruta)}` · "
                           f"precios_diarios: {'✅' if _tiene_pd else '❌ FALTA'}")
            if not CORIL_OK:
                st.error("El módulo coril_api.py no está disponible. Súbelo al repositorio.")
            elif not coril_api.api_configurada():
                st.warning("Faltan las credenciales. Configúralas en Settings → Secrets "
                           "(CORIL_API_BASE, CORIL_API_AUTH, CORIL_USER_ID).")
            else:
                st.caption("Prueba la descarga de una acción peruana desde la API de Coril.")
                _sym_test=st.text_input("Símbolo BVL de prueba",value="MINSURI1",key="coril_test_sym")
                if st.button("Probar descarga",key="coril_test_btn",use_container_width=True):
                    with st.spinner("Consultando la API de Coril…"):
                        _serie=coril_api.descargar_historico(_sym_test.strip().upper())
                    if _serie is None or len(_serie)<2:
                        st.error(f"No se obtuvo histórico para {_sym_test}. "
                                 "Verifica el símbolo (en mayúsculas) y las credenciales.")
                    else:
                        st.success(f"✅ Conexión OK · {len(_serie)} días de datos")
                        st.caption(f"Desde {_serie.index.min().date()} hasta {_serie.index.max().date()}")
                        st.caption(f"Último precio: {_serie.iloc[-1]:,.2f}")
                        st.line_chart(_serie)
                        _p_now=coril_api.precio_actual(_sym_test.strip().upper())
                        if _p_now: st.caption(f"Precio actual (endpoint): {_p_now:,.2f}")

# Descargar siempre con 15 años (fijo)
OPT_PERIOD = "15y"

# ═══════════════════ PANTALLA DE MODO ═════════════════════════════════════════
if st.session_state.mode is None:
    # Hero de bienvenida
    st.markdown("""
    <div style="text-align:center; padding: 24px 0 8px 0;">
        <div style="font-size:3.2rem; margin-bottom:4px;">📈</div>
        <h1 style="font-size:2.6rem; margin:0; color:#1a3a5c;">Simulador de inversiones</h1>
        <div style="font-size:1.1rem; color:#5a7290; font-weight:600; margin-top:2px;">Coril SAB</div>
        <p style="font-size:1.05rem; color:#64748b; max-width:640px; margin:16px auto 0 auto; line-height:1.6;">
            Diseña tu cartera de inversión y descubre cómo podría crecer tu dinero en el futuro.
            <b style="color:#2e5e8c;">Sencillo, visual, y sin necesidad de experiencia previa.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("<h3 style='text-align:center; color:#23405f;'>¿Cómo quieres empezar?</h3>", unsafe_allow_html=True)
    st.write("")
    cm1, cmg, cm2 = st.columns([1,0.08,1])
    with cm1:
        st.markdown("""
        <div style="background:white; border:2px solid #cfe0f0; border-radius:18px; padding:22px 24px;
                    box-shadow:0 4px 18px rgba(46,94,140,0.10); height:100%;">
            <div style="display:inline-block; background:#e8f2fc; color:#2e5e8c; font-size:0.75rem;
                        font-weight:700; padding:3px 10px; border-radius:20px; margin-bottom:10px;">
                ⭐ RECOMENDADO PARA EMPEZAR</div>
            <h3 style="margin:4px 0; color:#1a3a5c;">🤖 Modo Automático</h3>
            <p style="color:#64748b; font-size:0.95rem; line-height:1.55; margin:8px 0 14px 0;">
                Tú solo eliges en qué invertir y cuánto riesgo aceptas.
                El sistema hace todos los cálculos por ti.</p>
            <div style="color:#334155; font-size:0.92rem; line-height:1.9;">
                ✅ El sistema decide las mejores proporciones<br>
                ✅ Resultados y gráficos al instante<br>
                ✅ Perfecto para un primer análisis</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("🚀 Empezar con modo Automático", type="primary", use_container_width=True):
            st.session_state.mode="auto"; st.rerun()
    with cm2:
        st.markdown("""
        <div style="background:white; border:1.5px solid #e6edf5; border-radius:18px; padding:22px 24px;
                    box-shadow:0 2px 10px rgba(30,60,90,0.06); height:100%;">
            <div style="display:inline-block; background:#f1f5f9; color:#64748b; font-size:0.75rem;
                        font-weight:700; padding:3px 10px; border-radius:20px; margin-bottom:10px;">
                🎓 CON EXPERIENCIA</div>
            <h3 style="margin:4px 0; color:#1a3a5c;">🎛️ Modo Manual</h3>
            <p style="color:#64748b; font-size:0.95rem; line-height:1.55; margin:8px 0 14px 0;">
                Tú defines tus propias expectativas de retorno para cada
                inversión y controlas todos los supuestos.</p>
            <div style="color:#334155; font-size:0.92rem; line-height:1.9;">
                • Ingresas tus propias expectativas<br>
                • Ajustas el nivel de confianza de cada una<br>
                • Ideal si ya tienes una idea de inversión</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Usar modo Manual", type="secondary", use_container_width=True):
            st.session_state.mode="manual"; st.rerun()
    st.write("")
    st.info("💡 Podrás cambiar de modo cuando quieras. Nada es definitivo: "
            "es un simulador para explorar con tranquilidad.")
    st.stop()

AUTO = st.session_state.mode == "auto"

# Auto-descarga si no hay datos aún (primera vez tras añadir tickers)
if st.session_state.tickers and st.session_state.benchmarks and st.session_state.last_period and st.session_state.last_period!=OPT_PERIOD:
    with st.spinner("Actualizando datos (15y)…"): run_dl(OPT_PERIOD)

# ═══════════════════ MAIN ═════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
    <span style="font-size:1.8rem;">📈</span>
    <span style="font-size:1.7rem; font-weight:800; color:#1a3a5c;">Simulador de inversiones</span>
    <span style="font-size:0.9rem; color:#7a8ba0; font-weight:600; margin-top:6px;">Coril SAB</span>
</div>
""", unsafe_allow_html=True)

# Navegación tipo "pasos" (permite botones Siguiente/Atrás)
if AUTO:
    STEPS = ["Elige inversiones","Tu cartera","Proyección a futuro"]
else:
    STEPS = ["Elige inversiones","Tus expectativas","Tu cartera","Proyección a futuro"]

# Clamp del paso actual al rango válido
st.session_state.step = max(0, min(st.session_state.step, len(STEPS)-1))

# Barra de navegación superior (clicable) con estado visual
nav_cols = st.columns(len(STEPS))
for i,label in enumerate(STEPS):
    with nav_cols[i]:
        cur = st.session_state.step
        if i < cur:      icon="✓"      # completado
        elif i == cur:   icon="●"      # actual
        else:            icon=f"{i+1}"  # pendiente
        if st.button(f"{icon}  {label}", key=f"nav_{i}", use_container_width=True,
                     type="primary" if i==cur else "secondary"):
            st.session_state.step = i; st.rerun()
st.divider()

# Helpers de tab: cada bloque se ejecuta solo si es el paso activo.
# (Se usan flags en vez de st.tabs para poder navegar con botones.)
_step = st.session_state.step
if AUTO:
    show_tab1 = (_step==0); show_tab2=False; show_tab3=(_step==1); show_tab4=(_step==2)
else:
    show_tab1 = (_step==0); show_tab2=(_step==1); show_tab3=(_step==2); show_tab4=(_step==3)

def nav_buttons(back_to=None, next_to=None, next_label="Siguiente →", back_label="← Atrás"):
    """Dibuja botones de navegación Atrás / Siguiente."""
    c1,_,c3 = st.columns([1,3,1])
    if back_to is not None:
        with c1:
            if st.button(back_label, use_container_width=True, key=f"back_{back_to}"):
                st.session_state.step = back_to; st.rerun()
    if next_to is not None and next_label:
        with c3:
            if st.button(next_label, use_container_width=True, type="primary", key=f"next_{next_to}"):
                st.session_state.step = next_to; st.rerun()

# ═══════════════════ TAB 1 ════════════════════════════════════════════════════
if show_tab1:
    st.markdown("### 📥 Paso 1: Elige en qué invertir")
    st.write("Busca empresas o fondos y agrégalos a tu cartera. Si no sabes por dónde empezar, "
             "usa el botón **Cargar ejemplo** más abajo.")
    with st.expander("❓ No sé qué es renta variable ni renta fija: explícamelo"):
        st.markdown(
            """**Renta variable** 🔵 : inversiones que pueden subir o bajar bastante de valor,
como las **acciones** de empresas (Apple, Microsoft) o fondos que las agrupan (ETFs).
Ofrecen más ganancia potencial, pero también más riesgo.

**Renta fija** 🟢 : inversiones más estables y predecibles, como los **bonos** o los
**fondos de inversión** de deuda. Ganan menos, pero son más seguras. Sirven para dar
estabilidad a tu cartera.

**Benchmark** 📊 : un punto de referencia para comparar. Por ejemplo, el índice
**S&P 500** (^GSPC) representa a las 500 empresas más grandes de EE.UU. Sirve para saber
si tu cartera lo hace mejor o peor que "el mercado".

**Consejo:** una cartera equilibrada combina un poco de cada una. Cuánto de cada tipo
depende de tu **perfil de riesgo** (lo ajustas en la barra izquierda)."""
        )
    col_s,col_t=st.columns([4,1])
    with col_t: add_to=st.radio("Añadir como",["🔵 Renta variable","🟢 Renta fija","📊 Benchmark"],
                                help="Elige el tipo de inversión antes de buscar. "
                                     "Las sugerencias se filtran según lo que elijas.")
    with col_s: q=st.text_input("🔍 Buscar por nombre (escribe y presiona Enter)",
                                placeholder="Escribe el nombre de una empresa: Apple, Microsoft, Coca-Cola…")
    if q.strip():
        raw_res=search_yf(q.strip())
        res=filter_search(raw_res, add_to)
        if not res and raw_res:
            st.caption(f"ℹ️ No se encontraron resultados compatibles con **{add_to}**. "
                       f"Se encontraron {len(raw_res)} de otra clase.")
        if res:
            st.caption("Resultados de la búsqueda (pasa el cursor para ver detalles, pulsa para agregar):")
            # Traducción amigable del tipo de instrumento
            _tipo_map={"EQUITY":"Acción de empresa","ETF":"Fondo cotizado (ETF)",
                       "MUTUALFUND":"Fondo de inversión","INDEX":"Índice de mercado",
                       "CURRENCY":"Moneda","CRYPTOCURRENCY":"Criptomoneda"}
            cols=st.columns(min(len(res[:6]),3))
            for i,r in enumerate(res[:6]):
                with cols[i%len(cols)]:
                    _nombre = (r['nm'] or r['tk']).strip()
                    _nombre_corto = _nombre[:26] + ("…" if len(_nombre)>26 else "")
                    _tipo = _tipo_map.get(r.get('tp',''), r.get('tp','') or "Instrumento")
                    _bolsa = r.get('ex','')
                    _help = f"**{_nombre}**\n\nCódigo (ticker): {r['tk']}\n\nTipo: {_tipo}"
                    if _bolsa: _help += f"\n\nBolsa: {_bolsa}"
                    if st.button(f"➕ {_nombre_corto}  ({r['tk']})",key=f"a_{r['tk']}",
                                 use_container_width=True,help=_help):
                        tk=r['tk']
                        st.session_state.asset_names[tk]=_nombre   # recordar el nombre real
                        if add_to=="🔵 Renta variable":
                            if tk not in st.session_state.tickers:
                                _ok,_pn,_sa,_tot=puede_agregar(tk,capital)
                                if _ok:
                                    st.session_state.tickers.append(tk); st.toast(f"✓ {_nombre} agregado")
                                else:
                                    st.session_state["_bloqueo_precio"]=(_nombre,_pn,_sa,_tot,capital)
                        elif add_to=="🟢 Renta fija":
                            if tk not in st.session_state.rf_tickers: st.session_state.rf_tickers.append(tk); st.toast(f"✓ {_nombre} agregado")
                        else:
                            if tk not in st.session_state.benchmarks: st.session_state.benchmarks.append(tk); st.toast(f"✓ {_nombre} agregado")

    # Mensaje si se bloqueó una acción por exceder el monto
    if st.session_state.get("_bloqueo_precio"):
        _nb,_pn,_sa,_tot,_cap=st.session_state["_bloqueo_precio"]
        st.error(
            f"🚫 **No se agregó {_nb}.**\n\n"
            f"Su precio es **{usd(_pn)}** y, sumado a lo que ya elegiste (**{usd(_sa)}**), "
            f"llegaría a **{usd(_tot)}**, que supera tu monto de **{usd(_cap)}**.\n\n"
            f"👉 Sube el monto a invertir en la barra izquierda, o quita alguna acción antes de agregar esta."
        )
        del st.session_state["_bloqueo_precio"]

    # ── Inversiones populares (un clic, sin conocer tickers) ──────────────
    with st.expander("⭐ ¿No sabes qué agregar? Elige de las inversiones más populares", expanded=not st.session_state.tickers):
        if add_to=="🔵 Renta variable":
            _dest="tickers"; _lbl="RV"
            # Sub-selector: acciones peruanas (BVL) o extranjeras
            _mercado=st.radio("Mercado",["🌎 Extranjeras","🇵🇪 Peruanas (BVL)"],
                              horizontal=True,key="pop_mercado")
            _pop = POPULARES_BVL if "Peruanas" in _mercado else POPULARES_RV
            if "Peruanas" in _mercado and not (CORIL_OK and coril_api.api_configurada()):
                st.warning("La conexión con la API de Coril no está disponible. "
                           "Las acciones peruanas requieren esa conexión.")
                _pop=[]
        elif add_to=="🟢 Renta fija":
            _pop=POPULARES_RF; _dest="rf_tickers"; _lbl="RF"
        else:
            _pop=POPULARES_BK; _dest="benchmarks"; _lbl="Benchmark"
        st.caption(f"Mostrando opciones de **{add_to}**. Pulsa cualquiera para agregarla. "
                   "Cambia el tipo arriba a la derecha para ver otras.")
        pop_cols=st.columns(3)
        for i,(tk,nombre,emo) in enumerate(_pop):
            with pop_cols[i%3]:
                ya = tk in st.session_state.get(_dest,[])
                if st.button(f"{emo} {nombre}"+(" ✓" if ya else ""),
                             key=f"pop_{_dest}_{tk}",use_container_width=True,disabled=ya):
                    st.session_state.asset_names[tk]=nombre
                    if _dest=="tickers":
                        _ok,_pn,_sa,_tot=puede_agregar(tk,capital)
                        if _ok:
                            st.session_state[_dest].append(tk); st.toast(f"✓ {nombre} agregado"); st.rerun()
                        else:
                            st.session_state["_bloqueo_precio"]=(nombre,_pn,_sa,_tot,capital); st.rerun()
                    else:
                        st.session_state[_dest].append(tk); st.toast(f"✓ {nombre} agregado"); st.rerun()

    if not st.session_state.tickers and not st.session_state.rf_tickers:
        st.info("👇 ¿Primera vez? Pulsa aquí para cargar una cartera de ejemplo con empresas conocidas "
                "(Apple, Microsoft, Nvidia…) y explorar cómo funciona.")
        if st.button("🚀 Cargar ejemplo",type="primary"):
            st.session_state.tickers=list(EJ); st.session_state.views=[]; st.session_state.rf_tickers=[]

    # ── Listas: RV + RF + Benchmarks ─────────────────────────────────────
    # Precios actuales de las acciones de RV (para mostrar como guía)
    _precios_rv = fetch_precios(tuple(st.session_state.tickers)) if st.session_state.tickers else {}
    _suma_rv = sum(_precios_rv.values())
    la,lb,lc=st.columns(3)
    with la:
        st.caption(f"**🔵 Renta variable ({len(st.session_state.tickers)})**")
        for i,t in enumerate(st.session_state.tickers):
            c1,c2=st.columns([5,1])
            _n=nombre_activo(t)
            _pr=_precios_rv.get(t)
            _linea=f"**{_n}**" + (f"  ·  {t}" if _n!=t else "")
            if _pr is not None:
                _linea += f"<br><span style='color:#7a8ba0; font-size:0.8rem;'>Precio: ${_pr:,.2f}</span>"
            c1.markdown(_linea, unsafe_allow_html=True)
            if c2.button("✕",key=f"ra{i}"):
                rm=st.session_state.tickers.pop(i)
                st.session_state.views=[v for v in st.session_state.views if v.get("asset")!=rm and v.get("long")!=rm and v.get("short")!=rm]
                st.rerun()
        if _precios_rv:
            _color = "#2ca02c" if _suma_rv<=capital else "#d6604d"
            st.markdown(f"<div style='margin-top:6px; padding-top:6px; border-top:1px solid #e6edf5; "
                        f"font-size:0.85rem;'>Suma de precios: <b style='color:{_color};'>${_suma_rv:,.2f}</b> "
                        f"de ${capital:,.0f}</div>", unsafe_allow_html=True)
    with lb:
        st.caption(f"**🟢 Renta fija ({len(st.session_state.rf_tickers)})**")
        # Toggle FICO
        include_fico = st.checkbox("Incluir Fondo de inversión Coril (6.25%)", value=True, key="fico_toggle")
        st.session_state.include_fico = include_fico
        if include_fico:
            st.caption(f"✓ {FICO_DISPLAY} · {FICO.ret_annual:.2%} anual")
        for i,t in enumerate(st.session_state.rf_tickers):
            c1,c2=st.columns([5,1])
            _n=nombre_activo(t)
            c1.write(f"**{_n}**" + (f"  ·  {t}" if _n!=t else ""))
            if c2.button("✕",key=f"rrf{i}"): st.session_state.rf_tickers.pop(i)
        if not st.session_state.rf_tickers and not include_fico:
            st.warning("Sin activos de renta fija.")
    with lc:
        st.caption(f"**📊 Benchmarks ({len(st.session_state.benchmarks)})**")
        for i,b in enumerate(st.session_state.benchmarks):
            c1,c2=st.columns([5,1])
            _n=nombre_activo(b)
            c1.write(f"**{_n}**" + (f"  ·  {b}" if _n!=b else ""))
            if c2.button("✕",key=f"rb{i}"): st.session_state.benchmarks.pop(i)

    # ── Continuar (descarga automática) ──────────────────────────────────
    has_rf = bool(st.session_state.rf_tickers) or st.session_state.get("include_fico", True)
    has_any_asset = bool(st.session_state.tickers) or has_rf
    can_continue = bool(has_any_asset and st.session_state.benchmarks)

    # Verificar que el monto alcance para al menos una acción de cada una (acciones enteras)
    if st.session_state.tickers:
        _precios = fetch_precios(tuple(st.session_state.tickers))
        if _precios:
            _suma_min = sum(_precios.values())   # una unidad de cada acción
            if _suma_min > capital:
                _detalle = ", ".join(f"{nombre_activo(t)} ({usd(_precios[t])})"
                                     for t in st.session_state.tickers if t in _precios)
                st.warning(
                    f"⚠️ **El monto a invertir es menor que el precio de las acciones elegidas.**\n\n"
                    f"Comprar una unidad de cada acción costaría al menos **{usd(_suma_min)}**, "
                    f"pero tu monto es **{usd(capital)}**.\n\n"
                    f"Precios actuales: {_detalle}.\n\n"
                    f"👉 Sube el monto a invertir (en la barra izquierda) a por lo menos "
                    f"**{usd(_suma_min)}**, o quita alguna de las acciones más caras."
                )

    if st.session_state.data_range:
        st.success(f"📦 Datos cargados: {st.session_state.data_range}")

    st.divider()
    if not can_continue:
        st.caption("💡 Para continuar necesitas al menos un activo (renta variable o renta fija) "
                   "y un benchmark de referencia.")
    _next_label = "Continuar a Portafolio →" if AUTO else "Continuar a Expectativas →"
    cc_l,_,cc_r=st.columns([1,3,1])
    with cc_r:
        if st.button(_next_label, use_container_width=True, type="primary",
                     disabled=not can_continue, key="continue_dl"):
            # Descarga automática de datos si hace falta, luego avanza
            with st.spinner("Descargando datos y preparando el análisis…"):
                ok = run_dl(OPT_PERIOD)
            if ok:
                st.session_state.step = 1; st.rerun()
            else:
                _err=st.session_state.get("_dl_error") or "Verifica los tickers ingresados."
                st.error(f"No se pudieron descargar los datos. {_err}")
            # Diagnóstico BVL (temporal, para depurar la conexión Coril)
            _bvl_diag=st.session_state.get("_bvl_diag")
            if _bvl_diag:
                st.info(f"🔍 Diagnóstico BVL: {_bvl_diag}")

# ═══════════════════ TAB 2 (solo modo Manual) ═════════════════════════════════
if show_tab2:
    if True:
        if st.session_state.returns is None: st.info("⬅️ Descarga datos primero.")
        else:
            st.caption("Opcional: añade tus expectativas sobre algún activo.")
            vt=st.radio("",["Retorno de un activo","Un activo vs otro"],horizontal=True,label_visibility="collapsed")
            if vt=="Retorno de un activo":
                c1,c2,c3=st.columns([3,2,2])
                va=c1.selectbox("Activo",st.session_state.tickers,key="va")
                vq=c2.number_input("Ret. anual",value=0.10,step=0.01,format="%.2f",key="vq")
                vc=c3.slider("Confianza",0.1,1.0,0.5,0.1,key="vc")
                if st.button("Añadir"): st.session_state.views.append({"type":"absolute","asset":va,"q":float(vq),"confidence":float(vc)})
            else:
                c1,c2,c3,c4=st.columns(4)
                vl=c1.selectbox("Ganador",st.session_state.tickers,key="vl"); vs=c2.selectbox("Perdedor",st.session_state.tickers,key="vs")
                vq=c3.number_input("Dif.",value=0.05,step=0.01,format="%.2f",key="vqr"); vc=c4.slider("Conf.",0.1,1.0,0.5,0.1,key="vcr")
                if st.button("Añadir"):
                    if vl!=vs: st.session_state.views.append({"type":"relative","long":vl,"short":vs,"q":float(vq),"confidence":float(vc)})
            for i,v in enumerate(st.session_state.views):
                c1,c2=st.columns([6,1])
                if v["type"]=="absolute":
                    c1.caption(f"📌 {v['asset']} → {v['q']:.0%} (conf. {v['confidence']:.0%})")
                else:
                    c1.caption(f"📌 {v['long']} > {v['short']} por {v['q']:.0%} (conf. {v['confidence']:.0%})")
                if c2.button("✕",key=f"rv{i}"): st.session_state.views.pop(i)
            st.divider()
            st.caption("Las views son opcionales. Puedes continuar sin agregar ninguna.")
            nav_buttons(back_to=0, next_to=2, next_label="Ver Portafolio →")

# ═══════════════════ TAB 3 ════════════════════════════════════════════════════
if show_tab3:
    # Validación previa coherente con el perfil elegido.
    _eq_validos = [a for a in st.session_state.tickers
                   if st.session_state.returns is not None and a in st.session_state.returns.columns] \
                  if st.session_state.returns is not None else []
    _rf_validos = [a for a in st.session_state.rf_tickers
                   if st.session_state.returns is not None and a in st.session_state.returns.columns] \
                  if st.session_state.returns is not None else []
    _hay_rf = bool(_rf_validos) or st.session_state.get("include_fico",True)
    _needs_rv = eq_t > 1e-6
    _needs_rf = fi_t > 1e-6
    if st.session_state.returns is None:
        st.info("⬅️ Primero agrega activos en la pestaña **Activos** y pulsa Continuar.")
    elif _needs_rv and not _eq_validos:
        st.warning(f"⚠️ Tu perfil tiene **{eq_t:.0%}** en renta variable, pero no agregaste ningún "
                   "activo de renta variable. Agrega al menos uno en la pestaña **Activos**, "
                   "o baja el porcentaje de renta variable a 0% para un portafolio solo de renta fija.")
    elif _needs_rf and not _hay_rf:
        st.warning(f"⚠️ Tu perfil tiene **{fi_t:.0%}** en renta fija, pero no agregaste ningún activo "
                   "de renta fija ni activaste el Fondo de inversión. Agrega uno en la pestaña "
                   "**Activos**, o sube la renta variable a 100% para un portafolio solo de renta variable.")
    else:
        if AUTO:
            # Modo automático: optimiza al entrar (o al pulsar recalcular).
            # Auto-optimización: recalcula solo si cambió algún input relevante.
            sig = (tuple(sorted(st.session_state.tickers)),
                   tuple(sorted(st.session_state.rf_tickers)),
                   bool(st.session_state.get("include_fico",True)),
                   eq_t, fi_t,
                   round(float(st.session_state.get("gk_ic",0.05)),4))
            changed = st.session_state.get("_auto_sig") != sig
            if changed or not st.session_state.optimized or st.session_state.result is None:
                pb=list(st.session_state.bench_rets.values())[0]
                with st.spinner("Generando views (Grinold-Kahn) y optimizando…"):
                    try:
                        r=do_opt(st.session_state.tickers, st.session_state.rf_tickers,
                                 st.session_state.get("include_fico",True),
                                 st.session_state.views, eq_t, fi_t, pb, auto=True)
                    except ValueError as e:
                        st.error(f"No se pudo optimizar: {e}")
                        st.stop()
                for k in list(st.session_state.keys()):
                    if k.startswith("s_"): del st.session_state[k]
                st.session_state.result=r; st.session_state.manual_weights=r.weights.copy(); st.session_state.optimized=True
                st.session_state._auto_sig = sig
                st.session_state["_w_ver"] = st.session_state.get("_w_ver",0)+1  # nueva versión de campos
                for x in ["mc","stress"]:
                    if x in st.session_state: del st.session_state[x]
            st.markdown("### 📊 Tu cartera está lista")
            st.success("✨ Calculamos automáticamente las mejores proporciones para tu cartera, "
                       "según tu perfil de riesgo y el comportamiento reciente de cada inversión. "
                       "Puedes ajustar los porcentajes abajo si quieres · todo se actualiza al instante.")
        else:
            # Modo manual: recalcula solo cuando cambian activos, perfil o views.
            def _views_key(vs):
                return tuple((v.get("type"),v.get("asset"),v.get("long"),v.get("short"),
                             round(float(v.get("q",0)),4),round(float(v.get("confidence",0)),4)) for v in vs)
            man_sig=(tuple(sorted(st.session_state.tickers)),
                     tuple(sorted(st.session_state.rf_tickers)),
                     bool(st.session_state.get("include_fico",True)),
                     eq_t,fi_t,_views_key(st.session_state.views))
            man_changed = st.session_state.get("_man_sig")!=man_sig
            force = st.button("🔄 Recalcular ahora",use_container_width=True,
                              help="Vuelve a optimizar descartando ajustes manuales de pesos.")
            if man_changed or force or not st.session_state.optimized or st.session_state.result is None:
                pb=list(st.session_state.bench_rets.values())[0]
                with st.spinner("Optimizando…"):
                    r=do_opt(st.session_state.tickers, st.session_state.rf_tickers,
                             st.session_state.get("include_fico",True),
                             st.session_state.views, eq_t, fi_t, pb)
                for k in list(st.session_state.keys()):
                    if k.startswith("s_"): del st.session_state[k]
                st.session_state.result=r; st.session_state.manual_weights=r.weights.copy(); st.session_state.optimized=True
                st.session_state._man_sig=man_sig
                st.session_state["_w_ver"] = st.session_state.get("_w_ver",0)+1  # nueva versión de campos
                for x in ["mc","stress"]:
                    if x in st.session_state: del st.session_state[x]

        if st.session_state.optimized and st.session_state.result:
            res=st.session_state.result
            _wv=st.session_state.get("_w_ver",0)   # versión: refresca campos al recalcular
            st.markdown("##### ⚖️ Cuánto poner en cada inversión")
            st.caption("Estos son los porcentajes sugeridos para tu cartera. Puedes ajustarlos a mano "
                       "si quieres: todo se actualiza al instante. Los porcentajes deberían sumar 100%.")
            # Pesos en 2 columnas lado a lado
            assets=list(res.weights.index); mid=len(assets)//2+len(assets)%2
            col_a,col_b,col_r=st.columns([2,2,1.5])
            nw={}
            with col_a:
                for a in assets[:mid]:
                    ic="🟢" if a==FICO_TK else "🔵"
                    nw[a]=st.number_input(f"{ic} {disp(a)}",0.0,100.0,round(float(res.weights[a])*100,1),0.5,"%.1f",key=f"s_{_wv}_{a}")
            with col_b:
                for a in assets[mid:]:
                    ic="🟢" if a==FICO_TK else "🔵"
                    nw[a]=st.number_input(f"{ic} {disp(a)}",0.0,100.0,round(float(res.weights[a])*100,1),0.5,"%.1f",key=f"s_{_wv}_{a}")
            wn=pd.Series(nw); tot=wn.sum(); wnorm=wn/tot if tot>0 else wn/100; st.session_state.manual_weights=wnorm
            eqw=float(wnorm[[a for a in wnorm.index if a!=FICO_TK]].sum()); fiw=float(wnorm.get(FICO_TK,0))
            with col_r:
                st.metric("RV",f"{eqw:.1%}",delta=f"{eqw-eq_t:+.1%}"); st.metric("RF",f"{fiw:.1%}",delta=f"{fiw-fi_t:+.1%}")
                st.caption(f"Suma: {tot:.0f}%{'✅' if abs(tot-100)<0.5 else ' ⚠️→100%'}")

            # Métricas dinámicas
            w_np=wnorm.reindex(res.bl_returns.index).fillna(0).to_numpy()
            mu_np=res.bl_returns.to_numpy(); S_np=res.cov_matrix.to_numpy()
            b_np=st.session_state.betas.reindex(res.bl_returns.index).fillna(1).to_numpy()
            p_r=float(w_np@mu_np); p_v=float(np.sqrt(max(w_np@S_np@w_np,1e-10)))
            p_sh=(p_r-RF)/p_v if p_v>1e-10 else 0; p_bt=float(w_np@b_np)
            st.markdown("##### 🎯 Métricas esperadas del portafolio")
            st.caption("Esto es lo que el modelo espera de tu cartera. Pasa el cursor sobre el "
                       "signo ❓ de cada número para entender qué significa.")
            mbig, mrest = st.columns([1.3,2.2])
            with mbig:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#eef5fc 0%,#dcebf9 100%);
                            border:1.5px solid #cfe0f0; border-radius:14px; padding:16px 20px;">
                    <div style="font-size:0.85rem; color:#5a7290; font-weight:600;">Retorno esperado anual</div>
                    <div style="font-size:2.6rem; font-weight:800; color:#2e5e8c; line-height:1.1;">{p_r:.1%}</div>
                    <div style="font-size:0.8rem; color:#7a8ba0;">Lo que el modelo estima por año</div>
                </div>
                """, unsafe_allow_html=True)
            with mrest:
                mm1,mm2,mm3=st.columns(3)
                mm1.metric("Riesgo",f"{p_v:.1%}",
                           help="Volatilidad anual. Qué tanto puede moverse el portafolio en un año. "
                                "Más alto = más incertidumbre.")
                mm2.metric("Sharpe",f"{p_sh:.2f}",
                           help="Retorno por unidad de riesgo. Mide cuánto rendimiento extra obtienes por "
                                "cada punto de riesgo. Más alto es mejor; arriba de 1 se considera bueno.")
                mm3.metric("Beta",f"{p_bt:.2f}",
                           help="Sensibilidad al mercado. Beta 1 = se mueve igual que el mercado; "
                                "menor a 1 = más defensivo; mayor a 1 = más agresivo.")

            # Gráficos: composición por activo + por sector
            st.markdown("##### 🥧 Composición del portafolio")
            import plotly.express as px
            g1,g2=st.columns(2)
            with g1:
                st.caption("Por activo")

                ws=wnorm[wnorm>1e-4]
                colors=px.colors.qualitative.Set2[:len(ws)]
                _nombres=[nombre_activo(a) for a in ws.index.tolist()]
                fig=go.Figure(go.Pie(labels=[disp(a) for a in ws.index.tolist()],values=ws.values.tolist(),
                    marker_colors=colors,hole=.4,textinfo="label+percent",
                    customdata=_nombres,
                    hovertemplate="<b>%{customdata}</b><br>Peso: %{percent}<extra></extra>"))
                fig.update_layout(height=280,margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
                st.plotly_chart(fig,use_container_width=True,key="chart_pie_activo",config={"displayModeBar":False})
            with g2:
                st.caption("Por sector")
                sec=st.session_state.sectors
                sw={}                 # sector -> peso total
                sw_assets={}          # sector -> lista de nombres de activos
                for a in wnorm.index:
                    if wnorm[a]>1e-4:
                        # Sector con fallbacks útiles según el tipo de activo
                        if a==FICO_TK:
                            s="Fondo de inversión"
                        elif a in st.session_state.rf_tickers:
                            s="Renta fija"
                        else:
                            s=(sec.get(a) if sec is not None else None) or "Otros"
                            if s in ("–","",None): s="Otros"
                        sw[s]=sw.get(s,0)+wnorm[a]
                        sw_assets.setdefault(s,[]).append(nombre_activo(a))
                if sw and not (len(sw)==1 and list(sw.keys())[0]=="Otros"):
                    sec_colors=px.colors.qualitative.Pastel[:len(sw)]
                    _labels=list(sw.keys())
                    # Texto de hover: qué activos componen cada sector
                    _hover=["<b>"+lbl+"</b><br>"+", ".join(sw_assets[lbl][:6])
                            +(f" y {len(sw_assets[lbl])-6} más" if len(sw_assets[lbl])>6 else "")
                            for lbl in _labels]
                    fig=go.Figure(go.Pie(labels=_labels,values=list(sw.values()),
                        marker_colors=sec_colors,hole=.4,textinfo="label+percent",
                        customdata=_hover,
                        hovertemplate="%{customdata}<br>Peso: %{percent}<extra></extra>"))
                    fig.update_layout(height=280,margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
                    st.plotly_chart(fig,use_container_width=True,key="chart_pie_sector",config={"displayModeBar":False})
                    if "Otros" in sw:
                        st.caption("💡 Pasa el cursor sobre el gráfico para ver qué inversiones "
                                   "hay en cada sector, incluido \"Otros\".")
                else:
                    st.caption("ℹ️ No hay información de sectores disponible para estos activos "
                               "(por ejemplo, si son solo instrumentos de renta fija).")

            # ── PLAN DE COMPRA EN ACCIONES ENTERAS ──────────────────────────
            st.markdown("##### 🧾 Plan de compra — cuántas acciones comprar")
            st.caption("El portafolio óptimo se calcula en porcentajes, pero las acciones se compran "
                       "completas (no puedes comprar media acción). Aquí traducimos esos porcentajes "
                       "al número exacto de acciones que comprarías con tu monto de "
                       f"**{usd(capital)}**.")
            _acc_rv=[a for a in wnorm.index if a in st.session_state.tickers]
            _precios_plan=fetch_precios(tuple(_acc_rv)) if _acc_rv else {}
            if not _precios_plan:
                st.info("No hay precios disponibles para calcular el plan de compra "
                        "(puede pasar si solo tienes renta fija o fondos).")
            else:
                # Fraccionables: Fondo de inversión + ETFs de renta fija (aceptan monto libre)
                _fraccionables=set(st.session_state.rf_tickers) | {FICO_TK}
                _pesos={a: float(wnorm[a]) for a in wnorm.index}
                _unid,_mfrac,_efectivo,_gastado=acciones_enteras(_pesos,_precios_plan,capital,_fraccionables)

                st.markdown("**🔵 Acciones (se compran por unidades enteras)**")
                _filas_acc=[]
                _monto_optimo_acc=0.0   # costo de comprar el óptimo completo (acciones)
                _monto_alcanza_acc=0.0  # costo de lo que sí alcanza
                for a in wnorm.index:
                    if a not in _precios_plan: continue
                    _precio=_precios_plan[a]
                    _obj_monto=_pesos[a]*capital            # dinero que pide el óptimo
                    _ideal=_obj_monto/_precio                # acciones ideales (con decimales)
                    _necesarias=max(1,round(_ideal)) if _pesos[a]>1e-4 else 0  # óptimo en enteros
                    _alcanzan=_unid.get(a,0)                 # las que caben con tu monto
                    _monto_optimo_acc += _necesarias*_precio
                    _monto_alcanza_acc += _alcanzan*_precio
                    _filas_acc.append({
                        "Empresa": nombre_activo(a),
                        "Precio x acción": f"${_precio:,.2f}",
                        "Necesarias (óptimo)": f"{_necesarias}",
                        "Te alcanzan": f"{_alcanzan}",
                        "Faltan por comprar": f"{max(0,_necesarias-_alcanzan)}",
                        "Costo del óptimo": f"${_necesarias*_precio:,.0f}",
                    })
                if _filas_acc:
                    st.dataframe(pd.DataFrame(_filas_acc),use_container_width=True,hide_index=True)
                    st.caption("**Cómo leer la tabla:** *Necesarias* son las acciones que tendrías que "
                               "comprar de cada empresa para armar el portafolio óptimo. *Te alcanzan* son "
                               "las que puedes comprar con tu monto actual. *Faltan por comprar* es la "
                               "diferencia que te queda pendiente por falta de dinero.")

                # Fraccionables (Fondo / RF): se muestran aparte porque sí aceptan montos exactos
                _filas_frac=[]
                _monto_frac=0.0
                for a in wnorm.index:
                    if a in _precios_plan: continue
                    _m_obj=_pesos[a]*capital          # monto óptimo para este instrumento
                    _monto_frac += _m_obj
                    _filas_frac.append({
                        "Instrumento": nombre_activo(a),
                        "Objetivo %": f"{_pesos[a]:.1%}",
                        "Monto a invertir": f"${_mfrac.get(a,_m_obj):,.0f}",
                    })
                if _filas_frac:
                    st.markdown("**🟢 Renta fija y fondos (se invierte el monto exacto, sin comprar unidades)**")
                    st.dataframe(pd.DataFrame(_filas_frac),use_container_width=True,hide_index=True)

                # ── Monto mínimo para el portafolio óptimo completo ──
                # El óptimo completo = costo de las acciones necesarias + la parte de RF/fondos,
                # escalada para mantener las proporciones del perfil.
                _peso_acc=sum(_pesos[a] for a in wnorm.index if a in _precios_plan)
                if _peso_acc>1e-6:
                    # Si las acciones deben representar _peso_acc del total, el total mínimo es:
                    _monto_minimo = _monto_optimo_acc / _peso_acc
                else:
                    _monto_minimo = _monto_optimo_acc + _monto_frac

                st.markdown("**Resumen**")
                _invertido=capital-_efectivo
                pcol1,pcol2,pcol3=st.columns(3)
                pcol1.metric("💰 Total invertido",f"${_invertido:,.0f}",delta=f"{_invertido/capital:.0%} de tu monto")
                pcol2.metric("🏦 En acciones",f"${_gastado:,.0f}")
                pcol3.metric("💵 Te sobra (efectivo)",f"${_efectivo:,.0f}",
                             delta=None if _efectivo<1 else f"{_efectivo/capital:.1%}",delta_color="off")

                # Mensaje sobre el monto mínimo
                if _monto_minimo > capital*1.08:   # margen del 8% para redondeos de acciones enteras
                    _falta_dinero=_monto_minimo-capital
                    st.warning(
                        f"📌 **Para armar el portafolio óptimo completo necesitas al menos "
                        f"{usd(_monto_minimo)}.**\n\n"
                        f"Con tu monto actual de {usd(capital)} te faltan aproximadamente "
                        f"**{usd(_falta_dinero)}** para comprar todas las acciones necesarias "
                        f"en las proporciones ideales. Puedes subir el monto a invertir, o quedarte "
                        f"con la cartera que sí alcanza (mostrada arriba)."
                    )
                else:
                    st.success(
                        f"✅ **Tu monto de {usd(capital)} alcanza para armar el portafolio óptimo "
                        f"completo** (costo mínimo aproximado: {usd(_monto_minimo)}). "
                        f"El sobrante se coloca en el Fondo de inversión / renta fija."
                    )



            # Evolución histórica — filtrada por chart_years, siempre desde capital inicial
            st.markdown("##### 📈 Evolución histórica del capital")
            chart_years=st.selectbox("Ver gráfico desde hace",["1y","2y","3y","5y","10y","15y"],index=3,
                                     key="chart_years_sel",
                                     help="Solo afecta la vista de este gráfico. La optimización siempre usa 15 años.")
            pr_full,wl_full,dd_full,bw_full,bdd_full=wdd(wnorm,st.session_state.returns,st.session_state.bench_rets,capital)

            # Filtrar al rango visual seleccionado
            cy = int(chart_years.replace("y",""))
            cutoff = pr_full.index.max() - pd.DateOffset(years=cy)
            pr = pr_full.loc[pr_full.index >= cutoff]

            # Recalcular wealth desde capital inicial para el rango visible
            wl = np.exp(pr.cumsum()) * capital
            dd = wl / wl.cummax() - 1
            bw, bdd = {}, {}
            for n in bw_full:
                br = st.session_state.bench_rets[n].loc[pr.index].fillna(0) if n in st.session_state.bench_rets else pd.Series(0,index=pr.index)
                bw[n] = np.exp(br.cumsum()) * capital
                bdd[n] = bw[n] / bw[n].cummax() - 1

            fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.65,.35],vertical_spacing=.04,
                             subplot_titles=[f"Evolución de capital, últimos {cy} años (${capital:,.0f})","Drawdown"])
            fig.add_trace(go.Scatter(x=wl.index,y=wl.values,name="Portafolio",
                                    line=dict(color=C_RV,width=2.5)),row=1,col=1)
            for i,(n,v) in enumerate(bw.items()):
                fig.add_trace(go.Scatter(x=v.index,y=v.values,name=nombre_activo(n),
                    line=dict(color=BC[i%len(BC)],dash="dash",width=1.5)),row=1,col=1)
            fig.add_trace(go.Scatter(x=dd.index,y=dd.values,name="DD Portafolio",
                                    fill="tozeroy",fillcolor="rgba(214,96,77,0.3)",
                                    line=dict(color=C_OPT,width=1.5)),row=2,col=1)
            for i,(n,ddb) in enumerate(bdd.items()):
                fig.add_trace(go.Scatter(x=ddb.index,y=ddb.values,name=f"DD {n}",
                    line=dict(color=BC[i%len(BC)],dash="dot",width=1)),row=2,col=1)
            _prof=RiskProfile.for_split(eq_t,fi_t)
            fig.add_hline(y=-_prof.max_drawdown,line_dash="dash",line_color="black",
                         row=2,col=1,annotation_text=f"Límite {_prof.max_drawdown:.0%}")
            fig.update_yaxes(tickprefix="$",tickformat=",.0f",row=1,col=1)
            fig.update_yaxes(tickformat=".0%",row=2,col=1)
            fig.update_layout(height=520,margin=dict(l=0,r=0,t=25,b=0),
                             legend=dict(orientation="h",y=-0.08,font=dict(size=10)))
            st.plotly_chart(fig,use_container_width=True,key="chart_evol_hist",config={"displayModeBar":False})

            # Métricas históricas del rango visible
            ann_r=np.exp(pr.mean()*PPY)-1; ann_v=pr.std(ddof=1)*np.sqrt(PPY)
            from scipy.stats import norm as _norm
            mu_h=pr.mean()*PPY; sig_h=ann_v; z=_norm.ppf(0.05)
            var95=-(mu_h+z*sig_h); cvar95=-(mu_h-sig_h*_norm.pdf(z)/0.05)
            st.markdown("##### 📅 Comportamiento histórico del portafolio")
            st.caption("Estos números resumen cómo se habría comportado esta combinación de activos "
                       "en el pasado, según los datos descargados.")
            h1,h2,h3,h4,h5=st.columns(5)
            h1.metric("Retorno histórico anual",f"{ann_r:.2%}",
                      help="Cuánto habría rendido el portafolio en promedio por año, según su historia. "
                           "No es una promesa a futuro, es lo que ocurrió en el pasado.")
            h2.metric("Volatilidad anual",f"{ann_v:.2%}",
                      help="Qué tanto sube y baja el portafolio. Más alta = más movimiento y más riesgo. "
                           "Es la desviación estándar de los retornos, anualizada.")
            h3.metric("Caída máxima",f"{dd.min():.2%}",
                      help="La peor caída desde un punto alto hasta el punto más bajo que sufrió el "
                           "portafolio en el período. Mide el peor mal momento que habrías vivido.")
            h4.metric("VaR 95%",f"{var95:.2%}",
                      help="Value at Risk. En el 5% de los peores años, esperarías perder al menos "
                           "este porcentaje. Es el umbral de pérdida en un mal escenario.")
            h5.metric("CVaR 95%",f"{cvar95:.2%}",
                      help="Conditional VaR o Expected Shortfall. Cuando las cosas van realmente mal "
                           "(ese peor 5%), esta es la pérdida promedio. Siempre es peor que el VaR "
                           "porque mira el promedio de la cola, no solo el umbral.")

            # ── COMPARATIVA vs BENCHMARK(S) ────────────────────────────────
            st.divider()
            st.markdown("##### ⚖️ Tu portafolio frente a los benchmarks")
            st.caption("Comparación de las métricas clave contra los índices de referencia elegidos, "
                       "sobre el mismo período. Se actualiza sola al agregar o quitar benchmarks "
                       "en la pestaña Activos.")

            def _metrics_from_rets(r):
                """Retorno anual, vol anual, Sharpe y drawdown máx de una serie de log-retornos."""
                r = r.dropna()
                if len(r) < 2:
                    return None
                ar = np.exp(r.mean()*PPY)-1
                av = r.std(ddof=1)*np.sqrt(PPY)
                sh = (ar-RF)/av if av>1e-10 else float("nan")
                wl_ = np.exp(r.cumsum())
                mdd = float((wl_/wl_.cummax()-1).min())
                return {"ret":ar,"vol":av,"sharpe":sh,"dd":mdd}

            # Portafolio (rango visible)
            rows = []
            pm = _metrics_from_rets(pr)
            rows.append(("📊 Tu portafolio", pm, True))
            # Cada benchmark sobre el mismo rango de fechas
            for n in st.session_state.bench_rets:
                br = st.session_state.bench_rets[n].reindex(pr.index).dropna()
                bm = _metrics_from_rets(br)
                if bm is not None:
                    rows.append((nombre_activo(n), bm, False))

            if len(rows) > 1:
                # Tabla comparativa
                comp = pd.DataFrame(
                    {"Retorno anual":[f"{r['ret']:.2%}" for _,r,_ in rows],
                     "Volatilidad":[f"{r['vol']:.2%}" for _,r,_ in rows],
                     "Sharpe":[f"{r['sharpe']:.2f}" for _,r,_ in rows],
                     "Caída máx":[f"{r['dd']:.2%}" for _,r,_ in rows]},
                    index=[nm for nm,_,_ in rows])
                st.dataframe(comp, use_container_width=True)

                # Gráfico de barras agrupadas: retorno y volatilidad
                cbar1, cbar2 = st.columns(2)
                names = [nm for nm,_,_ in rows]
                bar_colors = [C_RV] + [BC[i%len(BC)] for i in range(len(rows)-1)]
                with cbar1:
                    st.caption("Retorno anual")
                    fr = go.Figure(go.Bar(x=names, y=[r['ret'] for _,r,_ in rows],
                        marker_color=bar_colors, text=[f"{r['ret']:.1%}" for _,r,_ in rows],
                        textposition="outside"))
                    fr.update_yaxes(tickformat=".0%"); fr.update_layout(height=260,margin=dict(l=0,r=0,t=5,b=0))
                    st.plotly_chart(fr,use_container_width=True,key="chart_bench_ret",config={"displayModeBar":False})
                with cbar2:
                    st.caption("Sharpe (retorno ajustado por riesgo)")
                    fs = go.Figure(go.Bar(x=names, y=[r['sharpe'] for _,r,_ in rows],
                        marker_color=bar_colors, text=[f"{r['sharpe']:.2f}" for _,r,_ in rows],
                        textposition="outside"))
                    fs.update_layout(height=260,margin=dict(l=0,r=0,t=5,b=0))
                    st.plotly_chart(fs,use_container_width=True,key="chart_bench_sharpe",config={"displayModeBar":False})

                # Lectura automática
                best_sh = max(rows, key=lambda x: (x[1]['sharpe'] if np.isfinite(x[1]['sharpe']) else -99))
                port_sh = rows[0][1]['sharpe']
                if best_sh[2]:
                    st.success("Tu portafolio tiene el **mejor retorno ajustado por riesgo (Sharpe)** "
                               "de la comparación. Eso significa que estás obteniendo más rendimiento "
                               "por cada unidad de riesgo que los índices de referencia.")
                else:
                    st.info(f"El benchmark **{best_sh[0]}** tiene mejor Sharpe ({best_sh[1]['sharpe']:.2f}) "
                            f"que tu portafolio ({port_sh:.2f}) en este período. Esto puede deberse a que "
                            f"tu portafolio prioriza otros objetivos (menor caída, perfil de riesgo, "
                            f"diversificación), no solo el retorno ajustado por riesgo.")

                # ── Rendimiento año por año (barras agrupadas) ──────────────
                st.markdown("###### 📊 Rendimiento por año")
                st.caption("Retorno de cada año calendario, comparado con los benchmarks. "
                           "Usa el mismo rango del gráfico de evolución de arriba.")

                def _yearly_returns(r):
                    """Retorno por año calendario a partir de log-retornos."""
                    r = r.dropna()
                    if len(r) < 2: return {}
                    by_year = {}
                    for yr, grp in r.groupby(r.index.year):
                        by_year[int(yr)] = float(np.exp(grp.sum())-1)
                    return by_year

                # Portafolio y benchmarks, restringidos al rango visible (pr ya está filtrado)
                port_yearly = _yearly_returns(pr)
                years_sorted = sorted(port_yearly.keys())
                if years_sorted:
                    figy = go.Figure()
                    figy.add_trace(go.Bar(x=[str(y) for y in years_sorted],
                        y=[port_yearly[y] for y in years_sorted],
                        name="📊 Tu portafolio", marker_color=C_RV,
                        text=[f"{port_yearly[y]:+.0%}" for y in years_sorted], textposition="outside"))
                    for i,n in enumerate(st.session_state.bench_rets):
                        bser = st.session_state.bench_rets[n].reindex(pr.index)
                        by = _yearly_returns(bser)
                        figy.add_trace(go.Bar(x=[str(y) for y in years_sorted],
                            y=[by.get(y) for y in years_sorted],
                            name=nombre_activo(n), marker_color=BC[i%len(BC)]))
                    figy.update_yaxes(tickformat=".0%")
                    figy.update_layout(barmode="group",height=320,margin=dict(l=0,r=0,t=5,b=0),
                                       legend=dict(orientation="h",y=1.12,font=dict(size=10)))
                    st.plotly_chart(figy,use_container_width=True,key="chart_yearly_bench",config={"displayModeBar":False})
            else:
                st.caption("Agrega uno o más benchmarks en la pestaña **Activos** para ver la comparación.")

        # Navegación al final del portafolio
        if st.session_state.optimized and st.session_state.result:
            st.divider()
            _proj_step = 2 if AUTO else 3
            _back_step = 0 if AUTO else 1
            nav_buttons(back_to=_back_step, next_to=_proj_step, next_label="Ver Proyecciones →")

# ═══════════════════ TAB 4 ════════════════════════════════════════════════════
if show_tab4:
    if not(st.session_state.optimized and st.session_state.result):
        st.info("⬅️ Primero define tu cartera en el paso anterior para ver la proyección.")
    else:
        st.markdown("### 🔮 ¿Cómo podría crecer tu inversión?")
        st.write("Aquí simulamos miles de futuros posibles para tu cartera y te mostramos el rango "
                 "de resultados que podrías esperar. Ajusta los años y la meta abajo.")
        res=st.session_state.result
        wnorm=st.session_state.manual_weights if st.session_state.manual_weights is not None else res.weights

        # ── CUADRO RESUMEN DESTACADO (arriba de todo — dato clave para el cliente) ──
        # Controles de la proyección (compactos, arriba)
        cc1,cc2,cc3=st.columns(3)
        mh=cc1.selectbox("Años a proyectar",[1,2,3,5,10],index=2,
                        help="Horizonte de la proyección. En cuántos años quieres ver cómo evoluciona la inversión.")
        mn=cc2.selectbox("Precisión",[1000,5000,10000],index=1,format_func=lambda x:f"{x:,}",
                        help="Cuántos futuros posibles simular. Más simulaciones = resultado más estable, pero más lento.")
        mt=cc3.number_input("Meta (USD)",value=int(capital*1.2),step=10_000,format="%d",
                           help="El capital que se desea alcanzar. Se calcula la probabilidad de llegar a esta meta.")
        # Proyección automática: recalcula si cambian parámetros o pesos
        mc_sig=(round(float(wnorm.sum()),6),tuple(round(float(x),6) for x in wnorm.values),
                int(mh),int(mn),int(mt),round(float(capital),2))
        if st.session_state.get("_mc_sig")!=mc_sig or "mc" not in st.session_state:
            with st.spinner("Calculando proyección…"):
                st.session_state["mc"]=monte_carlo(wnorm,res.bl_returns,res.cov_matrix,capital,mh,PPY,mn,mt)
            st.session_state._mc_sig=mc_sig
        mc=st.session_state.get("mc")
        if mc:
            p5_top=mc.percentiles[5][-1]; p50_top=mc.median_path[-1]; p95_top=mc.percentiles[95][-1]
            _g5=p5_top/mc.capital-1; _g50=p50_top/mc.capital-1; _g95=p95_top/mc.capital-1
            st.markdown(f"### 🔮 Proyección de tu inversión a {mh} año(s)")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#2e5e8c 0%,#3d7ab8 100%); border-radius:16px;
                        padding:20px 24px; color:white; box-shadow:0 6px 22px rgba(46,94,140,0.28); margin-bottom:14px;">
                <div style="font-size:0.9rem; opacity:0.85;">Tu inversión de ${mc.capital:,.0f} podría convertirse en</div>
                <div style="font-size:2.8rem; font-weight:800; line-height:1.1; margin:4px 0;">${p50_top:,.0f}</div>
                <div style="font-size:0.95rem; opacity:0.9;">en el escenario más probable
                    ({'ganancia' if _g50>=0 else 'pérdida'} de {abs(_g50):.1%})</div>
            </div>
            """, unsafe_allow_html=True)
            tcards = [
                ("😟 Escenario pesimista", p5_top, _g5, "#d6604d", "Si el mercado va mal"),
                ("🎯 Escenario base", p50_top, _g50, "#2e5e8c", "Lo más probable"),
                ("🚀 Escenario optimista", p95_top, _g95, "#2ca02c", "Si el mercado va bien"),
            ]
            tcols=st.columns(3)
            for col,(titulo,valor,gan,color,sub) in zip(tcols,tcards):
                col.markdown(f"""
                <div style="background:white; border-top:4px solid {color}; border-radius:12px;
                            padding:14px 16px; box-shadow:0 2px 10px rgba(30,60,90,0.07); text-align:center;">
                    <div style="font-size:0.82rem; color:#64748b; font-weight:600;">{titulo}</div>
                    <div style="font-size:1.7rem; font-weight:800; color:{color}; margin:4px 0;">${valor:,.0f}</div>
                    <div style="font-size:0.8rem; color:#94a3b8;">{sub} · {gan:+.1%}</div>
                </div>
                """, unsafe_allow_html=True)
        st.divider()

        # MONTE CARLO (detalle)
        st.subheader("🎲 ¿Cuánto podría valer el portafolio? (detalle)")
        if "mc" in st.session_state and st.session_state["mc"]:
            mc=st.session_state["mc"]; gain=mc.median_path[-1]-mc.capital
            c1,c2,c3=st.columns(3)
            c1.metric(f"💰 Valor proyectado a {mh} año(s)",f"${mc.median_path[-1]:,.0f}",delta=f"+${gain:,.0f} ({gain/mc.capital:+.1%})",
                      help="El valor estimado (escenario base) de la inversión al final del horizonte elegido.")
            c2.metric("🛡️ Probabilidad de no perder",f"{100-mc.prob_loss*100:.0f}%",
                      help="En qué porcentaje de los futuros simulados se termina con más dinero del invertido.")
            c3.metric("🎯 Probabilidad de alcanzar la meta",f"{mc.prob_target:.0%}",delta=f"${mc.target:,.0f}",delta_color="off",
                      help="En qué porcentaje de los futuros simulados se alcanza o supera la meta.")

            terminal = mc.terminal
            idx_best = int(np.argmax(terminal))
            idx_worst = int(np.argmin(terminal))
            idx_median = int(np.argsort(terminal)[len(terminal)//2])
            x = mc.dates

            # ── Métricas clave MC ──
            p5_val  = mc.percentiles[5][-1]
            p50_val = mc.percentiles[50][-1]
            p95_val = mc.percentiles[95][-1]
            gbm_min = terminal[idx_worst]
            gbm_max = terminal[idx_best]

            # ── GRÁFICO 1: Monte Carlo — bandas de percentiles ───────────
            st.markdown("#### 📊 Distribución de resultados: bandas de confianza")
            st.caption("Este gráfico resume el **rango probable** de tu inversión. "
                       "Las bandas muestran dónde caería tu capital en el 50%, 80% y 90% de los escenarios simulados. "
                       "La línea central (mediana) es el resultado más representativo.")
            fig1=go.Figure()
            for lo,hi,cl,nm in [(5,95,"rgba(46,94,140,0.08)","90% de escenarios"),
                                (10,90,"rgba(46,94,140,0.12)","80%"),
                                (25,75,"rgba(46,94,140,0.18)","50%")]:
                fig1.add_trace(go.Scatter(x=list(x)+list(x[::-1]),
                    y=list(mc.percentiles[hi])+list(mc.percentiles[lo][::-1]),
                    fill="toself",fillcolor=cl,line=dict(width=0),name=nm))
            fig1.add_trace(go.Scatter(x=x,y=mc.median_path,name="Mediana (P50)",
                                     line=dict(color=C_RV,width=2.5)))
            fig1.add_hline(y=mc.capital,line_dash="dot",line_color="gray",
                          annotation_text=f"Inversión ${mc.capital:,.0f}")
            if mc.target!=mc.capital:
                fig1.add_hline(y=mc.target,line_dash="dot",line_color=C_RF,
                              annotation_text=f"Meta ${mc.target:,.0f}")
            fig1.update_yaxes(tickprefix="$",tickformat=",.0f")
            fig1.update_layout(height=380,margin=dict(l=0,r=0,t=5,b=0),
                              legend=dict(orientation="h",y=-0.12))
            st.plotly_chart(fig1,use_container_width=True,key="chart_mc_bands",config={"displayModeBar":False})

            # Interpretación MC
            st.info(
                f"**Cómo leer este gráfico:** simulamos {len(terminal):,} futuros posibles para la inversión. "
                f"En 9 de cada 10, el capital terminó entre **{usd(p5_val)}** (escenario pesimista) y "
                f"**{usd(p95_val)}** (escenario optimista). "
                f"Bajo el escenario base, termina en **{usd(p50_val)}** "
                f"({'una ganancia' if p50_val>mc.capital else 'una pérdida'} de "
                f"**{abs(p50_val/mc.capital-1):.1%}**). "
                f"Entre más angosta sea la banda oscura del centro, más predecible es el portafolio."
            )

            # ── GRÁFICO 2: GBM — trayectorias individuales ───────────────
            st.markdown("#### 🔀 Caminos posibles de tu inversión")
            st.caption("Mientras el gráfico anterior resume los rangos, este muestra **caminos concretos** "
                       "que podría seguir tu inversión semana a semana. Cada línea gris es un escenario posible.")
            n_show = st.slider("Trayectorias a mostrar",10,200,50,10,
                               help="Cuántas simulaciones individuales dibujar de fondo.",
                               key="gbm_paths")

            fig2=go.Figure()
            rng_vis = np.random.default_rng(0)
            sample_idx = rng_vis.choice(len(terminal), size=min(n_show, len(terminal)), replace=False)
            for si in sample_idx:
                fig2.add_trace(go.Scatter(x=x,y=mc.paths[si],mode="lines",
                    line=dict(color="rgba(150,150,150,0.15)",width=0.5),
                    showlegend=False,hoverinfo="skip"))
            fig2.add_trace(go.Scatter(x=x,y=mc.paths[idx_best],
                name=f"🚀 Mejor: ${gbm_max:,.0f} ({gbm_max/mc.capital-1:+.1%})",
                line=dict(color="#2CA02C",width=2.5)))
            fig2.add_trace(go.Scatter(x=x,y=mc.paths[idx_worst],
                name=f"😟 Peor: ${gbm_min:,.0f} ({gbm_min/mc.capital-1:+.1%})",
                line=dict(color="#D6604D",width=2.5)))
            fig2.add_trace(go.Scatter(x=x,y=mc.paths[idx_median],
                name=f"📊 Mediana: ${terminal[idx_median]:,.0f} ({terminal[idx_median]/mc.capital-1:+.1%})",
                line=dict(color=C_RV,width=3)))
            fig2.add_hline(y=mc.capital,line_dash="dot",line_color="gray",
                          annotation_text=f"Inversión ${mc.capital:,.0f}")
            if mc.target!=mc.capital:
                fig2.add_hline(y=mc.target,line_dash="dot",line_color=C_RF,
                              annotation_text=f"Meta ${mc.target:,.0f}")
            fig2.update_yaxes(tickprefix="$",tickformat=",.0f")
            fig2.update_layout(height=420,margin=dict(l=0,r=0,t=5,b=0),
                              legend=dict(orientation="h",y=-0.12))
            st.plotly_chart(fig2,use_container_width=True,key="chart_mc_gbm",config={"displayModeBar":False})

            # Interpretación GBM
            st.info(
                f"**Cómo leer este gráfico:** cada línea gris es un camino posible que tu inversión "
                f"podría recorrer, semana a semana. En el mejor de todos llegó a **{usd(gbm_max)}** "
                f"y en el peor bajó hasta **{usd(gbm_min)}**. "
                f"Estos extremos son más amplios que el rango del gráfico anterior "
                f"({usd(p5_val)} – {usd(p95_val)}) porque aquí ves los casos más raros de "
                f"{len(terminal):,} simulaciones, no solo lo que pasa la mayoría de las veces."
            )

            with st.expander("🔬 ¿Qué es el Movimiento Browniano Geométrico y de dónde sale?"):
                st.markdown(
                    """El **Movimiento Browniano Geométrico (GBM)** es el modelo matemático estándar
para simular cómo evoluciona el precio de un activo en el tiempo. La idea: cada período, el
capital se multiplica por un factor de crecimiento aleatorio, de modo que nunca puede volverse
negativo (algo esencial, porque una inversión no puede valer menos de cero).

La fórmula que usa el sistema es:

`capital_final = capital_inicial × e^(suma de retornos aleatorios)`

Los retornos aleatorios de cada semana se sacan de una distribución normal multivariada
construida con el retorno esperado (μ) y la matriz de covarianza (Σ) del modelo Black-Litterman.
Al acumularlos y aplicarles la exponencial, se obtiene cada una de las líneas grises del gráfico.

**Importante:** los dos gráficos de esta sección (las bandas y las trayectorias) beben de las
**mismas** simulaciones GBM. No son dos modelos distintos · son dos formas de mirar el mismo
conjunto de futuros posibles."""
                )

            # ── Métricas resumen ──
            st.markdown(f"##### Rango de valor proyectado a {mh} año(s)")
            sc1,sc2,sc3=st.columns(3)
            sc1.metric("Escenario pesimista (P5)",f"${p5_val:,.0f}",delta=f"{p5_val/mc.capital-1:+.1%}",
                       help="Percentil 5: solo el 5% de los futuros simulados terminó peor que esto. "
                            "Representa un escenario adverso razonable.")
            sc2.metric("Escenario base (P50)",f"${p50_val:,.0f}",delta=f"{p50_val/mc.capital-1:+.1%}",
                       help="Mediana: la mitad de los futuros terminó por encima y la mitad por debajo. "
                            "Es el resultado central estimado.")
            sc3.metric("Escenario optimista (P95)",f"${p95_val:,.0f}",delta=f"{p95_val/mc.capital-1:+.1%}",
                       help="Percentil 95: solo el 5% de los futuros simulados terminó mejor que esto. "
                            "Representa un escenario favorable razonable.")

            # ── Explicación comparativa ──
            with st.expander("¿Por qué el primer gráfico y el segundo muestran rangos distintos?"):
                st.markdown(
                    f"""Los dos gráficos usan exactamente las mismas {len(terminal):,} simulaciones. Lo que cambia es qué parte te muestran.

**El primer gráfico (bandas)** te muestra lo que pasa la mayoría de las veces. Deja fuera el 5% de casos más extremos por arriba y por abajo. En palabras simples: *en 9 de cada 10 futuros posibles, tu inversión termina entre {usd(p5_val)} y {usd(p95_val)}*. Esta es la vista que conviene usar para planificar.

**El segundo gráfico (trayectorias)** te muestra todos los caminos, incluidos los más raros: el mejor de todos llegó a {usd(gbm_max)} y el peor bajó a {usd(gbm_min)}. Son casos muy poco probables (menos de 2 de cada 10,000), pero existen.

**¿En cuál fijarse?** Para tomar decisiones, usa el rango del primer gráfico. El segundo sirve para entender que el camino no es una línea recta: aunque termines cerca de lo esperado, en el trayecto puede haber subidas y bajadas fuertes. Si esos caminos grises se ven muy movidos, el portafolio podría estar más tranquilo con algo más de renta fija."""
                )

        # STRESS
        st.divider()
        st.subheader("🔥 ¿Cómo resistiría tu portafolio una crisis?")
        st.caption("Aplicamos a tu portafolio actual lo que realmente pasó en crisis históricas, "
                   "para ver cuánto habrías perdido o ganado en cada una.")
        ret_st=st.session_state.returns_full if st.session_state.returns_full is not None else st.session_state.returns
        bench_st=st.session_state.bench_full if st.session_state.bench_full is not None else (st.session_state.bench_rets if isinstance(st.session_state.bench_rets,dict) else {})
        if ret_st is not None: st.caption(f"📅 Datos disponibles: {ret_st.index.min().strftime('%Y-%m-%d')} → {ret_st.index.max().strftime('%Y-%m-%d')}")
        # Stress automático: recalcula si cambian los pesos o el set de benchmarks
        _bench_keys = tuple(sorted(bench_st.keys())) if isinstance(bench_st,dict) else ()
        stress_sig=(round(float(wnorm.sum()),6),tuple(round(float(x),6) for x in wnorm.values),round(float(capital),2),_bench_keys)
        if st.session_state.get("_stress_sig")!=stress_sig or "stress" not in st.session_state:
            pb=list(bench_st.values())[0] if bench_st else None
            with st.spinner("Evaluando crisis históricas…"):
                st.session_state["stress"]=stress_test(wnorm,ret_st,CRISIS_PERIODS,capital,{FICO_TK:FICO},PPY,pb)
            st.session_state._stress_sig=stress_sig
        if "stress" in st.session_state and st.session_state["stress"]:
            stres=st.session_state["stress"]; avail=[s for s in stres if s.available]
            if not avail: st.warning("No hay datos suficientes para evaluar crisis. Agrega más historia de datos.")
            else:
                # Retorno de CADA benchmark en cada crisis (para múltiples barras)
                def _bench_crisis_return(bser, crisis):
                    m=(bser.index>=crisis.start)&(bser.index<=crisis.end)
                    seg=bser.loc[m].dropna()
                    return float(np.exp(seg.sum())-1) if len(seg)>0 else None
                # Mapa: nombre_benchmark -> {crisis_name: retorno}
                bench_returns={}
                if isinstance(bench_st,dict):
                    for bn,bser in bench_st.items():
                        bench_returns[bn]={s.name:_bench_crisis_return(bser,
                            next(c for c in CRISIS_PERIODS if c.name==s.name)) for s in avail}

                worst=min(avail,key=lambda s:s.port_return)
                # ¿A cuántos benchmarks les ganó el portafolio en promedio?
                beats=sum(1 for s in avail if s.port_return>s.benchmark_return)
                st.info(f"**Resumen:** de {len(avail)} crisis evaluadas, tu portafolio resistió mejor que el "
                        f"benchmark principal en **{beats}**. La crisis que más te habría afectado es "
                        f"**{worst.name}**, con un impacto de **{worst.port_return:+.1%}** sobre tu capital.")
                st.caption("La barra de color es tu portafolio. Las demás barras son los benchmarks elegidos. "
                           "Mientras más arriba (o menos abajo) esté una barra, mejor resistió esa crisis.")
                fig=go.Figure()
                names=[s.name for s in avail]
                fig.add_trace(go.Bar(x=names,y=[s.port_return for s in avail],
                                     name="📊 Tu portafolio",marker_color=C_OPT))
                for i,(bn,bmap) in enumerate(bench_returns.items()):
                    fig.add_trace(go.Bar(x=names,
                        y=[bmap.get(nm) for nm in names],
                        name=nombre_activo(bn),marker_color=BC[i%len(BC)]))
                fig.update_yaxes(tickformat=".1%")
                fig.update_layout(barmode="group",height=340,margin=dict(l=0,r=0,t=5,b=0),
                                  legend=dict(orientation="h",y=1.12,font=dict(size=10)))
                st.plotly_chart(fig,use_container_width=True,key="chart_stress_bar",config={"displayModeBar":False})
                st.markdown("##### Detalle de cada crisis")
                for s in avail:
                    ic="🔴" if s.port_return<0 else "🟢"; diff=s.port_return-s.benchmark_return
                    cA,cB=st.columns([3,1])
                    with cA: st.markdown(f"{ic} **{s.name}**  ({s.start} a {s.end})"); st.caption(s.description)
                    with cB: st.metric("Impacto en tu capital",f"{s.port_return:+.1%}",delta=usd(s.port_loss),delta_color="off")
                    peor_activo=""
                    if not s.asset_returns.empty:
                        pa=s.asset_returns.sort_values()
                        peor_activo=f" El activo que más cayó fue **{disp(pa.index[0])}** ({pa.iloc[0]:+.1%})."
                    comparativa=("Te fue **mejor** que el mercado" if diff>0 else "Te fue **peor** que el mercado")
                    st.caption(f"{comparativa} por {abs(diff):.1%}. "
                               f"Caída máxima durante el episodio: {s.max_drawdown:.1%}.{peor_activo}")
                    st.divider()
            miss=[s for s in stres if not s.available]
            if miss:
                with st.expander(f"{len(miss)} crisis quedaron fuera del rango de datos"):
                    st.caption("Estas crisis ocurrieron antes de que empiecen tus datos, "
                               "así que no se pudieron evaluar:")
                    for s in miss: st.caption(f"• **{s.name}** ({s.start} → {s.end})")

        # Navegación: volver al portafolio
        st.divider()
        _port_step = 1 if AUTO else 2
        nav_buttons(back_to=_port_step, next_label="", back_label="← Volver a Portafolio")
