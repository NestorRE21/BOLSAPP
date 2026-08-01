# -*- coding: utf-8 -*-
"""Coril SAB — Optimizador BL v7 — Compacto"""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from optimizer import RiskProfile, ForcedAsset, View, BLConfig, run_profile, GKConfig, generate_gk_views, estimate_covariance, inject_forced_assets
from projections import monte_carlo, stress_test, CRISIS_PERIODS

st.set_page_config(page_title="Coril · Portafolios", page_icon="📈", layout="wide")
RF,PPY = 0.02,52
FICO_TK = "FICCMP13"
FICO_DISPLAY = "FICO"
FICO = ForcedAsset(ret_annual=0.0625,vol_annual=0.010,beta=0.30,sector="Factoring",region="Perú",moneda="USD",instrumento="Fondo")

def disp(ticker):
    """Nombre visible de un ticker (FICCMP13 → FICO). Para uso en la interfaz."""
    return FICO_DISPLAY if ticker == FICO_TK else ticker
PERFILES = {"Conservador (30/70)":(0.30,0.70),"Moderado-bajo (40/60)":(0.40,0.60),
            "Moderado (50/50)":(0.50,0.50),"Crecimiento (60/40)":(0.60,0.40),"Agresivo (70/30)":(0.70,0.30)}
P_DESC = {"Conservador (30/70)":"Preservar capital.","Moderado-bajo (40/60)":"Leve crecimiento.",
          "Moderado (50/50)":"Balance.","Crecimiento (60/40)":"Mayor exposición.","Agresivo (70/30)":"Máxima RV."}
EJ = ["AAPL","MSFT","NVDA","JNJ","KO","QQQ"]
C_RV,C_RF,C_OPT = "#2E5E8C","#2CA02C","#D6604D"
BC = ["#888","#E377C2","#FF7F0E","#9467BD","#17BECF"]

def usd(x):
    """Formatea monto en USD con el signo $ escapado para markdown de Streamlit."""
    return f"\\${x:,.0f}"

for k,v in {"tickers":[],"rf_tickers":[],"include_fico":True,"benchmarks":["^GSPC"],"views":[],"optimized":False,"result":None,
            "manual_weights":None,"returns":None,"bench_rets":None,"betas":None,"sectors":None,
            "returns_full":None,"bench_full":None,"last_period":None,"data_range":"",
            "mode":None,"gk_ic":0.05,"step":0}.items():
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
    params = _yf_period(period)
    raw=yf.download(tickers,**params,interval="1wk",auto_adjust=True,progress=False)
    if raw is None or raw.empty: return None
    px=raw["Close"].copy() if isinstance(raw.columns,pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":list(tickers)[0]})
    px=px.dropna(how="all").ffill(); px.index=pd.to_datetime(px.index).tz_localize(None)
    return np.log(px/px.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(how="all")

@st.cache_data(show_spinner=False,ttl=600)
def dl_bk(tks,period="15y"):
    import yfinance as yf
    out={}
    for b in tks:
        b=b.strip().upper()
        if not b: continue
        try:
            params = _yf_period(period)
            raw=yf.download(b,**params,interval="1wk",auto_adjust=True,progress=False)
            if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
            p=raw["Close"]; 
            if isinstance(p,pd.DataFrame): p=p.iloc[:,0]
            p.index=pd.to_datetime(p.index).tz_localize(None)
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

@st.cache_data(show_spinner=False,ttl=600)
def fetch_sec(tickers):
    import yfinance as yf
    out={}
    for t in tickers:
        try:
            i=yf.Ticker(t).info or {}; s=i.get("sector","")
            out[t]=s if s else (f"ETF · {i.get('category','')[:25]}" if i.get("quoteType")=="ETF" else i.get("industry","") or "–")
        except: out[t]="–"
    return pd.Series(out)

@st.cache_data(show_spinner=False,ttl=300)
def search_yf(q):
    import requests
    try:
        r=requests.get("https://query2.finance.yahoo.com/v1/finance/search",params={"q":q,"quotesCount":12,"newsCount":0},
                       headers={"User-Agent":"Mozilla/5.0"},timeout=5)
        return [{"tk":x["symbol"],"nm":x.get("shortname") or x.get("longname",""),
                 "tp":x.get("quoteType",""),"ex":x.get("exchange","")}
                for x in r.json().get("quotes",[]) if x.get("symbol")]
    except: return []

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

def _is_rf_candidate(r):
    """Heurística: ¿el resultado parece renta fija? (nunca apalancados)"""
    if _is_excluded(r) or _is_leveraged(r): return False   # RF nunca apalancado
    nm = (r.get("nm","") or "").lower()
    tk = (r.get("tk","") or "").upper()
    # Ticker conocido de RF
    if tk in _RF_TICKERS: return True
    # Keywords de RF en nombre
    if any(kw in nm for kw in _RF_KW): return True
    return False

def _is_rv_candidate(r):
    """Heurística: ¿el resultado parece renta variable? (apalancados sí permitidos)"""
    if _is_excluded(r): return False       # cripto/commodities no
    tp = r.get("tp","")
    # Apalancados de equity → sí van en RV
    if _is_leveraged(r): return True
    # Acciones individuales → siempre RV
    if tp == "EQUITY": return True
    # ETFs/fondos → solo si NO parece RF
    if tp in ("ETF","MUTUALFUND"):
        return not _is_rf_candidate(r)
    # Índices → pasan
    if tp == "INDEX": return True
    return True  # por defecto permitir

def filter_search(results, category):
    """Filtra resultados de búsqueda según la categoría seleccionada."""
    if category == "🔵 Renta variable":
        # RV: acciones, ETFs de equity y apalancados. No cripto/commodities.
        return [r for r in results if _is_rv_candidate(r)]
    elif category == "🟢 Renta fija":
        # RF: solo lo que positivamente parece RF, nunca apalancados.
        return [r for r in results if _is_rf_candidate(r)]
    else:  # Benchmark — permite apalancados; solo bloquea cripto/commodities.
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
    if not tks or not bks: return False
    # Descargar equity + RF de mercado juntos
    all_market = list(set(tks + rf_tks))  # sin duplicados
    lr=dl_eq(tuple(all_market),period)
    if lr is None or lr.empty: return False
    bd=dl_bk(tuple(bks),period)
    if not bd: return False
    common=lr.index
    for v in bd.values(): common=common.intersection(v.index)
    st.session_state.returns=lr.loc[common]; st.session_state.bench_rets={k:v.loc[common] for k,v in bd.items()}
    st.session_state.returns_full=lr; st.session_state.bench_full=bd
    b=calc_betas(lr.loc[common],list(bd.values())[0].loc[common]); b[FICO_TK]=FICO.beta; st.session_state.betas=b
    ok=[t for t in tks if t in lr.columns]
    s=fetch_sec(tuple(ok)); s[FICO_TK]=FICO.sector; st.session_state.sectors=s
    st.session_state.last_period=period
    st.session_state.data_range=f"{lr.index.min().strftime('%Y-%m-%d')} → {lr.index.max().strftime('%Y-%m-%d')}"
    for k in list(st.session_state.keys()):
        if k.startswith("s_"): del st.session_state[k]
    st.session_state.optimized=False; st.session_state.result=None; st.session_state.manual_weights=None
    for x in ["mc","stress"]:
        if x in st.session_state: del st.session_state[x]
    return True

# ═══════════════════ SIDEBAR ══════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 Coril")
    if st.session_state.mode is not None:
        _mode_label = "🤖 Automático" if st.session_state.mode=="auto" else "🎛️ Manual"
        st.caption(f"Modo: **{_mode_label}**")
        if st.button("↔️ Cambiar modo", use_container_width=True):
            st.session_state.mode=None; st.rerun()
        st.divider()
    ps=st.selectbox("Perfil",list(PERFILES.keys()),index=2); eq_t,fi_t=PERFILES[ps]
    st.caption(P_DESC[ps])
    c1,c2=st.columns(2); c1.metric("RV",f"{eq_t:.0%}"); c2.metric("RF",f"{fi_t:.0%}")
    st.divider()
    capital=st.slider("Inversión (USD)",1_000,1_000_000,100_000,1_000,format="$%d")
    chart_years=st.selectbox("Ver gráfico desde hace",["1y","2y","3y","5y","10y","15y"],index=3,
                             help="Solo afecta la vista del gráfico histórico. La optimización siempre usa 15 años.")
    with st.expander("⚙️ Avanzado"):
        _p=RiskProfile.for_split(eq_t,fi_t)
        st.caption(f"RF: {FICO_DISPLAY} · {FICO.ret_annual:.2%} | Beta: {_p.beta_min:.2f}–{_p.beta_max:.2f} | DD máx: {_p.max_drawdown:.0%}")
        st.caption("Optimización siempre usa **15 años** de datos.")
        if st.button("🗑️ Limpiar caché",use_container_width=True): st.cache_data.clear(); st.toast("✓")

# Descargar siempre con 15 años (fijo)
OPT_PERIOD = "15y"

# ═══════════════════ PANTALLA DE MODO ═════════════════════════════════════════
if st.session_state.mode is None:
    st.title("Optimizador de portafolios · Coril SAB")
    st.markdown("### ¿Cómo quieres construir el portafolio?")
    st.write("")
    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown("#### 🎛️ Manual")
        st.caption("Tú defines las expectativas de retorno de cada activo. "
                   "Control total sobre las views que alimentan el modelo Black-Litterman.")
        st.markdown("- Ingresas tus propias views (absolutas y relativas)\n"
                    "- Ajustas confianza por view\n"
                    "- Ideal si tienes una tesis de inversión propia")
        if st.button("Usar modo Manual", type="primary", use_container_width=True):
            st.session_state.mode="manual"; st.rerun()
    with cm2:
        st.markdown("#### 🤖 Automático")
        st.caption("El sistema genera las views por ti usando el framework de "
                   "Grinold-Kahn (α = volatilidad · IC · score), combinando momentum y baja volatilidad. Solo eliges activos y perfil.")
        st.markdown("- Views generadas por dos factores: momentum 12-1 y baja volatilidad\n"
                    "- IC conservador (0.05, forecaster \"bueno\")\n"
                    "- Portafolio y proyecciones automáticos")
        if st.button("Usar modo Automático", type="primary", use_container_width=True):
            st.session_state.mode="auto"; st.rerun()
    st.write("")
    st.info("💡 Podrás cambiar de modo en cualquier momento desde la barra lateral.")
    st.stop()

AUTO = st.session_state.mode == "auto"

# Auto-descarga si no hay datos aún (primera vez tras añadir tickers)
if st.session_state.tickers and st.session_state.benchmarks and st.session_state.last_period and st.session_state.last_period!=OPT_PERIOD:
    with st.spinner("Actualizando datos (15y)…"): run_dl(OPT_PERIOD)

# ═══════════════════ MAIN ═════════════════════════════════════════════════════
st.title("Optimizador de portafolios")

# Navegación tipo "pasos" (permite botones Siguiente/Atrás)
if AUTO:
    STEPS = ["1 · Activos","2 · Portafolio","3 · Proyecciones"]
else:
    STEPS = ["1 · Activos","2 · Expectativas","3 · Portafolio","4 · Proyecciones"]

# Clamp del paso actual al rango válido
st.session_state.step = max(0, min(st.session_state.step, len(STEPS)-1))

# Barra de navegación superior (clicable)
nav_cols = st.columns(len(STEPS))
for i,label in enumerate(STEPS):
    with nav_cols[i]:
        is_current = (i == st.session_state.step)
        if st.button(label, key=f"nav_{i}", use_container_width=True,
                     type="primary" if is_current else "secondary"):
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
    col_s,col_t=st.columns([4,1])
    with col_t: add_to=st.radio("Añadir como",["🔵 Renta variable","🟢 Renta fija","📊 Benchmark"])
    with col_s: q=st.text_input("🔍 Buscar",placeholder="Apple, TLT, SHY, AGG, ^GSPC…")
    if q.strip():
        raw_res=search_yf(q.strip())
        res=filter_search(raw_res, add_to)
        if not res and raw_res:
            st.caption(f"ℹ️ No se encontraron resultados compatibles con **{add_to}**. "
                       f"Se encontraron {len(raw_res)} de otra clase.")
        if res:
            cols=st.columns(min(len(res[:6]),3))
            for i,r in enumerate(res[:6]):
                with cols[i%len(cols)]:
                    if st.button(f"➕ {r['tk']} — {r['nm'][:18]}",key=f"a_{r['tk']}",use_container_width=True):
                        tk=r['tk']
                        if add_to=="🔵 Renta variable":
                            if tk not in st.session_state.tickers: st.session_state.tickers.append(tk); st.toast(f"✓ {tk} → RV")
                        elif add_to=="🟢 Renta fija":
                            if tk not in st.session_state.rf_tickers: st.session_state.rf_tickers.append(tk); st.toast(f"✓ {tk} → RF")
                        else:
                            if tk not in st.session_state.benchmarks: st.session_state.benchmarks.append(tk); st.toast(f"✓ {tk} → Benchmark")
    if not st.session_state.tickers:
        if st.button("🚀 Cargar ejemplo",type="primary"):
            st.session_state.tickers=list(EJ); st.session_state.views=[]; st.session_state.rf_tickers=[]

    # ── Listas: RV + RF + Benchmarks ─────────────────────────────────────
    la,lb,lc=st.columns(3)
    with la:
        st.caption(f"**🔵 Renta variable ({len(st.session_state.tickers)})**")
        for i,t in enumerate(st.session_state.tickers):
            c1,c2=st.columns([5,1]); c1.write(t)
            if c2.button("✕",key=f"ra{i}"):
                rm=st.session_state.tickers.pop(i)
                st.session_state.views=[v for v in st.session_state.views if v.get("asset")!=rm and v.get("long")!=rm and v.get("short")!=rm]
    with lb:
        st.caption(f"**🟢 Renta fija ({len(st.session_state.rf_tickers)})**")
        # Toggle FICO
        include_fico = st.checkbox("Incluir FICO Coril (6.25%)", value=True, key="fico_toggle")
        st.session_state.include_fico = include_fico
        if include_fico:
            st.caption(f"✓ {FICO_DISPLAY} · {FICO.ret_annual:.2%} forzado")
        for i,t in enumerate(st.session_state.rf_tickers):
            c1,c2=st.columns([5,1]); c1.write(t)
            if c2.button("✕",key=f"rrf{i}"): st.session_state.rf_tickers.pop(i)
        if not st.session_state.rf_tickers and not include_fico:
            st.warning("Sin activos de renta fija.")
    with lc:
        st.caption(f"**📊 Benchmarks ({len(st.session_state.benchmarks)})**")
        for i,b in enumerate(st.session_state.benchmarks):
            c1,c2=st.columns([5,1]); c1.write(b)
            if c2.button("✕",key=f"rb{i}"): st.session_state.benchmarks.pop(i)

    # ── Descarga ─────────────────────────────────────────────────────────
    has_rf = bool(st.session_state.rf_tickers) or st.session_state.get("include_fico", True)
    can_dl = bool(st.session_state.tickers and st.session_state.benchmarks and has_rf)
    if st.session_state.data_range: st.success(f"📦 {st.session_state.data_range} ({st.session_state.last_period})")
    if st.button("📥 Descargar datos",type="primary",use_container_width=True, disabled=not can_dl):
        with st.spinner("Descargando…"):
            if run_dl(OPT_PERIOD): st.success(f"✅ {st.session_state.data_range}")
            else: st.error("Error. Verifica tickers.")
    if not can_dl:
        st.caption("💡 Para continuar necesitas: al menos un activo de renta variable, "
                   "uno de renta fija (o FICO activado) y un benchmark.")
    # Botón para avanzar cuando ya hay datos
    if st.session_state.returns is not None:
        st.divider()
        _next_label = "Continuar a Portafolio →" if AUTO else "Continuar a Expectativas →"
        nav_buttons(next_to=1, next_label=_next_label)

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
    # Validación previa: necesitamos datos y al menos un activo de RV con historia.
    _eq_validos = [a for a in st.session_state.tickers
                   if st.session_state.returns is not None and a in st.session_state.returns.columns] \
                  if st.session_state.returns is not None else []
    if st.session_state.returns is None:
        st.info("⬅️ Primero agrega activos y descarga los datos en la pestaña **Activos**.")
    elif not _eq_validos:
        st.warning("⚠️ Necesitas al menos un activo de **renta variable** (acción o ETF de equity) "
                   "con datos descargados. Ve a la pestaña **Activos**, agrégalo y espera a que "
                   "cargue el histórico.")
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
                for x in ["mc","stress"]:
                    if x in st.session_state: del st.session_state[x]
            st.success("Las expectativas de retorno se calcularon automáticamente combinando "
                       "dos señales: el momentum de los últimos 12 meses y la baja volatilidad "
                       "de cada activo. Cualquier cambio en activos o perfil actualiza todo al instante.")
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
                for x in ["mc","stress"]:
                    if x in st.session_state: del st.session_state[x]

        if st.session_state.optimized and st.session_state.result:
            res=st.session_state.result
            # Pesos en 2 columnas lado a lado
            assets=list(res.weights.index); mid=len(assets)//2+len(assets)%2
            col_a,col_b,col_r=st.columns([2,2,1.5])
            nw={}
            with col_a:
                for a in assets[:mid]:
                    ic="🟢" if a==FICO_TK else "🔵"
                    nw[a]=st.number_input(f"{ic} {disp(a)}",0.0,100.0,round(float(res.weights[a])*100,1),0.5,"%.1f",key=f"s_{a}")
            with col_b:
                for a in assets[mid:]:
                    ic="🟢" if a==FICO_TK else "🔵"
                    nw[a]=st.number_input(f"{ic} {disp(a)}",0.0,100.0,round(float(res.weights[a])*100,1),0.5,"%.1f",key=f"s_{a}")
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
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Retorno esperado anual",f"{p_r:.2%}",
                      help="Cuánto se espera que rinda el portafolio por año, según el modelo "
                           "Black-Litterman. Es una expectativa a futuro, no una garantía.")
            m2.metric("Riesgo (volatilidad anual)",f"{p_v:.2%}",
                      help="Qué tanto puede moverse el portafolio en un año. Más alto = más incertidumbre.")
            m3.metric("Sharpe",f"{p_sh:.2f}",
                      help="Retorno por unidad de riesgo. Mide cuánto rendimiento extra obtienes por "
                           "cada punto de riesgo que asumes. Más alto es mejor; arriba de 1 se considera bueno.")
            m4.metric("Beta",f"{p_bt:.2f}",
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
                fig=go.Figure(go.Pie(labels=[disp(a) for a in ws.index.tolist()],values=ws.values.tolist(),
                    marker_colors=colors,hole=.4,textinfo="label+percent"))
                fig.update_layout(height=280,margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
                st.plotly_chart(fig,use_container_width=True)
            with g2:
                st.caption("Por sector")
                sec=st.session_state.sectors
                if sec is not None and not sec.empty:
                    sw={}
                    for a in wnorm.index:
                        if wnorm[a]>1e-4: s=sec.get(a,"–"); sw[s]=sw.get(s,0)+wnorm[a]
                    sec_colors=px.colors.qualitative.Pastel[:len(sw)]
                    fig=go.Figure(go.Pie(labels=list(sw.keys()),values=list(sw.values()),
                        marker_colors=sec_colors,hole=.4,textinfo="label+percent"))
                    fig.update_layout(height=280,margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
                    st.plotly_chart(fig,use_container_width=True)

            # Evolución histórica — filtrada por chart_years, siempre desde capital inicial
            st.markdown("##### 📈 Evolución histórica del capital")
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
                             subplot_titles=[f"Evolución de capital · últimos {cy} años (${capital:,.0f})","Drawdown"])
            fig.add_trace(go.Scatter(x=wl.index,y=wl.values,name="Portafolio",
                                    line=dict(color=C_RV,width=2.5)),row=1,col=1)
            for i,(n,v) in enumerate(bw.items()):
                fig.add_trace(go.Scatter(x=v.index,y=v.values,name=n,
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
            st.plotly_chart(fig,use_container_width=True)

            # Métricas históricas del rango visible
            ann_r=np.exp(pr.mean()*PPY)-1; ann_v=pr.std(ddof=1)*np.sqrt(PPY)
            from scipy.stats import norm as _norm
            mu_h=pr.mean()*PPY; sig_h=ann_v; z=_norm.ppf(0.05)
            var95=-(mu_h+z*sig_h); cvar95=-(mu_h-sig_h*_norm.pdf(z)/0.05)
            st.markdown("##### 📅 Comportamiento histórico del portafolio")
            st.caption("Estos números resumen cómo se habría comportado esta combinación de activos "
                       "en el pasado, según los datos descargados.")
            h1,h2,h3=st.columns(3)
            h1.metric("Retorno histórico anual",f"{ann_r:.2%}",
                      help="Cuánto habría rendido el portafolio en promedio por año, según su historia. "
                           "No es una promesa a futuro, es lo que ocurrió en el pasado.")
            h2.metric("Volatilidad anual",f"{ann_v:.2%}",
                      help="Qué tanto sube y baja el portafolio. Más alta = más movimiento y más riesgo. "
                           "Es la desviación estándar de los retornos, anualizada.")
            h3.metric("Caída máxima (drawdown)",f"{dd.min():.2%}",
                      help="La peor caída desde un punto alto hasta el punto más bajo que sufrió el "
                           "portafolio en el período. Mide el peor mal momento que habrías vivido.")
            h4,h5=st.columns(2)
            h4.metric("VaR 95% (pérdida esperada)",f"{var95:.2%}",
                      help="Value at Risk. En el 5% de los peores años, esperarías perder al menos "
                           "este porcentaje. Es el umbral de pérdida en un mal escenario.")
            h5.metric("CVaR 95% (pérdida extrema)",f"{cvar95:.2%}",
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
                    rows.append((n, bm, False))

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
                    st.plotly_chart(fr,use_container_width=True)
                with cbar2:
                    st.caption("Sharpe (retorno ajustado por riesgo)")
                    fs = go.Figure(go.Bar(x=names, y=[r['sharpe'] for _,r,_ in rows],
                        marker_color=bar_colors, text=[f"{r['sharpe']:.2f}" for _,r,_ in rows],
                        textposition="outside"))
                    fs.update_layout(height=260,margin=dict(l=0,r=0,t=5,b=0))
                    st.plotly_chart(fs,use_container_width=True)

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
    if not(st.session_state.optimized and st.session_state.result): st.info("⬅️ Optimiza primero.")
    else:
        res=st.session_state.result
        wnorm=st.session_state.manual_weights if st.session_state.manual_weights is not None else res.weights

        # MONTE CARLO
        st.subheader("🎲 ¿Cuánto podría valer tu portafolio?")
        c1,c2,c3=st.columns(3)
        mh=c1.selectbox("Años a proyectar",[1,2,3,5,10],index=2,
                        help="Horizonte de la proyección. En cuántos años quieres ver cómo evoluciona tu inversión.")
        mn=c2.selectbox("Precisión",[1000,5000,10000],index=1,format_func=lambda x:f"{x:,}",
                        help="Cuántos futuros posibles simular. Más simulaciones = resultado más estable, pero más lento.")
        mt=c3.number_input("Meta (USD)",value=int(capital*1.2),step=10_000,format="%d",
                           help="El capital que te gustaría alcanzar. Se calcula la probabilidad de llegar a esta meta.")
        # Proyección automática: recalcula si cambian parámetros o pesos
        mc_sig=(round(float(wnorm.sum()),6),tuple(round(float(x),6) for x in wnorm.values),
                int(mh),int(mn),int(mt),round(float(capital),2))
        if st.session_state.get("_mc_sig")!=mc_sig or "mc" not in st.session_state:
            with st.spinner("Calculando proyección…"):
                st.session_state["mc"]=monte_carlo(wnorm,res.bl_returns,res.cov_matrix,capital,mh,PPY,mn,mt)
            st.session_state._mc_sig=mc_sig
        if "mc" in st.session_state and st.session_state["mc"]:
            mc=st.session_state["mc"]; gain=mc.median_path[-1]-mc.capital
            c1,c2,c3=st.columns(3)
            c1.metric(f"💰 Valor proyectado a {mh} año(s)",f"${mc.median_path[-1]:,.0f}",delta=f"+${gain:,.0f} ({gain/mc.capital:+.1%})",
                      help="El valor más probable (mediana) de tu inversión al final del horizonte elegido.")
            c2.metric("🛡️ Probabilidad de no perder",f"{100-mc.prob_loss*100:.0f}%",
                      help="En qué porcentaje de los futuros simulados terminas con más dinero del que invertiste.")
            c3.metric("🎯 Probabilidad de alcanzar la meta",f"{mc.prob_target:.0%}",delta=f"${mc.target:,.0f}",delta_color="off",
                      help="En qué porcentaje de los futuros simulados alcanzas o superas tu meta.")
            st.success(f"En **{mc.horizon_years:.0f} año(s)**, tu inversión de {usd(mc.capital)} probablemente valdrá "
                       f"entre **{usd(mc.percentiles[5][-1])}** (si el mercado va mal) y "
                       f"**{usd(mc.percentiles[95][-1])}** (si va bien). "
                       f"El resultado más probable es **{usd(mc.median_path[-1])}**.")

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
            st.markdown("#### 📊 Distribución de resultados — Bandas de confianza")
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
            st.plotly_chart(fig1,use_container_width=True)

            # Interpretación MC
            st.info(
                f"**Cómo leer este gráfico:** simulamos {len(terminal):,} futuros posibles para tu inversión. "
                f"En 9 de cada 10, el capital terminó entre **{usd(p5_val)}** y **{usd(p95_val)}**. "
                f"Lo más probable es que termine en **{usd(p50_val)}** "
                f"({'una ganancia' if p50_val>mc.capital else 'una pérdida'} de "
                f"**{abs(p50_val/mc.capital-1):.1%}**). "
                f"Entre más angosta sea la banda oscura del centro, más predecible es el portafolio."
            )

            # ── GRÁFICO 2: GBM — trayectorias individuales ───────────────
            st.markdown("#### 🔀 Trayectorias individuales — Movimiento Browniano Geométrico")
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
            st.plotly_chart(fig2,use_container_width=True)

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
**mismas** simulaciones GBM. No son dos modelos distintos — son dos formas de mirar el mismo
conjunto de futuros posibles."""
                )

            # ── Métricas resumen ──
            st.markdown(f"##### Rango de valor proyectado a {mh} año(s)")
            sc1,sc2,sc3=st.columns(3)
            sc1.metric("😟 Escenario malo (P5)",f"${p5_val:,.0f}",delta=f"{p5_val/mc.capital-1:+.1%}",
                       help="Percentil 5: solo el 5% de los futuros simulados terminó peor que esto. "
                            "Es tu escenario pesimista razonable.")
            sc2.metric("📊 Escenario más probable (P50)",f"${p50_val:,.0f}",delta=f"{p50_val/mc.capital-1:+.1%}",
                       help="Mediana: la mitad de los futuros terminó por encima y la mitad por debajo. "
                            "El resultado central esperado.")
            sc3.metric("🚀 Escenario bueno (P95)",f"${p95_val:,.0f}",delta=f"{p95_val/mc.capital-1:+.1%}",
                       help="Percentil 95: solo el 5% de los futuros simulados terminó mejor que esto. "
                            "Es tu escenario optimista razonable.")

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
                        name=bn,marker_color=BC[i%len(BC)]))
                fig.update_yaxes(tickformat=".1%")
                fig.update_layout(barmode="group",height=340,margin=dict(l=0,r=0,t=5,b=0),
                                  legend=dict(orientation="h",y=1.12,font=dict(size=10)))
                st.plotly_chart(fig,use_container_width=True)
                st.markdown("##### Detalle de cada crisis")
                for s in avail:
                    ic="🔴" if s.port_return<0 else "🟢"; diff=s.port_return-s.benchmark_return
                    cA,cB=st.columns([3,1])
                    with cA: st.markdown(f"{ic} **{s.name}** · {s.start} → {s.end}"); st.caption(s.description)
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
