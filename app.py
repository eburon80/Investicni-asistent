import base64
import datetime as dt
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Investiční asistent", page_icon="📈", layout="wide")

# ============================ DATA A KONFIGURACE ============================
STATIC = {
    "ETF (UCITS)": {"CSPX.L": "iShares S&P 500", "IWDA.L": "iShares MSCI World", "EQQQ.L": "Invesco Nasdaq 100",
                    "EIMI.L": "iShares MSCI Emerging Markets", "VWRL.L": "Vanguard FTSE All-World",
                    "IDTL.L": "iShares USD Treasury 20+"},
    "Komodity (ETC)": {"SGLN.L": "Zlato (iShares Physical Gold)", "SSLN.L": "Stříbro (iShares Physical Silver)",
                       "CRUD.L": "Ropa WTI (WisdomTree)", "COPA.L": "Měď (WisdomTree)"},
    "Evropské akcie (ADR)": {"ASML": "ASML", "SAP": "SAP", "NVO": "Novo Nordisk", "SHEL": "Shell", "AZN": "AstraZeneca",
                             "SNY": "Sanofi", "UL": "Unilever", "TM": "Toyota"},
    "Krypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"},
}
INDEX_SRC = {"Nasdaq 100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
             "S&P 500 (velké, pomalé)": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")}
FALLBACK_US = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "GOOGL": "Alphabet", "AMZN": "Amazon",
               "META": "Meta", "AVGO": "Broadcom", "TSLA": "Tesla", "JPM": "JPMorgan", "XOM": "Exxon Mobil",
               "JNJ": "Johnson & Johnson", "V": "Visa", "COST": "Costco", "NFLX": "Netflix", "AMD": "AMD"}
NAMES = {t: n for g in STATIC.values() for t, n in g.items()}

HORIZONS = {
    "kratka": dict(label="Krátkodobá (týdny až 3 měsíce)", period="1y", fast=20, slow=50, mom=63, vol=63,
                   cyc=(5, 60), cyc_win=252),
    "stredni": dict(label="Střednědobá (3 až 12 měsíců)", period="3y", fast=50, slow=100, mom=126, vol=126,
                    cyc=(20, 150), cyc_win=504),
    "dlouha": dict(label="Dlouhodobá (1 rok a více)", period="6y", fast=100, slow=200, mom=252, vol=252,
                   cyc=(60, 400), cyc_win=1008),
}
METHODS = {"trend": "Trend (klouzavé průměry)", "mom": "Momentum (výnos za období)",
           "rsi": "RSI (překoupenost / přeprodanost)", "macd": "MACD (změna hybnosti)",
           "risk": "Riziko (volatilita, Sharpe, pokles od maxima)", "cyc": "Pravidelné oscilace (cykly)"}
METHOD_COLS = ["Trend", "Momentum", "RSI", "MACD", "Riziko", "Cykly"]

PLANS = {  # orientační výchozí hodnoty, uživatel je může upravit
    "Standard": dict(free=0, fixed=1.0, pct=0.0, fx=1.0, fx_lim=1000.0, crypto=1.49),
    "Plus": dict(free=0, fixed=1.0, pct=0.0, fx=0.5, fx_lim=3000.0, crypto=1.49),
    "Premium": dict(free=5, fixed=1.0, pct=0.0, fx=0.0, fx_lim=1e9, crypto=0.99),
    "Metal": dict(free=10, fixed=1.0, pct=0.0, fx=0.0, fx_lim=1e9, crypto=0.99),
    "Ultra": dict(free=10, fixed=1.0, pct=0.0, fx=0.0, fx_lim=1e9, crypto=0.49),
}


@st.cache_data(ttl=86400, show_spinner=False)
def index_members(name: str) -> dict:
    url, col = INDEX_SRC[name]
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
        for tb in pd.read_html(io.StringIO(html)):
            if col in tb.columns and len(tb) > 50:
                nc = next((c for c in ("Security", "Company") if c in tb.columns), col)
                tick = tb[col].astype(str).str.replace(".", "-", regex=False).str.strip()
                return dict(zip(tick, tb[nc].astype(str)))
    except Exception:
        pass
    return {}


def build_universe(groups, extra) -> dict:
    uni = {}
    for g in groups:
        if g in STATIC:
            uni.update(STATIC[g])
        elif g in INDEX_SRC:
            uni.update(index_members(g) or FALLBACK_US)
    for x in extra.split(","):
        if x.strip():
            uni.setdefault(x.strip().upper(), x.strip().upper())
    NAMES.update(uni)
    return uni


@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(tickers: tuple, period: str = "2y") -> pd.DataFrame:
    df = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True)
    close = df["Close"] if "Close" in df else df
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    return close.dropna(how="all").dropna(axis=1, how="all")


# ============================ CYKLY (OSCILACE) ============================
def _prep(x, win):
    x = np.asarray(x, dtype=float)[-win:]
    x = x[np.isfinite(x) & (x > 0)]
    n = len(x)
    if n < 60:
        return None, None, n
    t = np.arange(n)
    y = np.log(x)
    return y - np.polyval(np.polyfit(t, y, 1), t), t, n


def find_cycles(x, pmin, pmax, win, n_peaks=3):
    """Dominantní periodické složky (FFT na odtrendovaném logaritmu ceny) + fit sinusoidy."""
    y, t, n = _prep(x, win)
    if y is None:
        return []
    P = np.abs(np.fft.rfft(y * np.hanning(n)))[1:] ** 2
    per = 1 / np.fft.rfftfreq(n)[1:]
    ok = np.where((per >= pmin) & (per <= min(pmax, n / 2.5)))[0]
    if len(ok) == 0:
        return []
    pk = [i for i in ok if (i == 0 or P[i] >= P[i - 1]) and (i == len(P) - 1 or P[i] >= P[i + 1])]
    pk = sorted(pk, key=lambda i: -P[i])[:n_peaks]
    tv, out = np.var(y), []
    for i in pk:
        w = 2 * np.pi / per[i]
        X = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones(n)])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - np.var(y - X @ c) / tv if tv > 0 else 0.0
        amp = float(np.hypot(c[0], c[1]))
        th = float((w * (n - 1) + np.arctan2(c[1], c[0])) % (2 * np.pi))
        out.append(dict(period=float(per[i]), r2=float(r2), amp=(np.exp(2 * amp) - 1) * 100, theta=th, w=float(w),
                        cycles=n / per[i], coef=[float(v) for v in c]))
    return out


def cycle_score(c) -> float:
    if c["r2"] < 0.15:
        return 50.0
    th = c["theta"]
    return float(np.clip(50 - 40 * np.sin(th) + 10 * np.sign(np.cos(th)), 0, 100))


def phase_text(th) -> str:
    s, c = np.sin(th), np.cos(th)
    if s < -0.6:
        return "blízko dna cyklu"
    if s > 0.6:
        return "blízko vrcholu cyklu"
    return "na vzestupné fázi cyklu" if c > 0 else "na sestupné fázi cyklu"


def cycle_pvalue(x, pmin, pmax, win, r2, n=100) -> float:
    """Podíl náhodných řad se stejným rozdělením denních změn, které mají cyklus aspoň tak silný."""
    lr = np.diff(np.log(np.asarray(x, dtype=float)[-win:]))
    rng, cnt = np.random.default_rng(0), 0
    for _ in range(n):
        sx = np.exp(np.concatenate([[0.0], np.cumsum(rng.permutation(lr))]))
        c = find_cycles(sx, pmin, pmax, 10 ** 9, 1)
        cnt += bool(c) and c[0]["r2"] >= r2
    return (cnt + 1) / (n + 1)


# ============================ SKÓRE ============================
def compute_scores(prices: pd.DataFrame, h: dict, methods) -> pd.DataFrame:
    p = prices.ffill()
    need = max(h["slow"], h["mom"], h["vol"]) + 20
    alive = prices.iloc[-5:].notna().any()
    ok = [c for c in p.columns if p[c].notna().sum() >= need and alive[c]]
    if not ok or not methods:
        return pd.DataFrame()
    p = p[ok]
    last = p.iloc[-1]
    fast, slow = p.rolling(h["fast"]).mean().iloc[-1], p.rolling(h["slow"]).mean().iloc[-1]
    r = p / p.shift(1) - 1
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).iloc[-1].fillna(50)
    macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    hn, hp = hist.iloc[-1], hist.iloc[-2]
    mom = last / p.iloc[-1 - h["mom"]] - 1
    w = h["vol"]
    vol = r.iloc[-w:].std() * np.sqrt(252)
    sharpe = r.iloc[-w:].mean() * 252 / vol.replace(0, np.nan)
    dd = last / p.iloc[-252:].max() - 1
    sc = {}
    if "trend" in methods:
        sc["Trend"] = ((last > fast).astype(int) + (fast > slow).astype(int) + (last > slow).astype(int)) / 3 * 100
    if "mom" in methods:
        sc["Momentum"] = mom.rank(pct=True) * 100
    if "rsi" in methods:
        sc["RSI"] = pd.Series(np.select([rsi.between(45, 65), rsi.between(30, 45), rsi.between(65, 75)],
                                        [100, 70, 50], 20), index=rsi.index)
    if "macd" in methods:
        sc["MACD"] = pd.Series(np.select([(hn > 0) & (hn >= hp), hn > 0, hn > hp], [100, 70, 40], 0), index=hn.index)
    if "risk" in methods:
        sc["Riziko"] = sharpe.rank(pct=True).fillna(0) * 50 + pd.Series(
            np.select([dd > -0.10, dd > -0.20], [50, 25], 0), index=dd.index)
    if "cyc" in methods:
        cs = {}
        for t in p.columns:
            cy = find_cycles(p[t].dropna().to_numpy(), h["cyc"][0], h["cyc"][1], h["cyc_win"], 1)
            cs[t] = cycle_score(cy[0]) if cy else 50.0
        sc["Cykly"] = pd.Series(cs)
    df = pd.DataFrame(sc)
    df["skore"] = df.mean(axis=1).round(0)
    df["verdikt"] = np.where(df.skore >= 70, "Silný signál", np.where(df.skore >= 50, "Sledovat", "Slabý"))
    df["price"], df["mom"], df["vol"], df["rsi"], df["dd"] = last, mom, vol, rsi, dd
    df["fast"], df["slow"] = fast, slow
    df["nazev"] = [NAMES.get(t, t) for t in df.index]
    return df.sort_values("skore", ascending=False)


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(252)
    return {"Roční výnos (CAGR)": eq.iloc[-1] ** (252 / max(len(r), 1)) - 1, "Volatilita": vol,
            "Sharpe": (r.mean() * 252) / vol if vol else np.nan,
            "Max. pokles": (eq / eq.cummax() - 1).min(), "Celkový výnos": eq.iloc[-1] - 1}


@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(tickers: tuple, top_n: int, cost: float, hk: str, methods: tuple) -> dict:
    """Měsíční rebalancování; skóre v každém měsíci vychází jen z dat dostupných k tomu dni."""
    h = HORIZONS[hk]
    prices = load_prices(tickers, "6y")
    p = prices.ffill()
    rets = p / p.shift(1) - 1
    idx = p.index
    me = [d for d in idx.to_series().groupby(idx.to_period("M")).last().tolist() if idx.get_loc(d) >= 260]
    strat = pd.Series(0.0, index=idx)
    prev = pd.Series(0.0, index=p.columns)
    for k, d in enumerate(me):
        tbl = compute_scores(prices.loc[:d], h, methods)
        if tbl.empty:
            continue
        sel = tbl.head(top_n).index
        w = pd.Series(0.0, index=p.columns)
        w[sel] = 1 / len(sel)
        s0 = idx.get_loc(d) + 1
        e0 = idx.get_loc(me[k + 1]) if k + 1 < len(me) else len(idx) - 1
        if s0 > e0:
            continue
        days = idx[s0:e0 + 1]
        r = rets.loc[days, sel].mean(axis=1).fillna(0)
        r.iloc[0] -= cost * (w - prev).abs().sum()
        strat.loc[days] = r
        prev = w
    first = strat[strat != 0].index[0]
    out = {"Strategie (top skóre)": strat.loc[first:],
           "Všechny tituly rovnoměrně": rets.loc[first:].mean(axis=1).fillna(0)}
    for b, nm in (("CSPX.L", "S&P 500 (CSPX)"), ("SPY", "S&P 500 (SPY)")):
        if b in rets:
            out[nm] = rets.loc[first:, b].fillna(0)
            break
    return out


# ============================ ZPRÁVY, MOMENTY, SOUHRN ============================
@st.cache_data(ttl=1800, show_spinner=False)
def get_news(t: str) -> list:
    out = []
    try:
        for n in (yf.Ticker(t).news or [])[:6]:
            c = n.get("content", n)
            url = (c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url") or c.get("link")
            pub = c.get("pubDate") or c.get("providerPublishTime")
            if isinstance(pub, (int, float)):
                pub = dt.datetime.fromtimestamp(pub).strftime("%Y-%m-%d")
            out.append((str(pub or "")[:10], c.get("title"), url,
                        (c.get("provider") or {}).get("displayName") or c.get("publisher") or ""))
    except Exception:
        pass
    return [o for o in out if o[1]]


@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings(t: str):
    try:
        cal = yf.Ticker(t).calendar
        d = cal.get("Earnings Date") if isinstance(cal, dict) else None
        return ", ".join(str(x) for x in d) if d else None
    except Exception:
        return None


def key_moments(s: pd.Series, n_big=3) -> list:
    s = s.dropna()
    ev = []
    r = (s / s.shift(1) - 1).iloc[-1260:]
    for d, v in r.abs().nlargest(n_big).items():
        ev.append((d, f"Výrazný denní pohyb: {r[d]:+.1%}"))
    s50, s200 = s.rolling(50).mean(), s.rolling(200).mean()
    cr = (s50 > s200).astype(int).diff()
    for d in cr[cr == 1].index[-1:]:
        ev.append((d, "Golden cross (SMA50 nad SMA200)"))
    for d in cr[cr == -1].index[-1:]:
        ev.append((d, "Death cross (SMA50 pod SMA200)"))
    y = s.iloc[-252:]
    ev += [(y.idxmax(), f"52týdenní maximum ({y.max():,.2f})"), (y.idxmin(), f"52týdenní minimum ({y.min():,.2f})")]
    return sorted(ev, key=lambda x: x[0], reverse=True)


def facts_text(t: str, name: str, s: pd.Series) -> str:
    """Strukturovaná fakta o titulu z Yahoo Finance + historie cen."""
    s = s.dropna()
    L = [f"**{name} ({t})**"]
    try:
        info = yf.Ticker(t).info or {}
    except Exception:
        info = {}
    if info.get("longBusinessSummary"):
        L.append(info["longBusinessSummary"][:450].rsplit(" ", 1)[0] + "…")
    meta = [f"{k}: {info[f]}" for k, f in (("sektor", "sector"), ("odvětví", "industry"), ("země", "country"),
                                          ("doporučení analytiků", "recommendationKey")) if info.get(f)]
    for k, f, fmt in (("P/E", "trailingPE", "{:.1f}"), ("forward P/E", "forwardPE", "{:.1f}"),
                      ("dividenda", "dividendYield", "{:.2f} %"), ("cílová cena analytiků", "targetMeanPrice", "{:.2f}")):
        if isinstance(info.get(f), (int, float)):
            meta.append(f"{k}: " + fmt.format(info[f]))
    if meta:
        L.append("Základní údaje: " + ", ".join(meta))
    try:
        ud = yf.Ticker(t).upgrades_downgrades
        if ud is not None and len(ud):
            L.append("Poslední změny doporučení: " + "; ".join(
                f"{i:%Y-%m-%d} {r.get('Firm', '')} {r.get('FromGrade', '')}→{r.get('ToGrade', '')}"
                for i, r in ud.head(4).iterrows()))
    except Exception:
        pass
    nw = get_news(t)
    if nw:
        L.append("**Aktuální zprávy:** " + " | ".join(f"{d} {ti}" for d, ti, _, _ in nw[:5]))
    r = (s / s.shift(1) - 1)
    big = r.abs().nlargest(5).index.sort_values()
    L.append("**Největší pohyby v historii:** " + "; ".join(f"{d:%d.%m.%Y} {r[d]:+.1%}" for d in big))
    dd = s / s.cummax() - 1
    tr = dd.idxmin()
    pk = s.loc[:tr].idxmax()
    rec = s.loc[tr:][s.loc[tr:] >= s.loc[pk]]
    L.append(f"**Nejhlubší pokles:** {dd.min():.0%} mezi {pk:%m/%Y} a {tr:%m/%Y}; "
             + (f"návrat na předchozí maximum {rec.index[0]:%m/%Y}." if len(rec) else "předchozí maximum dosud nedosaženo."))
    return "\n\n".join(L)


def api_key():
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def make_summary(t: str, name: str, s: pd.Series):
    facts = facts_text(t, name, s)
    key = api_key()
    if key:
        try:
            import anthropic
            msg = (f"Napiš česky stručný souhrn (max 250 slov) o titulu {name} ({t}) pro investora. Použij webové "
                   "vyhledávání pro aktuální dění. Struktura: 1) co to je, 2) aktuální dění a zprávy (poslední "
                   "týdny), 3) zásadní události v minulosti, které vysvětlují největší pohyby ceny (vysvětli "
                   "uvedená data), 4) co může ovlivnit další vývoj. Jen podstatné, bez rad ke koupi.\n\n"
                   "Data z grafu a Yahoo Finance:\n" + facts)
            resp = anthropic.Anthropic(api_key=key).messages.create(
                model="claude-sonnet-5-5", max_tokens=1500,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}],
                messages=[{"role": "user", "content": msg}])
            txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
            if txt:
                return txt, "AI souhrn (Claude + webové vyhledávání)"
        except Exception as e:
            facts += f"\n\n_(AI souhrn se nepodařil: {e})_"
    return facts, "Souhrn z dat Yahoo Finance (pro AI souhrn nastavte ANTHROPIC_API_KEY)"


# ============================ POPLATKY A SIMULACE ============================
def is_crypto(t: str) -> bool:
    return t.endswith("-USD")


def apply_fees(positions: list, fees: dict) -> list:
    """Přepočítá poplatky chronologicky (obchody zdarma a limit směny se počítají za měsíc)."""
    used, rows = {}, []
    for p in sorted(positions, key=lambda x: x["date"]):
        u = used.setdefault(p["date"][:7], {"n": 0, "fx": 0.0})
        amt = p["amount"]
        if is_crypto(p["ticker"]):
            fee = amt * fees["crypto"] / 100
        else:
            fee = 0.0 if u["n"] < fees["free"] else max(fees["fixed"], amt * fees["pct"] / 100)
            u["n"] += 1
            if p["fx"]:
                fee += max(0.0, amt - max(0.0, fees["fx_lim"] - u["fx"])) * fees["fx"] / 100
                u["fx"] += amt
        fee = min(fee, amt)
        rows.append({**p, "fee": fee, "units": (amt - fee) / p["price"]})
    return rows


def sell_fee(p: dict, value: float, fees: dict) -> float:
    if is_crypto(p["ticker"]):
        return value * fees["crypto"] / 100
    return max(fees["fixed"], value * fees["pct"] / 100) + (value * fees["fx"] / 100 if p["fx"] else 0.0)


# ---------- trvalé ukládání portfolia do soukromého GitHub Gistu ----------
GIST_DESC, GIST_FILE = "investicni-asistent-portfolio", "portfolio.json"


def secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def _gh(method, path, **kw):
    r = requests.request(method, "https://api.github.com" + path, timeout=20, headers={
        "Authorization": f"Bearer {secret('GITHUB_TOKEN')}", "Accept": "application/vnd.github+json"}, **kw)
    r.raise_for_status()
    return r.json()


def gist_find():
    for page in (1, 2, 3):
        items = _gh("GET", f"/gists?per_page=100&page={page}")
        for g in items:
            if g.get("description") == GIST_DESC and GIST_FILE in g.get("files", {}):
                return g["id"]
        if len(items) < 100:
            break
    return None


def gist_load() -> list:
    gid = ss.get("gist_id") or gist_find()
    if not gid:
        return []
    ss["gist_id"] = gid
    return json.loads(_gh("GET", f"/gists/{gid}")["files"][GIST_FILE]["content"])


def load_portfolio():
    ss["pos"], ss["load_ok"], ss["sync_err"] = [], True, None
    if secret("GITHUB_TOKEN"):
        try:
            ss["pos"] = gist_load()
        except Exception as e:
            ss["load_ok"], ss["sync_err"] = False, f"Načtení z GitHubu selhalo: {e}"


def persist():
    """Uloží portfolio do Gistu; při neúspěšném načtení neukládá, aby se nepřepsala stará data."""
    if not secret("GITHUB_TOKEN"):
        return
    if not ss.get("load_ok", True):
        ss["sync_err"] = "Portfolio se nepodařilo načíst, ukládání je pozastaveno (ochrana uložených dat)."
        return
    try:
        body = {"files": {GIST_FILE: {"content": json.dumps(ss["pos"])}}}
        gid = ss.get("gist_id") or gist_find()
        if gid:
            _gh("PATCH", f"/gists/{gid}", json=body)
        else:
            gid = _gh("POST", "/gists", json={"description": GIST_DESC, "public": False, **body})["id"]
        ss["gist_id"], ss["sync_err"] = gid, None
    except Exception as e:
        ss["sync_err"] = f"Uložení na GitHub selhalo: {e}"


def fee_settings() -> dict:
    with st.expander("⚙️ Poplatky Revolut (upravitelné)"):
        plan = st.selectbox("Tarif", list(PLANS), key="plan")
        d = PLANS[plan]
        k = plan
        f = dict(
            free=st.number_input("Obchody zdarma za měsíc", 0, 100, int(d["free"]), key=f"free_{k}"),
            fixed=st.number_input("Pevný poplatek za obchod", 0.0, 100.0, float(d["fixed"]), 0.1, key=f"fix_{k}"),
            pct=st.number_input("Procentní poplatek za obchod (%)", 0.0, 5.0, float(d["pct"]), 0.01, key=f"pct_{k}"),
            fx=st.number_input("Poplatek za směnu měny (%) nad limit", 0.0, 5.0, float(d["fx"]), 0.1, key=f"fx_{k}"),
            fx_lim=st.number_input("Měsíční limit směny bez poplatku", 0.0, 1e9, float(d["fx_lim"]), 100.0,
                                   key=f"fxl_{k}"),
            crypto=st.number_input("Poplatek / spread u kryptoměn (%)", 0.0, 10.0, float(d["crypto"]), 0.01,
                                   key=f"cr_{k}"))
        st.caption("Výchozí hodnoty vycházejí z veřejných informací z podzimu 2026: od 5. 10. 2026 pevně 1 € za "
                   "obchod místo 0,25 %, od 24. 11. 2026 Standard a Plus bez obchodů zdarma, směna měny a kryptospread "
                   "podle tarifu. Poplatky se liší podle země a mění se, ověřte je v aplikaci Revolut a upravte. "
                   "Pevný poplatek zadávejte ve stejné měně jako investované částky.")
    return f


def chart(s: pd.Series, title: str, fast: int, slow: int):
    full = s.dropna()
    d = (full - full.shift(1)).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    u = (full.shift(1) - full).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    rs = 100 - 100 / (1 + d / u.replace(0, np.nan))
    v = full.iloc[-400:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_scatter(x=v.index, y=v, name="Cena", row=1, col=1)
    fig.add_scatter(x=v.index, y=full.rolling(fast).mean().loc[v.index], name=f"SMA {fast}", row=1, col=1)
    fig.add_scatter(x=v.index, y=full.rolling(slow).mean().loc[v.index], name=f"SMA {slow}", row=1, col=1)
    fig.add_scatter(x=v.index, y=rs.loc[v.index], name="RSI", row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), title=title, legend=dict(orientation="h"))
    return fig


# ============================ UI ============================
st.title("📈 Investiční asistent")
st.caption("Analýza grafů z aktuálních tržních dat. Výstup je podklad k vašemu zvážení, nikoli investiční poradenství.")
ss = st.session_state
if "pos" not in ss:
    load_portfolio()
tab_an, tab_det, tab_sim, tab_bt = st.tabs(["Analýza", "Detail titulu", "Simulace nákupu", "Backtest"])

# ---------- Analýza ----------
with tab_an:
    groups = st.multiselect("Skupiny titulů", list(STATIC) + list(INDEX_SRC),
                            default=["ETF (UCITS)", "Komodity (ETC)", "Krypto", "Nasdaq 100"])
    extra = st.text_input("Další tickery (čárkou, např. KO, NFLX)")
    st.markdown("**Analytické metody** (zaškrtněte jednu nebo více)")
    methods = [k for k, lab in METHODS.items() if st.checkbox(lab, value=k in ("trend", "mom", "rsi"), key=f"m_{k}")]
    hk = st.radio("Časový rámec investice", list(HORIZONS), format_func=lambda k: HORIZONS[k]["label"])
    top_n = st.slider("Počet doporučení", 3, 15, 5)
    uni = build_universe(groups, extra)
    st.caption(f"Vybráno titulů: {len(uni)}. Analýza se spustí až tlačítkem.")
    if st.button("▶ Spustit analýzu", type="primary"):
        if not methods:
            st.warning("Zaškrtněte alespoň jednu analytickou metodu.")
        elif not uni:
            st.warning("Vyberte alespoň jednu skupinu titulů.")
        else:
            h = HORIZONS[hk]
            with st.spinner("Stahuji data a počítám…"):
                px = load_prices(tuple(uni), h["period"])
                tb = compute_scores(px, h, methods)
            if tb.empty:
                st.error("Data se nepodařilo získat nebo je titulů s dostatečnou historií málo.")
            else:
                ss["res"] = dict(table=tb, prices=px, hk=hk, methods=list(methods), top_n=top_n)
    res = ss.get("res")
    if res:
        tb, h = res["table"], HORIZONS[res["hk"]]
        top = tb.head(res["top_n"]).copy()
        inv = 1 / top.vol.replace(0, np.nan)
        top["podil"] = (inv / inv.sum() * 100).fillna(0).round(1)
        st.subheader(f"Top {len(top)} k zvážení")
        st.caption(f"{h['label']} | metody: {', '.join(METHODS[m].split(' (')[0] for m in res['methods'])} | "
                   f"data do {res['prices'].index[-1]:%d.%m.%Y} | analyzováno {len(tb)} titulů")
        mc = [c for c in METHOD_COLS if c in top.columns]
        show = top[["nazev", "skore", "verdikt", "podil"] + mc + ["mom", "vol", "rsi", "dd"]].rename(columns={
            "nazev": "Název", "skore": "Skóre", "verdikt": "Signál", "podil": "Podíl %", "mom": "Výnos",
            "vol": "Volatilita", "rsi": "RSI hodnota", "dd": "Pokles od max."})
        fm = {"Skóre": "{:.0f}", "Podíl %": "{:.1f}", "Výnos": "{:+.1%}", "Volatilita": "{:.1%}", "RSI hodnota": "{:.0f}",
              "Pokles od max.": "{:.1%}", **{c: "{:.0f}" for c in mc}}
        st.dataframe(show.style.format(fm))
        st.caption("Podíl je vážený nepřímo úměrně volatilitě. Detail, souhrn a oscilace najdete v záložce Detail titulu.")
        with st.expander("Celý žebříček"):
            st.dataframe(tb[["nazev", "skore", "verdikt"] + [c for c in METHOD_COLS if c in tb.columns]])

# ---------- Detail ----------
with tab_det:
    opts = list(res["table"].index) if res else list(uni)
    t = st.selectbox("Titul", opts, format_func=lambda x: f"{NAMES.get(x, x)} ({x})") if opts else None
    cu = st.text_input("Nebo jiný ticker", key="cust")
    t = cu.strip().upper() if cu.strip() else t
    if not t:
        st.info("Vyberte titul.")
    else:
        pxd = load_prices((t,), "5y")
        s = pxd[t].dropna() if t in pxd else pd.Series(dtype=float)
        if len(s) < 120:
            st.error("Pro tento titul nejsou dostupná data.")
        else:
            name = NAMES.get(t, t)
            hh = HORIZONS[res["hk"]] if res else HORIZONS["stredni"]
            c1, c2 = st.columns(2)
            c1.metric("Poslední cena", f"{s.iloc[-1]:,.2f}", f"{s.iloc[-1] / s.iloc[-2] - 1:+.2%}")
            if res and t in res["table"].index:
                c2.metric("Skóre", f"{res['table'].loc[t, 'skore']:.0f}", res["table"].loc[t, "verdikt"], delta_color="off")
            st.plotly_chart(chart(s, name, hh["fast"], hh["slow"]))
            st.markdown("**Zásadní momenty v grafu**")
            for d, txt in key_moments(s):
                st.write(f"• {d:%d.%m.%Y}: {txt}")
            e = get_earnings(t)
            if e:
                st.write(f"• Očekávané výsledky hospodaření: {e}")
            st.markdown("**Aktuální zprávy**")
            for d, ti, url, src in get_news(t) or [("", "Zprávy se nepodařilo načíst.", None, "")]:
                st.markdown(f"• {d} [{ti}]({url}) {src}" if url else f"• {d} {ti}")

            st.divider()
            st.subheader("Souhrn o titulu")
            if st.button("Vygenerovat souhrn", key=f"bs_{t}"):
                with st.spinner("Připravuji souhrn…"):
                    ss[f"sum_{t}"] = make_summary(t, name, s)
            if f"sum_{t}" in ss:
                txt, src = ss[f"sum_{t}"]
                st.markdown(txt)
                st.caption(src)

            st.divider()
            st.subheader("Pravidelné oscilace")
            wopt = {"1 rok": 252, "2 roky": 504, "5 let": 1260}
            wl = st.radio("Okno analýzy", list(wopt), index=1, horizontal=True, key=f"w_{t}")
            if st.button("Analyzovat oscilace", key=f"bc_{t}"):
                win = wopt[wl]
                cy = find_cycles(s.to_numpy(), 5, 500, win, 3)
                pv = cycle_pvalue(s.to_numpy(), 5, 500, win, cy[0]["r2"]) if cy else None
                ss[f"cyc_{t}"] = (cy, pv, win)
            if f"cyc_{t}" in ss:
                cy, pv, win = ss[f"cyc_{t}"]
                if not cy:
                    st.info("V tomto okně nebyl nalezen použitelný cyklus.")
                else:
                    st.dataframe(pd.DataFrame([{"Perioda (obch. dny)": round(c["period"]), "Počet cyklů v okně": round(c["cycles"], 1),
                                                "Amplituda (dno–vrchol, %)": round(c["amp"], 1),
                                                "Vysvětlená variabilita (R²)": round(c["r2"], 2)} for c in cy]))
                    c0 = cy[0]
                    verdict = ("statisticky významný" if pv < 0.05 else "slabý, může jít o náhodu")
                    st.markdown(f"**Dominantní cyklus:** ~{c0['period']:.0f} obchodních dnů, {verdict} "
                                f"(p ≈ {pv:.2f}). Cena je nyní **{phase_text(c0['theta'])}**.")
                    sx = s.iloc[-win:]
                    y, tt, n = _prep(sx.to_numpy(), win)
                    if y is not None and len(y) == len(sx):
                        a, b, cc = c0["coef"]
                        wv = c0["w"]
                        fut = pd.bdate_range(sx.index[-1] + pd.Timedelta(days=1), periods=int(c0["period"]))
                        tf = np.arange(n, n + len(fut))
                        wave = lambda q: (a * np.sin(wv * q) + b * np.cos(wv * q) + cc) * 100
                        th = c0["theta"]
                        dtr = ((1.5 * np.pi - th) % (2 * np.pi)) / wv
                        dpk = ((0.5 * np.pi - th) % (2 * np.pi)) / wv
                        st.write(f"Pokud cyklus přetrvá: nejbližší dno ~{sx.index[-1] + pd.offsets.BDay(round(dtr)):%d.%m.%Y}, "
                                 f"nejbližší vrchol ~{sx.index[-1] + pd.offsets.BDay(round(dpk)):%d.%m.%Y}.")
                        fg = go.Figure()
                        fg.add_scatter(x=sx.index, y=y * 100, name="Odchylka od trendu (%)")
                        fg.add_scatter(x=sx.index, y=wave(tt), name="Cyklus")
                        fg.add_scatter(x=fut, y=wave(tf), name="Předpoklad", line=dict(dash="dot"))
                        fg.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"))
                        st.plotly_chart(fg)
                    st.caption("Cykly nalezené v minulosti nemusí přetrvat. Slabý nebo nevýznamný cyklus neberte jako signál.")

# ---------- Simulace ----------
with tab_sim:
    st.caption("Virtuální nákupy se sledováním zhodnocení. Ceny jsou z Yahoo Finance (denní uzávěrky, bez kurzových "
               "změn). Pro trvalé uložení nastavte GITHUB_TOKEN (viz návod), jinak použijte kód dole.")
    if secret("GITHUB_TOKEN"):
        st.caption("☁️ Portfolio se automaticky ukládá do soukromého Gistu na vašem GitHubu.")
        if ss.get("sync_err"):
            st.warning(ss["sync_err"])
        if not ss.get("load_ok", True) and st.button("Zkusit znovu načíst z GitHubu"):
            load_portfolio()
            st.rerun()
    else:
        st.info("Trvalé ukládání je vypnuté (chybí GITHUB_TOKEN v nastavení Secrets). Použijte kód dole.")
    fees = fee_settings()
    sopts = list(res["table"].index) if res else list(uni)
    st_t = st.selectbox("Titul", sopts, format_func=lambda x: f"{NAMES.get(x, x)} ({x})", key="sim_t") if sopts else None
    st_c = st.text_input("Nebo jiný ticker", key="sim_c")
    amt = st.number_input("Investovaná částka (včetně poplatků)", 1.0, 1e8, 1000.0, 100.0)
    bd = st.date_input("Datum nákupu (i zpětně)", value=dt.date.today(), max_value=dt.date.today(),
                       min_value=dt.date.today() - dt.timedelta(days=365 * 5 - 10))
    fxflag = st.checkbox("Nákup vyžaduje směnu měny", value=True)
    if st.button("Přidat virtuální nákup"):
        tk = st_c.strip().upper() or st_t
        try:
            sp = load_prices((tk,), "5y")[tk].dropna()
            sp = sp.loc[:pd.Timestamp(bd)]
            if sp.empty:
                raise ValueError("pro dané datum není cena")
            ss["pos"].append(dict(ticker=tk, name=NAMES.get(tk, tk), date=str(sp.index[-1].date()),
                                  amount=float(amt), fx=bool(fxflag), price=float(sp.iloc[-1])))
            persist()
            st.success(f"Přidáno: {tk} za cenu {sp.iloc[-1]:,.2f} ({sp.index[-1]:%d.%m.%Y}).")
        except Exception as ex:
            st.error(f"Nákup se nepodařilo přidat: {ex}")
    st.button("Obnovit ceny", on_click=st.cache_data.clear)
    if ss["pos"]:
        rows = apply_fees(ss["pos"], fees)
        tks = tuple(sorted({r["ticker"] for r in rows}))
        pxs = load_prices(tks, "5y").ffill()
        tot_in = tot_val = 0.0
        out = []
        for r in rows:
            cur = float(pxs[r["ticker"]].dropna().iloc[-1])
            val = r["units"] * cur
            sf = sell_fee(r, val, fees)
            tot_in += r["amount"]
            tot_val += val - sf
            out.append({"Titul": r["name"], "Datum": r["date"], "Vloženo": r["amount"], "Poplatek nákup": r["fee"],
                        "Hodnota nyní": val, "Čistá hodnota po prodeji": val - sf,
                        "Zisk/ztráta %": (val - sf) / r["amount"] - 1})
        st.subheader("Moje virtuální portfolio")
        st.metric("Čistá hodnota po odečtení poplatků při prodeji", f"{tot_val:,.2f}",
                  f"{tot_val - tot_in:+,.2f} ({tot_val / tot_in - 1:+.1%})")
        st.dataframe(pd.DataFrame(out).style.format({"Vloženo": "{:,.2f}", "Poplatek nákup": "{:.2f}",
                                                      "Hodnota nyní": "{:,.2f}", "Čistá hodnota po prodeji": "{:,.2f}",
                                                      "Zisk/ztráta %": "{:+.1%}"}))
        t0 = pd.Timestamp(min(r["date"] for r in rows))
        pv = pxs.loc[t0:]
        val_s = sum((pxs[r["ticker"]] * r["units"]).loc[t0:].where(pv.index >= pd.Timestamp(r["date"]), 0.0).fillna(0.0)
                    for r in rows)
        inv_s = sum(pd.Series(np.where(pv.index >= pd.Timestamp(r["date"]), r["amount"], 0.0), index=pv.index)
                    for r in rows)
        fg = go.Figure()
        fg.add_scatter(x=pv.index, y=val_s, name="Hodnota (hrubá)")
        fg.add_scatter(x=pv.index, y=inv_s, name="Vloženo", line=dict(dash="dot"))
        fg.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fg)
        c1, c2 = st.columns(2)
        if c1.button("Smazat poslední nákup"):
            ss["pos"].pop()
            persist()
            st.rerun()
        if c2.button("Smazat vše"):
            ss["pos"] = []
            persist()
            st.rerun()
    with st.expander("Uložit / obnovit portfolio (kód)"):
        st.text_area("Kód k uložení (zkopírujte)", base64.b64encode(json.dumps(ss["pos"]).encode()).decode(), height=90)
        code = st.text_input("Vložte kód k obnovení")
        if st.button("Obnovit z kódu") and code.strip():
            try:
                ss["pos"] = json.loads(base64.b64decode(code.strip()).decode())
                persist()
                st.rerun()
            except Exception:
                st.error("Kód je neplatný.")

# ---------- Backtest ----------
with tab_bt:
    st.caption("Ověří zvolené metody a časový rámec z Analýzy v minulosti: každý měsíc se koupí top N titulů podle "
               "skóre spočítaného jen z dat dostupných k danému dni (až 6 let). U velkých skupin trvá minuty.")
    bcost = st.number_input("Náklady na obchod při přeskupení (%)", 0.0, 3.0, 0.25, 0.05) / 100
    if st.button("Spustit backtest"):
        if not methods or not uni:
            st.warning("Nejdřív v záložce Analýza vyberte tituly a aspoň jednu metodu.")
        else:
            with st.spinner("Počítám backtest…"):
                try:
                    bt = run_backtest(tuple(uni), top_n, bcost, hk, tuple(methods))
                except Exception as ex:
                    bt = None
                    st.error(f"Backtest se nepodařilo spočítat: {ex}")
            if bt:
                fg = go.Figure()
                for nm, r in bt.items():
                    fg.add_scatter(x=r.index, y=(1 + r).cumprod() * 100, name=nm)
                fg.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"),
                                 yaxis_title="Hodnota (start = 100)")
                st.plotly_chart(fg)
                st.dataframe(pd.DataFrame({k: perf_stats(v) for k, v in bt.items()}).T.style.format(
                    {"Roční výnos (CAGR)": "{:+.1%}", "Volatilita": "{:.1%}", "Sharpe": "{:.2f}",
                     "Max. pokles": "{:.1%}", "Celkový výnos": "{:+.1%}"}))
                st.warning("Seznam titulů obsahuje dnešní známé firmy (survivorship bias), výsledek je spíše "
                           "optimistický. Minulá výkonnost nezaručuje budoucí.")
