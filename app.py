import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Investiční asistent", page_icon="📈", layout="wide")

UNIVERSE = {
    "Akcie USA": {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "GOOGL": "Alphabet",
                  "AMZN": "Amazon", "META": "Meta", "JPM": "JPMorgan", "XOM": "Exxon Mobil",
                  "JNJ": "Johnson & Johnson", "TSLA": "Tesla"},
    "Akcie Evropa/ČR": {"ASML": "ASML", "SAP": "SAP", "NVO": "Novo Nordisk", "CEZ.PR": "ČEZ",
                        "KOMB.PR": "Komerční banka"},
    "ETF": {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "VT": "Světové akcie", "EEM": "Rozvíjející se trhy",
            "TLT": "Dluhopisy USA 20+", "VNQ": "Nemovitosti USA"},
    "Komodity": {"GC=F": "Zlato", "SI=F": "Stříbro", "CL=F": "Ropa WTI", "NG=F": "Zemní plyn",
                 "HG=F": "Měď", "ZW=F": "Pšenice"},
    "Krypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"},
}
NAMES = {t: n for g in UNIVERSE.values() for t, n in g.items()}


@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(tickers: tuple, period: str = "2y") -> pd.DataFrame:
    df = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True)
    close = df["Close"] if "Close" in df else df
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how="all")


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def indicators(s: pd.Series) -> dict | None:
    s = s.dropna()
    if len(s) < 220:
        return None
    r = s.pct_change().dropna()
    sma50, sma200 = s.rolling(50).mean(), s.rolling(200).mean()
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    last1y = s.iloc[-252:]
    vol = r.iloc[-252:].std() * np.sqrt(252)
    ret6m = s.iloc[-1] / s.iloc[-127] - 1
    ret12m = s.iloc[-1] / s.iloc[-253] - 1 if len(s) > 253 else np.nan
    return dict(price=s.iloc[-1], sma50=sma50.iloc[-1], sma200=sma200.iloc[-1],
                rsi=rsi(s).iloc[-1], macd_hist=hist.iloc[-1], vol=vol, ret6m=ret6m, ret12m=ret12m,
                sharpe=(r.iloc[-252:].mean() * 252) / vol if vol else 0,
                dd=s.iloc[-1] / last1y.max() - 1)


def score_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for t in prices.columns:
        i = indicators(prices[t])
        if i:
            rows[t] = i
    df = pd.DataFrame(rows).T.astype(float)
    if df.empty:
        return df
    trend = (df.price > df.sma50) * 10 + (df.sma50 > df.sma200) * 10 + (df.price > df.sma200) * 10
    mom = df.ret6m.rank(pct=True) * 30
    macd = (df.macd_hist > 0) * 10
    rs = np.where(df.rsi.between(45, 65), 10, np.where(df.rsi.between(30, 45) | df.rsi.between(65, 75), 5, 0))
    risk = df.sharpe.rank(pct=True) * 10 + np.where(df.dd > -0.10, 10, np.where(df.dd > -0.20, 5, 0))
    df["skore"] = (trend + mom + macd + rs + risk).round(0)
    df["verdikt"] = np.where(df.skore >= 70, "Silný signál", np.where(df.skore >= 50, "Sledovat", "Slabý"))
    df["nazev"] = [NAMES.get(t, t) for t in df.index]
    return df.sort_values("skore", ascending=False)


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
def get_earnings(t: str) -> str | None:
    try:
        cal = yf.Ticker(t).calendar
        d = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if d:
            return ", ".join(str(x) for x in d)
    except Exception:
        pass
    return None


def key_moments(s: pd.Series) -> list:
    s = s.dropna()
    ev = []
    r = s.pct_change().iloc[-252:]
    for d, v in r.abs().nlargest(3).items():
        ev.append((d, f"Největší denní pohyb: {r[d]:+.1%}"))
    sma50, sma200 = s.rolling(50).mean(), s.rolling(200).mean()
    up = (sma50 > sma200).astype(int).diff()
    for d in up[up == 1].index[-1:]:
        ev.append((d, "Golden cross (SMA50 prorazila SMA200 vzhůru)"))
    for d in up[up == -1].index[-1:]:
        ev.append((d, "Death cross (SMA50 klesla pod SMA200)"))
    y = s.iloc[-252:]
    ev.append((y.idxmax(), f"52týdenní maximum ({y.max():,.2f})"))
    ev.append((y.idxmin(), f"52týdenní minimum ({y.min():,.2f})"))
    return sorted(ev, key=lambda x: x[0], reverse=True)


def chart(s: pd.Series, title: str):
    s = s.dropna().iloc[-400:]
    full = s
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_scatter(x=s.index, y=s, name="Cena", row=1, col=1)
    fig.add_scatter(x=s.index, y=full.rolling(50).mean(), name="SMA 50", row=1, col=1)
    fig.add_scatter(x=s.index, y=full.rolling(200).mean(), name="SMA 200", row=1, col=1)
    fig.add_scatter(x=s.index, y=rsi(full), name="RSI", row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), title=title, legend=dict(orientation="h"))
    return fig


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = max(len(r) / 252, 1e-9)
    vol = r.std() * np.sqrt(252)
    return {"Roční výnos (CAGR)": eq.iloc[-1] ** (1 / years) - 1, "Volatilita": vol,
            "Sharpe": (r.mean() * 252) / vol if vol else np.nan,
            "Max. pokles": (eq / eq.cummax() - 1).min(), "Celkový výnos": eq.iloc[-1] - 1}


@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(tickers: tuple, top_n: int, cost: float) -> dict:
    """Měsíční rebalancování: v každém měsíci se skóre spočítá jen z dat dostupných k tomu dni."""
    prices = load_prices(tickers, "6y").ffill()
    rets = prices.pct_change()
    idx = prices.index
    month_ends = idx.to_series().groupby(idx.to_period("M")).last().tolist()
    month_ends = [d for d in month_ends if idx.get_loc(d) >= 260]
    strat = pd.Series(0.0, index=idx)
    prev_w = pd.Series(0.0, index=prices.columns)
    for k, d in enumerate(month_ends):
        tbl = score_table(prices.loc[:d])
        if tbl.empty:
            continue
        sel = tbl.head(top_n).index
        w = pd.Series(0.0, index=prices.columns)
        w[sel] = 1 / len(sel)
        start = idx.get_loc(d) + 1
        end = idx.get_loc(month_ends[k + 1]) if k + 1 < len(month_ends) else len(idx) - 1
        if start > end:
            continue
        days = idx[start:end + 1]
        r = rets.loc[days, sel].mean(axis=1).fillna(0)
        r.iloc[0] -= cost * (w - prev_w).abs().sum()
        strat.loc[days] = r
        prev_w = w
    first = strat[strat != 0].index[0]
    out = {"Strategie (top skóre)": strat.loc[first:],
           "Všechna aktiva rovnoměrně": rets.loc[first:].mean(axis=1).fillna(0)}
    if "SPY" in rets:
        out["S&P 500 (SPY)"] = rets.loc[first:, "SPY"].fillna(0)
    return out


st.title("📈 Investiční asistent")
st.caption("Analýza grafů z aktuálních tržních dat. Výstup je podklad k vašemu zvážení, nikoli investiční poradenství.")

with st.sidebar:
    groups = st.multiselect("Skupiny aktiv", list(UNIVERSE), default=list(UNIVERSE))
    top_n = st.slider("Počet doporučení", 3, 10, 5)
    extra = st.text_input("Další tickery (čárkou, např. KO, NFLX)")
    st.button("Obnovit data", on_click=st.cache_data.clear)

tickers = [t for g in groups for t in UNIVERSE[g]]
tickers += [x.strip().upper() for x in extra.split(",") if x.strip()]
tickers = tuple(dict.fromkeys(tickers))
if not tickers:
    st.info("Vyberte alespoň jednu skupinu aktiv.")
    st.stop()

with st.spinner("Stahuji aktuální data z trhů…"):
    prices = load_prices(tickers)
table = score_table(prices)
if table.empty:
    st.error("Nepodařilo se získat data. Zkontrolujte připojení k internetu a zkuste to znovu.")
    st.stop()

st.caption(f"Poslední data: {prices.index[-1]:%d.%m.%Y}  |  Analyzováno titulů: {len(table)}")

top = table.head(top_n).copy()
inv = 1 / top.vol
top["podil"] = (inv / inv.sum() * 100).round(1)

st.subheader(f"Top {top_n} k zvážení")
show = top[["nazev", "skore", "verdikt", "podil", "ret6m", "vol", "rsi", "dd"]].rename(columns={
    "nazev": "Název", "skore": "Skóre", "verdikt": "Signál", "podil": "Návrh podílu %",
    "ret6m": "Výnos 6M", "vol": "Volatilita", "rsi": "RSI", "dd": "Pokles od maxima"})
st.dataframe(show.style.format({"Výnos 6M": "{:+.1%}", "Volatilita": "{:.1%}", "RSI": "{:.0f}",
                                "Pokles od maxima": "{:.1%}", "Návrh podílu %": "{:.1f}"}),
             use_container_width=True)
st.caption("Podíly jsou váženy nepřímo úměrně volatilitě (méně rizikové tituly dostávají větší váhu).")

st.subheader("Detail a zásadní momenty")
for t, row in top.iterrows():
    with st.expander(f"{row.nazev} ({t}): skóre {row.skore:.0f}, {row.verdikt}", expanded=False):
        st.plotly_chart(chart(prices[t], row.nazev), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Zásadní momenty v grafu**")
            for d, txt in key_moments(prices[t]):
                st.write(f"• {d:%d.%m.%Y}: {txt}")
            e = get_earnings(t)
            if e:
                st.write(f"• Očekávané výsledky hospodaření: {e}")
        with c2:
            st.markdown("**Aktuální zprávy**")
            news = get_news(t)
            if not news:
                st.write("Zprávy se nepodařilo načíst.")
            for d, title, url, src in news:
                st.markdown(f"• {d} [{title}]({url}) {('(' + src + ')') if src else ''}" if url else f"• {d} {title}")
        sig = []
        sig.append("cena nad SMA 200" if row.price > row.sma200 else "cena pod SMA 200")
        sig.append("SMA 50 nad SMA 200" if row.sma50 > row.sma200 else "SMA 50 pod SMA 200")
        sig.append("MACD pozitivní" if row.macd_hist > 0 else "MACD negativní")
        sig.append("RSI přeprodáno" if row.rsi < 30 else "RSI překoupeno" if row.rsi > 70 else "RSI neutrální")
        st.info("Technický obraz: " + ", ".join(sig) + ".")

with st.expander("Celý žebříček"):
    st.dataframe(table[["nazev", "skore", "verdikt", "ret6m", "ret12m", "vol", "rsi", "dd"]], use_container_width=True)

with st.expander("Jak se počítá skóre (0 až 100)"):
    st.markdown(
        "- **Trend (30):** cena vs. SMA 50 a SMA 200, pozice SMA 50 vůči SMA 200\n"
        "- **Momentum (30):** pořadí 6měsíčního výnosu v rámci analyzovaných aktiv\n"
        "- **MACD (10):** kladný histogram\n- **RSI (10):** zdravé pásmo 45 až 65\n"
        "- **Riziko (20):** Sharpe za 1 rok a pokles od ročního maxima\n\n"
        "Historická data nezaručují budoucí výnosy. Skóre neřeší oceňování, fundamenty ani vaši daňovou situaci.")

st.subheader("Backtest strategie")
st.caption("Ověří, jak by si skórovací systém vedl v minulosti: každý měsíc by se koupilo top N titulů "
           "podle skóre spočítaného jen z dat dostupných k danému dni. Použije se až 6 let historie.")
bt_cost = st.number_input("Poplatek za obchod (%)", 0.0, 2.0, 0.1, 0.05) / 100
if st.button("Spustit backtest"):
    with st.spinner("Počítám backtest (může trvat desítky sekund)…"):
        try:
            res = run_backtest(tickers, top_n, bt_cost)
        except Exception as e:
            res = None
            st.error(f"Backtest se nepodařilo spočítat: {e}")
    if res:
        fig = go.Figure()
        for name, r in res.items():
            fig.add_scatter(x=r.index, y=(1 + r).cumprod() * 100, name=name)
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Hodnota (start = 100)",
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({k: perf_stats(v) for k, v in res.items()}).T.style.format(
            {"Roční výnos (CAGR)": "{:+.1%}", "Volatilita": "{:.1%}", "Sharpe": "{:.2f}",
             "Max. pokles": "{:.1%}", "Celkový výnos": "{:+.1%}"}), use_container_width=True)
        st.warning("Pozor na zkreslení: seznam titulů obsahuje dnešní známé firmy (survivorship bias), "
                   "které v minulosti mohly vyjít lépe, než by se dalo předem vědět. Výsledek je tedy spíše "
                   "optimistický. Minulá výkonnost nezaručuje budoucí.")
