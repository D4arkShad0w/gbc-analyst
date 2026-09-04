#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GBC ANALYST — all-in-one AI Gold & Bitcoin Market Analyst (Iran-focused)
=========================================================================
Single file. Runs on GitHub Actions. Reports to Telegram.

Two separate decision chains:
  GOLD_IR : Global gold → USD/IRR (free market) → implied local price → premium → prediction
  XAUUSD  : Global gold (technicals + NDS structure + macro + correlations + news AI)
  BTC     : Macro/liquidity → risk sentiment → BTC → Iranian exchange premium

Every prediction is PROBABILISTIC with explicit invalidation conditions.
Never certainty. Not financial advice.

Usage:
  python gbc_analyst.py --full          # comprehensive analysis
  python gbc_analyst.py --quick         # cheap watchdog (shocks, invalidations, hot news)
  python gbc_analyst.py --auto          # recommended for CI (quick, escalates to full when due)
  python gbc_analyst.py --test-notify   # send a Telegram test message

Env vars (or .env file):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, FRED_API_KEY,
  GEMINI_API_KEY (or GOOGLE_API_KEY),
  GEMINI_MODELS   (optional comma-separated override),
  CRYPTOPANIC_TOKEN (optional), NOTIFY=0 to mute Telegram.
"""

import os, re, sys, json, csv, time, html, hashlib
from calendar import timegm
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev

import requests
import feedparser
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

# ======================================================================
# 1. CONFIG
# ======================================================================
def _env_models():
    default = ("gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash-lite,"
               "gemini-2.5-flash,gemini-2.0-flash")
    return [m.strip() for m in os.getenv("GEMINI_MODELS", default).split(",") if m.strip()]

CFG = {
    "TELEGRAM_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT":  os.getenv("TELEGRAM_CHAT_ID", ""),
    "FRED_KEY":       os.getenv("FRED_API_KEY", ""),
    "GEMINI_KEY":     os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
    "GEMINI_MODELS":  _env_models(),
    "CRYPTOPANIC_TOKEN": os.getenv("CRYPTOPANIC_TOKEN", ""),
    "STATE_DIR":      os.getenv("STATE_DIR", "state"),
    "NOTIFY":         os.getenv("NOTIFY", "1") == "1",
    # Base module weights (the grading loop adapts these from real hit rates)
    "WEIGHTS": {"technical": 1.0, "nds": 1.2, "correlation": 0.8, "macro": 1.5,
                "sentiment": 0.9, "ai_events": 1.8, "iran_local": 1.6},
    "MIN_CONF": 40, "MAX_CONF": 92,           # confidence is clamped — never 100%
    "SHOCK": {"XAUUSD": 0.015, "BTC": 0.035, "USDIRR": 0.010},
    "INTEL_MAX_AGE_H": 8,                     # reuse cached LLM news analysis for 8h
}

P_STATE  = os.path.join(CFG["STATE_DIR"], "market_state.json")
P_SCHED  = os.path.join(CFG["STATE_DIR"], "schedule.json")
P_GRADE  = os.path.join(CFG["STATE_DIR"], "grading.json")
P_ALERTS = os.path.join(CFG["STATE_DIR"], "alerts.json")
P_GEM    = os.path.join(CFG["STATE_DIR"], "gemini.json")
P_HIST   = os.path.join(CFG["STATE_DIR"], "history", "snapshots.csv")

# ======================================================================
# 2. SMALL UTILITIES
# ======================================================================
def now():     return datetime.now(timezone.utc)
def nowiso():  return now().isoformat(timespec="seconds")

def jload(path, default):
    try:
        with open(path, encoding="utf-8") as f: return json.load(f)
    except Exception:
        return default

def jsave(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)

def retry(fn, tries=3, wait=2):
    last = None
    for i in range(tries):
        try: return fn()
        except Exception as e:
            last = e; time.sleep(wait * (i + 1))
    raise last

def sig(module, asset, score, conf, why, w=1.0, horizon="2-7d"):
    """A signal: direction (-1..+1), own confidence (0..1), weight, plain-language why."""
    return {"module": module, "asset": asset,
            "score": max(-1.0, min(1.0, float(score))), "conf": float(conf),
            "w": float(w), "why": why, "horizon": horizon}

def fmt(v):
    return f"{v:,.2f}" if abs(v) < 10000 else f"{v:,.0f}"

def esc(t):
    return html.escape(str(t), quote=False)

# ======================================================================
# 3. DATA — GLOBAL MARKETS (yfinance)
# ======================================================================
TICKERS = {"XAUUSD": "GC=F", "BTC": "BTC-USD", "DXY": "DX-Y.NYB", "US10Y": "^TNX",
           "OIL": "CL=F", "SPX": "^GSPC", "VIX": "^VIX", "GLD": "GLD"}

def fetch_history(sym, period="1y"):
    def _d():
        df = yf.download(TICKERS.get(sym, sym), period=period, interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    df = retry(_d, tries=3, wait=2)
    if df.empty and sym == "XAUUSD":                     # fallback: GLD scaled to gold
        g = fetch_history("GLD", period)
        return g * (2400.0 / float(g["Close"].iloc[-1]))
    return df

def _chg(c, n):
    try:
        v = float(c.pct_change(n).iloc[-1])
        return v if pd.notna(v) else 0.0
    except Exception:
        return 0.0

def global_snapshots():
    snaps, frames = {}, {}
    for key in ["XAUUSD", "BTC", "DXY", "US10Y", "OIL", "SPX", "VIX"]:
        try:
            df = fetch_history(key)
            frames[key] = df
            c = df["Close"].squeeze()
            snaps[key] = {"price": float(c.iloc[-1]),
                          "chg_1d": _chg(c, 1), "chg_7d": _chg(c, 5)}
        except Exception as e:
            snaps[key] = {"error": str(e)}
    return snaps, frames

# ======================================================================
# 4. DATA — IRAN (TGJU free-market dollar / 18k gold / coin, Nobitex USDT/BTC)
# ======================================================================
TGJU_URL = "https://api.tgju.org/v1/market/indicator/summary-table-data/{code}"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

def tgju(code):
    def _g():
        r = requests.get(TGJU_URL.format(code=code), headers=UA, timeout=15)
        r.raise_for_status()
        digits = re.sub(r"[^\d.]", "", str(r.json()["data"][0][1]))
        return float(digits) if digits else None
    return retry(_g, tries=2, wait=1)

def nobitex():
    def _g():
        r = requests.get("https://api.nobitex.ir/market/stats",
                         params={"srcCurrency": "usdt,btc", "dstCurrency": "irt"},
                         headers=UA, timeout=15)
        r.raise_for_status()
        s = r.json()["stats"]
        return float(s["usdt-irt"]["latest"]) * 10, float(s["btc-irt"]["latest"]) * 10  # → RIAL
    return retry(_g, tries=2, wait=1)

def iran_snapshot(xau):
    snap = {"usdirr_free": None, "geram18_rial": None, "emami_rial": None,
            "usdt_irt_rial": None, "btc_irt_rial": None,
            "geram18_implied_rial": None, "geram18_premium_pct": None,
            "usdt_premium_pct": None, "btc_ir_premium_pct": None}
    try:
        snap["usdirr_free"]  = tgju("price_dollar_rl")
        snap["geram18_rial"] = tgju("geram18")
        snap["emami_rial"]   = tgju("sekee")
    except Exception:
        pass
    try:
        usdt, btc = nobitex()
        snap["usdt_irt_rial"], snap["btc_irt_rial"] = usdt, btc
        if not snap["usdirr_free"] and usdt:            # fallback dollar proxy (typical USDT discount)
            snap["usdirr_free"] = round(usdt * 0.985)
    except Exception:
        pass
    usd, g18 = snap["usdirr_free"], snap["geram18_rial"]
    if usd and g18 and xau:
        implied = xau * usd * 0.75 / 31.1035            # implied 18k gram in Rial
        snap["geram18_implied_rial"] = implied
        snap["geram18_premium_pct"] = (g18 / implied - 1.0) * 100
    if snap["usdt_irt_rial"] and usd:
        snap["usdt_premium_pct"] = (snap["usdt_irt_rial"] / usd - 1.0) * 100
    return snap

def btc_ir_premium(btc_irt_rial, btc_usd, usd):
    if btc_irt_rial and btc_usd and usd:
        return (btc_irt_rial / (btc_usd * usd) - 1.0) * 100
    return None

# ======================================================================
# 5. DATA — US MACRO (FRED)
# ======================================================================
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

def fred_series(sid, limit=400):
    r = requests.get(FRED_URL, params={"series_id": sid, "api_key": CFG["FRED_KEY"],
                                       "file_type": "json", "sort_order": "desc",
                                       "limit": limit}, timeout=15)
    r.raise_for_status()
    obs = [(o["date"], float(o["value"])) for o in r.json()["observations"]
           if o["value"] not in (".", "")]
    obs.reverse()
    return obs

def macro_snapshot():
    out = {}
    if not CFG["FRED_KEY"]:
        return out
    try:
        d10 = fred_series("DGS10", 30); bie = fred_series("T10YIE", 30)
        out["us10y"] = d10[-1][1]
        out["us10y_chg_5d"] = d10[-1][1] - d10[-6][1]
        if len(bie) >= 6:
            out["real10y"] = d10[-1][1] - bie[-1][1]
            out["real10y_chg_5d"] = out["real10y"] - (d10[-6][1] - bie[-6][1])
    except Exception:
        pass
    try:
        cpi = fred_series("CPIAUCSL", 60)
        if len(cpi) >= 13: out["cpi_yoy"] = (cpi[-1][1] / cpi[-13][1] - 1) * 100
    except Exception:
        pass
    try:
        walcl = fred_series("WALCL", 12)
        out["fed_bs_chg_4w_pct"] = (walcl[-1][1] / walcl[-5][1] - 1) * 100
    except Exception:
        pass
    try:
        m2 = fred_series("M2SL", 30)
        if len(m2) >= 13: out["m2_yoy"] = (m2[-1][1] / m2[-13][1] - 1) * 100
    except Exception:
        pass
    return out

# ======================================================================
# 6. DATA — NEWS (Google News RSS + GDELT + CryptoPanic)
# ======================================================================
QUERIES = ["gold price", "bitcoin", "federal reserve interest rate decision",
           "US inflation CPI", "iran sanctions", "iran rial dollar",
           "middle east conflict oil", "central bank gold purchases",
           "treasury yields dollar index", "crypto regulation"]

def _md5(t): return hashlib.md5(t.lower().encode()).hexdigest()[:12]

def google_news(q, limit=8):
    url = (f"https://news.google.com/rss/search?q={requests.utils.quote(q)}"
           f"&hl=en-US&gl=US&ceid=US:en")
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:limit]:
        pub = getattr(e, "published_parsed", None)
        ts = datetime.fromtimestamp(timegm(pub), tz=timezone.utc) if pub else now()
        out.append({"id": _md5(e.title), "title": html.unescape(e.title),
                    "source": "GoogleNews", "url": e.get("link", ""),
                    "published": ts.isoformat(timespec="seconds")})
    return out

def gdelt():
    url = ("https://api.gdeltproject.org/api/v2/doc/doc?query=" +
           requests.utils.quote("(gold OR bitcoin OR sanctions OR iran) market sourcelang:english") +
           "&mode=artlist&maxrecords=20&format=json&timespan=24h")
    r = requests.get(url, timeout=20, headers=UA)
    r.raise_for_status()
    return [{"id": _md5(a["title"]), "title": a["title"], "source": "GDELT",
             "url": a.get("url", ""), "published": a.get("seendate", "")}
            for a in r.json().get("articles", [])]

def cryptopanic():
    if not CFG["CRYPTOPANIC_TOKEN"]: return []
    try:
        r = requests.get("https://cryptopanic.com/api/v1/posts/",
                         params={"auth_token": CFG["CRYPTOPANIC_TOKEN"],
                                 "public": "true", "currencies": "BTC"}, timeout=15)
        return [{"id": _md5(p["title"]), "title": p["title"], "source": "CryptoPanic",
                 "url": p.get("url", ""), "published": p.get("published_at", "")}
                for p in r.json().get("results", [])[:15]]
    except Exception:
        return []

def collect_news():
    items = []
    for q in QUERIES:
        try: items += google_news(q)
        except Exception: pass
    try: items += gdelt()
    except Exception: pass
    items += cryptopanic()
    seen, uniq = set(), []
    for it in items:
        if it["id"] in seen: continue
        seen.add(it["id"]); uniq.append(it)
    return uniq[:60]

# ======================================================================
# 7. INDICATORS (hand-rolled, textbook formulas — no pandas_ta needed)
# ======================================================================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, 1e-12)
    return 100 - 100 / (1 + rs)

def macd_hist(s):
    line = ema(s, 12) - ema(s, 26)
    return line - ema(line, 9)

def atr_series(df, n=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def atr_pctile_now(df):
    c = df["Close"].squeeze()
    a = (atr_series(df, 14) / c * 100).dropna()
    return float((a < a.iloc[-1]).mean() * 100) if len(a) > 20 else 50.0

# ======================================================================
# 8. ANALYSIS — TECHNICAL
# ======================================================================
def technical_block(df):
    c = df["Close"].squeeze()
    e200 = float(ema(c, 200).iloc[-1]) if len(c) >= 200 else None
    h = macd_hist(c)
    return {"close": float(c.iloc[-1]),
            "ema20": float(ema(c, 20).iloc[-1]), "ema50": float(ema(c, 50).iloc[-1]),
            "ema200": e200, "rsi": float(rsi(c).iloc[-1]),
            "atr_pct": float((atr_series(df, 14) / c * 100).iloc[-1]),
            "atr_pctile": atr_pctile_now(df),
            "macd_hist": float(h.iloc[-1]), "macd_hist_prev": float(h.iloc[-2])}

def technical_signals(asset, t):
    out, stack = [], 0.0
    if t["close"] > t["ema50"] and t["ema20"] > t["ema50"]: stack += 0.5
    else: stack -= 0.5
    if t["ema200"] is not None:
        stack += 0.5 if t["close"] > t["ema200"] else -0.5
    ema_txt = f"{t['ema20']:,.2f}/{t['ema50']:,.2f}"
    if t["ema200"] is not None: ema_txt += f"/{t['ema200']:,.2f}"
    out.append(sig("technical", asset, stack, 0.7,
                   f"Price {fmt(t['close'])} sits {'above' if stack > 0 else 'below'} its main moving "
                   f"averages ({ema_txt}) — trend is {'up' if stack > 0 else 'down or weak'}."))
    r = t["rsi"]
    rs = (r - 50) / 25 if 30 <= r <= 70 else (0.3 if r > 70 else -0.3)
    out.append(sig("technical", asset, rs, 0.5,
                   f"RSI(14) = {r:.1f} — " +
                   ("overbought, momentum stretched." if r > 70 else
                    "oversold, momentum stretched." if r < 30 else "balanced momentum.")))
    mh, mp = t["macd_hist"], t["macd_hist_prev"]
    out.append(sig("technical", asset, 1.0 if mh > 0 else -1.0, 0.6,
                   f"MACD momentum is {'positive' if mh > 0 else 'negative'} and "
                   f"{'rising' if mh > mp else 'fading'}."))
    return out

# ======================================================================
# 9. ANALYSIS — NDS / MARKET STRUCTURE + SUPPORT & RESISTANCE
#    NDS implemented deterministically: swing nodes → displacement legs
#    (range ÷ ATR) → structure sequence (HH/HL vs LH/LL) → score.
# ======================================================================
def swings(df, w=3):
    hi, lo, n = df["High"].squeeze(), df["Low"].squeeze(), len(df)
    nodes = []
    for i in range(w, max(w, n - w)):
        if hi.iloc[i] >= hi.iloc[i - w:i + w + 1].max(): nodes.append([i, float(hi.iloc[i]), "H"])
        if lo.iloc[i] <= lo.iloc[i - w:i + w + 1].min(): nodes.append([i, float(lo.iloc[i]), "L"])
    out = []
    for nd in nodes:                                   # enforce H/L alternation
        if out and out[-1][2] == nd[2]:
            if (nd[2] == "H" and nd[1] >= out[-1][1]) or (nd[2] == "L" and nd[1] <= out[-1][1]):
                out[-1] = nd
        else:
            out.append(nd)
    return out

def structure_block(df):
    a = float(atr_series(df, 14).iloc[-1])
    nodes = swings(df)
    highs = [p for _, p, t in nodes if t == "H"][-2:]
    lows  = [p for _, p, t in nodes if t == "L"][-2:]
    close = float(df["Close"].iloc[-1])
    struct, seq = 0.0, "not enough swing points yet"
    if len(highs) == 2 and len(lows) == 2:
        hh, hl = highs[1] > highs[0], lows[1] > lows[0]
        if hh and hl:                 struct, seq = 1.0, "higher highs + higher lows (uptrend structure)"
        elif not hh and not hl:       struct, seq = -1.0, "lower highs + lower lows (downtrend structure)"
        else:                         struct, seq = 0.0, "mixed swings (range)"
    disp, dsign = 0.0, 0
    if nodes:
        move = close - nodes[-1][1]
        disp = abs(move) / a if a > 0 else 0.0
        dsign = 1 if move > 0 else -1
    score = max(-1.0, min(1.0, 0.6 * struct + 0.4 * dsign * min(1.0, disp / 2)))
    return {"score": score, "seq": seq, "atr": a, "close": close, "disp": disp}

def nds_signal(asset, b):
    return sig("nds", asset, b["score"], 0.75,
               f"Market structure: {b['seq']}. Last displacement leg is {b['disp']:.1f}× the "
               f"average daily range ({'strong' if b['disp'] > 1.5 else 'moderate' if b['disp'] > 0.8 else 'weak'}).")

def support_resistance(df, atr_v, max_each=3):
    pts = sorted([p for _, p, _ in swings(df)] + [float(df["Close"].iloc[-1])])
    tol = max(0.004 * pts[-1], 0.5 * atr_v)
    clusters, cur = [], [pts[0]]
    for p in pts[1:]:
        if p - cur[-1] <= tol: cur.append(p)
        else: clusters.append(cur); cur = [p]
    clusters.append(cur)
    zones = [sum(c) / len(c) for c in clusters]
    close = float(df["Close"].iloc[-1])
    return ([z for z in zones if z < close][-max_each:],
            [z for z in zones if z >= close][:max_each])

# ======================================================================
# 10. ANALYSIS — CORRELATION
# ======================================================================
def correlation_signals(frames):
    closes = {}
    for k, df in frames.items():
        try: closes[k] = df["Close"].squeeze()
        except Exception: pass
    if len(closes) < 3: return []
    ret = pd.DataFrame(closes).dropna().tail(40).pct_change().dropna()
    if len(ret) < 25: return []
    out = []
    def add(asset, peers):
        vals = []
        for peer, expect in peers:
            if peer in ret.columns and asset in ret.columns:
                c = ret[asset].corr(ret[peer])
                if pd.notna(c) and abs(c) > 0.3:
                    vals.append(expect * c * ret[peer].tail(5).sum() * 3)
        if vals:
            s = max(-1.0, min(1.0, sum(vals) / len(vals)))
            out.append(sig("correlation", asset, s, min(0.8, 0.4 + 0.1 * len(vals)),
                           f"Cross-market correlations (30-day) are currently "
                           f"{'supportive of' if s > 0 else 'pressuring'} {asset}."))
    add("XAUUSD", [("DXY", -1), ("US10Y", -1), ("OIL", 0.5)])
    add("BTC",    [("SPX", 1), ("DXY", -1), ("XAUUSD", 0.3)])
    return out

# ======================================================================
# 11. ANALYSIS — MACRO
# ======================================================================
def macro_signals(mac, g):
    out = []
    if not mac: return out
    parts, why = [], []
    if "real10y_chg_5d" in mac:
        parts.append(-mac["real10y_chg_5d"] * 15)
        why.append(f"10Y real yield {mac.get('real10y', 0):.2f}% (5d Δ {mac['real10y_chg_5d']:+.2f})")
    if g.get("DXY", {}).get("chg_7d") is not None:
        parts.append(-g["DXY"]["chg_7d"] * 20)
        why.append(f"dollar index {'weaker' if g['DXY']['chg_7d'] < 0 else 'stronger'} "
                   f"({g['DXY']['chg_7d'] * 100:+.1f}% / 5d)")
    if "fed_bs_chg_4w_pct" in mac:
        parts.append(mac["fed_bs_chg_4w_pct"] * 10)
        why.append(f"Fed balance sheet {mac['fed_bs_chg_4w_pct']:+.2f}% / 4w")
    if parts:
        out.append(sig("macro", "XAUUSD", max(-1, min(1, sum(parts) / len(parts))), 0.75,
                       "Macro backdrop: " + "; ".join(why) +
                       ". Falling real yields and a weaker dollar are tailwinds for gold.",
                       horizon="weekly"))
    bparts, bwhy = [], []
    if "fed_bs_chg_4w_pct" in mac or "m2_yoy" in mac:
        bparts.append((mac.get("fed_bs_chg_4w_pct", 0) * 12 + mac.get("m2_yoy", 0)) * 0.05)
        bwhy.append(f"liquidity (Fed BS {mac.get('fed_bs_chg_4w_pct', 0):+.2f}%/4w, "
                    f"M2 YoY {mac.get('m2_yoy', 0):+.1f}%)")
    if g.get("SPX", {}).get("chg_7d") is not None and g.get("VIX", {}).get("chg_7d") is not None:
        bparts.append(g["SPX"]["chg_7d"] * 15 - g["VIX"]["chg_7d"] * 0.3)
        bwhy.append(f"risk appetite (S&P 5d {g['SPX']['chg_7d'] * 100:+.1f}%, "
                    f"VIX {g['VIX'].get('price', 0):.0f})")
    if bparts:
        out.append(sig("macro", "BTC", max(-1, min(1, sum(bparts) / len(bparts))), 0.7,
                       "Macro backdrop for crypto: " + "; ".join(bwhy) + ".", horizon="weekly"))
    return out

# ======================================================================
# 12. ANALYSIS — SENTIMENT (VADER + Fear & Greed)
# ======================================================================
_VADER = SentimentIntensityAnalyzer()

def headline_sentiment(items):
    if not items: return 0.0
    comp = [_VADER.polarity_scores(i["title"])["compound"] for i in items]
    return max(-1.0, min(1.0, sum(comp) / len(comp) * 2))

def fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
        return int(r["data"][0]["value"])
    except Exception:
        return None

def sentiment_signals(items, fng=None):
    out = []
    vs = headline_sentiment(items)
    if items:
        out.append(sig("sentiment", "XAUUSD", vs * 0.5, 0.4,
                       f"News headline sentiment across {len(items)} recent stories is "
                       f"{'positive' if vs > 0.05 else 'negative' if vs < -0.05 else 'neutral'}."))
    if fng is not None:
        out.append(sig("sentiment", "BTC", (fng - 50) / 50, 0.5,
                       f"Crypto Fear & Greed index at {fng}/100 "
                       f"({'greed' if fng > 60 else 'fear' if fng < 40 else 'neutral'})."))
    return out

# ======================================================================
# 13. ANALYSIS — IRAN-SPECIFIC MODELS
#     Iranian gold ≈ (XAU/USD) × (USD/IRR) × local premium.
#     USD/IRR is usually the dominant driver and can fully offset a
#     global gold drop — that's why it carries the highest weight.
# ======================================================================
def iran_levels(rows):
    """Support/resistance for the 18k gram (Rial) from accumulated daily history."""
    seq = [float(r["geram18_rial"]) for r in rows[-60:]
           if r.get("geram18_rial") not in ("", None)]
    if len(seq) < 15: return [], []
    srt, close = sorted(seq), seq[-1]
    q = lambda p: srt[int(p * (len(srt) - 1))]
    sup = [v for v in dict.fromkeys((q(.25), q(.40))) if v < close * 0.999]
    res = [v for v in dict.fromkeys((q(.75), q(.90))) if v > close * 1.001]
    return sup, res

def iran_gold_signals(snap, rows, ai_usdirr):
    out = []
    usd, prem = snap.get("usdirr_free"), snap.get("geram18_premium_pct")
    chg7 = snap.get("usdirr_chg_7d") or 0
    if usd:
        s = max(-1.0, min(1.0, chg7 * 40))
        out.append(sig("iran_local", "GOLD_IR", s, 0.8,
                       f"Free-market USD/IRR at {usd:,.0f} rial is "
                       f"{'rising' if s > 0.1 else 'falling' if s < -0.1 else 'flat'} "
                       f"({chg7 * 100:+.2f}% over ~5 days) — the single biggest driver of "
                       f"Iranian gold prices.", w=1.6))
    xs = snap.get("xau_score", 0)
    if xs:
        out.append(sig("iran_local", "GOLD_IR", xs * 0.6, 0.6,
                       "Global gold trend partially passes through to local prices after "
                       "currency conversion.", w=1.2))
    if prem is not None and len(rows) >= 20:
        hist = [float(r["geram18_premium_pct"]) for r in rows
                if r.get("geram18_premium_pct") not in ("", None)]
        if len(hist) >= 20 and pstdev(hist) > 0.01:
            z = (prem - mean(hist)) / pstdev(hist)
            out.append(sig("iran_local", "GOLD_IR", max(-1.0, min(1.0, -z * 0.6)), 0.5,
                           f"Local 18k gold trades at {prem:+.1f}% ({z:+.1f}σ) vs its implied "
                           f"value (global gold × USD/IRR). " +
                           ("Stretched premium raises pullback risk." if z > 1 else
                            "Discount leaves room for local catch-up." if z < -1 else
                            "Premium is near its normal range.")))
    elif prem is not None:
        out.append(sig("iran_local", "GOLD_IR", 0, 0.4,
                       f"Local premium vs implied value: {prem:+.1f}% "
                       f"(needs ~3 more weeks of history for a z-score)."))
    if abs(ai_usdirr) > 0.15:
        out.append(sig("ai_events", "GOLD_IR", ai_usdirr, 0.7,
                       "Recent Iran-related news (sanctions, conflict, domestic policy) is "
                       "expected to pressure the rial and, with it, local gold and coin prices.",
                       w=1.8))
    return out

def btc_ir_signal(prem):
    if prem is None: return None
    return sig("iran_local", "BTC", max(-1.0, min(1.0, prem / 8)), 0.5,
               f"BTC trades at {prem:+.1f}% vs its converted global price on Iranian "
               f"exchanges — {'strong local demand' if prem > 2 else 'weak local demand' if prem < -2 else 'normal spread'}.")

# ======================================================================
# 14. AI LAYER — GEMINI (REST, model rotation, quota-friendly)
# ======================================================================
GEM_SYS = """You are a financial news analyst for GOLD and BITCOIN markets, with special
expertise in IRAN (USD/IRR free-market rate, Iranian gold & coin market, Iranian crypto market).

Classify each headline. Return ONLY JSON:
{"items":[{"id":"...","category":"monetary|inflation|geopolitics|iran_domestic|sanctions|military|crypto|market|other",
 "impact":"high|medium|low|noise|structural",
 "gold":-1..1,"bitcoin":-1..1,"usdirr":-1..1,
 "timing":"next 24-72h|this week|next week|ongoing|unknown"}],
 "digest":"3-sentence summary of the current market narrative",
 "narrative_changed": true|false,
 "high_impact_events":[{"name":"...","asset":"gold|bitcoin|usdirr","timing":"..."}],
 "invalidations":{"gold":["specific condition that would flip the current gold view"],
                  "bitcoin":["..."],"usdirr":["..."]}}

Rules:
- impact=high ONLY if it can move gold/BTC/USD-IRR by more than 1-2% within 24-72h.
- impact=structural for long-term regime shifts (e.g., central banks accelerating gold buying).
- impact=noise for routine chatter and opinion pieces.
- usdirr: +1 means news pressures USD/IRR UP (rial weakens → Iranian gold/coin prices tend UP).
- Be conservative. Scores reflect the direction of the EXPECTED price move, not moral judgment."""

SAFETY = [{"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_ONLY_HIGH"}
          for c in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")]

def gemini_chat(system, user, max_tokens=4096, json_mode=True):
    """Tries each configured Gemini model in order (starting from the last one that
    worked). On quota errors (429) / model errors (403, 404) it rotates to the next."""
    if not CFG["GEMINI_KEY"] or not CFG["GEMINI_MODELS"]:
        return ""
    models = CFG["GEMINI_MODELS"]
    start = jload(P_GEM, {}).get("idx", 0) % len(models)
    last_err = "no models configured"
    for i in range(len(models)):
        m = models[(start + i) % len(models)]
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "safetySettings": SAFETY,
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens,
                                     **({"responseMimeType": "application/json"} if json_mode else {})}}
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                              params={"key": CFG["GEMINI_KEY"]}, json=body, timeout=120)
            if r.status_code == 200:
                cands = r.json().get("candidates") or []
                if not cands:
                    last_err = f"{m}: blocked/empty response"; continue
                txt = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
                jsave(P_GEM, {"idx": (start + i) % len(models), "model": m})
                print(f"[gemini] used {m}")
                return txt
            last_err = f"{m}: HTTP {r.status_code}"
            print(f"[gemini] {last_err} → rotating")
        except Exception as e:
            last_err = f"{m}: {e}"
            print(f"[gemini] {last_err} → rotating")
    print("[gemini] all models failed:", last_err)
    return ""

def parse_json(text):
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r"\{.*\}", text or "", re.S)
    try: return json.loads(m.group(0)) if m else {}
    except Exception: return {}

def fallback_intel():
    return {"scores": {"gold": 0, "bitcoin": 0, "usdirr": 0},
            "digest": "(LLM unavailable — AI event analysis disabled this run.)",
            "changed": False, "high": [], "invalidations": {}, "ok": False}

def analyze_news(items, prev_digest=""):
    lines = "\n".join(f"[{i['id']}] {i['title']}" for i in items[:45])
    user = (f"PREVIOUS NARRATIVE (compare against it):\n{prev_digest or 'none yet'}\n\n"
            f"HEADLINES (last ~40h):\n{lines}\n\nReturn the JSON now.")
    data = parse_json(gemini_chat(GEM_SYS, user))
    if not data or "items" not in data:
        return fallback_intel()
    agg = {"gold": 0.0, "bitcoin": 0.0, "usdirr": 0.0}
    n = {"gold": 0.0, "bitcoin": 0.0, "usdirr": 0.0}
    wmap = {"high": 1.0, "structural": 0.8, "medium": 0.5, "low": 0.2, "noise": 0.0}
    for it in data.get("items", []):
        w = wmap.get(it.get("impact"), 0)
        for k in agg:
            v = it.get(k)
            if w and isinstance(v, (int, float)) and v != 0:
                agg[k] += v * w; n[k] += w
    return {"scores": {k: (max(-1.0, min(1.0, agg[k] / n[k])) if n[k] else 0.0) for k in agg},
            "digest": data.get("digest", ""),
            "changed": bool(data.get("narrative_changed")),
            "high": data.get("high_impact_events", []) or [],
            "invalidations": data.get("invalidations", {}) or {},
            "ok": True}

def get_intel(items, st):
    """LLM quota saver: reuse cached news intelligence if headlines unchanged recently."""
    if not items:
        return fallback_intel(), True
    ids_hash = hashlib.md5("|".join(sorted(i["id"] for i in items)).encode()).hexdigest()
    cache = st.get("intel", {})
    age_h = 1e9
    if cache.get("fetched"):
        try: age_h = (now() - datetime.fromisoformat(cache["fetched"])).total_seconds() / 3600
        except Exception: pass
    if cache.get("ids_hash") == ids_hash and cache.get("ok") and age_h < CFG["INTEL_MAX_AGE_H"]:
        return cache, True
    intel = analyze_news(items, st.get("digest", "")) if CFG["GEMINI_KEY"] else fallback_intel()
    intel.update({"ids_hash": ids_hash, "fetched": nowiso()})
    return intel, False

# ======================================================================
# 15. DECISION ENGINE — signal fusion, confidence, risk, zones
# ======================================================================
def adapted_weights():
    """Base weight × accuracy multiplier learned from closed predictions."""
    W = dict(CFG["WEIGHTS"])
    for mod, s in jload(P_GRADE, {}).get("module_stats", {}).items():
        if mod in W and s.get("total", 0) >= 5:
            acc = s["hits"] / s["total"]
            W[mod] = round(min(2.0, max(0.5, W[mod] * (0.5 + acc))), 3)
    return W

def fuse(signals, W=None):
    W = W or adapted_weights()
    num = den = 0.0
    for s in signals:
        w = W.get(s["module"], 1.0) * s["w"] * max(s["conf"], 0.1)
        num += w * s["score"]; den += w
    net = num / den if den else 0.0
    d = 1 if net >= 0 else -1
    aligned = sum(W.get(s["module"], 1.0) * s["w"] * abs(s["score"])
                  for s in signals if s["score"] * d > 0)
    total = sum(W.get(s["module"], 1.0) * s["w"] * abs(s["score"]) for s in signals) or 1.0
    return net, aligned / total

def make_prediction(asset, signals, levels, intel, entry, horizon="2-7d", days=5):
    net, agree = fuse(signals)
    direction = "bullish" if net > 0.12 else "bearish" if net < -0.12 else "neutral"
    conf = CFG["MIN_CONF"] + (CFG["MAX_CONF"] - CFG["MIN_CONF"]) * abs(net) * (0.45 + 0.55 * agree)
    if direction == "neutral": conf = min(conf, 55)
    conf = int(max(CFG["MIN_CONF"], min(CFG["MAX_CONF"], conf)))
    rs_ = levels.get("atr_pctile", 50) + len(intel.get("high", [])) * 20
    risk = "high" if rs_ > 100 else "medium" if rs_ > 60 else "low"
    W = adapted_weights()
    ranked = sorted(signals, key=lambda s: -(abs(s["score"]) * W.get(s["module"], 1.0) * s["w"]))[:5]
    sup, res = levels.get("supports", []), levels.get("resistances", [])
    zone = lambda v: f"{fmt(v * 0.998)} – {fmt(v * 1.002)}"
    buy_zones  = [zone(v) for v in (sup[-2:] if direction != "bearish" else sup[-1:])]
    sell_zones = [zone(v) for v in (res[:2] if direction != "bullish" else res[:1])]
    inv, inv_levels = [], []
    if direction == "bullish" and sup:
        inv.append(f"A daily close below {fmt(sup[-1])} breaks the nearest support / market structure.")
        inv_levels.append(sup[-1])
    elif direction == "bearish" and res:
        inv.append(f"A daily close above {fmt(res[0])} reclaims the broken structure.")
        inv_levels.append(res[0])
    keymap = {"XAUUSD": "gold", "BTC": "bitcoin", "GOLD_IR": "usdirr"}
    inv += [str(x) for x in intel.get("invalidations", {}).get(keymap.get(asset, "gold"), [])][:2]
    return {"asset": asset, "direction": direction, "net": round(net, 3),
            "confidence": conf, "risk": risk, "horizon": horizon,
            "reasons": [s["why"] for s in ranked if s["why"]],
            "buy_zones": buy_zones, "sell_zones": sell_zones,
            "supports": [fmt(v) for v in sup], "resistances": [fmt(v) for v in res],
            "invalidations": inv, "invalidation_levels": inv_levels,
            "entry_price": entry or 0, "created": nowiso(),
            "valid_until": (now() + timedelta(days=days)).isoformat(timespec="seconds")}

# ======================================================================
# 16. GRADING LOOP — the system checks itself and adapts weights
# ======================================================================
def grade_and_adapt(prices_now):
    g = jload(P_GRADE, {"predictions": [], "module_stats": {}})
    still_open, closed = [], []
    for p in g["predictions"]:
        try: matured = now() >= datetime.fromisoformat(p["valid_until"])
        except Exception: matured = False
        (closed if matured else still_open).append(p)
    for p in closed:
        px0, px1 = p.get("entry_price"), prices_now.get(p["asset"])
        if not px0 or not px1: continue
        move = px1 / px0 - 1
        d = p["direction"]
        hit = (move > 0.002) if d == "bullish" else (move < -0.002) if d == "bearish" else abs(move) <= 0.005
        for mod in p.get("modules", []):
            m = g["module_stats"].setdefault(mod, {"hits": 0, "total": 0})
            m["hits"] += int(hit); m["total"] += 1
        print(f"[grade] {p['asset']} {d} → {'HIT' if hit else 'MISS'} ({move * 100:+.2f}%)")
    g["predictions"] = still_open[-60:]
    jsave(P_GRADE, g)

def register_predictions(preds):
    g = jload(P_GRADE, {"predictions": [], "module_stats": {}})
    for a, p in preds.items():
        g["predictions"].append({"asset": a, "direction": p["direction"],
                                 "entry_price": p["entry_price"],
                                 "valid_until": p["valid_until"],
                                 "modules": p.get("modules", [])})
    g["predictions"] = g["predictions"][-60:]
    jsave(P_GRADE, g)

# ======================================================================
# 17. TELEGRAM — formatting, dedupe, sending
# ======================================================================
ARROWS = {"bullish": "📈 Bullish", "bearish": "📉 Bearish", "neutral": "↔️ Range-bound"}
RISKI  = {"low": "🟢 Low", "medium": "🟡 Medium", "high": "🔴 High"}
TITLES = {"GOLD_IR": "🥇 GOLD — IRAN", "XAUUSD": "🌍 GOLD — GLOBAL", "BTC": "₿ BITCOIN"}

def _chunks(t, n):
    while t:
        cut = t.rfind("\n", 0, n)
        cut = cut if cut > 500 else n
        yield t[:cut]; t = t[cut:].lstrip("\n")

def tg_send(text):
    if not CFG["NOTIFY"] or not CFG["TELEGRAM_TOKEN"] or not CFG["TELEGRAM_CHAT"]:
        print("[telegram] muted / missing credentials — suppressed:\n", text[:200])
        return False
    for chunk in _chunks(text, 3900):
        for _ in range(3):
            try:
                r = requests.post(f"https://api.telegram.org/bot{CFG['TELEGRAM_TOKEN']}/sendMessage",
                                  json={"chat_id": CFG["TELEGRAM_CHAT"], "text": chunk,
                                        "parse_mode": "HTML", "disable_web_page_preview": True},
                                  timeout=20)
                if r.status_code == 429:
                    time.sleep(int(r.json().get("parameters", {}).get("retry_after", 3)))
                    continue
                break
            except Exception:
                time.sleep(2)
    return True

def _ts(iso):
    try: return datetime.fromisoformat(iso).timestamp()
    except Exception: return 0

def fp(kind, *parts):
    return kind + ":" + hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:10]

def should_send(key, cooldown_h):
    a = jload(P_ALERTS, {"sent": {}})
    last = a["sent"].get(key)
    return not (last and time.time() - _ts(last) < cooldown_h * 3600)

def mark_sent(key):
    a = jload(P_ALERTS, {"sent": {}})
    a["sent"][key] = nowiso()
    a["sent"] = {k: v for k, v in a["sent"].items() if time.time() - _ts(v) < 7 * 86400}
    jsave(P_ALERTS, a)

def prediction_block(title, p):
    z = ""
    if p["buy_zones"]:  z += f"\n🟩 <b>Buy zone:</b> {' | '.join(p['buy_zones'])}"
    if p["sell_zones"]: z += f"\n🟥 <b>Sell zone:</b> {' | '.join(p['sell_zones'])}"
    if p["supports"]:   z += f"\n<b>Supports:</b> {', '.join(p['supports'][-2:])}"
    if p["resistances"]:z += f"\n<b>Resistances:</b> {', '.join(p['resistances'][:2])}"
    inv = "\n".join("• " + esc(i) for i in p["invalidations"][:3]) or "• N/A"
    why = "\n".join("• " + esc(r) for r in p["reasons"][:5]) or "• —"
    return (f"<b>{title}</b>\n"
            f"Direction: {ARROWS[p['direction']]}\n"
            f"Confidence: <b>{p['confidence']}%</b> (probability — not certainty)\n"
            f"Risk: {RISKI[p['risk']]}  •  Timeframe: {p['horizon']}\n\n"
            f"<b>Why?</b>\n{why}{z}\n\n<b>❌ Invalidation:</b>\n{inv}")

def send_daily_report(preds, intel, weekly):
    key = fp("daily", nowiso()[:10])
    if not should_send(key, 20): return
    body = ""
    for a in ["GOLD_IR", "XAUUSD", "BTC"]:
        if a in preds: body += "\n\n" + prediction_block(TITLES[a], preds[a])
    ev = "\n".join(f"• [{esc(e.get('timing', '?'))}] {esc(e.get('name', '?'))} → {esc(e.get('asset', '?'))}"
                   for e in intel.get("high", [])[:6]) or "• None flagged right now."
    wk = "\n".join(f"• {esc(k)}: {esc(v)}" for k, v in weekly.items()) or "• —"
    tg_send(f"🤖 <b>Daily Market Analysis — {nowiso()[:10]}</b>\n" + body +
            f"\n\n<b>📅 Week Ahead</b>\n{wk}\n\n<b>⏰ Events to Watch</b>\n{ev}"
            f"\n\n<b>📰 Narrative</b>\n{esc(intel.get('digest', ''))}"
            f"\n\n<i>Predictions are probabilistic with explicit invalidation conditions — "
            f"never guarantees. Not financial advice.</i>")
    mark_sent(key)

def send_prediction_change(p, reason):
    key = fp("pred", p["asset"], p["direction"], p["confidence"] // 10, round(p["net"], 1))
    if not should_send(key, 4): return
    t = TITLES.get(p["asset"], p["asset"])
    tg_send(f"🔁 <b>Prediction update — {t}</b>\n({esc(reason)})\n\n" + prediction_block(t, p))
    mark_sent(key)

def send_alert(text, kind="alert", cooldown_h=1):
    key = fp(kind, text[:120])
    if not should_send(key, cooldown_h): return
    tg_send("🚨 <b>ALERT</b>\n" + text)
    mark_sent(key)

# ======================================================================
# 18. STATE / HISTORY PERSISTENCE
# ======================================================================
HIST_COLS = ["date", "xauusd", "btc_usd", "dxy", "us10y", "oil", "usdirr_free",
             "usdt_irt", "geram18_rial", "emami_rial", "btc_irt",
             "geram18_premium_pct", "usdt_premium_pct"]

def _history_rows():
    if not os.path.exists(P_HIST): return []
    with open(P_HIST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def append_history(snap):
    os.makedirs(os.path.dirname(P_HIST), exist_ok=True)
    today = nowiso()[:10]
    rows = []
    if os.path.exists(P_HIST):
        with open(P_HIST, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != today]
    row = {"date": today}
    for c in HIST_COLS[1:]:
        v = snap.get(c)
        row[c] = round(v, 4) if isinstance(v, (int, float)) else (v or "")
    rows.append(row); rows = rows[-730:]
    with open(P_HIST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HIST_COLS)
        w.writeheader(); w.writerows(rows)

# ======================================================================
# 19. PIPELINE
# ======================================================================
def full():
    print(f"=== FULL ANALYSIS @ {nowiso()} ===")
    st = jload(P_STATE, {})
    prev_snap = st.get("snap", {})
    g, frames = global_snapshots()
    if "price" not in g.get("XAUUSD", {}):
        print("!! No XAUUSD data — aborting this run."); return {}
    xau = g["XAUUSD"]["price"]

    ir = iran_snapshot(xau)
    if ir.get("btc_irt_rial"):
        ir["btc_ir_premium_pct"] = btc_ir_premium(ir["btc_irt_rial"],
                                                  g["BTC"].get("price"), ir.get("usdirr_free"))
    rows = _history_rows()
    if len(rows) >= 6 and ir.get("usdirr_free"):
        prev = rows[-6].get("usdirr_free")
        try:
            if prev: ir["usdirr_chg_7d"] = ir["usdirr_free"] / float(prev) - 1
        except Exception: pass

    mac = macro_snapshot()
    items = collect_news()
    intel, reused = get_intel(items, st)
    print(f"[intel] {'reused cache' if reused else 'fresh LLM analysis'}; news items={len(items)}")

    # ---- deterministic analysis ----
    signals = {"XAUUSD": [], "GOLD_IR": [], "BTC": []}
    tech = {}
    for a in ("XAUUSD", "BTC"):
        if a not in frames: continue
        try:
            t = technical_block(frames[a])
            tech[a] = t
            signals[a] += technical_signals(a, t)
            sb = structure_block(frames[a])
            signals[a].append(nds_signal(a, sb))
            sup, res = support_resistance(frames[a], sb["atr"])
            t["supports"], t["resistances"] = sup, res
            if a == "XAUUSD": ir["xau_score"] = sb["score"]
        except Exception as e:
            print(f"[analysis] {a} failed: {e}")

    signals["XAUUSD"] += correlation_signals(frames)
    signals["BTC"]    += correlation_signals(frames)
    signals["XAUUSD"] += macro_signals(mac, g)
    signals["BTC"]    += macro_signals(mac, g)
    signals["BTC"]    += sentiment_signals(items, fear_greed())
    signals["XAUUSD"] += sentiment_signals(items)
    for asset, k in (("XAUUSD", "gold"), ("BTC", "bitcoin")):
        s = intel["scores"].get(k, 0)
        if abs(s) > 0.1:
            signals[asset].append(sig("ai_events", asset, s, 0.8,
                                      "AI reading of the latest news flow points net in this direction."))

    # ---- Iran-specific chains ----
    signals["GOLD_IR"] += iran_gold_signals(ir, rows, intel["scores"].get("usdirr", 0))
    bps = btc_ir_signal(ir.get("btc_ir_premium_pct"))
    if bps: signals["BTC"].append(bps)

    # ---- grade old predictions first, then build new ones ----
    prices_now = {"XAUUSD": xau, "BTC": g["BTC"].get("price"), "GOLD_IR": ir.get("geram18_rial")}
    grade_and_adapt(prices_now)

    preds = {}
    for a in ("GOLD_IR", "XAUUSD", "BTC"):
        if not signals[a]: continue
        if a == "GOLD_IR":
            sup, res = iran_levels(rows)
            levels = {"supports": sup, "resistances": res,
                      "atr_pctile": tech.get("XAUUSD", {}).get("atr_pctile", 50)}
        else:
            t = tech.get(a, {})
            levels = {"supports": t.get("supports", []), "resistances": t.get("resistances", []),
                      "atr_pctile": t.get("atr_pctile", 50)}
        p = make_prediction(a, signals[a], levels, intel, prices_now.get(a))
        p["modules"] = sorted({s["module"] for s in signals[a]})
        preds[a] = p
        print(f"[pred] {a}: {p['direction'].upper()}  conf={p['confidence']}%  net={p['net']}  risk={p['risk']}")

    # ---- alerts: prediction changes & invalidations ----
    for a, p in preds.items():
        o = st.get("preds", {}).get(a)
        if o and (o["direction"] != p["direction"] or abs(o["confidence"] - p["confidence"]) >= 15):
            why = ("direction changed" if o["direction"] != p["direction"]
                   else f"confidence {o['confidence']}% → {p['confidence']}%")
            send_prediction_change(p, why)
    for a, p in preds.items():
        px = prices_now.get(a)
        for lvl in p["invalidation_levels"]:
            if px and ((p["direction"] == "bullish" and px < lvl) or
                       (p["direction"] == "bearish" and px > lvl)):
                send_alert(f"<b>{TITLES.get(a, a)}</b> prediction invalidated — "
                           f"price {fmt(px)} crossed {fmt(lvl)}.", kind="invalid", cooldown_h=6)

    # ---- weekly view + daily report ----
    weekly = {}
    for a, label, mods in (("XAUUSD", "Global gold", ("macro", "ai_events")),
                           ("BTC", "Bitcoin", ("macro", "ai_events", "sentiment")),
                           ("GOLD_IR", "Iranian gold", ("iran_local", "ai_events"))):
        w = [s for s in signals[a] if s["horizon"] == "weekly" or s["module"] in mods]
        if w:
            net, _ = fuse(w)
            weekly[label] = ("Higher bias" if net > 0.1 else "Lower bias" if net < -0.1 else "Mixed") \
                            + f" (score {net:+.2f})"
    send_daily_report(preds, intel, weekly)

    # ---- persist state & history, set adaptive schedule ----
    st.update({"preds": preds, "digest": intel.get("digest", ""),
               "intel": {k: intel.get(k) for k in ("scores", "digest", "changed", "high",
                                                   "invalidations", "ids_hash", "fetched", "ok")},
               "updated": nowiso(),
               "snap": {"xauusd": xau, "btc": g["BTC"].get("price"),
                        "usdirr": ir.get("usdirr_free"), "geram18": ir.get("geram18_rial")}})
    jsave(P_STATE, st)
    append_history({"xauusd": xau, "btc_usd": g.get("BTC", {}).get("price"),
                    "dxy": g.get("DXY", {}).get("price"), "us10y": g.get("US10Y", {}).get("price"),
                    "oil": g.get("OIL", {}).get("price"), "usdirr_free": ir.get("usdirr_free"),
                    "usdt_irt": ir.get("usdt_irt_rial"), "geram18_rial": ir.get("geram18_rial"),
                    "emami_rial": ir.get("emami_rial"), "btc_irt": ir.get("btc_irt_rial"),
                    "geram18_premium_pct": ir.get("geram18_premium_pct"),
                    "usdt_premium_pct": ir.get("usdt_premium_pct")})
    register_predictions(preds)
    write_policy(prev_snap, g, ir, intel)
    return preds

# ---- adaptive execution frequency ----
def write_policy(prev_snap, g, ir, intel):
    shock = any(base and cur and abs(cur / base - 1) > thr for _, base, cur, thr in (
        ("XAUUSD", prev_snap.get("xauusd"), g.get("XAUUSD", {}).get("price"), CFG["SHOCK"]["XAUUSD"]),
        ("BTC",    prev_snap.get("btc"),    g.get("BTC", {}).get("price"),    CFG["SHOCK"]["BTC"]),
        ("USDIRR", prev_snap.get("usdirr"), ir.get("usdirr_free"),            CFG["SHOCK"]["USDIRR"])))
    vol_hot = False
    try: vol_hot = structure_block(fetch_history("XAUUSD"))["disp"] > 2.0
    except Exception: pass
    breaking = bool(intel.get("changed")) and bool(intel.get("high"))
    mode = "critical" if (breaking or shock) else \
           "elevated" if (vol_hot or len(intel.get("high", [])) >= 2) else "normal"
    every = {"normal": 360, "elevated": 120, "critical": 60}[mode]
    jsave(P_SCHED, {"mode": mode, "full_every_min": every,
                    "next_full_due": (now() + timedelta(minutes=every)).isoformat(timespec="seconds"),
                    "decided": nowiso()})
    print(f"[policy] mode={mode}, next full analysis in ~{every} min")

# ---- cheap watchdog ----
HOT_KEYWORDS = ("sanction", "strike", "attack", "war", "emergency", "rate decision", "cpi",
                "inflation surprise", "default", "devaluation", "intervention",
                "nuclear", "ceasefire")

def quick():
    st = jload(P_STATE, {})
    if not st.get("preds"):
        print("[quick] no state yet → bootstrap full run")
        return full()
    g, _ = global_snapshots()
    snap, alerts = st.get("snap", {}), []
    for k, path in (("XAUUSD", "xauusd"), ("BTC", "btc")):
        base, cur = snap.get(path), g.get(k, {}).get("price")
        if base and cur and abs(cur / base - 1) > CFG["SHOCK"][k]:
            alerts.append(f"<b>{k}</b> moved {(cur / base - 1) * 100:+.1f}% since last check "
                          f"({fmt(base)} → {fmt(cur)}).")
    for a, p in st.get("preds", {}).items():
        px = {"XAUUSD": g.get("XAUUSD", {}).get("price"),
              "BTC": g.get("BTC", {}).get("price")}.get(a)
        for lvl in p.get("invalidation_levels", []):
            if px and ((p["direction"] == "bullish" and px < lvl) or
                       (p["direction"] == "bearish" and px > lvl)):
                alerts.append(f"<b>{TITLES.get(a, a)}</b>: invalidation level {fmt(lvl)} "
                              f"crossed (price {fmt(px)}).")
    try:  # cheap Iranian gold check
        g18, p = tgju("geram18"), st["preds"].get("GOLD_IR")
        if p and g18:
            for lvl in p.get("invalidation_levels", []):
                if (p["direction"] == "bullish" and g18 < lvl) or \
                   (p["direction"] == "bearish" and g18 > lvl):
                    alerts.append(f"<b>🥇 GOLD — IRAN</b>: invalidation level {fmt(lvl)} "
                                  f"crossed (18k gram {fmt(g18)} rial).")
    except Exception:
        pass
    items = collect_news()
    seen = set(st.get("seen_news", []))
    for i in [i for i in items if i["id"] not in seen
              and any(k in i["title"].lower() for k in HOT_KEYWORDS)][:3]:
        alerts.append("Possible market-moving news: " + esc(i["title"]))
    if alerts:
        send_alert("\n".join("• " + a for a in alerts), kind="quick", cooldown_h=2)
    st["seen_news"] = ([i["id"] for i in items] + st.get("seen_news", []))[:400]
    jsave(P_STATE, st)
    pol = jload(P_SCHED, {})
    due = False
    if pol.get("next_full_due"):
        try: due = now() >= datetime.fromisoformat(pol["next_full_due"])
        except Exception: due = False
    if due:
        print("[quick] full analysis due → running")
        full()

def auto():
    st = jload(P_STATE, {})
    if not st.get("preds") or not os.path.exists(P_SCHED):
        return full()
    return quick()

def test_notify():
    ok = tg_send("✅ <b>GBC Analyst test</b> — Telegram pipeline works. "
                 "Full analysis arrives at the next scheduled run.")
    print("telegram:", "sent" if ok else "muted or failed")

# ======================================================================
# 20. CLI
# ======================================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--auto"
    if mode in ("--full", "--force"): full()
    elif mode == "--quick":           quick()
    elif mode == "--test-notify":     test_notify()
    else:                             auto()

if __name__ == "__main__":
    main()
