#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GBC ANALYST v3.1 — تحلیلگر طلایی/بیت‌کوین (تک‌فایل، فارسی، لاگ کامل، تیکرهای اصلاح‌شده)
Usage: python gbc_analyst.py --full | --quick | --auto | --test-notify
Log:   state/logs/last_run.log
"""

import os, re, sys, json, csv, time, html, hashlib, logging, traceback
from calendar import timegm
from contextlib import contextmanager
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

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
DIAG = {"run_id": RUN_ID, "steps": {}, "sources": {}, "llm": {},
        "counts": {"warn": 0, "err": 0}}

# ======================================================================
# 1. LOGGING
# ======================================================================
LOG_PATH = os.path.join(os.getenv("STATE_DIR", "state"), "logs", "last_run.log")
log = logging.getLogger("gbc")

class _CountFilter(logging.Filter):
    def filter(self, rec):
        if rec.levelno >= logging.ERROR:
            DIAG["counts"]["err"] += 1
        elif rec.levelno >= logging.WARNING:
            DIAG["counts"]["warn"] += 1
        return True

def setup_logging():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    # ساکت‌کردن لاگ‌های پرحرف کتابخانه‌ها
    for noisy in ("yfinance", "peewee", "urllib3", "multiprocessing"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    log.addFilter(_CountFilter())

def src(name, ok, err=None):
    d = DIAG["sources"].setdefault(name, {"ok": 0, "fail": 0, "last_err": None})
    if ok:
        d["ok"] += 1
    else:
        d["fail"] += 1
        d["last_err"] = str(err)[:250]
        log.warning(f"[source:{name}] FAILED: {err}")

@contextmanager
def step(name):
    log.info(f"▶ {name} — شروع")
    t0 = time.time()
    try:
        yield
        sec = round(time.time() - t0, 1)
        DIAG["steps"][name] = {"ok": True, "sec": sec}
        log.info(f"✔ {name} — OK ({sec}s)")
    except Exception as e:
        sec = round(time.time() - t0, 1)
        DIAG["steps"][name] = {"ok": False, "sec": sec, "err": str(e)[:300]}
        log.error(f"✖ {name} — FAILED ({sec}s): {e}")
        log.debug(traceback.format_exc())
        raise

def retry(fn, tries=3, wait=2, name="?"):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            log.debug(f"[retry:{name}] attempt {i + 1}/{tries} failed: {e}")
            time.sleep(wait * (i + 1))
    raise last

# ======================================================================
# 2. CONFIG  (تمام مقادیر env با strip — فاصله/اینتر اضافه حذف می‌شود)
# ======================================================================
def _e(key, default=""):
    return (os.getenv(key) or "").strip() or default

def _env_models():
    default = ("gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash-lite,"
               "gemini-2.5-flash,gemini-2.0-flash")
    return [m.strip() for m in _e("GEMINI_MODELS", default).split(",") if m.strip()]

CFG = {
    "TELEGRAM_TOKEN": _e("TELEGRAM_BOT_TOKEN"),
    "TELEGRAM_CHAT":  _e("TELEGRAM_CHAT_ID"),
    "FRED_KEY":       _e("FRED_API_KEY"),
    "GEMINI_KEY":     _e("GEMINI_API_KEY") or _e("GOOGLE_API_KEY"),
    "GEMINI_MODELS":  _env_models(),
    "CRYPTOPANIC_TOKEN": _e("CRYPTOPANIC_TOKEN"),
    "STATE_DIR":      _e("STATE_DIR", "state"),
    "NOTIFY":         _e("NOTIFY", "1") == "1",
    "WEIGHTS": {"technical": 1.0, "nds": 1.2, "nds_w": 0.8, "correlation": 0.8,
                "macro": 1.5, "sentiment": 0.9, "ai_events": 1.8, "iran_local": 1.6},
    "MIN_CONF": 40, "MAX_CONF": 92,
    "SHOCK": {"XAUUSD": 0.015, "BTC": 0.035, "USDIRR": 0.010},
    "INTEL_MAX_AGE_H": 8,
}

P_STATE  = os.path.join(CFG["STATE_DIR"], "market_state.json")
P_SCHED  = os.path.join(CFG["STATE_DIR"], "schedule.json")
P_GRADE  = os.path.join(CFG["STATE_DIR"], "grading.json")
P_ALERTS = os.path.join(CFG["STATE_DIR"], "alerts.json")
P_GEM    = os.path.join(CFG["STATE_DIR"], "gemini.json")
P_HIST   = os.path.join(CFG["STATE_DIR"], "history", "snapshots.csv")

# ======================================================================
# 3. UTILS + فارسی‌سازی
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

def sig(module, asset, score, conf, why, w=1.0, horizon="2-7d"):
    return {"module": module, "asset": asset,
            "score": max(-1.0, min(1.0, float(score))), "conf": float(conf),
            "w": float(w), "why": why, "horizon": horizon}

def fmt(v):
    try:
        return f"{v:,.2f}" if abs(v) < 10000 else f"{v:,.0f}"
    except Exception:
        return str(v)

def esc(t):
    return html.escape(str(t), quote=False)

FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
def fa(v):
    return str(v).translate(FA)

DIR_FA  = {"bullish": "📈 انتظار بالا رفتن قیمت",
           "bearish": "📉 انتظار پایین آمدن قیمت",
           "neutral": "↔️ بدون تغییر بزرگ (بلاتکلیف)"}
DIR_S   = {"bullish": "صعودی", "bearish": "نزولی", "neutral": "بلاتکلیف"}
RISK_FA = {"low": "🟢 کم", "medium": "🟡 متوسط", "high": "🔴 بالا"}
TITLES  = {"GOLD_IR": "🥇 طلا در ایران", "XAUUSD": "🌍 طلای جهانی", "BTC": "₿ بیت‌کوین"}
SHORTN  = {"GOLD_IR": "طلا ایران", "XAUUSD": "طلا جهانی", "BTC": "بیت‌کوین"}
UNITS   = {"GOLD_IR": "ریال", "XAUUSD": "دلار", "BTC": "دلار"}
TIMING_FA = {"next 24-72h": "۲۴ تا ۷۲ ساعت آینده", "this week": "همین هفته",
             "next week": "هفتهٔ بعد", "ongoing": "در جریان", "unknown": ""}

# ======================================================================
# 4. DATA — GLOBAL MARKETS (نگاشت درست تیکرها + زنجیرهٔ جایگزین)
# ======================================================================
CHAINS = {
    "XAUUSD": ["XAUUSD=X", "GC=F", "GLD"],
    "BTC":    ["BTC-USD", "BTC=F"],
    "DXY":    ["DX-Y.NYB", "DX=F"],
    "US10Y":  ["^TNX"],
    "OIL":    ["CL=F", "BZ=F"],
    "SPX":    ["^GSPC", "SPY"],
    "VIX":    ["^VIX"],
}
# ^TNX بازده×10 است → برای نمایش درصد واقعی ÷10 می‌کنیم
SCALE = {"US10Y": 0.10}
# چک سلامت قیمت: اگر قیمت کمتر از این باشد، داده قطعاً غلط است
SANITY_MIN = {"XAUUSD": 100, "BTC": 1000, "DXY": 50, "US10Y": 0.1,
              "OIL": 5, "SPX": 500, "VIX": 5}

def fetch_history(sym, period="2y"):
    chain = CHAINS.get(sym, [sym])
    last_err = None
    for t in chain:
        try:
            df = yf.download(t, period=period, interval="1d", auto_adjust=True,
                             progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if df.empty:
                last_err = f"{t}: empty"; continue
            last_px = float(df["Close"].iloc[-1])
            mn = SANITY_MIN.get(sym)
            if mn is not None and last_px < mn:
                last_err = (f"{t}: sanity fail — price {last_px} < min {mn} "
                            f"(probably wrong asset)"); continue
            if t == "GLD":
                sc = 2400.0 / last_px
                df = df * sc
                log.warning(f"[yf] {sym} via GLD scaled x{sc:.1f} (fallback)")
                last_px = float(df["Close"].iloc[-1])
            if sym in SCALE:
                df = df * SCALE[sym]
                last_px = float(df["Close"].iloc[-1])
            log.debug(f"[yf] {sym} <- {t}: {len(df)} rows, last={last_px:,.2f}")
            src("yfinance", True)
            return df, t
        except Exception as e:
            last_err = e
    src("yfinance", False, f"{sym}: {last_err}")
    raise RuntimeError(f"{sym}: all ticker sources failed — last error: {last_err}")

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
            df, used = fetch_history(key)
            frames[key] = df
            c = df["Close"].squeeze()
            snaps[key] = {"price": float(c.iloc[-1]), "sym": used,
                          "chg_1d": _chg(c, 1), "chg_7d": _chg(c, 5)}
            log.info(f"[data] {key} = {snaps[key]['price']:,.2f} (via {used}, "
                     f"1d {snaps[key]['chg_1d'] * 100:+.2f}%)")
        except Exception as e:
            snaps[key] = {"error": str(e)}
            log.warning(f"[data] {key} unavailable: {e}")
    return snaps, frames

def gold_spot_fallback():
    try:
        df = yf.download("XAUUSD=X", period="5d", interval="1d", auto_adjust=True,
                         progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception as e:
        log.debug(f"[yf] XAUUSD=X spot failed: {e}")
    return None

def sanity_checks(g, ir):
    prob = []
    btc = g.get("BTC", {}).get("price")
    if btc is not None and btc < SANITY_MIN["BTC"]:
        prob.append(f"قیمت بیت‌کوین مشکوک است: {btc} (احتمالاً تیکر اشتباه)")
    xau = g.get("XAUUSD", {}).get("price")
    if xau is not None and xau < SANITY_MIN["XAUUSD"]:
        prob.append(f"قیمت طلا مشکوک است: {xau}")
    usd = ir.get("usdirr_free")
    if usd is not None and usd < 10000:
        prob.append(f"دلار آزاد مشکوک است: {usd}")
    for p in prob:
        log.error("[sanity] " + p)
    if not prob:
        log.info("[sanity] all key prices look reasonable ✔")
    return prob

# ======================================================================
# 5. DATA — IRAN (TGJU + Nobitex با جایگزین Wallex)
# ======================================================================
TGJU_URL = "https://api.tgju.org/v1/market/indicator/summary-table-data/{code}"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

def tgju_one(code):
    def _g():
        r = requests.get(TGJU_URL.format(code=code), headers=UA, timeout=15)
        r.raise_for_status()
        digits = re.sub(r"[^\d.]", "", str(r.json()["data"][0][1]))
        return float(digits) if digits else None
    return retry(_g, tries=2, wait=1, name=f"tgju:{code}")

def nobitex():
    def _g():
        r = requests.get("https://api.nobitex.ir/market/stats",
                         params={"srcCurrency": "usdt,btc", "dstCurrency": "irt"},
                         headers=UA, timeout=15)
        r.raise_for_status()
        s = r.json()["stats"]
        return float(s["usdt-irt"]["latest"]) * 10, float(s["btc-irt"]["latest"]) * 10
    return retry(_g, tries=2, wait=1, name="nobitex")

def wallex():
    """جایگزین نوبیتکس — قیمت تتر و بیت‌کوین تومانی از والکس"""
    def _g():
        r = requests.get("https://api.wallex.ir/v1/markets", headers=UA, timeout=15)
        r.raise_for_status()
        mk = r.json().get("result", {}).get("markets", {})
        def last(key):
            for k, v in mk.items():
                if str(k).upper() == key:
                    return float(v["stats"]["lastPrice"]) * 10  # تومان → ریال
            return None
        usdt, btc = last("USDTTM"), last("BTCTMN")
        if usdt is None:
            raise RuntimeError("USDTTM not found in wallex response")
        return usdt, btc
    return retry(_g, tries=2, wait=1, name="wallex")

def iran_snapshot(xau_main):
    snap = {"usdirr_free": None, "geram18_rial": None, "geram24_rial": None,
            "emami_rial": None, "nim_rial": None, "rob_rial": None,
            "usdt_irt_rial": None, "btc_irt_rial": None,
            "geram18_implied_rial": None, "geram18_premium_pct": None,
            "sekee_bubble_pct": None, "usdt_premium_pct": None,
            "btc_ir_premium_pct": None, "spot_used": None, "crypto_src": None}
    # --- TGJU ---
    got = 0
    for code, key in (("price_dollar_rl", "usdirr_free"), ("geram18", "geram18_rial"),
                      ("geram24", "geram24_rial"), ("sekee", "emami_rial"),
                      ("nim", "nim_rial"), ("rob", "rob_rial")):
        try:
            v = tgju_one(code)
            snap[key] = v
            got += (v is not None)
            if v is not None: src("tgju", True)
        except Exception as e:
            src("tgju", False, f"{code}: {e}")
    log.info(f"[data] tgju ok={got}/6 dollar={snap['usdirr_free']} "
             f"g18={snap['geram18_rial']} sekee={snap['emami_rial']}")
    # --- Nobitex → Wallex ---
    try:
        usdt, btc = nobitex()
        snap["usdt_irt_rial"], snap["btc_irt_rial"] = usdt, btc
        snap["crypto_src"] = "nobitex"
        src("nobitex", True)
        log.info(f"[data] nobitex usdt={usdt:,.0f}R btc={btc:,.0f}R")
    except Exception as e1:
        src("nobitex", False, e1)
        try:
            usdt, btc = wallex()
            snap["usdt_irt_rial"], snap["btc_irt_rial"] = usdt, btc
            snap["crypto_src"] = "wallex"
            src("wallex", True)
            log.info(f"[data] wallex(fallback) usdt={usdt:,.0f}R btc={btc:,.0f}R")
        except Exception as e2:
            src("wallex", False, e2)
    # --- dollar fallback via USDT ---
    if not snap["usdirr_free"] and snap["usdt_irt_rial"]:
        snap["usdirr_free"] = round(snap["usdt_irt_rial"] * 0.985)
        log.warning(f"[data] dollar missing -> USDT proxy ({snap['usdirr_free']:,.0f}R)")
    # --- implied & premia ---
    spot, where = gold_spot_fallback(), "XAUUSD=X"
    if spot is None:
        spot, where = xau_main, "main-chain"
    snap["spot_used"] = where
    usd, g18 = snap["usdirr_free"], snap["geram18_rial"]
    if usd and g18 and spot:
        implied = spot * usd * 0.75 / 31.1035
        snap["geram18_implied_rial"] = implied
        snap["geram18_premium_pct"] = (g18 / implied - 1.0) * 100
    if snap["emami_rial"] and g18:
        snap["sekee_bubble_pct"] = (snap["emami_rial"] / (g18 * 9.76) - 1.0) * 100
    if snap["usdt_irt_rial"] and usd:
        snap["usdt_premium_pct"] = (snap["usdt_irt_rial"] / usd - 1.0) * 100
    log.info(f"[data] premium18={snap['geram18_premium_pct']} "
             f"bubble_sekee={snap['sekee_bubble_pct']} "
             f"usdt_prem={snap['usdt_premium_pct']} spot={where} crypto={snap['crypto_src']}")
    return snap

def set_btc_ir_premium(snap, btc_usd):
    if snap.get("btc_irt_rial") and btc_usd and snap.get("usdirr_free"):
        snap["btc_ir_premium_pct"] = (snap["btc_irt_rial"] /
                                      (btc_usd * snap["usdirr_free"]) - 1.0) * 100

# ======================================================================
# 6. DATA — MACRO (FRED) با پیام راهنما برای خطای کلید
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

def _fred_err_hint(e):
    s = str(e)
    if "400" in s:
        return s + " ← کلید FRED نامعتبر است (فاصله/کاراکتر اضافه یا کلید غلط)"
    if "403" in s:
        return s + " ← کلید FRED هنوز فعال نشده (ایمیل تایید را چک کن)"
    return s

def macro_snapshot():
    out = {}
    if not CFG["FRED_KEY"]:
        log.warning("[fred] no API key — macro skipped (سکرت FRED_API_KEY را بساز)")
        src("fred", False, "no key")
        return out
    try:
        d10 = fred_series("DGS10", 30); bie = fred_series("T10YIE", 30)
        out["us10y"] = d10[-1][1]
        out["us10y_chg_5d"] = d10[-1][1] - d10[-6][1]
        if len(bie) >= 6:
            out["real10y"] = d10[-1][1] - bie[-1][1]
            out["real10y_chg_5d"] = out["real10y"] - (d10[-6][1] - bie[-6][1])
        src("fred", True)
        log.info(f"[data] fred 10y={out['us10y']}% real={out.get('real10y')}")
    except Exception as e:
        src("fred", False, "yields: " + _fred_err_hint(e))
    try:
        cpi = fred_series("CPIAUCSL", 60)
        if len(cpi) >= 13: out["cpi_yoy"] = (cpi[-1][1] / cpi[-13][1] - 1) * 100
        src("fred", True)
    except Exception as e:
        src("fred", False, "cpi: " + _fred_err_hint(e))
    try:
        walcl = fred_series("WALCL", 12)
        out["fed_bs_chg_4w_pct"] = (walcl[-1][1] / walcl[-5][1] - 1) * 100
        src("fred", True)
    except Exception as e:
        src("fred", False, "walcl: " + _fred_err_hint(e))
    try:
        m2 = fred_series("M2SL", 30)
        if len(m2) >= 13: out["m2_yoy"] = (m2[-1][1] / m2[-13][1] - 1) * 100
        src("fred", True)
    except Exception as e:
        src("fred", False, "m2: " + _fred_err_hint(e))
    return out

# ======================================================================
# 7. DATA — NEWS
# ======================================================================
QUERIES = ["gold price", "bitcoin price", "federal reserve rate decision",
           "Powell speech Fed", "FOMC minutes meeting", "US inflation CPI report",
           "US jobs report payroll", "iran sanctions", "iran rial dollar exchange",
           "iran nuclear talks", "Israel Iran strike", "Strait of Hormuz oil",
           "central bank gold buying", "dollar index treasury yields",
           "bitcoin ETF regulation", "middle east conflict oil price"]

def _md5(t): return hashlib.md5(t.lower().encode()).hexdigest()[:12]

def google_news(q, limit=6):
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
           requests.utils.quote("(gold OR bitcoin OR sanctions OR iran OR FOMC) "
                                "market sourcelang:english") +
           "&mode=artlist&maxrecords=25&format=json&timespan=24h")
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
    except Exception as e:
        src("cryptopanic", False, e)
        return []

def collect_news():
    items, fails = [], 0
    for q in QUERIES:
        try:
            got = google_news(q)
            items += got
            if not got: fails += 1
        except Exception as e:
            fails += 1
            src("google_news", False, f"{q}: {e}")
    try:
        items += gdelt(); src("gdelt", True)
    except Exception as e:
        src("gdelt", False, e)   # 429 = محدودیت نرخ؛ بی‌خطر چون Google News پوشش می‌دهد
    items += cryptopanic()
    seen, uniq = set(), []
    for it in items:
        if it["id"] in seen: continue
        seen.add(it["id"]); uniq.append(it)
    log.info(f"[news] collected={len(uniq)} (empty-queries={fails})")
    if fails >= len(QUERIES) - 2:
        src("google_news", False, f"{fails}/{len(QUERIES)} queries empty")
    return uniq[:70]

WHO_RX = re.compile(r"(powell|fomc|federal reserve|fed chair|lagarde|ecb|boj|boe|"
                    r"central bank|treasury secretary|yellen)", re.I)
WHAT_RX = re.compile(r"(speech|testimon|press conference|hearing|remarks|decision|"
                     r"minutes|rate cut|rate hike|hold rates)", re.I)
HOTWORDS = ("sanction", "strike", "attack", "missile", "drone", "war", "nuclear",
            "ceasefire", "hormuz", "opec", "default", "devaluation", "intervention",
            "emergency", "rate decision", "cpi", "inflation", "payroll", "jobs report",
            "nfp", "nonfarm", "powell", "fomc", "rate cut", "rate hike",
            "bitcoin etf", "crypto ban", "hack", "securities and exchange")

def is_hot(title):
    t = title.lower()
    if any(k in t for k in HOTWORDS): return True
    if WHO_RX.search(t) and WHAT_RX.search(t): return True
    return False

# ======================================================================
# 8. INDICATORS
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

def bollinger_pb(c, n=20, k=2.0):
    ma = c.rolling(n).mean(); sd = c.rolling(n).std()
    up, dn = ma + k * sd, ma - k * sd
    rng = (up - dn).replace(0, 1e-12)
    return float(((c - dn) / rng).iloc[-1])

# ======================================================================
# 9. ANALYSIS — TECHNICAL
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
            "macd_hist": float(h.iloc[-1]), "macd_hist_prev": float(h.iloc[-2]),
            "bb_pb": bollinger_pb(c)}

def technical_signals(asset, t):
    out, stack = [], 0.0
    if t["close"] > t["ema50"] and t["ema20"] > t["ema50"]: stack += 0.5
    else: stack -= 0.5
    if t["ema200"] is not None:
        stack += 0.5 if t["close"] > t["ema200"] else -0.5
    if stack > 0:
        why = f"قیمت ({fmt(t['close'])}) بالای میانگین‌های قیمتی‌اش نشسته؛ روند کلی رو به بالا است."
    else:
        why = f"قیمت ({fmt(t['close'])}) زیر میانگین‌های قیمتی‌اش است؛ روند کلی رو به پایین است."
    out.append(sig("technical", asset, stack, 0.7, why))
    r = t["rsi"]
    rs = (r - 50) / 25 if 30 <= r <= 70 else (0.3 if r > 70 else -0.3)
    if r > 70:
        txt = "قیمت چند روز است خیلی تند بالا رفته؛ احتمال توقف یا اصلاح کوتاه وجود دارد."
    elif r < 30:
        txt = "قیمت چند روز است تند پایین آمده؛ احتمال برگشت کوتاه وجود دارد."
    else:
        txt = "فشار خرید و فروش نسبتاً متعادل است."
    out.append(sig("technical", asset, rs, 0.5, txt))
    mh, mp = t["macd_hist"], t["macd_hist_prev"]
    if mh > 0:
        txt = "نیروی خرید بیشتر از فروش است" + (" و هنوز قوی‌تر می‌شود." if mh > mp else " ولی دارد کم‌رنگ می‌شود.")
    else:
        txt = "نیروی فروش بیشتر از خرید است" + (" و دارد شدیدتر می‌شود." if mh < mp else " ولی دارد کم‌رنگ می‌شود.")
    out.append(sig("technical", asset, 1.0 if mh > 0 else -1.0, 0.6, txt))
    pb = t.get("bb_pb")
    if pb is not None and (pb > 0.95 or pb < 0.05):
        if pb > 0.95:
            out.append(sig("technical", asset, -0.35, 0.4,
                           "قیمت چسبیده به سقف کانال ۲۰ روزه؛ بیشترِ خریدهای احتمالی انجام شده."))
        else:
            out.append(sig("technical", asset, 0.35, 0.4,
                           "قیمت چسبیده به کف کانال ۲۰ روزه؛ فروشندگان بیش از حد فعال شده‌اند."))
    return out

# ======================================================================
# 10. ANALYSIS — NDS / STRUCTURE
# ======================================================================
SEQ_FA = {
    "higher highs + higher lows (uptrend structure)":
        "سقف‌ها و کف‌ها هر بار بالاتر از قبل ساخته می‌شوند ← روند صعودی",
    "lower highs + lower lows (downtrend structure)":
        "سقف‌ها و کف‌ها هر بار پایین‌تر از قبل ساخته می‌شوند ← روند نزولی",
    "mixed swings (range)": "سقف‌ها و کف‌ها نامنظم‌اند ← بازار در یک محدوده گیر کرده",
    "not enough swing points yet": "دادهٔ کافی برای تشخیص الگو نیست",
}

def swings(df, w=3):
    hi, lo, n = df["High"].squeeze(), df["Low"].squeeze(), len(df)
    nodes = []
    for i in range(w, max(w, n - w)):
        if hi.iloc[i] >= hi.iloc[i - w:i + w + 1].max(): nodes.append([i, float(hi.iloc[i]), "H"])
        if lo.iloc[i] <= lo.iloc[i - w:i + w + 1].min(): nodes.append([i, float(lo.iloc[i]), "L"])
    out = []
    for nd in nodes:
        if out and out[-1][2] == nd[2]:
            if (nd[2] == "H" and nd[1] >= out[-1][1]) or (nd[2] == "L" and nd[1] <= out[-1][1]):
                out[-1] = nd
        else:
            out.append(nd)
    return out

def structure_block(df, w=3):
    a = float(atr_series(df, 14).iloc[-1]) if len(df) > 15 else float((df["High"] - df["Low"]).mean())
    nodes = swings(df, w)
    highs = [p for _, p, t in nodes if t == "H"][-2:]
    lows  = [p for _, p, t in nodes if t == "L"][-2:]
    close = float(df["Close"].iloc[-1])
    struct, seq = 0.0, "not enough swing points yet"
    if len(highs) == 2 and len(lows) == 2:
        hh, hl = highs[1] > highs[0], lows[1] > lows[0]
        if hh and hl:           struct, seq = 1.0, "higher highs + higher lows (uptrend structure)"
        elif not hh and not hl: struct, seq = -1.0, "lower highs + lower lows (downtrend structure)"
        else:                   struct, seq = 0.0, "mixed swings (range)"
    disp, dsign = 0.0, 0
    if nodes:
        move = close - nodes[-1][1]
        disp = abs(move) / a if a > 0 else 0.0
        dsign = 1 if move > 0 else -1
    score = max(-1.0, min(1.0, 0.6 * struct + 0.4 * dsign * min(1.0, disp / 2)))
    return {"score": score, "seq": seq, "atr": a, "close": close, "disp": disp}

def nds_signal(asset, b):
    strength = "قدرتمند" if b["disp"] > 1.5 else "متوسط" if b["disp"] > 0.8 else "ضعیف"
    disp_txt = fa(f"{b['disp']:.1f}")
    why = (f"الگوی سقف و کف: {SEQ_FA.get(b['seq'], b['seq'])}. "
           f"آخرین حرکت {disp_txt} برابرِ میانگین روزهای قبل بوده ({strength}).")
    return sig("nds", asset, b["score"], 0.75, why)

def nds_weekly_signal(asset, b):
    strength = "قدرتمند" if b["disp"] > 1.5 else "متوسط" if b["disp"] > 0.8 else "ضعیف"
    disp_txt = fa(f"{b['disp']:.1f}")
    why = (f"در بازهٔ هفتگی هم الگو همین است: {SEQ_FA.get(b['seq'], b['seq'])} "
           f"(حرکت {disp_txt} برابر میانگین — {strength}).")
    return sig("nds_w", asset, b["score"], 0.65, why, horizon="weekly")

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

def resample_weekly(df):
    o = df["Open"].resample("W-FRI").first()
    h = df["High"].resample("W-FRI").max()
    l = df["Low"].resample("W-FRI").min()
    cl = df["Close"].resample("W-FRI").last()
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": cl}).dropna()

# ======================================================================
# 11. CORRELATION / MACRO / SENTIMENT
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
            why = ("حرکت اخیر دلار، نرخ بهره و بقیهٔ بازارها در جهت‌هایی است که معمولاً "
                   "به نفع این بازار تمام می‌شود." if s > 0 else
                   "حرکت اخیر دلار، نرخ بهره و بقیهٔ بازارها در جهت‌هایی است که معمولاً "
                   "علیه این بازار تمام می‌شود.")
            out.append(sig("correlation", asset, s, min(0.8, 0.4 + 0.1 * len(vals)), why))
    add("XAUUSD", [("DXY", -1), ("US10Y", -1), ("OIL", 0.5)])
    add("BTC",    [("SPX", 1), ("DXY", -1), ("XAUUSD", 0.3), ("VIX", -0.5)])
    return out

def macro_signals(mac, g, frames):
    out = []
    if not mac: return out
    bits, parts = [], []
    if "real10y_chg_5d" in mac:
        parts.append(-mac["real10y_chg_5d"] * 15)
        bits.append("بهرهٔ واقعی آمریکا " + ("پایین آمده" if mac["real10y_chg_5d"] < 0 else "بالا رفته"))
    if g.get("DXY", {}).get("chg_7d") is not None:
        parts.append(-g["DXY"]["chg_7d"] * 20)
        bits.append("دلار " + ("ضعیف شده" if g["DXY"]["chg_7d"] < 0 else "قوی شده"))
    try:
        dxy_c = frames["DXY"]["Close"].squeeze()
        below = float(dxy_c.iloc[-1]) < float(ema(dxy_c, 20).iloc[-1])
        parts.append(8 if below else -8)
        bits.append("دلار " + ("زیر میانگین ۲۰ روزه" if below else "بالای میانگین ۲۰ روزه"))
    except Exception:
        pass
    if "fed_bs_chg_4w_pct" in mac:
        parts.append(mac["fed_bs_chg_4w_pct"] * 10)
        bits.append("نقدینگی جهانی " + ("زیاد شده" if mac["fed_bs_chg_4w_pct"] > 0 else "کم شده"))
    if parts:
        s = max(-1.0, min(1.0, sum(parts) / len(parts)))
        out.append(sig("macro", "XAUUSD", s, 0.75,
                       "شرایط کلی: " + "، ".join(bits) +
                       f" — این ترکیب معمولاً قیمت طلا را {'بالا' if s > 0 else 'پایین'} می‌برد.",
                       horizon="weekly"))
    bbits, bparts = [], []
    if "fed_bs_chg_4w_pct" in mac or "m2_yoy" in mac:
        bparts.append((mac.get("fed_bs_chg_4w_pct", 0) * 12 + mac.get("m2_yoy", 0)) * 0.05)
        bbits.append("پولِ ارزان در جهان " +
                     ("زیاد شده" if mac.get("fed_bs_chg_4w_pct", 0) > 0 else "کم شده"))
    if g.get("SPX", {}).get("chg_7d") is not None and g.get("VIX", {}).get("chg_7d") is not None:
        bparts.append(g["SPX"]["chg_7d"] * 15 - g["VIX"]["chg_7d"] * 0.3)
        bbits.append("بازار سهام آمریکا " + ("سرحال است" if g["SPX"]["chg_7d"] > 0 else "ترسیده"))
    if bparts:
        s = max(-1.0, min(1.0, sum(bparts) / len(bparts)))
        out.append(sig("macro", "BTC", s, 0.7,
                       "شرایط کلی: " + "، ".join(bbits) +
                       f" — این ترکیب معمولاً {'به نفع' if s > 0 else 'علیه'} بیت‌کوین است.",
                       horizon="weekly"))
    return out

_VADER = SentimentIntensityAnalyzer()

def headline_sentiment(items):
    if not items: return 0.0
    comp = [_VADER.polarity_scores(i["title"])["compound"] for i in items]
    return max(-1.0, min(1.0, sum(comp) / len(comp) * 2))

def fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
        v = int(r["data"][0]["value"])
        src("fear_greed", True)
        return v
    except Exception as e:
        src("fear_greed", False, e)
        return None

def sentiment_signals(items, fng=None):
    out = []
    vs = headline_sentiment(items)
    if items:
        tone = "مثبت" if vs > 0.05 else "منفی" if vs < -0.05 else "بی‌طرف"
        out.append(sig("sentiment", "XAUUSD", vs * 0.5, 0.4,
                       f"لحن اخبار اخیر ({fa(len(items))} خبر) در مجموع {tone} است."))
    if fng is not None:
        mood = "طمع — جمعیت خریدار است" if fng > 60 else \
               "ترس — جمعیت فروشنده است" if fng < 40 else "بی‌طرف"
        out.append(sig("sentiment", "BTC", (fng - 50) / 50, 0.5,
                       f"شاخص ترس و طمع = {fa(fng)} از ۱۰۰ ({mood})."))
    return out

# ======================================================================
# 12. IRAN MODELS
# ======================================================================
def usdirr_momentum(rows):
    vals = [float(r["usdirr_free"]) for r in rows
            if r.get("usdirr_free") not in ("", None)]
    if len(vals) < 3: return 0.0, 0
    c = vals
    def chg(n):
        return (c[-1] / c[-1 - n] - 1) if len(c) > n else None
    num = den = 0.0
    for n, wt in ((1, .2), (3, .3), (5, .3), (10, .2)):
        x = chg(n)
        if x is not None:
            num += x * wt; den += wt
    return (num / den if den else 0.0), len(vals)

def coin_momentum(rows):
    c = [float(r["emami_rial"]) for r in rows
         if r.get("emami_rial") not in ("", None)]
    if len(c) < 6: return None
    return c[-1] / c[-6] - 1

def iran_levels(rows):
    seq = [float(r["geram18_rial"]) for r in rows[-60:]
           if r.get("geram18_rial") not in ("", None)]
    if len(seq) < 15: return [], []
    srt, close = sorted(seq), seq[-1]
    q = lambda p: srt[int(p * (len(srt) - 1))]
    sup = [v for v in dict.fromkeys((q(.25), q(.40))) if v < close * 0.999]
    res = [v for v in dict.fromkeys((q(.75), q(.90))) if v > close * 1.001]
    return sup, res

def iran_gold_signals(snap, rows, ai_usdirr, xau_nds, xau_chg7):
    out = []
    usd = snap.get("usdirr_free")
    mom, nhist = usdirr_momentum(rows)
    mom_pct = mom * 100
    if usd:
        s = max(-1.0, min(1.0, mom * 40))
        if s > 0.1:
            why = (f"دلار آزاد {fmt(usd)} ریال است و در روزهای اخیر روند صعودی داشته "
                   f"(میانگین وزنی {mom_pct:+.2f}٪) — مهم‌ترین عامل گران شدن طلای داخل.")
        elif s < -0.1:
            why = (f"دلار آزاد {fmt(usd)} ریال است و روند نزولی داشته "
                   f"({mom_pct:+.2f}٪) — فشار پایین روی طلای داخل.")
        else:
            why = (f"دلار آزاد {fmt(usd)} ریال است و تقریباً ثابت مانده — موتور اصلی "
                   f"حرکت طلای داخل فعلاً خاموش است.")
        out.append(sig("iran_local", "GOLD_IR", s, 0.8, why, w=1.6))
    uprem = snap.get("usdt_premium_pct")
    if uprem is not None:
        if uprem > -0.5:
            out.append(sig("iran_local", "GOLD_IR", 0.3, 0.4,
                           f"تتر با پریمیوم {uprem:+.1f}٪ معامله می‌شود — تقاضای دلار در داخل بالاست."))
        elif uprem < -2.5:
            out.append(sig("iran_local", "GOLD_IR", -0.3, 0.4,
                           f"تتر با دیسکانت {uprem:+.1f}٪ معامله می‌شود — تقاضای دلار ضعیف است."))
    cmom = coin_momentum(rows)
    if cmom is not None:
        cpct = cmom * 100
        out.append(sig("iran_local", "GOLD_IR", max(-1.0, min(1.0, cmom * 25)), 0.6,
                       f"سکهٔ امامی در ~۵ روز {cpct:+.1f}٪ تغییر کرده — "
                       f"{'حرارت بازار داخلی' if cmom > 0 else 'سردی بازار داخلی'} را نشان می‌دهد."))
    bub = snap.get("sekee_bubble_pct")
    if bub is not None:
        if bub > 15:
            out.append(sig("iran_local", "GOLD_IR", -0.5, 0.5,
                           f"حباب سکه {bub:+.1f}٪ است — سکه بیش از ارزش طلای داخلش گران است "
                           f"و احتمال سرد شدن دارد."))
        elif bub < 6:
            out.append(sig("iran_local", "GOLD_IR", 0.3, 0.5,
                           f"حباب سکه فقط {bub:+.1f}٪ است — سکه به قیمت طلای خودش نزدیک است."))
    xs = (xau_nds or 0) * 0.5 + max(-1.0, min(1.0, (xau_chg7 or 0) * 10)) * 0.5
    if xs:
        out.append(sig("iran_local", "GOLD_IR", xs * 0.6, 0.6,
                       "طلای جهانی روند " + ("صعودی" if xs > 0 else "نزولی") +
                       " دارد و بخشی از آن به قیمت داخل منتقل می‌شود.", w=1.2))
    prem = snap.get("geram18_premium_pct")
    if prem is not None and nhist >= 20:
        hist = [float(r["geram18_premium_pct"]) for r in rows
                if r.get("geram18_premium_pct") not in ("", None)]
        if len(hist) >= 20 and pstdev(hist) > 0.01:
            z = (prem - mean(hist)) / pstdev(hist)
            if z > 1:
                txt = (f"طلای داخل {prem:+.1f}٪ گران‌تر از ارزش واقعی‌اش است؛ بیش از حد داغ "
                       f"شده و احتمال عقب‌نشینی کوتاه دارد.")
            elif z < -1:
                txt = (f"طلای داخل {prem:+.1f}٪ ارزان‌تر از ارزش واقعی‌اش است؛ عقب مانده و "
                       f"اگر اوضاع عادی شود جا برای بالا آمدن دارد.")
            else:
                txt = "قیمت داخل تقریباً همان ارزش واقعی‌اش است؛ حاشیهٔ خاصی وجود ندارد."
            out.append(sig("iran_local", "GOLD_IR", max(-1.0, min(1.0, -z * 0.6)), 0.5, txt))
    elif prem is not None:
        side = "ارزان‌تر" if prem < 0 else "گران‌تر"
        out.append(sig("iran_local", "GOLD_IR", 0, 0.4,
                       f"طلای داخل حدود {abs(prem):.1f}٪ {side} از ارزش واقعی‌اش است "
                       f"(ارزش واقعی از اسپات جهانی محاسبه شد، نه فیوچرز)."))
    if abs(ai_usdirr) > 0.15:
        out.append(sig("ai_events", "GOLD_IR", ai_usdirr, 0.7,
                       "بررسی اخبار مهم (تحریم، تنش، سیاست داخلی) فشار فضای خبری را " +
                       ("به سمت گران شدن" if ai_usdirr > 0 else "به سمت ارزان شدن") +
                       " بازار داخل نشان می‌دهد.", w=1.8))
    log.debug(f"[iran] signals={len(out)} mom={mom_pct:+.2f}% bubble={bub}")
    return out

def btc_ir_signal(prem):
    if prem is None: return None
    if prem > 2:
        txt = (f"بیت‌کوین در صرافی‌های داخل {prem:+.1f}٪ گران‌تر از قیمت جهانی (با دلار آزاد) "
               f"معامله می‌شود؛ تقاضای داخل قوی است.")
    elif prem < -2:
        txt = (f"بیت‌کوین در صرافی‌های داخل {abs(prem):.1f}٪ ارزان‌تر از قیمت جهانی است؛ "
               f"تقاضای داخل ضعیف است.")
    else:
        txt = "تفاوت قیمت بیت‌کوین داخل با جهانی در حالت عادی است."
    return sig("iran_local", "BTC", max(-1.0, min(1.0, prem / 8)), 0.5, txt)

# ======================================================================
# 13. GEMINI
# ======================================================================
GEM_SYS = """You are a financial news analyst for GOLD and BITCOIN markets, with special
expertise in IRAN (USD/IRR free-market rate, Iranian gold & coin market, Iranian crypto market).

Classify each headline. Return ONLY JSON:
{"items":[{"id":"...","category":"monetary|inflation|geopolitics|iran_domestic|sanctions|military|crypto|market|other",
 "impact":"high|medium|low|noise|structural",
 "gold":-1..1,"bitcoin":-1..1,"usdirr":-1..1,
 "timing":"next 24-72h|this week|next week|ongoing|unknown"}],
 "digest":"3-sentence summary — WRITE IT IN SIMPLE PERSIAN (FARSI)",
 "narrative_changed": true|false,
 "high_impact_events":[{"name":"... — IN SIMPLE PERSIAN (include speeches, FOMC, CPI dates if mentioned)","asset":"gold|bitcoin|usdirr","timing":"..."}],
 "invalidations":{"gold":["condition — IN SIMPLE PERSIAN"],
                  "bitcoin":["... — PERSIAN"],"usdirr":["... — PERSIAN"]}}

Rules:
- impact=high ONLY if it can move gold/BTC/USD-IRR by more than 1-2% within 24-72h.
- impact=structural for long-term regime shifts.
- impact=noise for routine chatter and opinion pieces.
- Central-bank speeches/testimony/FOMC/CPI scheduling are important — do not mark them noise.
- usdirr: +1 means news pressures USD/IRR UP (rial weakens -> Iranian gold/coin tend UP).
- Be conservative. Scores reflect the direction of the EXPECTED price move.
- Persian text must be simple enough for a non-expert (no jargon)."""

SIMPLE_SYS = """You write in VERY simple Persian (Farsi) for people who know nothing about trading.
Input: JSON of market predictions already computed. Do NOT change or invent any number.
For each asset write ONE short sentence (max 22 words) in Persian: what is likely to happen
in the next few days, and the single main reason. Everyday words only — never say
'resistance', 'support', 'RSI', 'MACD', 'structure'.
Return ONLY JSON with keys: GOLD_IR, XAUUSD, BTC."""

SAFETY = [{"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_ONLY_HIGH"}
          for c in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")]

def gemini_chat(system, user, max_tokens=4096, json_mode=True):
    if not CFG["GEMINI_KEY"] or not CFG["GEMINI_MODELS"]:
        miss = []
        if not CFG["GEMINI_KEY"]:
            miss.append("GEMINI_API_KEY خالی است → Settings > Secrets and variables > Actions "
                        "→ New repository secret با نام GEMINI_API_KEY و کلید AIza... از aistudio.google.com/apikey")
        if not CFG["GEMINI_MODELS"]:
            miss.append("GEMINI_MODELS خالی است")
        DIAG["llm"] = {"ok": False, "err": "; ".join(miss)}
        log.warning("[gemini] SKIP — " + " | ".join(miss))
        return ""
    models = CFG["GEMINI_MODELS"]
    start = jload(P_GEM, {}).get("idx", 0) % len(models)
    last_err = "no models"
    for i in range(len(models)):
        m = models[(start + i) % len(models)]
        t0 = time.time()
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
                    last_err = f"{m}: blocked/empty"
                    log.warning(f"[gemini] {last_err} -> rotating")
                    continue
                txt = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
                jsave(P_GEM, {"idx": (start + i) % len(models), "model": m})
                DIAG["llm"] = {"ok": True, "model": m, "ms": int((time.time() - t0) * 1000),
                               "out_len": len(txt)}
                log.info(f"[gemini] OK model={m} out={len(txt)} chars "
                         f"({int((time.time() - t0) * 1000)}ms)")
                return txt
            last_err = f"{m}: HTTP {r.status_code}"
            log.warning(f"[gemini] {last_err} -> rotating")
        except Exception as e:
            last_err = f"{m}: {e}"
            log.warning(f"[gemini] {last_err} -> rotating")
    DIAG["llm"] = {"ok": False, "err": last_err}
    log.error(f"[gemini] ALL models failed: {last_err}")
    return ""

def parse_json(text):
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r"\{.*\}", text or "", re.S)
    try: return json.loads(m.group(0)) if m else {}
    except Exception: return {}

def fallback_intel():
    return {"scores": {"gold": 0, "bitcoin": 0, "usdirr": 0},
            "digest": "(تحلیل هوش مصنوعی در این اجرا در دسترس نبود.)",
            "changed": False, "high": [], "invalidations": {}, "ok": False}

def analyze_news(items, prev_digest=""):
    lines = "\n".join(f"[{i['id']}] {i['title']}" for i in items[:45])
    log.debug(f"[gemini] prompt headlines={min(len(items), 45)} chars={len(lines)}")
    user = (f"PREVIOUS NARRATIVE (compare against it):\n{prev_digest or 'none yet'}\n\n"
            f"HEADLINES (last ~40h):\n{lines}\n\nReturn the JSON now.")
    data = parse_json(gemini_chat(GEM_SYS, user))
    if not data or "items" not in data:
        log.warning("[gemini] news intel invalid JSON/empty")
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
    scores = {k: (max(-1.0, min(1.0, agg[k] / n[k])) if n[k] else 0.0) for k in agg}
    log.info(f"[gemini] news scores gold={scores['gold']:+.2f} "
             f"btc={scores['bitcoin']:+.2f} usdirr={scores['usdirr']:+.2f} "
             f"high_events={len(data.get('high_impact_events') or [])}")
    return {"scores": scores, "digest": data.get("digest", ""),
            "changed": bool(data.get("narrative_changed")),
            "high": data.get("high_impact_events", []) or [],
            "invalidations": data.get("invalidations", {}) or {}, "ok": True}

def get_intel(items, st):
    if not items:
        return fallback_intel(), True
    ids_hash = hashlib.md5("|".join(sorted(i["id"] for i in items)).encode()).hexdigest()
    cache = st.get("intel", {})
    age_h = 1e9
    if cache.get("fetched"):
        try: age_h = (now() - datetime.fromisoformat(cache["fetched"])).total_seconds() / 3600
        except Exception: pass
    if cache.get("ids_hash") == ids_hash and cache.get("ok") and age_h < CFG["INTEL_MAX_AGE_H"]:
        log.info(f"[intel] reused cache (age {age_h:.1f}h) — saved 1 LLM call")
        return cache, True
    intel = analyze_news(items, st.get("digest", "")) if CFG["GEMINI_KEY"] else fallback_intel()
    intel.update({"ids_hash": ids_hash, "fetched": nowiso()})
    return intel, False

def gemini_plain(preds):
    if not CFG["GEMINI_KEY"] or not preds:
        return {}
    payload = {}
    for a, p in preds.items():
        payload[a] = {"direction": DIR_FA[p["direction"]],
                      "confidence_percent": p["confidence"],
                      "current_price": p["entry_price"],
                      "top_reasons": p["reasons"][:2]}
    try:
        txt = gemini_chat(SIMPLE_SYS, json.dumps(payload, ensure_ascii=False),
                          json_mode=True, max_tokens=800)
        d = parse_json(txt)
        return {k: v.strip() for k, v in d.items() if isinstance(v, str) and v.strip()}
    except Exception as e:
        log.warning(f"[gemini] plain-summary failed: {e}")
        return {}

# ======================================================================
# 14. DECISION ENGINE
# ======================================================================
def adapted_weights():
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

def quality_caps(asset, intel, snap, n_signals):
    cap, notes = 92, []
    if asset == "GOLD_IR" and not snap.get("usdirr_free"):
        cap = min(cap, 45)
        notes.append("دسترسی به قیمت لحظه‌ای دلار/طلای داخل نبود؛ اطمینان محدود شد.")
    if not intel.get("ok"):
        cap = min(cap, 65)
        notes.append("لایهٔ هوش مصنوعی (اخبار) در این اجرا فعال نبود؛ تحلیل ناقص است.")
    if n_signals < 4:
        cap = min(cap, 60)
    return cap, notes

def make_prediction(asset, signals, levels, intel, entry, snap, zone_w=0.004):
    net, agree = fuse(signals)
    direction = "bullish" if net > 0.12 else "bearish" if net < -0.12 else "neutral"
    conf = CFG["MIN_CONF"] + (CFG["MAX_CONF"] - CFG["MIN_CONF"]) * abs(net) * (0.45 + 0.55 * agree)
    if direction == "neutral": conf = min(conf, 55)
    cap, notes = quality_caps(asset, intel, snap, len(signals))
    conf = int(min(max(CFG["MIN_CONF"], conf), cap))
    rs_ = levels.get("atr_pctile", 50) + len(intel.get("high", [])) * 20
    risk = "high" if rs_ > 100 else "medium" if rs_ > 60 else "low"
    W = adapted_weights()
    ranked = sorted(signals, key=lambda s: -(abs(s["score"]) * W.get(s["module"], 1.0) * s["w"]))[:5]
    sup, res = levels.get("supports", []), levels.get("resistances", [])
    zone = lambda v: f"{fmt(v * (1 - zone_w))} – {fmt(v * (1 + zone_w))}"
    buy_zones  = [zone(v) for v in (sup[-2:] if direction != "bearish" else sup[-1:])]
    sell_zones = [zone(v) for v in (res[:2] if direction != "bullish" else res[:1])]
    inv, inv_levels = [], []
    if direction == "bullish" and sup:
        inv.append(f"اگر قیمت یک روزِ کامل پایین‌تر از {fmt(sup[-1])} بسته شود، این تحلیل غلط است.")
        inv_levels.append(sup[-1])
    elif direction == "bearish" and res:
        inv.append(f"اگر قیمت یک روزِ کامل بالاتر از {fmt(res[0])} بسته شود، این تحلیل غلط است.")
        inv_levels.append(res[0])
    keymap = {"XAUUSD": "gold", "BTC": "bitcoin", "GOLD_IR": "usdirr"}
    inv += [str(x) for x in intel.get("invalidations", {}).get(keymap.get(asset, "gold"), [])][:2]
    reasons = (["⚠️ " + n for n in notes] + [s["why"] for s in ranked if s["why"]])
    log.info(f"[pred] {asset}: {direction.upper()} conf={conf}% (cap={cap}) "
             f"net={net:+.3f} agree={agree:.2f} risk={risk} signals={len(signals)}")
    return {"asset": asset, "direction": direction, "net": round(net, 3),
            "confidence": conf, "risk": risk, "horizon": "2-7d",
            "reasons": reasons,
            "buy_zones": buy_zones, "sell_zones": sell_zones,
            "entry_price": entry or 0, "created": nowiso(),
            "valid_until": (now() + timedelta(days=5)).isoformat(timespec="seconds"),
            "invalidations": inv, "invalidation_levels": inv_levels}

# ======================================================================
# 15. GRADING
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
        log.info(f"[grade] {p['asset']} {d} -> {'HIT' if hit else 'MISS'} ({move * 100:+.2f}%)")
    g["predictions"] = still_open[-60:]
    jsave(P_GRADE, g)
    stats = g.get("module_stats", {})
    if stats:
        txt = ", ".join(f"{k} {v['hits']}/{v['total']}" for k, v in stats.items())
        log.info(f"[grade] module accuracy so far: {txt}")

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
# 16. TELEGRAM
# ======================================================================
def _chunks(t, n):
    while t:
        cut = t.rfind("\n", 0, n)
        cut = cut if cut > 500 else n
        yield t[:cut]; t = t[cut:].lstrip("\n")

def tg_send(text):
    if not CFG["NOTIFY"] or not CFG["TELEGRAM_TOKEN"] or not CFG["TELEGRAM_CHAT"]:
        log.warning("[telegram] muted/missing credentials — suppressed")
        return False
    okc = 0
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
                if r.status_code == 200:
                    okc += 1
                else:
                    log.warning(f"[telegram] HTTP {r.status_code}: {r.text[:200]}")
                break
            except Exception as e:
                log.warning(f"[telegram] send error: {e}")
                time.sleep(2)
    src("telegram", okc > 0, f"sent={okc}")
    return okc > 0

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

def prediction_block(asset, p, plain=""):
    title = TITLES.get(asset, asset)
    lines = [f"<b>{title}</b>",
             DIR_FA[p["direction"]],
             f"احتمال درست بودن: <b>{fa(p['confidence'])}٪</b> (احتمال است، نه قطعیت)",
             f"ریسک: {RISK_FA[p['risk']]}  •  بازه: ۲ تا ۷ روز آینده"]
    if p["entry_price"]:
        lines.append(f"قیمت فعلی: {fmt(p['entry_price'])} {UNITS.get(asset, '')}".strip())
    if plain:
        lines += ["", f"🤖 <b>خلاصهٔ ساده:</b> {esc(plain)}"]
    why = "\n".join("• " + esc(r) for r in p["reasons"][:6]) or "• —"
    lines += ["", "<b>🔍 چرا؟ (به زبان ساده)</b>", why]
    if p["buy_zones"]:
        lines.append(f"🟩 <b>ناحیهٔ مناسب خرید:</b> {' | '.join(p['buy_zones'])}")
    if p["sell_zones"]:
        lines.append(f"🟥 <b>ناحیهٔ احتیاط/فروش:</b> {' | '.join(p['sell_zones'])}")
    if p["invalidations"]:
        lines.append("<b>❌ این تحلیل کِی غلط می‌شود؟</b>")
        lines += ["• " + esc(i) for i in p["invalidations"][:3]]
    else:
        lines.append("❌ فعلاً شرط باطل‌کننده ندارم (چون جهت مشخصی نگفتم).")
    return "\n".join(lines)

LEGEND = ("\n\n📖 <b>راهنمای ساده:</b>\n"
          "• «ناحیهٔ خرید» یعنی قیمتی که بازار قبلاً آن‌جا واکنش مثبت نشان داده؛ "
          "«ناحیهٔ احتیاط/فروش» یعنی جایی که معمولاً قیمت برمی‌گردد.\n"
          "• احتمال ۷۰٪ یعنی از هر ۱۰ بار مشابه، حدود ۷ بار بازار همان‌طور می‌رود — تضمین نیست.\n"
          "• ⚠️ این گزارش توصیهٔ مالی نیست.")

def iran_prices_block(snap, g, prev_row):
    def chg(key):
        try:
            p, c = (prev_row or {}).get(key), snap.get(key)
            if p and c: return (c / float(p) - 1) * 100
        except Exception:
            pass
        return None
    L = ["", "<b>💱 قیمت‌های لحظه‌ای بازار ایران</b>"]
    items = [("دلار آزاد", "usdirr_free", "ریال"),
             ("طلای ۱۸ عیار", "geram18_rial", "ریال"),
             ("طلای ۲۴ عیار", "geram24_rial", "ریال"),
             ("سکهٔ امامی", "emami_rial", "ریال"),
             ("نیم‌سکه", "nim_rial", "ریال"),
             ("ربع‌سکه", "rob_rial", "ریال")]
    any_price = False
    for name, key, unit in items:
        v = snap.get(key)
        if not v: continue
        any_price = True
        s = f"• {name}: <b>{fmt(v)}</b> {unit}"
        c = chg(key)
        if c is not None: s += f" ({c:+.1f}٪)"
        L.append(s)
    if snap.get("usdt_irt_rial"):
        s = f"• تتر: <b>{fmt(snap['usdt_irt_rial'] / 10)}</b> تومان"
        if snap.get("usdt_premium_pct") is not None:
            s += f" (پریمیوم {snap['usdt_premium_pct']:+.1f}٪ نسبت به دلار)"
        L.append(s); any_price = True
    if snap.get("btc_irt_rial"):
        s = f"• بیت‌کوین: <b>{fmt(snap['btc_irt_rial'] / 10)}</b> تومان"
        if snap.get("btc_ir_premium_pct") is not None:
            s += f" (اختلاف با جهانی: {snap['btc_ir_premium_pct']:+.1f}٪)"
        L.append(s); any_price = True
    if g.get("XAUUSD", {}).get("price"):
        L.append(f"• اونس جهانی: <b>{fmt(g['XAUUSD']['price'])}</b> دلار")
        any_price = True
    bub = snap.get("sekee_bubble_pct")
    if bub is not None:
        L.append(f"• حباب سکهٔ امامی: {bub:+.1f}٪")
    return "\n".join(L) if any_price else ""

def macro_lines(mac, g):
    L = []
    if mac.get("us10y"):
        L.append(f"• بازده اوراق ۱۰ساله آمریکا: {mac['us10y']:.1f}٪ "
                 f"(۵روزه {mac.get('us10y_chg_5d', 0):+.2f})")
    if "cpi_yoy" in mac:
        L.append(f"• تورم سالانه آمریکا: {mac['cpi_yoy']:.1f}٪")
    if "fed_bs_chg_4w_pct" in mac:
        L.append(f"• ترازنامه فدرال‌رزرو (۴هفته): {mac['fed_bs_chg_4w_pct']:+.2f}٪")
    if "m2_yoy" in mac:
        L.append(f"• رشد پول در گردش آمریکا: {mac['m2_yoy']:+.1f}٪ سالانه")
    if g.get("DXY", {}).get("chg_7d") is not None:
        L.append(f"• شاخص دلار (۵روزه): {g['DXY']['chg_7d'] * 100:+.1f}٪")
    if g.get("OIL", {}).get("chg_7d") is not None:
        L.append(f"• نفت (۵روزه): {g['OIL']['chg_7d'] * 100:+.1f}٪")
    return "\n".join(L)

def send_daily_report(preds, intel, weekly, plain, prices_block, macro_block):
    key = fp("daily", nowiso()[:10])
    if not should_send(key, 20): return
    ov = "  |  ".join(f"{SHORTN[a]}: {DIR_S[p['direction']]} ({fa(p['confidence'])}٪)"
                      for a, p in preds.items() if a in SHORTN)
    body = ""
    for a in ["GOLD_IR", "XAUUSD", "BTC"]:
        if a in preds:
            body += "\n\n━━━━━━━━━━━━━\n" + prediction_block(a, preds[a], plain.get(a, ""))
    ev = "\n".join(f"• {esc(e.get('name', '?'))} "
                   f"({TIMING_FA.get(e.get('timing', ''), e.get('timing', ''))})"
                   for e in intel.get("high", [])[:6]) or "• فعلاً اتفاق مهمی علامت‌گذاری نشده."
    wk = "\n".join(f"• {esc(k)}: {esc(v)}" for k, v in weekly.items()) or "• —"
    text = (f"🤖 <b>گزارش روزانه بازار — {nowiso()[:10]}</b>\n"
            f"⚡ خلاصه: {esc(ov)}\n" + body + prices_block +
            f"\n\n<b>📊 داده‌های کلان</b>\n{macro_block or '• —'}"
            f"\n\n<b>📅 هفتهٔ پیشِ رو</b>\n{wk}"
            f"\n\n<b>⏰ اتفاق‌های مهم پیشِ رو</b>\n{ev}"
            f"\n\n<b>📖 خلاصهٔ خبرها</b>\n{esc(intel.get('digest', ''))}"
            f"{LEGEND}")
    tg_send(text)
    mark_sent(key)

def send_prediction_change(p, reason):
    key = fp("pred", p["asset"], p["direction"], p["confidence"] // 10, round(p["net"], 1))
    if not should_send(key, 4): return
    tg_send(f"🔁 <b>نظر ربات عوض شد — {TITLES.get(p['asset'], p['asset'])}</b>\n"
            f"({esc(reason)})\n\n" + prediction_block(p["asset"], p))
    mark_sent(key)

def send_alert(text, kind="alert", cooldown_h=1):
    key = fp(kind, text[:120])
    if not should_send(key, cooldown_h): return
    tg_send("🚨 <b>هشدار</b>\n" + text)
    mark_sent(key)

# ======================================================================
# 17. STATE / HISTORY
# ======================================================================
HIST_COLS = ["date", "xauusd", "btc_usd", "dxy", "us10y", "oil", "usdirr_free",
             "usdt_irt", "geram18_rial", "geram24_rial", "emami_rial", "nim_rial",
             "rob_rial", "btc_irt", "geram18_premium_pct", "usdt_premium_pct",
             "sekee_bubble_pct"]

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
    log.info(f"[state] history now {len(rows)} rows")

# ======================================================================
# 18. PIPELINE
# ======================================================================
def full():
    with step("FULL ANALYSIS"):
        st = jload(P_STATE, {})
        prev_snap = st.get("snap", {})
        with step("داده‌های بازار جهانی"):
            g, frames = global_snapshots()
        if "price" not in g.get("XAUUSD", {}):
            log.error("!! No XAUUSD data — aborting run")
            return {}
        xau = g["XAUUSD"]["price"]
        btc_usd = g["BTC"].get("price")

        with step("داده‌های بازار ایران"):
            ir = iran_snapshot(xau)
            set_btc_ir_premium(ir, btc_usd)
            rows = _history_rows()
            prev_row = None
            today = nowiso()[:10]
            for r in reversed(rows):
                if r["date"] < today: prev_row = r; break

        sanity_checks(g, ir)

        with step("مکرو (FRED)"):
            mac = macro_snapshot()

        with step("جمع‌آوری اخبار"):
            items = collect_news()
        with step("تحلیل خبری Gemini"):
            intel, reused = get_intel(items, st)
            log.info(f"[intel] {'cached' if reused else 'fresh'}; "
                     f"ok={intel.get('ok')} changed={intel.get('changed')}")

        signals = {"XAUUSD": [], "GOLD_IR": [], "BTC": []}
        tech = {}
        with step("تحلیل تکنیکال و ساختار"):
            for a in ("XAUUSD", "BTC"):
                if a not in frames: continue
                try:
                    t = technical_block(frames[a])
                    tech[a] = t
                    signals[a] += technical_signals(a, t)
                    sb = structure_block(frames[a])
                    signals[a].append(nds_signal(a, sb))
                    try:
                        wsb = structure_block(resample_weekly(frames[a]), w=2)
                        signals[a].append(nds_weekly_signal(a, wsb))
                    except Exception as e:
                        log.debug(f"[nds-weekly] {a} skipped: {e}")
                    sup, res = support_resistance(frames[a], sb["atr"])
                    t["supports"], t["resistances"] = sup, res
                    log.info(f"[tech] {a} close={t['close']:,.2f} rsi={t['rsi']:.1f} "
                             f"bb%b={t['bb_pb']:.2f} atr_pctile={t['atr_pctile']:.0f} "
                             f"sup={[round(s, 1) for s in sup]} res={[round(r, 1) for r in res]}")
                    if a == "XAUUSD": ir["xau_nds"] = sb["score"]
                except Exception as e:
                    log.error(f"[analysis] {a} failed: {e}")
                    src("analysis", False, f"{a}: {e}")
        signals["XAUUSD"] += correlation_signals(frames)
        signals["BTC"]    += correlation_signals(frames)
        signals["XAUUSD"] += macro_signals(mac, g, frames)
        signals["BTC"]    += macro_signals(mac, g, frames)
        fng = fear_greed()
        signals["BTC"]    += sentiment_signals(items, fng)
        signals["XAUUSD"] += sentiment_signals(items)
        for asset, k in (("XAUUSD", "gold"), ("BTC", "bitcoin")):
            s = intel["scores"].get(k, 0)
            if abs(s) > 0.1:
                signals[asset].append(sig("ai_events", asset, s, 0.8,
                                          "بررسی هوشمندانهٔ اخبار مهم فضای خبری را " +
                                          ("به نفع" if s > 0 else "علیه") + " این بازار نشان می‌دهد."))

        with step("مدل‌های ایران"):
            signals["GOLD_IR"] += iran_gold_signals(ir, rows, intel["scores"].get("usdirr", 0),
                                                    ir.get("xau_nds"), g.get("XAUUSD", {}).get("chg_7d"))
            bps = btc_ir_signal(ir.get("btc_ir_premium_pct"))
            if bps: signals["BTC"].append(bps)
            log.info(f"[signals] counts: XAUUSD={len(signals['XAUUSD'])} "
                     f"GOLD_IR={len(signals['GOLD_IR'])} BTC={len(signals['BTC'])}")

        prices_now = {"XAUUSD": xau, "BTC": btc_usd, "GOLD_IR": ir.get("geram18_rial")}
        grade_and_adapt(prices_now)
        preds = {}
        with step("ساخت پیش‌بینی‌ها"):
            for a in ("GOLD_IR", "XAUUSD", "BTC"):
                if not signals[a]: continue
                if a == "GOLD_IR":
                    sup, res = iran_levels(rows)
                    levels = {"supports": sup, "resistances": res,
                              "atr_pctile": tech.get("XAUUSD", {}).get("atr_pctile", 50)}
                    zone_w = 0.005
                else:
                    t = tech.get(a, {})
                    levels = {"supports": t.get("supports", []),
                              "resistances": t.get("resistances", []),
                              "atr_pctile": t.get("atr_pctile", 50)}
                    zone_w = 0.003
                preds[a] = make_prediction(a, signals[a], levels, intel,
                                           prices_now.get(a), ir, zone_w=zone_w)
                preds[a]["modules"] = sorted({s["module"] for s in signals[a]})

        with step("خلاصهٔ سادهٔ Gemini"):
            plain = gemini_plain(preds)

        for a, p in preds.items():
            o = st.get("preds", {}).get(a)
            if o and (o["direction"] != p["direction"] or abs(o["confidence"] - p["confidence"]) >= 15):
                why = ("جهتِ پیش‌بینی عوض شد" if o["direction"] != p["direction"]
                       else f"احتمال از {fa(o['confidence'])}٪ به {fa(p['confidence'])}٪ رسید")
                send_prediction_change(p, why)
        for a, p in preds.items():
            px = prices_now.get(a)
            for lvl in p["invalidation_levels"]:
                if px and ((p["direction"] == "bullish" and px < lvl) or
                           (p["direction"] == "bearish" and px > lvl)):
                    send_alert(f"<b>{TITLES.get(a, a)}</b>: سطح هشدار {fmt(lvl)} شکست "
                               f"(قیمت {fmt(px)}) — تحلیل قبلی باطل است.",
                               kind="invalid", cooldown_h=6)
        if intel.get("changed") and intel.get("high"):
            names = "; ".join(e.get("name", "") for e in intel["high"][:3])
            send_alert("روایت خبری بازار عوض شد:\n" + esc(names), kind="narrative", cooldown_h=6)

        weekly = {}
        for a, label in (("XAUUSD", "طلای جهانی"), ("BTC", "بیت‌کوین"), ("GOLD_IR", "طلای ایران")):
            w = [s for s in signals[a] if s["horizon"] == "weekly" or
                 s["module"] in ("macro", "ai_events")]
            if w:
                net, _ = fuse(w)
                weekly[label] = ("تمایل به بالا رفتن" if net > 0.1 else
                                 "تمایل به پایین آمدن" if net < -0.1 else "نامشخص / در یک محدوده")

        prices_block = iran_prices_block(ir, g, prev_row)
        macro_block = macro_lines(mac, g)
        with step("ارسال گزارش تلگرام"):
            send_daily_report(preds, intel, weekly, plain, prices_block, macro_block)

        st.update({"preds": preds, "digest": intel.get("digest", ""),
                   "intel": {k: intel.get(k) for k in ("scores", "digest", "changed", "high",
                                                       "invalidations", "ids_hash", "fetched", "ok")},
                   "updated": nowiso(), "diag": DIAG,
                   "snap": {"xauusd": xau, "btc": btc_usd,
                            "usdirr": ir.get("usdirr_free"), "geram18": ir.get("geram18_rial")}})
        jsave(P_STATE, st)
        append_history({"xauusd": xau, "btc_usd": btc_usd,
                        "dxy": g.get("DXY", {}).get("price"),
                        "us10y": g.get("US10Y", {}).get("price"),
                        "oil": g.get("OIL", {}).get("price"),
                        "usdirr_free": ir.get("usdirr_free"),
                        "usdt_irt": ir.get("usdt_irt_rial"),
                        "geram18_rial": ir.get("geram18_rial"),
                        "geram24_rial": ir.get("geram24_rial"),
                        "emami_rial": ir.get("emami_rial"),
                        "nim_rial": ir.get("nim_rial"), "rob_rial": ir.get("rob_rial"),
                        "btc_irt": ir.get("btc_irt_rial"),
                        "geram18_premium_pct": ir.get("geram18_premium_pct"),
                        "usdt_premium_pct": ir.get("usdt_premium_pct"),
                        "sekee_bubble_pct": ir.get("sekee_bubble_pct")})
        register_predictions(preds)
        write_policy(prev_snap, g, ir, intel)
        log.info("════ SUMMARY ════")
        for a, p in preds.items():
            log.info(f"  {a}: {p['direction'].upper()} conf={p['confidence']}% risk={p['risk']}")
        for name, d in DIAG["sources"].items():
            log.info(f"  src {name}: ok={d['ok']} fail={d['fail']} last_err={d['last_err']}")
        log.info(f"  llm: {DIAG.get('llm')}")
        log.info(f"  warns={DIAG['counts']['warn']} errors={DIAG['counts']['err']}")
        return preds

def write_policy(prev_snap, g, ir, intel):
    shock = any(base and cur and abs(cur / base - 1) > thr for _, base, cur, thr in (
        ("XAUUSD", prev_snap.get("xauusd"), g.get("XAUUSD", {}).get("price"), CFG["SHOCK"]["XAUUSD"]),
        ("BTC",    prev_snap.get("btc"),    g.get("BTC", {}).get("price"),    CFG["SHOCK"]["BTC"]),
        ("USDIRR", prev_snap.get("usdirr"), ir.get("usdirr_free"),            CFG["SHOCK"]["USDIRR"])))
    vol_hot = False
    try: vol_hot = structure_block(fetch_history("XAUUSD")[0])["disp"] > 2.0
    except Exception: pass
    breaking = bool(intel.get("changed")) and bool(intel.get("high"))
    mode = "critical" if (breaking or shock) else \
           "elevated" if (vol_hot or len(intel.get("high", [])) >= 2) else "normal"
    every = {"normal": 360, "elevated": 120, "critical": 60}[mode]
    jsave(P_SCHED, {"mode": mode, "full_every_min": every,
                    "next_full_due": (now() + timedelta(minutes=every)).isoformat(timespec="seconds"),
                    "decided": nowiso()})
    log.info(f"[policy] mode={mode} -> next full in ~{every} min "
             f"(shock={shock} vol_hot={vol_hot} breaking={breaking})")

def quick():
    st = jload(P_STATE, {})
    if not st.get("preds"):
        log.info("[quick] no state yet -> bootstrap full")
        return full()
    g, _ = global_snapshots()
    snap, alerts = st.get("snap", {}), []
    for k, path, name in (("XAUUSD", "xauusd", "طلای جهانی"), ("BTC", "btc", "بیت‌کوین")):
        base, cur = snap.get(path), g.get(k, {}).get("price")
        if base and cur and abs(cur / base - 1) > CFG["SHOCK"][k]:
            alerts.append(f"<b>{name}</b> ناگهان {(cur / base - 1) * 100:+.1f}٪ جابه‌جا شد "
                          f"({fmt(base)} → {fmt(cur)}).")
    for a, p in st.get("preds", {}).items():
        px = {"XAUUSD": g.get("XAUUSD", {}).get("price"),
              "BTC": g.get("BTC", {}).get("price")}.get(a)
        for lvl in p.get("invalidation_levels", []):
            if px and ((p["direction"] == "bullish" and px < lvl) or
                       (p["direction"] == "bearish" and px > lvl)):
                alerts.append(f"<b>{TITLES.get(a, a)}</b>: سطح هشدار {fmt(lvl)} شکست "
                              f"(قیمت {fmt(px)}) — تحلیل قبلی باطل است.")
    try:
        g18 = tgju_one("geram18")
        p = st["preds"].get("GOLD_IR")
        if p and g18:
            for lvl in p.get("invalidation_levels", []):
                if (p["direction"] == "bullish" and g18 < lvl) or \
                   (p["direction"] == "bearish" and g18 > lvl):
                    alerts.append(f"<b>🥇 طلا در ایران</b>: سطح هشدار {fmt(lvl)} شکست "
                                  f"(۱۸ عیار {fmt(g18)} ریال) — تحلیل قبلی باطل است.")
    except Exception as e:
        src("tgju", False, f"quick geram18: {e}")
    try:
        items = collect_news()
        seen = set(st.get("seen_news", []))
        for i in [i for i in items if i["id"] not in seen and is_hot(i["title"])][:3]:
            alerts.append("خبر/رویداد مهم: " + esc(i["title"]))
        st["seen_news"] = ([i["id"] for i in items] + st.get("seen_news", []))[:400]
    except Exception as e:
        log.warning(f"[quick] news scan failed: {e}")
    if alerts:
        send_alert("\n".join("• " + a for a in alerts), kind="quick", cooldown_h=2)
    jsave(P_STATE, st)
    pol = jload(P_SCHED, {})
    due = False
    if pol.get("next_full_due"):
        try: due = now() >= datetime.fromisoformat(pol["next_full_due"])
        except Exception: due = False
    if due:
        log.info("[quick] full analysis due -> running")
        full()

def auto():
    st = jload(P_STATE, {})
    if not st.get("preds") or not os.path.exists(P_SCHED):
        return full()
    return quick()

def test_notify():
    ok = tg_send("✅ <b>ربات GBC آنلاین است.</b>\nگزارش کامل در اجرای بعدی ارسال می‌شود.")
    log.info(f"telegram test: {'sent' if ok else 'failed'}")

# ======================================================================
# 19. CLI
# ======================================================================
def main():
    setup_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else "--auto"
    log.info(f"════ RUN {RUN_ID} mode={mode} ════")
    code = 0
    try:
        if mode in ("--full", "--force"): full()
        elif mode == "--quick":           quick()
        elif mode == "--test-notify":     test_notify()
        else:                             auto()
    except Exception as e:
        log.exception("FATAL ERROR — run aborted")
        try:
            tg_send("⛔ <b>اجرای ربات خطا داد</b>\n" + esc(str(e)[:300]) +
                    "\n\nجزئیات کامل: فایل state/logs/last_run.log در ریپو.")
        except Exception:
            pass
        code = 1
    finally:
        log.info(f"════ END {RUN_ID} warns={DIAG['counts']['warn']} "
                 f"errors={DIAG['counts']['err']} ════")
    sys.exit(code)

if __name__ == "__main__":
    main()
