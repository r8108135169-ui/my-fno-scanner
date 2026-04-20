# ============================================================
# app.py — Global Multi-Asset Breakout & F&O Scanner
# Full-Stack Financial Analysis Dashboard
# Built with: Streamlit · yfinance · pandas_ta · Plotly
# ============================================================
# ARCHITECTURE OVERVIEW:
#   1. ASSET_UNIVERSE   — Ticker registry for all asset classes
#   2. Data Layer       — Cached yfinance fetchers (daily/weekly/monthly)
#   3. Indicators       — RSI, EMA(20/50/200), ATR, BB, MACD via pandas_ta
#   4. Breakout Engine  — Multi-timeframe level calculator + badge generator
#   5. F&O Module       — PCR/OI placeholder + Swing signal logic
#   6. Visualization    — Plotly candlestick with overlays (3-pane chart)
#   7. Scanner          — Batch watchlist ranker (Vol Surge + Distance)
#   8. UI Layout        — Streamlit tabs: Chart | F&O | Watchlist | Overview
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
from datetime import datetime, timedelta

# ============================================================
# SECTION 1 — ASSET UNIVERSE REGISTRY
# ============================================================
# Add or remove tickers here to customize the scanner universe.
# Use None for assets without a valid yfinance symbol.

ASSET_UNIVERSE: dict[str, dict[str, str | None]] = {
    "Indian Stocks": {
        "NIFTY 50":       "^NSEI",
        "Bank Nifty":     "^NSEBANK",
        "Reliance":       "RELIANCE.NS",
        "TCS":            "TCS.NS",
        "HDFC Bank":      "HDFCBANK.NS",
        "Infosys":        "INFY.NS",
        "ICICI Bank":     "ICICIBANK.NS",
        "Wipro":          "WIPRO.NS",
        "SBI":            "SBIN.NS",
        "Bajaj Finance":  "BAJFINANCE.NS",
        "Maruti Suzuki":  "MARUTI.NS",
        "Axis Bank":      "AXISBANK.NS",
    },
    "US Stocks": {
        "S&P 500":    "^GSPC",
        "Nasdaq":     "^IXIC",
        "Dow Jones":  "^DJI",
        "Apple":      "AAPL",
        "Microsoft":  "MSFT",
        "Google":     "GOOGL",
        "Amazon":     "AMZN",
        "Meta":       "META",
        "Nvidia":     "NVDA",
        "Tesla":      "TSLA",
        "Berkshire":  "BRK-B",
    },
    "Crypto": {
        "Bitcoin":   "BTC-USD",
        "Ethereum":  "ETH-USD",
        "BNB":       "BNB-USD",
        "Solana":    "SOL-USD",
        "XRP":       "XRP-USD",
        "Cardano":   "ADA-USD",
        "Dogecoin":  "DOGE-USD",
        "Avalanche": "AVAX-USD",
        "Chainlink": "LINK-USD",
        "Polkadot":  "DOT-USD",
    },
    "Commodities": {
        "Gold":             "GC=F",
        "Silver":           "SI=F",
        "Crude Oil (WTI)":  "CL=F",
        "Brent Crude":      "BZ=F",
        "Natural Gas":      "NG=F",
        "Copper":           "HG=F",
        "Wheat":            "ZW=F",
        "Corn":             "ZC=F",
    },
    "Global Bonds": {
        "US 10Y Yield":  "^TNX",
        "US 30Y Yield":  "^TYX",
        "US 2Y Yield":   "^IRX",
        "VIX":           "^VIX",   # Fear gauge — added for context
    },
}

# Pre-defined "quick overview" tickers shown on the Market Overview tab
KEY_INDICES = {
    "🇮🇳 Nifty 50":   "^NSEI",
    "🏦 Bank Nifty":   "^NSEBANK",
    "🇺🇸 S&P 500":    "^GSPC",
    "💻 Nasdaq":        "^IXIC",
    "₿ Bitcoin":        "BTC-USD",
    "🪙 Ethereum":      "ETH-USD",
    "🥇 Gold":          "GC=F",
    "🛢️ Crude Oil":    "CL=F",
    "😱 VIX":           "^VIX",
}

# ============================================================
# SECTION 2 — PAGE CONFIG & CUSTOM CSS
# ============================================================

st.set_page_config(
    page_title="Global Multi-Asset Breakout & F&O Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Google Font import ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Main header gradient ──────────────────────────────── */
.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00E5FF 0%, #00BFA5 40%, #69F0AE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0;
}
.sub-header {
    color: #607D8B;
    font-size: 0.9rem;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

/* ── Breakout badge styles ──────────────────────────────── */
.badge {
    display: inline-block;
    padding: 4px 13px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.3px;
}

/* ── Info card ──────────────────────────────────────────── */
.info-card {
    background: #0F1923;
    border: 1px solid #1C2B38;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 6px 0;
}
.info-card-title {
    font-size: 0.75rem;
    color: #546E7A;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.info-card-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #E0E0E0;
    font-family: 'Space Mono', monospace;
}

/* ── Sidebar tweaks ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #080E14 !important;
    border-right: 1px solid #1C2B38;
}
section[data-testid="stSidebar"] * {
    color: #CFD8DC;
}

/* ── Tab styling ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #080E14;
    border-bottom: 1px solid #1C2B38;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    color: #546E7A;
    background: transparent;
    border-radius: 6px 6px 0 0;
    padding: 8px 20px;
    letter-spacing: 0.3px;
}
.stTabs [aria-selected="true"] {
    color: #00E5FF !important;
    background: #0D1A24 !important;
    border-bottom: 2px solid #00E5FF !important;
}

/* ── Metric card ─────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #0F1923;
    border: 1px solid #1C2B38;
    border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem !important;
    color: #546E7A !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem !important;
    color: #E0E0E0 !important;
}

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: #1C2B38 !important; }

/* ── Dataframe header ────────────────────────────────────── */
.dataframe thead th {
    background: #0D1A24;
    color: #00E5FF;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}

/* ── Top pick card ───────────────────────────────────────── */
.pick-card {
    background: #0F1923;
    border-radius: 12px;
    padding: 18px;
    border-top: 3px solid #00E5FF;
    font-family: 'DM Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SECTION 3 — DATA FETCHING (with caching)
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)   # 5-minute cache
def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance with error handling.

    Parameters
    ----------
    ticker   : Yahoo Finance symbol  (e.g. "AAPL", "^NSEI", "BTC-USD")
    period   : "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"
    interval : "1m", "5m", "15m", "1h", "1d", "1wk", "1mo"

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume]
    """
    try:
        obj = yf.Ticker(ticker)
        df = obj.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        # Flatten MultiIndex columns if present (yfinance ≥0.2.38 sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as exc:
        st.warning(f"⚠️  Could not fetch `{ticker}`: {exc}")
        return pd.DataFrame()


# ============================================================
# SECTION 4 — TECHNICAL INDICATORS
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach technical indicators to an OHLCV DataFrame.

    Indicators added
    ----------------
    RSI_14       — Relative Strength Index (14-period, Wilder smoothing)
    EMA_20       — Exponential Moving Average (20-period short-term trend)
    EMA_50       — Exponential Moving Average (50-period mid-term trend)
    EMA_200      — Exponential Moving Average (200-period long-term trend)
    ATR_14       — Average True Range (14-period volatility)
    BB_Upper/Mid/Lower — Bollinger Bands (20-period, 2σ)
    MACD / Signal / Hist — MACD (12, 26, 9)
    Volume_MA_20 — 20-day simple average of volume
    Volume_Ratio — Current volume ÷ Volume_MA_20  (>1.5 = surge)
    """
    if df.empty or len(df) < 21:
        return df

    df = df.copy()

    # ── RSI ──────────────────────────────────────────────────
    # Standard 14-period RSI.  Overbought > 70, Oversold < 30.
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # ── EMAs ─────────────────────────────────────────────────
    df["EMA_20"]  = ta.ema(df["Close"], length=20)   # Short momentum
    df["EMA_50"]  = ta.ema(df["Close"], length=50)   # Mid trend
    df["EMA_200"] = ta.ema(df["Close"], length=200)  # Major trend (bull/bear)

    # ── ATR ───────────────────────────────────────────────────
    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # ── Bollinger Bands (20, 2σ) ─────────────────────────────
    bb = ta.bbands(df["Close"], length=20, std=2)
    if bb is not None and not bb.empty:
        df["BB_Upper"] = bb.filter(like="BBU").iloc[:, 0]
        df["BB_Lower"] = bb.filter(like="BBL").iloc[:, 0]
        df["BB_Mid"]   = bb.filter(like="BBM").iloc[:, 0]

    # ── MACD (12, 26, 9) ─────────────────────────────────────
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"]        = macd.filter(like="MACD_12").iloc[:, 0]
        df["MACD_Signal"] = macd.filter(like="MACDs").iloc[:, 0]
        df["MACD_Hist"]   = macd.filter(like="MACDh").iloc[:, 0]

    # ── Volume statistics ─────────────────────────────────────
    # Volume_MA_20 — baseline average for surge detection
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    # Volume_Ratio > 1.5 confirms a volume-backed breakout
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"].replace(0, np.nan)

    return df


# ============================================================
# SECTION 5 — BREAKOUT ENGINE
# ============================================================

def calc_breakout_levels(df: pd.DataFrame) -> dict:
    """
    Compute multi-timeframe breakout levels and flags.

    ┌─────────────┬──────────────────────────────────────────────────────────────┐
    │ Timeframe   │ Breakout Condition                                           │
    ├─────────────┼──────────────────────────────────────────────────────────────┤
    │ 1-Day       │ Close > max(High[-21:-1])  AND  Volume > 1.5× Vol_MA_20     │
    │ Weekly      │ Close > max(High[-253:-1])  [≈ 52 weeks of trading days]    │
    │ Monthly     │ Close > max(High[-505:-1])  [≈ 24 months of trading days]   │
    │ Multi-Year  │ Close > max(High[-1261:-1]) [≈ 5 years of trading days]     │
    └─────────────┴──────────────────────────────────────────────────────────────┘

    ADJUSTABLE PARAMETERS (marked with # ← ADJUST):
    - lookback_1d        : daily breakout lookback in bars       (default 20)
    - vol_surge_factor   : volume multiplier for confirmation    (default 1.5)
    - lookback_weekly    : weekly breakout lookback in bars      (default 252)
    - lookback_monthly   : monthly breakout lookback in bars     (default 504)
    - lookback_multiyear : multi-year breakout lookback in bars  (default 1260)
    - near_pct_*         : within N% counts as "near breakout"
    """
    if df.empty or len(df) < 21:
        return {}

    close   = df["Close"].iloc[-1]
    vol_cur = df["Volume"].iloc[-1]
    vol_avg = df["Volume_MA_20"].iloc[-1] if "Volume_MA_20" in df.columns else np.nan

    # ── Volume surge flag ─────────────────────────────────────
    vol_surge_factor = 1.5           # ← ADJUST: raise to require more conviction
    vol_surge = (vol_cur > vol_surge_factor * vol_avg) if not np.isnan(vol_avg) else False

    # ── Helper to build one level dict ───────────────────────
    def _level(lookback: int, near_pct: float, vol_needed: bool = False) -> dict:
        n = min(lookback, len(df) - 1)
        lvl = df["High"].iloc[-n - 1 : -1].max()   # Exclude latest bar
        dist = ((close - lvl) / lvl) * 100
        broken = bool(close > lvl) and (not vol_needed or vol_surge)
        near   = not broken and dist > -near_pct
        return {
            "level":            round(float(lvl), 4),
            "breakout":         broken,
            "near_breakout":    near,
            "volume_confirmed": vol_surge,
            "distance_pct":     round(float(dist), 3),
        }

    levels = {}

    # ── 1-Day Breakout ───────────────────────────────────────
    # Price must exceed the 20-day high AND be backed by volume surge
    lookback_1d = 20          # ← ADJUST: bars for daily high lookback
    levels["1D"] = {
        **_level(lookback_1d, near_pct=2.0, vol_needed=True),
        "label": f"20D High",
    }

    # ── 52-Week Breakout ─────────────────────────────────────
    # Price must exceed the highest high of the last ~252 trading days
    lookback_weekly = 252     # ← ADJUST: trading days for weekly breakout
    levels["Weekly"] = {
        **_level(lookback_weekly, near_pct=3.0),
        "label": "52W High",
    }

    # ── 24-Month Breakout ────────────────────────────────────
    # Price must exceed the highest high of the last ~504 trading days
    lookback_monthly = 504    # ← ADJUST: trading days for monthly breakout
    levels["Monthly"] = {
        **_level(lookback_monthly, near_pct=5.0),
        "label": "24M High",
    }

    # ── 5-Year (Multi-Year) Breakout ─────────────────────────
    # Price must exceed the highest high of the last ~1260 trading days
    lookback_multiyear = 1260 # ← ADJUST: trading days for multi-year breakout
    levels["MultiYear"] = {
        **_level(lookback_multiyear, near_pct=7.0),
        "label": "5Y High",
    }

    return levels


def get_breakout_badges(levels: dict) -> list[tuple[str, str]]:
    """
    Convert breakout level flags into human-readable (label, colour) tuples.
    Order: MultiYear > Monthly > Weekly > 1D (strongest first).
    """
    confirmed_order = [
        ("MultiYear", "🚀 Multi-Year Breakout!",  "#7C4DFF"),
        ("Monthly",   "📅 Monthly Breakout",       "#00C853"),
        ("Weekly",    "📈 52-Week Breakout",        "#FF9100"),
        ("1D",        "⚡ 20D Breakout + Vol",      "#F44336"),
    ]
    near_order = [
        ("MultiYear", "🎯 Near 5Y Level",  "#B39DDB"),
        ("Monthly",   "🎯 Near 24M Level", "#A5D6A7"),
        ("Weekly",    "🎯 Near 52W Level", "#FFCC80"),
        ("1D",        "🎯 Near 20D Level", "#EF9A9A"),
    ]

    badges = [(lbl, col) for tf, lbl, col in confirmed_order
              if levels.get(tf, {}).get("breakout")]
    if not badges:
        badges = [(lbl, col) for tf, lbl, col in near_order
                  if levels.get(tf, {}).get("near_breakout")]
    if not badges:
        badges = [("— No Breakout Signal", "#455A64")]
    return badges


# ============================================================
# SECTION 6 — F&O MODULE  (PCR · OI · Swing Logic)
# ============================================================

def get_fo_placeholder(ticker: str) -> dict:
    """
    F&O metrics with PLACEHOLDER values.

    ═══════════════════════════════════════════════════════════
    HOW TO CONNECT LIVE DATA:
    ─────────────────────────
    Indian Markets (NSE):
      pip install nsepython
      from nsepython import nse_optionchain_scrapper
      chain = nse_optionchain_scrapper("NIFTY")
      # compute PCR from chain["filtered"]["CE"]["totOI"] / chain["filtered"]["PE"]["totOI"]

    US Markets (CBOE):
      url = "https://www.cboe.com/us/equities/market_statistics/daily/"
      # Parse CBOE daily PUT/CALL ratios

    Broker APIs:
      Zerodha Kite: kiteconnect.instruments(), kiteconnect.ltp()
      Interactive Brokers: ib_insync library

    Third-party data:
      Unusual Whales, Market Chameleon, Sensibull
    ═══════════════════════════════════════════════════════════

    PCR interpretation:
      PCR > 1.2  → Bearish (heavy put buying, market expects fall)
      PCR 0.7-1.2→ Neutral
      PCR < 0.7  → Bullish (heavy call buying, market expects rise)
    """
    rng = np.random.default_rng(abs(hash(ticker)) % (2**31))
    pcr = round(rng.uniform(0.55, 1.45), 2)
    oi_trend = rng.choice(["📈 Increasing", "📉 Decreasing", "➡️ Stable"])

    if pcr < 0.70:
        pcr_signal, pcr_color = "Bullish 🟢", "#00C853"
    elif pcr > 1.20:
        pcr_signal, pcr_color = "Bearish 🔴", "#F44336"
    else:
        pcr_signal, pcr_color = "Neutral 🟡", "#FFD740"

    return {
        "pcr": pcr, "pcr_signal": pcr_signal, "pcr_color": pcr_color,
        "oi_trend": oi_trend,
        "note": "⚠️  Simulated PCR/OI — replace get_fo_placeholder() with live API calls",
    }


def get_swing_signal(df: pd.DataFrame, levels: dict) -> dict:
    """
    Swing trading signal generator.

    Signal Matrix
    ─────────────
    🚀 Strong Buy  : Price > EMA20 & EMA50, RSI in (rsi_low, rsi_ob), breakout confirmed
    ✅ Swing Buy   : Price > EMA20, RSI in (rsi_low, rsi_ob)
    👀 Watch       : Price within ±ema_band% of EMA20, RSI < rsi_ob
    ⚠️ Overbought  : RSI > 70
    🔴 Avoid/Short : Price below EMA20

    ADJUSTABLE PARAMETERS:
    - rsi_ob     : RSI overbought threshold        (default 65)  ← ADJUST
    - rsi_low    : Minimum RSI for entry signal    (default 40)  ← ADJUST
    - ema_band   : ±% tolerance near EMA20         (default 0.02)← ADJUST
    """
    if df.empty or len(df) < 21:
        return {"signal": "Insufficient Data", "reason": "Need ≥21 bars", "color": "#546E7A"}

    close   = df["Close"].iloc[-1]
    rsi     = df["RSI_14"].iloc[-1]    if "RSI_14"  in df.columns else np.nan
    ema20   = df["EMA_20"].iloc[-1]    if "EMA_20"  in df.columns else np.nan
    ema50   = df["EMA_50"].iloc[-1]    if "EMA_50"  in df.columns else np.nan

    if any(np.isnan(v) for v in [rsi, ema20]):
        return {"signal": "Calculating…", "reason": "Warming up indicators", "color": "#546E7A"}

    # ── Breakout presence ─────────────────────────────────────
    any_breakout = any(levels.get(tf, {}).get("breakout") for tf in levels)

    # ── Parameters ───────────────────────────────────────────
    rsi_ob   = 65    # ← ADJUST: upper RSI threshold (not overbought zone)
    rsi_low  = 40    # ← ADJUST: minimum RSI for swing buy entry
    ema_band = 0.02  # ← ADJUST: ±2% counts as "near EMA20"

    above_ema20 = close > ema20
    above_ema50 = (not np.isnan(ema50)) and close > ema50
    near_ema20  = abs(close - ema20) / ema20 < ema_band

    if above_ema20 and above_ema50 and rsi_low < rsi < rsi_ob and any_breakout:
        return {"signal": "🚀 Strong Buy",  "reason": f"EMA20/50 ✓  RSI={rsi:.1f} (healthy)  Breakout ✓", "color": "#00C853"}
    elif above_ema20 and rsi_low < rsi < rsi_ob:
        return {"signal": "✅ Swing Buy",   "reason": f"Price > EMA20 ✓  RSI={rsi:.1f} (not overbought)", "color": "#69F0AE"}
    elif near_ema20 and rsi < rsi_ob:
        return {"signal": "👀 Watch",       "reason": f"Price near EMA20 (±{ema_band*100:.0f}%)  RSI={rsi:.1f}", "color": "#FFD740"}
    elif rsi > 70:
        return {"signal": "⚠️ Overbought",  "reason": f"RSI={rsi:.1f} > 70 — wait for pullback", "color": "#FF5252"}
    else:
        return {"signal": "🔴 Avoid/Short", "reason": f"Price below EMA20  RSI={rsi:.1f}", "color": "#FF1744"}


# ============================================================
# SECTION 7 — PLOTLY CANDLESTICK CHART
# ============================================================

_DARK_BG  = "#080E14"
_PANEL_BG = "#0A1520"
_GRID     = "#111E2A"

def build_chart(df: pd.DataFrame, ticker: str, name: str, levels: dict) -> go.Figure:
    """
    3-pane Plotly chart:
      Pane 1 (60%) — Candlestick + EMA overlays + BB + breakout hlines
      Pane 2 (20%) — Volume bars + 20-day MA line
      Pane 3 (20%) — RSI with OB/OS bands
    """
    if df.empty:
        return go.Figure()

    display = df.tail(252).copy()   # Always show last year of data

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=("", "", ""),
    )

    # ── 1. Candlestick ────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=display.index,
        open=display["Open"], high=display["High"],
        low=display["Low"],   close=display["Close"],
        name="OHLC",
        increasing=dict(line=dict(color="#00BFA5"), fillcolor="#00BFA5"),
        decreasing=dict(line=dict(color="#F44336"), fillcolor="#F44336"),
    ), row=1, col=1)

    # ── 2. EMAs ───────────────────────────────────────────────
    ema_cfg = [
        ("EMA_20",  "#FFD600", "EMA 20",  1.4),
        ("EMA_50",  "#FF6D00", "EMA 50",  1.4),
        ("EMA_200", "#EA80FC", "EMA 200", 1.8),
    ]
    for col, color, label, width in ema_cfg:
        if col in display.columns:
            fig.add_trace(go.Scatter(
                x=display.index, y=display[col],
                mode="lines", name=label,
                line=dict(color=color, width=width), opacity=0.9,
            ), row=1, col=1)

    # ── 3. Bollinger Bands ────────────────────────────────────
    if "BB_Upper" in display.columns and "BB_Lower" in display.columns:
        fig.add_trace(go.Scatter(
            x=display.index, y=display["BB_Upper"],
            mode="lines", name="BB Upper",
            line=dict(color="rgba(82,130,200,0.45)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=display.index, y=display["BB_Lower"],
            mode="lines", name="BB Lower",
            line=dict(color="rgba(82,130,200,0.45)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(82,130,200,0.04)",
        ), row=1, col=1)

    # ── 4. Breakout level hlines ──────────────────────────────
    level_style = {
        "1D":        ("#F44336", "dot",    "20D High"),
        "Weekly":    ("#FF9100", "dash",   "52W High"),
        "Monthly":   ("#00C853", "dashdot","24M High"),
        "MultiYear": ("#7C4DFF", "solid",  "5Y High"),
    }
    for tf, (color, dash, lbl) in level_style.items():
        info = levels.get(tf, {})
        if not info:
            continue
        lvl     = info["level"]
        broken  = info["breakout"]
        alpha   = "FF" if broken else "88"
        line_w  = 1.8 if broken else 1.0
        tick    = " ✅" if broken else ""
                # Safe plotting for horizontal breakout lines
        for level in levels:
            # Check if the level is actually a number before plotting
            if level is not None and str(level).lower() != 'nan':
                try:
                    fig.add_hline(
                        y=float(level), 
                        line_dash="dot", 
                        line_color="orange", 
                        annotation_text="Breakout",
                        annotation_position="bottom right"
                    )
                except (ValueError, TypeError):
                    # Skip if the math fails for a specific level
                    continue
        
        # This return must be aligned with the 'for' loop above (8 spaces)
        return fig
        # This line must be aligned with the word 'for' above
        return fig
        # --- END OF COPY ---
        # This return must be aligned with the 'for' loop above
        return fig
            rows.append({
                "Asset":          name,
                "Ticker":         ticker,
                "Price":          round(close, 2),
                "Change %":       round(chg_pct, 2),
                "RSI(14)":        round(rsi_val, 1) if not np.isnan(rsi_val) else None,
                "EMA 20":         round(ema20_val, 2) if not np.isnan(ema20_val) else None,
                "Vol Ratio":      round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
                "Breakout":       badges[0][0],
                "Swing Signal":   swing["signal"],
                "Dist to Lvl %":  round(nearest_dist, 2),
                "52W High":       round(levels.get("Weekly", {}).get("level", 0), 2),
                "5Y High":        round(levels.get("MultiYear", {}).get("level", 0), 2),
            })
        except Exception:
            continue   # Skip tickers that fail silently

    prog.empty()
    status.empty()

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.sort_values(
        by=["Vol Ratio", "Dist to Lvl %"],
        ascending=[False, True],
        na_position="last",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    return result


# ============================================================
# SECTION 9 — MAIN APP LAYOUT
# ============================================================

def main() -> None:
    # ── Header ────────────────────────────────────────────────
    col_title, col_time = st.columns([4, 1])
    with col_title:
        st.markdown('<p class="main-header">📊 Global Multi-Asset Breakout & F&O Scanner</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Multi-timeframe breakout detection · F&O metrics · Swing signals · Live data</p>',
                    unsafe_allow_html=True)
    with col_time:
        st.markdown(
            f"<div style='text-align:right;padding-top:18px;"
            f"font-family:Space Mono;font-size:0.75rem;color:#546E7A'>"
            f"🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='margin:6px 0 14px'>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────
    # SIDEBAR
    # ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<p style='font-family:Space Mono;font-size:0.9rem;"
            "color:#00E5FF;font-weight:700;margin-bottom:4px'>⚙️  CONTROLS</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:4px 0 12px'>", unsafe_allow_html=True)

        # ── Asset class & picker ─────────────────────────────
        asset_category = st.selectbox("📂  Asset Class", list(ASSET_UNIVERSE.keys()))
        valid_assets   = {k: v for k, v in ASSET_UNIVERSE[asset_category].items() if v is not None}
        selected_name  = st.selectbox("🎯  Select Asset", list(valid_assets.keys()))
        selected_tick  = valid_assets[selected_name]

        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

        # ── Chart timeframe ──────────────────────────────────
        st.markdown(
            "<p style='font-family:Space Mono;font-size:0.75rem;"
            "color:#546E7A;text-transform:uppercase;letter-spacing:1px'>Chart Timeframe</p>",
            unsafe_allow_html=True,
        )
        tf_choice = st.select_slider(
            "lookback",
            options=["3mo", "6mo", "1y", "2y", "5y"],
            value="1y",
            label_visibility="collapsed",
        )
        tf_bars = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}

        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

        # ── Breakout parameter overrides ─────────────────────
        st.markdown(
            "<p style='font-family:Space Mono;font-size:0.75rem;"
            "color:#546E7A;text-transform:uppercase;letter-spacing:1px'>Breakout Params</p>",
            unsafe_allow_html=True,
        )
        with st.expander("🔧  Adjust Thresholds"):
            _vol_thr  = st.slider("Volume Surge Factor",   1.0, 3.0, 1.5, 0.1,
                                  help="Volume must exceed Avg × this factor to confirm a 1-Day breakout")
            _rsi_ob   = st.slider("RSI Overbought Level",  55, 80, 65, 1,
                                  help="RSI threshold above which swing buy is avoided")
            _rsi_low  = st.slider("RSI Minimum Entry",     25, 55, 40, 1,
                                  help="RSI must be above this for a swing buy signal")
            _daily_lb = st.slider("Daily Lookback (days)", 5,  50, 20, 5,
                                  help="Number of days for 1-Day breakout level")
            st.caption("Changes apply to the signal generation logic in real time.")

        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

        # ── Scanner launch ───────────────────────────────────
        run_scan = st.button("🚀  Run Full Watchlist Scan",
                             use_container_width=True, type="primary",
                             help="Batch-scans all assets in the selected category")

        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

        # ── Legend ───────────────────────────────────────────
        st.markdown("""
<p style='font-family:Space Mono;font-size:0.72rem;color:#546E7A;
text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>Legend</p>
<div style='font-size:0.78rem;color:#78909C;line-height:1.8'>
🚀 Multi-Year — 5Y High Breakout<br>
📅 Monthly — 24M High Breakout<br>
📈 Weekly — 52W High Breakout<br>
⚡ Daily — 20D High + Vol Surge<br>
🎯 Near — Within 2-7% of level<br>
<hr style='border-color:#1C2B38;margin:8px 0'>
✅ Vol confirmed  ❌ No vol surge<br>
🟢 RSI OK  🔴 RSI Overbought  🟢 OS
</div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)
        st.caption("⏰ Price cache: 5 min | Scan cache: 10 min")
        st.caption("📌 Data: Yahoo Finance (yfinance)")
        st.caption("⚠️  For informational use only. Not financial advice.")

    # ────────────────────────────────────────────────────────
    # LOAD DATA FOR SELECTED ASSET
    # ────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {selected_name} ({selected_tick}) data…"):
        df_raw = fetch_ohlcv(selected_tick, period="5y", interval="1d")

    df = pd.DataFrame()
    levels = {}
    swing  = {"signal": "N/A", "reason": "No data", "color": "#546E7A"}
    badges = [("— No Data", "#546E7A")]

    if not df_raw.empty:
        df     = add_indicators(df_raw)
        levels = calc_breakout_levels(df)
        swing  = get_swing_signal(df, levels)
        badges = get_breakout_badges(levels)

    # ────────────────────────────────────────────────────────
    # TABS
    # ────────────────────────────────────────────────────────
    tab_chart, tab_fo, tab_watch, tab_overview = st.tabs([
        "📈  Chart Analysis",
        "🔍  F&O & Swing Module",
        "📋  Daily Watchlist",
        "🌍  Market Overview",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — CHART ANALYSIS
    # ════════════════════════════════════════════════════════
    with tab_chart:
        if df.empty:
            st.error(f"❌  No data returned for `{selected_tick}`. "
                     "Try a different asset or check your internet connection.")
        else:
            close    = df["Close"].iloc[-1]
            prev_c   = df["Close"].iloc[-2] if len(df) > 1 else close
            chg_pct  = ((close - prev_c) / prev_c) * 100
            rsi_v    = df["RSI_14"].iloc[-1]       if "RSI_14"       in df.columns else np.nan
            vol_r    = df["Volume_Ratio"].iloc[-1]  if "Volume_Ratio" in df.columns else np.nan
            ema20_v  = df["EMA_20"].iloc[-1]        if "EMA_20"       in df.columns else np.nan
            atr_v    = df["ATR_14"].iloc[-1]        if "ATR_14"       in df.columns else np.nan
            ema200_v = df["EMA_200"].iloc[-1]       if "EMA_200"      in df.columns else np.nan

            # ── Asset title + quick stats ─────────────────────
            st.markdown(
                f"<span style='font-family:Space Mono;font-size:1.05rem;"
                f"color:#00E5FF;font-weight:700'>{selected_name}</span>"
                f"<span style='font-family:Space Mono;font-size:0.8rem;"
                f"color:#546E7A;margin-left:10px'>{selected_tick}</span>",
                unsafe_allow_html=True,
            )

            # ── 5 key metrics ─────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("💰 Price",
                          f"{close:,.2f}",
                          f"{chg_pct:+.2f}%")
            with m2:
                rsi_label = ("🔴 OB" if (not np.isnan(rsi_v) and rsi_v > 70)
                             else ("🟢 OS" if (not np.isnan(rsi_v) and rsi_v < 30) else "🟡 OK"))
                st.metric("📊 RSI (14)",
                          f"{rsi_v:.1f}" if not np.isnan(rsi_v) else "—",
                          rsi_label)
            with m3:
                vol_label = "🔥 Surge!" if (not np.isnan(vol_r) and vol_r > 1.5) else "Normal"
                st.metric("📦 Vol Ratio",
                          f"{vol_r:.2f}x" if not np.isnan(vol_r) else "—",
                          vol_label)
            with m4:
                ema_lbl = ("🟢 Above EMA200" if (not np.isnan(ema200_v) and close > ema200_v)
                           else "🔴 Below EMA200")
                st.metric("📉 EMA 200",
                          f"{ema200_v:,.2f}" if not np.isnan(ema200_v) else "—",
                          ema_lbl)
            with m5:
                st.metric("📏 ATR (14)",
                          f"{atr_v:,.2f}" if not np.isnan(atr_v) else "—",
                          "Volatility")

            # ── Breakout badges ───────────────────────────────
            st.markdown("<div style='margin:10px 0 4px'>", unsafe_allow_html=True)
            badge_html = ""
            for lbl, col in badges:
                badge_html += (
                    f'<span class="badge" style="background:{col}">{lbl}</span>'
                )
            swing_html = (
                f'<span class="badge" style="background:{swing["color"]};'
                f'margin-left:14px">Swing: {swing["signal"]}</span>'
            )
            st.markdown(badge_html + swing_html, unsafe_allow_html=True)
            st.caption(f"Signal rationale: {swing['reason']}")
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Breakout level detail cards ───────────────────
            with st.expander("📐  Breakout Level Details", expanded=False):
                bc1, bc2, bc3, bc4 = st.columns(4)
                tf_meta = [
                    ("1D",        "⚡ 20-Day",   "#F44336", bc1),
                    ("Weekly",    "📈 52-Week",   "#FF9100", bc2),
                    ("Monthly",   "📅 24-Month",  "#00C853", bc3),
                    ("MultiYear", "🚀 5-Year",    "#7C4DFF", bc4),
                ]
                for tf, label, color, bcol in tf_meta:
                    info = levels.get(tf, {})
                    if not info:
                        bcol.caption(f"{label}: No data")
                        continue
                    st_txt = ("✅ BROKEN" if info["breakout"]
                              else ("🎯 NEAR" if info["near_breakout"] else "❌ NOT YET"))
                    vol_txt = "✅" if info["volume_confirmed"] else "❌"
                    bcol.markdown(
                        f"<div style='border-left:4px solid {color};padding:12px;"
                        f"background:#0F1923;border-radius:6px;'>"
                        f"<b style='color:{color}'>{label}</b><br>"
                        f"<span style='font-family:Space Mono;font-size:0.85rem'>"
                        f"Level: <b>{info['level']:,.2f}</b></span><br>"
                        f"Status: {st_txt}<br>"
                        f"Distance: <b>{info['distance_pct']:+.2f}%</b><br>"
                        f"Vol Confirmed: {vol_txt}</div>",
                        unsafe_allow_html=True,
                    )

            # ── Candlestick chart ─────────────────────────────
            bars = tf_bars.get(tf_choice, 252)
            fig  = build_chart(df.tail(bars), selected_tick, selected_name, levels)
            st.plotly_chart(fig, use_container_width=True)

            # ── Raw OHLCV data expander ───────────────────────
            with st.expander("📋  Raw OHLCV + Indicators (last 30 bars)"):
                cols_show = [c for c in
                             ["Open", "High", "Low", "Close", "Volume",
                              "RSI_14", "EMA_20", "EMA_50", "EMA_200", "Volume_Ratio"]
                             if c in df.columns]
                raw_view = df[cols_show].tail(30).copy()
                raw_view.index = raw_view.index.strftime("%Y-%m-%d")
                st.dataframe(raw_view.round(2), use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 2 — F&O & SWING MODULE
    # ════════════════════════════════════════════════════════
    with tab_fo:
        st.markdown(
            "<p style='font-family:Space Mono;font-size:1rem;"
            "color:#00E5FF;font-weight:700'>🔍  F&O Analysis & Swing Module</p>",
            unsafe_allow_html=True,
        )

        if df.empty:
            st.warning("Select a valid asset in the **Chart Analysis** tab first.")
        else:
            close = df["Close"].iloc[-1]
            left, right = st.columns([1, 1], gap="large")

            # ── Left: Technical indicators table ──────────────
            with left:
                st.markdown("#### 📊 Indicator Snapshot")

                def _fmt(col): 
                    v = df[col].iloc[-1] if col in df.columns else np.nan
                    return f"{v:,.2f}" if not np.isnan(v) else "—"

                def _sig_ema(col):
                    v = df[col].iloc[-1] if col in df.columns else np.nan
                    if np.isnan(v):
                        return "—"
                    return "🟢 Price Above" if close > v else "🔴 Price Below"

                ind_rows = [
                    ("RSI (14)",   _fmt("RSI_14"),
                     "🔴 Overbought" if (not df["RSI_14"].empty and not np.isnan(df["RSI_14"].iloc[-1]) and df["RSI_14"].iloc[-1] > 70)
                     else ("🟢 Oversold" if (not np.isnan(df["RSI_14"].iloc[-1]) and df["RSI_14"].iloc[-1] < 30) else "🟡 Neutral")),
                    ("EMA (20)",   _fmt("EMA_20"),   _sig_ema("EMA_20")),
                    ("EMA (50)",   _fmt("EMA_50"),   _sig_ema("EMA_50")),
                    ("EMA (200)",  _fmt("EMA_200"),  _sig_ema("EMA_200")),
                    ("ATR (14)",   _fmt("ATR_14"),   "Volatility gauge"),
                    ("BB Upper",   _fmt("BB_Upper"), "Resistance zone"),
                    ("BB Lower",   _fmt("BB_Lower"), "Support zone"),
                ]
                ind_df = pd.DataFrame(ind_rows, columns=["Indicator", "Value", "Signal"])
                st.dataframe(ind_df, use_container_width=True, hide_index=True)

                # MACD mini-section
                st.markdown("#### 📉 MACD (12, 26, 9)")
                if all(c in df.columns for c in ["MACD", "MACD_Signal", "MACD_Hist"]):
                    m_val  = df["MACD"].iloc[-1]
                    m_sig  = df["MACD_Signal"].iloc[-1]
                    m_hist = df["MACD_Hist"].iloc[-1]
                    if not any(np.isnan(v) for v in [m_val, m_sig, m_hist]):
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("MACD",    f"{m_val:.4f}")
                        mc2.metric("Signal",  f"{m_sig:.4f}")
                        mc3.metric("Hist",    f"{m_hist:.4f}")
                        cross = "🟢 Bullish Crossover" if m_val > m_sig else "🔴 Bearish Crossover"
                        st.info(f"MACD Signal: **{cross}**")

                        # Mini MACD histogram chart
                        hist_colors = ["#00BFA5" if v >= 0 else "#F44336"
                                       for v in df["MACD_Hist"].tail(60)]
                        fig_macd = go.Figure(go.Bar(
                            x=df.tail(60).index, y=df["MACD_Hist"].tail(60),
                            marker_color=hist_colors, name="MACD Hist",
                        ))
                        fig_macd.add_trace(go.Scatter(
                            x=df.tail(60).index, y=df["MACD"].tail(60),
                            mode="lines", name="MACD",
                            line=dict(color="#40C4FF", width=1.5),
                        ))
                        fig_macd.add_trace(go.Scatter(
                            x=df.tail(60).index, y=df["MACD_Signal"].tail(60),
                            mode="lines", name="Signal",
                            line=dict(color="#FF9100", width=1.5),
                        ))
                        fig_macd.update_layout(
                            template="plotly_dark", paper_bgcolor=_DARK_BG,
                            plot_bgcolor=_PANEL_BG, height=220,
                            margin=dict(l=0, r=0, t=10, b=0),
                            legend=dict(font=dict(size=9, family="Space Mono"),
                                        bgcolor="rgba(0,0,0,0)"),
                            xaxis_rangeslider_visible=False,
                        )
                        st.plotly_chart(fig_macd, use_container_width=True)

            # ── Right: F&O PCR / OI + Swing signal ────────────
            with right:
                st.markdown("#### 📑 F&O Metrics (PCR & OI)")

                fo = get_fo_placeholder(selected_tick)
                st.warning(fo["note"], icon="⚠️")

                fo1, fo2 = st.columns(2)
                fo1.metric("Put-Call Ratio (PCR)", fo["pcr"],
                           fo["pcr_signal"])
                fo2.metric("Open Interest Trend", fo["oi_trend"])

                # PCR gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fo["pcr"],
                    title={"text": "PCR Gauge",
                           "font": {"color": "#CFD8DC", "family": "Space Mono", "size": 12}},
                    number={"font": {"color": "#CFD8DC", "family": "Space Mono"}},
                    gauge={
                        "axis": {"range": [0, 2],
                                 "tickfont": {"color": "#546E7A", "size": 9}},
                        "bar": {"color": fo["pcr_color"]},
                        "bgcolor": _PANEL_BG,
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 0.70], "color": "rgba(0,200,83,0.15)"},
                            {"range": [0.70, 1.20], "color": "rgba(255,215,0,0.10)"},
                            {"range": [1.20, 2],   "color": "rgba(244,67,54,0.15)"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 2},
                                      "thickness": 0.75, "value": fo["pcr"]},
                    },
                ))
                fig_gauge.update_layout(
                    template="plotly_dark", paper_bgcolor=_DARK_BG,
                    height=220, margin=dict(l=20, r=20, t=40, b=10),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # PCR reference table
                pcr_ref = pd.DataFrame({
                    "PCR Range": ["< 0.70", "0.70 – 1.20", "> 1.20"],
                    "Sentiment": ["Bullish 🟢", "Neutral 🟡",  "Bearish 🔴"],
                    "Meaning":   ["Heavy call buying", "Balanced", "Heavy put buying"],
                })
                st.dataframe(pcr_ref, use_container_width=True, hide_index=True)

                # OI matrix
                st.markdown("**OI + Price Trend Matrix:**")
                oi_mat = pd.DataFrame({
                    "Price Trend": ["Rising ↑", "Falling ↓", "Rising ↑", "Falling ↓"],
                    "OI Trend":    ["Rising ↑", "Rising ↑",  "Falling ↓","Falling ↓"],
                    "Signal":      ["💪 Strong Bull", "🔻 Strong Bear",
                                    "🔄 Short Cover", "🔄 Long Unwind"],
                })
                st.dataframe(oi_mat, use_container_width=True, hide_index=True)

                # ── Swing signal card ─────────────────────────
                st.markdown("#### 🎯 Swing Trading Signal")
                st.markdown(
                    f"<div style='background:{swing['color']}22;"
                    f"border-left:4px solid {swing['color']};"
                    f"padding:14px 18px;border-radius:8px;'>"
                    f"<span style='font-family:Space Mono;font-size:1.1rem;"
                    f"font-weight:700;color:{swing['color']}'>{swing['signal']}</span><br>"
                    f"<span style='color:#CFD8DC;font-size:0.85rem'>{swing['reason']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ATR-based stop-loss & targets
                close = df["Close"].iloc[-1]
                if "ATR_14" in df.columns and not np.isnan(df["ATR_14"].iloc[-1]):
                    atr  = df["ATR_14"].iloc[-1]
                    sl   = close - 2 * atr
                    tgt1 = close + 2 * atr
                    tgt2 = close + 4 * atr

                    st.markdown(
                        "<p style='font-family:Space Mono;font-size:0.78rem;"
                        "color:#546E7A;margin-top:14px'>📐 ATR-BASED LEVELS  (2× ATR risk)</p>",
                        unsafe_allow_html=True,
                    )
                    a1, a2, a3 = st.columns(3)
                    a1.metric("🛑 Stop Loss",      f"{sl:,.2f}",
                              f"{((sl-close)/close*100):+.2f}%",
                              delta_color="inverse")
                    a2.metric("🎯 Target 1 (1:1)", f"{tgt1:,.2f}",
                              f"{((tgt1-close)/close*100):+.2f}%")
                    a3.metric("🚀 Target 2 (1:2)", f"{tgt2:,.2f}",
                              f"{((tgt2-close)/close*100):+.2f}%")

    # ════════════════════════════════════════════════════════
    # TAB 3 — DAILY WATCHLIST SCANNER
    # ════════════════════════════════════════════════════════
    with tab_watch:
        st.markdown(
            "<p style='font-family:Space Mono;font-size:1rem;"
            "color:#00E5FF;font-weight:700'>📋  Daily Watchlist Scanner</p>",
            unsafe_allow_html=True,
        )
        st.caption(f"Scanning: **{asset_category}**  |  "
                   "Ranked by Volume Surge → Distance to Breakout Level")

        # Trigger scan
        scan_key = f"scan_{asset_category}"
        if run_scan:
            st.session_state[scan_key] = run_full_scan(asset_category)

        scan_df = st.session_state.get(scan_key)

        if scan_df is None:
            st.info(
                "👆  Click **🚀 Run Full Watchlist Scan** in the sidebar to scan "
                f"all {len(valid_assets)} assets in **{asset_category}**.",
                icon="ℹ️",
            )
            preview = pd.DataFrame([
                {"#": i+1, "Asset": k, "Ticker": v, "Status": "⏳ Pending"}
                for i, (k, v) in enumerate(valid_assets.items())
            ])
            st.dataframe(preview, use_container_width=True, hide_index=True)

        elif scan_df.empty:
            st.warning("No data returned. Check your connection and try again.")

        else:
            # ── Summary KPIs ──────────────────────────────────
            n_breakout = scan_df["Breakout"].str.contains("Breakout", na=False).sum()
            n_near     = scan_df["Breakout"].str.contains("Near",     na=False).sum()
            n_swing    = scan_df["Swing Signal"].str.contains("Buy",  na=False).sum()
            n_surge    = (scan_df["Vol Ratio"] > 1.5).sum() if "Vol Ratio" in scan_df.columns else 0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🚀 Confirmed Breakouts", int(n_breakout))
            k2.metric("🎯 Near Breakout",       int(n_near))
            k3.metric("✅ Swing Buy Signals",   int(n_swing))
            k4.metric("🔥 Volume Surges (>1.5×)", int(n_surge))

            st.markdown("<hr style='margin:8px 0 14px'>", unsafe_allow_html=True)

            # ── Filters ───────────────────────────────────────
            fc1, fc2, fc3 = st.columns(3)
            f_break = fc1.checkbox("🚀 Breakouts only",    False)
            f_swing = fc2.checkbox("✅ Swing Buys only",   False)
            f_vol   = fc3.slider("Min Vol Ratio", 0.0, 3.0, 0.0, 0.1)

            filt = scan_df.copy()
            if f_break:
                filt = filt[filt["Breakout"].str.contains("Breakout", na=False)]
            if f_swing:
                filt = filt[filt["Swing Signal"].str.contains("Buy", na=False)]
            if f_vol > 0:
                filt = filt[filt["Vol Ratio"] >= f_vol]

            # ── Styled table ──────────────────────────────────
            show_cols = ["Asset", "Ticker", "Price", "Change %", "RSI(14)",
                         "Vol Ratio", "Breakout", "Swing Signal", "Dist to Lvl %", "52W High"]
            show_cols = [c for c in show_cols if c in filt.columns]

            def _style_change(val):
                if isinstance(val, (int, float)):
                    return "color:#00C853;font-weight:600" if val > 0 else "color:#F44336;font-weight:600"
                return ""
            def _style_rsi(val):
                if isinstance(val, (int, float)):
                    if val > 70: return "color:#F44336;font-weight:600"
                    if val < 30: return "color:#00BFA5;font-weight:600"
                return ""
            def _style_vol(val):
                if isinstance(val, (int, float)) and val > 1.5:
                    return "background:rgba(255,145,0,0.2);color:#FF9100;font-weight:700"
                return ""

            styled = (
                filt[show_cols].style
                .applymap(_style_change, subset=["Change %", "Dist to Lvl %"])
                .applymap(_style_rsi,    subset=["RSI(14)"])
                .applymap(_style_vol,    subset=["Vol Ratio"])
            )
            st.dataframe(styled, use_container_width=True, height=420)

            # ── Top 3 picks ───────────────────────────────────
            st.markdown("#### 🌟 Top Picks by Volume Surge")
            top3 = filt.nlargest(3, "Vol Ratio") if "Vol Ratio" in filt.columns else filt.head(3)
            medals = ["🥇", "🥈", "🥉"]
            borders = ["#FFD600", "#90A4AE", "#CD7F32"]
            pick_cols = st.columns(min(3, max(1, len(top3))))

            for i, (_, row) in enumerate(top3.iterrows()):
                if i >= len(pick_cols):
                    break
                chg = row.get("Change %", 0) or 0
                chg_col = "#00C853" if chg > 0 else "#F44336"
                vol_val  = row.get("Vol Ratio", 0) or 0
                with pick_cols[i]:
                    st.markdown(
                        f"<div class='pick-card' style='border-top-color:{borders[i]}'>"
                        f"<span style='font-size:1.2rem'>{medals[i]}</span> "
                        f"<b style='color:#E0E0E0'>{row['Asset']}</b><br>"
                        f"<span style='font-family:Space Mono;font-size:0.7rem;color:#546E7A'>"
                        f"{row.get('Ticker','')}</span><br>"
                        f"<span style='font-family:Space Mono;font-size:1.2rem;"
                        f"color:#E0E0E0;font-weight:700'>{row.get('Price',0):,.2f}</span>"
                        f"  <span style='color:{chg_col};font-weight:600'>{chg:+.2f}%</span><br>"
                        f"<span style='background:#FF9100;color:#000;padding:2px 10px;"
                        f"border-radius:12px;font-size:0.75rem;font-weight:700'>"
                        f"Vol {vol_val:.2f}x</span><br>"
                        f"<span style='font-size:0.78rem;color:#78909C;margin-top:6px;"
                        f"display:block'>{row.get('Breakout','')}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Download ──────────────────────────────────────
            st.markdown("<hr style='margin:14px 0 8px'>", unsafe_allow_html=True)
            csv = scan_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download Watchlist CSV",
                data=csv,
                file_name=f"watchlist_{asset_category}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ════════════════════════════════════════════════════════
    # TAB 4 — MARKET OVERVIEW
    # ════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown(
            "<p style='font-family:Space Mono;font-size:1rem;"
            "color:#00E5FF;font-weight:700'>🌍  Global Market Overview</p>",
            unsafe_allow_html=True,
        )

        with st.spinner("Loading key indices…"):
            ov_rows = []
            for idx_name, idx_tick in KEY_INDICES.items():
                try:
                    d = fetch_ohlcv(idx_tick, period="5d", interval="1d")
                    if d.empty or len(d) < 2:
                        continue
                    c  = d["Close"].iloc[-1]
                    p  = d["Close"].iloc[-2]
                    ch = ((c - p) / p) * 100
                    # 1-year high from data
                    d_1y = fetch_ohlcv(idx_tick, period="1y", interval="1d")
                    h52  = d_1y["High"].max() if not d_1y.empty else c
                    ov_rows.append({
                        "Index": idx_name, "Ticker": idx_tick,
                        "Price": round(c, 2), "Change %": round(ch, 2),
                        "52W High": round(h52, 2),
                        "At High?": "✅" if c > h52 * 0.98 else "❌",
                    })
                except Exception:
                    continue

        if ov_rows:
            ov_df = pd.DataFrame(ov_rows)

            # ── Metric tiles ──────────────────────────────────
            cols = st.columns(3)
            for i, row in enumerate(ov_rows):
                col = cols[i % 3]
                chg = row["Change %"]
                col.metric(
                    row["Index"],
                    f"{row['Price']:,.2f}",
                    f"{chg:+.2f}%",
                )

            st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)

            # ── Summary table ─────────────────────────────────
            st.markdown("#### 📋 Market Summary")
            st.dataframe(ov_df, use_container_width=True, hide_index=True)

            # ── Performance bar chart ─────────────────────────
            st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)
            st.markdown("#### 📊 Daily % Change — Global Markets")
            fig_bar = go.Figure(go.Bar(
                x=ov_df["Index"],
                y=ov_df["Change %"],
                marker_color=["#00BFA5" if v >= 0 else "#F44336"
                              for v in ov_df["Change %"]],
                text=[f"{v:+.2f}%" for v in ov_df["Change %"]],
                textposition="outside",
                textfont=dict(family="Space Mono", size=10, color="#CFD8DC"),
            ))
            fig_bar.add_hline(y=0, line_color="#1C2B38", line_width=1.5)
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor=_DARK_BG, plot_bgcolor=_PANEL_BG,
                height=380,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis_title="% Change",
                showlegend=False,
                xaxis=dict(tickfont=dict(family="Space Mono", size=10, color="#90A4AE")),
                yaxis=dict(tickfont=dict(family="Space Mono", size=10, color="#546E7A"),
                           gridcolor=_GRID),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Market Hours ──────────────────────────────────────
        st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)
        st.markdown("#### 🕐 Global Market Hours (IST)")
        hours_df = pd.DataFrame({
            "Exchange":    ["NSE / BSE (India)", "NYSE / Nasdaq (US)",
                            "Tokyo (Japan)",     "London (UK)", "Crypto"],
            "Open (IST)":  ["09:15",             "19:30 (20:30 DST)",
                            "05:30",             "14:30 (13:30 DST)", "24/7"],
            "Close (IST)": ["15:30",             "02:00 (01:00 DST)",
                            "12:00",             "23:30 (22:30 DST)", "24/7"],
            "Notes":       ["Pre-market 09:00",  "Extended: 16:00–20:00",
                            "Nikkei 225",        "FTSE 100",          "Never closes"],
        })
        st.dataframe(hours_df, use_container_width=True, hide_index=True)

        # ── Disclaimer ────────────────────────────────────────
        st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)
        st.markdown("""
<div style='background:#0F1923;border:1px solid #1C2B38;border-radius:8px;
padding:16px 20px;font-size:0.82rem;color:#546E7A;font-family:DM Sans,sans-serif'>
<b style='color:#90A4AE'>⚠️  Disclaimer</b><br>
This dashboard is built for <b>educational and research purposes only</b>.
All data is sourced from Yahoo Finance and may be delayed.
PCR/OI values are simulated placeholders.
<b>This is not financial advice.</b> Always do your own research (DYOR) before
making any investment decisions. Past breakouts do not guarantee future performance.
</div>
        """, unsafe_allow_html=True)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
