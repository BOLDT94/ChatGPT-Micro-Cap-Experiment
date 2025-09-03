import os
import sys
import io
import datetime as dt
from typing import Optional, Tuple, List

import pandas as pd
import requests
import yfinance as yf

WATCH = "holdings/watchlist.csv"
OUTDIR = "data/daily"
os.makedirs(OUTDIR, exist_ok=True)

TODAY = dt.date.today()
OUTPATH = f"{OUTDIR}/prices_raw_{TODAY.strftime('%Y%m%d')}.csv"

# --------------------- Helpers ---------------------

def to_stooq_symbol(yahoo_symbol: str) -> Optional[str]:
    """
    Konvertera Yahoo (.ST) till Stooq (.ST -> .st, bindestreck små bokstäver).
    Ex: 'EMBRAC-B.ST' -> 'embrac-b.st', 'INTRUM.ST' -> 'intrum.st'
    Återvänder None om symbolen inte ser svensk ut.
    """
    if not isinstance(yahoo_symbol, str):
        return None
    s = yahoo_symbol.strip()
    if s.endswith(".ST"):
        base = s[:-3]
        return base.lower() + ".st"
    # Utländska ETF:er etc (ACWI) lämnas
    return None

def stooq_last_close(symbol: str) -> Optional[Tuple[float, dt.date]]:
    """
    Hämta sista giltiga close från Stooq historik-CSV.
    Endpoint-format: https://stooq.com/q/d/l/?s=<symbol>&i=d
    Returnerar (pris, datum) eller None.
    """
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200 or len(r.text) < 10:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        # Stooq kolumner: Date,Open,High,Low,Close,Volume
        if "Close" not in df.columns or "Date" not in df.columns or df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"])
        if df.empty:
            return None
        last = df.iloc[-1]
        return float(last["Close"]), last["Date"].date()
    except Exception:
        return None

def yahoo_last_close(symbol: str, periods: List[str] = ["10d","1mo","6mo"]) -> Optional[Tuple[float, dt.date]]:
    """
    Hämtar sista giltiga Close (eller Adj Close) via yfinance, med fallback-perioder.
    """
    def pick_close(df: pd.DataFrame) -> Optional[Tuple[float, dt.date]]:
        if df is None or df.empty:
            return None
        df = df.sort_index()
        col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
        if col is None:
            return None
        s = df[col].dropna()
        if s.empty:
            return None
        return float(s.iloc[-1]), s.index[-1].date()

    for per in periods:
        try:
            h = yf.download(symbol, period=per, interval="1d", auto_adjust=False, progress=False)
            got = pick_close(h)
            if got:
                return got
        except Exception:
            pass
    return None

def fx_yahoo(symbol: str) -> Optional[float]:
    """
    Hämtar sista giltiga Close för FX via yfinance (SEK=X, EURSEK=X).
    """
    got = yahoo_last_close(symbol, periods=["10d","1mo","6mo"])
    return got[0] if got else None

# --------------------- Körning ---------------------

# Läs watchlist
try:
    wl = pd.read_csv(WATCH)
except Exception as e:
    print("ERROR: kunde inte läsa", WATCH, e); sys.exit(0)

if "ticker" not in wl.columns:
    print("ERROR: 'ticker' kolumn saknas i watchlist.csv"); sys.exit(0)

wl["currency"] = wl.get("currency", pd.Series(["SEK"]*len(wl))).fillna("SEK")

tickers = [t for t in wl["ticker"].dropna().astype(str).str.strip().tolist() if t]
rows, stooq_hits, yahoo_hits, fails = [], 0, 0, 0

for ysym in tickers:
    price, asof = None, None

    # 1) Försök Stooq
    stsym = to_stooq_symbol(ysym)
    if stsym:
        got = stooq_last_close(stsym)
        if got:
            price, asof = got
            stooq_hits += 1

    # 2) Fallback – Yahoo
    if price is None:
        got = yahoo_last_close(ysym)
        if got:
            price, asof = got
            yahoo_hits += 1
        else:
            fails += 1
            print(f"VARNING: ingen close-data för {ysym} (Stooq+Yahoo)")

    rows.append({"ticker": ysym, "close": price, "asof_date": (asof.isoformat() if asof else None)})

print(f"DEBUG: Stooq OK={stooq_hits}, Yahoo OK={yahoo_hits}, Fails={fails}")

# DataFrame även om allt misslyckas
df = pd.DataFrame(rows)

# FX via Yahoo (stabilt för valutapar)
usdsek = fx_yahoo("SEK=X")      # USD/SEK
eursek = fx_yahoo("EURSEK=X")   # EUR/SEK
print(f"DEBUG FX: USD/SEK={usdsek}, EUR/SEK={eursek}")

# Merge valuta och SEK-konvertera
df = df.merge(wl[["ticker","currency"]], on="ticker", how="left")

def to_sek(px, ccy):
    if pd.isna(px):
        return None
    if ccy == "USD" and usdsek:
        return px * usdsek
    if ccy == "EUR" and eursek:
        return px * eursek
    return px  # SEK eller okänt

df["close_sek"] = [to_sek(px, c) for px, c in zip(df["close"], df["currency"])]
df["source"] = ["stooq" if (to_stooq_symbol(t) and not pd.isna(px)) else ("yfinance" if not pd.isna(px) else "none")
                for t, px in zip(df["ticker"], df["close"])]

df.to_csv(OUTPATH, index=False, encoding="utf-8")
print("OK:", OUTPATH, "rader:", len(df))
