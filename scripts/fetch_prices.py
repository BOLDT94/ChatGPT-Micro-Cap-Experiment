#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import date
import pandas as pd
import yfinance as yf

WATCHLIST_PATH = os.path.join("holdings", "watchlist.csv")

def read_watchlist(path: str):
    try:
        wl = pd.read_csv(path)
        tickers = wl["ticker"].dropna().astype(str).str.strip().tolist()
        print(f"✅ Läste watchlist från {path} ({len(tickers)} tickers)")
        if not tickers:
            raise ValueError("watchlist.csv är tom.")
        return tickers
    except Exception as e:
        print(f"⚠️ Kunde inte läsa {path}: {e} — fallback till ['ACWI']")
        return ["ACWI"]

def last_fx(symbol: str) -> float | None:
    """Hämtar senaste Close för ett FX-par via yfinance."""
    try:
        h = yf.download(symbol, period="2mo", interval="1d", auto_adjust=False, progress=False)
        if h is None or h.empty:
            return None
        col = "Close" if "Close" in h.columns else ("Adj Close" if "Adj Close" in h.columns else None)
        s = h[col].dropna() if col else None
        return float(s.iloc[-1]) if s is not None and not s.empty else None
    except Exception:
        return None

def detect_currency(ticker: str) -> str:
    """Försök få valuta från yfinance info, annars heuristik."""
    try:
        info = yf.Ticker(ticker).fast_info  # snabbare än .info
        cur = getattr(info, "currency", None) or getattr(info, "last_price", None)  # dummy access
        cur = yf.Ticker(ticker).fast_info.currency
        if cur:
            return str(cur).upper()
    except Exception:
        pass
    # Heuristik
    if ticker.endswith(".ST"):
        return "SEK"
    return "USD"

def fetch_close(ticker: str):
    """Hämtar senaste stängningspris och currency."""
    try:
        h = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
        if h is None or h.empty:
            print(f"Misslyckades: {ticker} – ingen data.")
            return None
        col = "Close" if "Close" in h.columns else ("Adj Close" if "Adj Close" in h.columns else None)
        s = h[col].dropna() if col else None
        if s is None or s.empty:
            print(f"Misslyckades: {ticker} – saknar Close.")
            return None
        close = float(s.iloc[-1])
        currency = detect_currency(ticker)
        return close, currency
    except Exception as e:
        print(f"Misslyckades: {ticker} – {e}")
        return None

def to_sek(price: float, currency: str, fx: dict[str, float | None]) -> float:
    c = (currency or "SEK").upper()
    if c == "SEK":
        return price
    if c == "USD" and fx.get("USDSEK"):
        return price * fx["USDSEK"]
    if c == "EUR" and fx.get("EURSEK"):
        return price * fx["EURSEK"]
    # Fallback: lämna priset oförändrat om vi saknar FX (hellre data än fail)
    return price

def main():
    tickers = read_watchlist(WATCHLIST_PATH)

    # FX (hämtas en gång per körning)
    usdsek = last_fx("SEK=X")        # USD/SEK
    eursek = last_fx("EURSEK=X")     # EUR/SEK
    fx = {"USDSEK": usdsek, "EURSEK": eursek}
    print(f"FX USDSEK={usdsek}  EURSEK={eursek}")

    rows = []
    for t in tickers:
        res = fetch_close(t)
        if not res:
            continue
        close, currency = res
        close_sek = to_sek(close, currency, fx)
        rows.append({
            "ticker": t,
            "close": close,
            "currency": currency,
            "asof_date": date.today().isoformat(),
            "source": "yahoo",
            "close_sek": close_sek
        })

    df = pd.DataFrame(rows)
    out_dir = "data/daily"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"prices_raw_{date.today().strftime('%Y%m%d')}.csv")
    df.to_csv(out_file, index=False)
    print(f"📁 Sparade {len(df)} rader till {out_file}")

if __name__ == "__main__":
    main()
