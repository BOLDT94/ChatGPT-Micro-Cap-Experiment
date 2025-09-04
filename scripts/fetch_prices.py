#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import yfinance as yf
import os
from datetime import date

# --- Nytt: läs watchlist från holdings/watchlist.csv ---
WATCHLIST_PATH = os.path.join("holdings", "watchlist.csv")

try:
    wl = pd.read_csv(WATCHLIST_PATH)
    tickers = wl["ticker"].dropna().astype(str).str.strip().tolist()
    print(f"✅ Läste watchlist från {WATCHLIST_PATH} ({len(tickers)} tickers)")
except Exception as e:
    print(f"⚠️ Kunde inte läsa {WATCHLIST_PATH}: {e}")
    # fallback lista om filen saknas eller är trasig
    tickers = ["ACWI"]

# --- funktion för att hämta prisdata ---
def fetch_prices(ticker: str):
    try:
        hist = yf.download(ticker, period="10d", interval="1d", progress=False)
        if hist.empty:
            print(f"Misslyckades: {ticker} – ingen data")
            return None

        close = hist["Close"].iloc[-1]
        return {
            "ticker": ticker,
            "close": float(close),
            "asof_date": date.today().isoformat(),
            "currency": "SEK" if ticker.endswith(".ST") else "USD",  # enkel heuristik
            "source": "yfinance"
        }
    except Exception as e:
        print(f"Misslyckades: {ticker} – {e}")
        return None

# --- kör på alla tickers ---
rows = []
for t in tickers:
    row = fetch_prices(t)
    if row:
        rows.append(row)

# --- skapa DataFrame ---
df = pd.DataFrame(rows)

# --- extra: beräkna close_sek (enkel konvertering, här bara dummy 1:1) ---
df["close_sek"] = df["close"]  # TODO: valutaomräkning vid behov

# --- skriv utfil ---
out_dir = "data/daily"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, f"prices_raw_{date.today().strftime('%Y%m%d')}.csv")
df.to_csv(out_file, index=False)

print(f"📁 Sparade {len(df)} rader till {out_file}")
