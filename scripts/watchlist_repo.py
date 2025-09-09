#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io
from datetime import date, timedelta
import numpy as np
import pandas as pd

PRICES_DIR       = "data/daily"
WATCHLIST_FILE   = "holdings/watchlist.csv"
HOLDINGS_FILE    = "holdings/holdings_repo.csv"   # för in_portfolio-flagga
OUT_DIR          = "eod/watchlist"
MASTER_LOG       = os.path.join(OUT_DIR, "watchlist_log.csv")

def _load_latest_prices(max_back_days: int = 4) -> tuple[pd.DataFrame, date]:
    for k in range(max_back_days):
        d = date.today() - timedelta(days=k)
        path = os.path.join(PRICES_DIR, f"prices_raw_{d.strftime('%Y%m%d')}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            # för säkerhets skull: normalisera kolumnnamn
            df.columns = [c.strip().lower() for c in df.columns]
            return df, d
    raise FileNotFoundError("Hittar ingen prices_raw_* de senaste dagarna.")

def _read_watchlist() -> pd.DataFrame:
    if not os.path.exists(WATCHLIST_FILE):
        raise FileNotFoundError(f"Saknar {WATCHLIST_FILE}")
    wl = pd.read_csv(WATCHLIST_FILE)
    wl.columns = [c.strip().lower() for c in wl.columns]
    if "ticker" not in wl.columns:
        raise ValueError("watchlist.csv saknar kolumnen 'ticker'.")
    wl = wl[["ticker"]].dropna()
    wl["ticker"] = wl["ticker"].astype(str).str.strip()
    return wl

def _read_holdings() -> set[str]:
    if not os.path.exists(HOLDINGS_FILE):
        return set()
    h = pd.read_csv(HOLDINGS_FILE)
    if "Ticker" not in h.columns:
        return set()
    tickers = (
        h["Ticker"].astype(str).str.strip().str.upper()
        .loc[lambda s: s.ne("CASH")]  # exkludera CASH
        .tolist()
    )
    return set(tickers)

def _safe_read_master(path: str, parse_dates: list[str] = None) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception:
        return pd.DataFrame()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    prices, pdate = _load_latest_prices()
    wl = _read_watchlist()
    owned = _read_holdings()

    # normalisera kolumner från prices
    # minsta vi behöver: ticker, close, currency, close_sek, asof_date, source
    # men vi skyddar oss mot saknade kolumner
    for col in ["ticker","close","currency","close_sek","asof_date","source"]:
        if col not in prices.columns:
            prices[col] = np.nan

    prices["ticker"] = prices["ticker"].astype(str).str.strip()
    snap = wl.merge(
        prices[["ticker","close","currency","close_sek","asof_date","source"]],
        on="ticker", how="left"
    )

    # flagga om tickern finns i holdings_repo
    snap["in_portfolio"] = snap["ticker"].astype(str).str.upper().isin(owned)
    snap["date"] = pdate.isoformat()

    out_daily = os.path.join(OUT_DIR, f"watchlist_eod_{pdate.strftime('%Y%m%d')}.csv")
    cols = ["date","ticker","close","currency","close_sek","asof_date","source","in_portfolio"]

    snap[cols].to_csv(out_daily, index=False, encoding="utf-8")
    print(f"✓ Skrev daglig snapshot: {out_daily}  ({len(snap)} rader)")

    # uppdatera master-logg (idempotent per date+ticker)
    log = _safe_read_master(MASTER_LOG, parse_dates=["date"])
    if not log.empty:
        # ta bort ev. duplikat för samma (date,ticker) innan vi lägger till
        key = ["date","ticker"]
        log = log[~log.set_index(key).index.isin(snap[cols].set_index(key).index)].copy()
        log = pd.concat([log, snap[cols]], ignore_index=True)
    else:
        log = snap[cols].copy()

    log.to_csv(MASTER_LOG, index=False, encoding="utf-8")
    print(f"✓ Uppdaterade masterlogg: {MASTER_LOG}  (nu {len(log)} rader)")

if __name__ == "__main__":
    main()
