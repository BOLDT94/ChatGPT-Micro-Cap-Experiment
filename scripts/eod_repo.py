#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EOD logger (repo-driven)

- Läser senaste data/daily/prices_raw_YYYYMMDD.csv (fallback några dagar bakåt)
- Säkerställer close_sek
- Läser holdings/holdings_repo.csv (repo-”sanning” för Actions)
- Beräknar Marknadsvärde, CASH, Stop-loss
- Hämtar benchmark ACWI i SEK
- Upsert till eod/eod_log.csv (en rad per datum)
- Räknar Dag N + %-avkastning och %-avkastning vs benchmark

Krav i repo:
- holdings/holdings_repo.csv  (kolumner: Ticker,Antal,Inköpskurs,Stop-loss)
- scripts/fetch_prices.py     (valfritt men bra att köra före EOD i workflow)
"""

from __future__ import annotations
import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

PRICES_DIR      = "data/daily"
HOLDINGS_FILE   = "holdings/holdings_repo.csv"
EOD_FILE        = "eod/eod_log.csv"
PORTFOLIO_NAME  = "AI Portfölj 1"

# ---------- Helpers ----------

def _as_float(x):
    try:
        return float(str(x).replace(" ", "").replace(",", "."))
    except Exception:
        return np.nan

def _last_fx(symbol: str) -> float | None:
    """Senaste Close (daglig) via yfinance (t.ex. 'SEK=X', 'EURSEK=X')."""
    try:
        h = yf.download(symbol, period="2mo", interval="1d", auto_adjust=False, progress=False)
        col = "Close" if "Close" in h.columns else ("Adj Close" if "Adj Close" in h.columns else None)
        s = h[col].dropna() if col else None
        return float(s.iloc[-1]) if s is not None and not s.empty else None
    except Exception:
        return None

def _last_close(ticker: str) -> float | None:
    try:
        h = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
        col = "Close" if "Close" in h.columns else ("Adj Close" if "Adj Close" in h.columns else None)
        s = h[col].dropna() if col else None
        return float(s.iloc[-1]) if s is not None and not s.empty else None
    except Exception:
        return None

def ensure_close_sek(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "currency" not in out.columns:
        out["currency"] = "SEK"
    out["currency"] = out["currency"].fillna("SEK").astype(str).str.upper().str.strip()

    # redan beräknat?
    if "close_sek" in out.columns and out["close_sek"].notna().any():
        need = out["close_sek"].isna()
        if need.any():
            usdsek = _last_fx("SEK=X"); eursek = _last_fx("EURSEK=X")
            fx = {"USD": usdsek, "EUR": eursek}
            out.loc[need, "close_sek"] = out.loc[need].apply(
                lambda r: (r["close"] * fx.get(r["currency"], 1.0)) if pd.notna(r["close"]) else np.nan, axis=1
            )
        return out

    if "close" not in out.columns:
        raise RuntimeError("prices_raw saknar både 'close_sek' och 'close' – kan inte beräkna.")

    usdsek = _last_fx("SEK=X"); eursek = _last_fx("EURSEK=X")
    fx = {"USD": usdsek, "EUR": eursek}
    out["close_sek"] = out.apply(
        lambda r: (r["close"] * fx.get(r["currency"], 1.0)) if pd.notna(r["close"]) else np.nan, axis=1
    )
    return out

def load_latest_prices(max_back_days: int = 4) -> tuple[pd.DataFrame, date]:
    """Hitta senaste prices_raw_YYYYMMDD.csv i data/daily (idag -> bakåt)."""
    for k in range(max_back_days):
        d = date.today() - timedelta(days=k)
        p = os.path.join(PRICES_DIR, f"prices_raw_{d.strftime('%Y%m%d')}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = ensure_close_sek(df)
            return df, d
    raise FileNotFoundError("Hittade ingen prices_raw_* de senaste dagarna.")

def load_holdings_repo() -> pd.DataFrame:
    if not os.path.exists(HOLDINGS_FILE):
        raise FileNotFoundError(f"Saknar {HOLDINGS_FILE}")
    df = pd.read_csv(HOLDINGS_FILE)
    need = {"Ticker","Antal","Inköpskurs","Stop-loss"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{HOLDINGS_FILE} saknar kolumner: {miss}")

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    for c in ["Antal","Inköpskurs"]:
        df[c] = df[c].apply(_as_float)

    # Säkra CASH
    is_cash = df["Ticker"].str.upper().eq("CASH")
    if is_cash.any():
        df.loc[is_cash, "Inköpskurs"] = 1.0
        df.loc[is_cash, "Antal"] = df.loc[is_cash, "Antal"].fillna(0.0)

    return df

def stop_hit(row) -> bool:
    sl = str(row.get("Stop-loss","")).strip()
    price, buy = row.get("close_sek"), row.get("Inköpskurs")
    if pd.isna(price) or pd.isna(buy):
        return False
    if sl.endswith("%"):
        try:
            p = float(sl[:-1]) / 100.0
        except Exception:
            return False
        return price <= (1 - p) * buy
    try:
        return price <= float(sl)
    except Exception:
        return False

def upsert_eod(eod_path: str, row: dict) -> pd.DataFrame:
    cols = [
        "date","day_index","day_tag","portfolio_name",
        "cash_SEK","total_value_SEK",
        "benchmark_SEK","return_total_pct","return_vs_bm_pct",
        "notes"
    ]
    if os.path.exists(eod_path):
        eod = pd.read_csv(eod_path, parse_dates=["date"])
        for c in cols:
            if c not in eod.columns:
                eod[c] = np.nan
    else:
        os.makedirs(os.path.dirname(eod_path), exist_ok=True)
        eod = pd.DataFrame(columns=cols)

    mask = (eod["date"].dt.date == row["date"])
    if mask.any():
        idx = eod.index[mask][0]
        for k, v in row.items():
            eod.at[idx, k] = v
    else:
        eod = pd.concat([eod, pd.DataFrame([row])], ignore_index=True)

    # Re-räkna Dag N och avkastning
    eod = eod.sort_values("date").reset_index(drop=True)
    if not eod.empty:
        base_date = eod["date"].iloc[0].date()
        eod["day_index"] = eod["date"].dt.date.apply(lambda d: (d - base_date).days)
        eod["day_tag"] = eod.apply(lambda r: f"Dag {int(r['day_index'])} – {r['date'].date().isoformat()}", axis=1)

        base_port = float(eod["total_value_SEK"].iloc[0]) if pd.notna(eod["total_value_SEK"].iloc[0]) else np.nan
        base_bm = eod["benchmark_SEK"].dropna().iloc[0] if eod["benchmark_SEK"].notna().any() else np.nan

        def pct(a,b):
            try:
                if b and b != 0 and not np.isnan(a) and not np.isnan(b):
                    return (a/b - 1.0) * 100.0
            except Exception:
                pass
            return np.nan

        eod["return_total_pct"] = eod["total_value_SEK"].apply(lambda v: pct(v, base_port))
        if not np.isnan(base_bm):
            eod["return_vs_bm_pct"] = eod.apply(
                lambda r: (pct(r["total_value_SEK"], base_port) - pct(r["benchmark_SEK"], base_bm))
                if pd.notna(r["benchmark_SEK"]) else np.nan,
                axis=1
            )

    eod.to_csv(eod_path, index=False)
    return eod

# ---------- Main ----------

def main():
    prices_raw, asof = load_latest_prices()
    holdings = load_holdings_repo()

    # Merge holdings ↔ prices
    px = prices_raw.rename(columns={"ticker":"Ticker"}).copy()
    px["Ticker"] = px["Ticker"].astype(str).str.strip()
    holdings["Ticker"] = holdings["Ticker"].astype(str).str.strip()

    if "close_sek" in holdings.columns:
        holdings = holdings.drop(columns=["close_sek"])

    merged = holdings.merge(px[["Ticker","close_sek"]], on="Ticker", how="left")

    is_cash = merged["Ticker"].str.upper().eq("CASH")
    merged.loc[is_cash, "close_sek"] = 1.0
    merged["Marknadsvärde"] = (merged["Antal"] * merged["close_sek"]).fillna(0.0)
    merged["StopLossTriggad"] = merged.apply(stop_hit, axis=1)

    cash = float(merged.loc[is_cash, "Marknadsvärde"].sum()) if is_cash.any() else 0.0
    total = float(merged["Marknadsvärde"].sum())
    n_sl = int(merged["StopLossTriggad"].sum())

    # Benchmark (ACWI i SEK)
    # 1) försök med prices_raw om 'ACWI' finns
    bm = np.nan
    try:
        if "ACWI" in px["Ticker"].values:
            bm = float(px.loc[px["Ticker"]=="ACWI","close_sek"].iloc[0])
    except Exception:
        pass
    # 2) fallback yfinance
    if np.isnan(bm):
        acwi_usd = _last_close("ACWI")
        usdsek = _last_fx("SEK=X")
        if acwi_usd is not None and usdsek is not None:
            bm = acwi_usd * usdsek

    row = {
        "date": pd.to_datetime(asof),
        "portfolio_name": PORTFOLIO_NAME,
        "cash_SEK": cash,
        "total_value_SEK": total,
        "benchmark_SEK": bm,
        "notes": f"Auto (Actions) | Stop-loss triggar: {n_sl}",
    }

    eod = upsert_eod(EOD_FILE, row)

    print(f"✓ EOD {asof}  Cash={cash:.2f}  Total={total:.2f}  BM={'NaN' if np.isnan(bm) else round(bm,2)}")
    print(eod.tail().to_string(index=False))

if __name__ == "__main__":
    main()
