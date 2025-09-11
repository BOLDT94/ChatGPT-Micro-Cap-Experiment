#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weekly report builder for AI Portfölj 1

Producerar:
- reports/weekly_report_YYYY-WW.html  (mailvänlig HTML)
- reports/weekly_chart_YYYY-WW.png    (graf)

Input:
- eod/eod_log.csv                         (dagliga portföljvärden)
- eod/holdings/holdings_YYYYMMDD.csv      (dagliga snapshots, för förändringar)
- eod/watchlist/watchlist_log.csv         (alla watchlist-snapshots, för movers)

Beräknar:
- Veckans avkastning (portfölj & benchmark), sharpe/sortino, max drawdown
- Förändringar i holdings under veckan (nya/stängda/ändrade)
- Top 5 winners/losers (watchlist, % i SEK) för veckan
"""

from __future__ import annotations
import os, glob, math
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EOD_FILE        = "eod/eod_log.csv"
HOLD_SNAP_DIR   = "eod/holdings"
WL_LOG_FILE     = "eod/watchlist/watchlist_log.csv"
REPORTS_DIR     = "reports"

# ----- Helpers -----

def _read_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception:
        return pd.DataFrame()

def _fmt_pct(x):
    return "—" if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))) else f"{x:.2f}%"

def _fmt_money(x):
    try: return f"{float(x):,.0f}".replace(",", " ").replace("."," ")
    except: return str(x)

def _isocal(d: pd.Timestamp | date) -> tuple[int,int]:
    if isinstance(d, pd.Timestamp):
        y,w,_ = d.isocalendar()
        return int(y), int(w)
    y,w,_ = d.isocalendar()
    return int(y), int(w)

def _week_bounds(target: date | None = None):
    """Return (monday, friday) for the ISO-week containing target (default: today)."""
    if target is None:
        target = date.today()
    # ISO: Monday=1 ... Sunday=7
    wd = target.isoweekday()
    monday = target - timedelta(days=wd-1)
    friday = monday + timedelta(days=4)
    return monday, friday

def _weekly_slice(df: pd.DataFrame, monday: date, friday: date, col="date"):
    return df[(df[col].dt.date >= monday) & (df[col].dt.date <= friday)].copy()

def _drawdown(series: pd.Series) -> float:
    """Max drawdown in % for a value series."""
    if series.empty or series.isna().all():
        return float("nan")
    peak = -np.inf
    max_dd = 0.0
    for v in series:
        if pd.isna(v): 
            continue
        peak = max(peak, v)
        if peak > 0:
            dd = (v/peak - 1.0) * 100.0
            max_dd = min(max_dd, dd)
    return max_dd

def _sharpe_sortino(daily_values: pd.Series, risk_free_daily: float = 0.0) -> tuple[float,float]:
    """Sharpe & Sortino baserat på dagliga totalvärden (enkla log-returns)."""
    s = daily_values.dropna()
    if len(s) < 3:
        return float("nan"), float("nan")
    rets = np.log(s/s.shift(1)).dropna()
    if rets.empty:
        return float("nan"), float("nan")
    excess = rets - risk_free_daily
    sharpe = excess.mean() / (excess.std(ddof=1) + 1e-12) * np.sqrt(252)
    downside = excess[excess < 0]
    sortino = excess.mean() / (downside.std(ddof=1) + 1e-12) * np.sqrt(252)
    return float(sharpe), float(sortino)

def _make_chart(eod_week: pd.DataFrame, out_png: str, title: str):
    if eod_week.empty:
        return
    plt.figure(figsize=(10,5))
    plt.plot(eod_week["date"], eod_week["total_value_SEK"], marker="o", label="Portfölj (SEK)")
    if "benchmark_SEK" in eod_week.columns and eod_week["benchmark_SEK"].notna().any():
        plt.plot(eod_week["date"], eod_week["benchmark_SEK"], linestyle="--", label="Benchmark ACWI (SEK)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def _holdings_file_for(d: date) -> str | None:
    path = os.path.join(HOLD_SNAP_DIR, f"holdings_{d.strftime('%Y%m%d')}.csv")
    return path if os.path.exists(path) else None

# ----- Main -----

def main():
    today = date.today()
    monday, friday = _week_bounds(today)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1) Läs EOD och veckosnitt
    eod = _read_csv_safe(EOD_FILE, parse_dates=["date"]).sort_values("date")
    eod_week = _weekly_slice(eod, monday, friday, col="date")

    # robust: om fredag är helgdag/saknas, ta sista raden <= fredag
    if eod_week.empty and not eod.empty:
        eod_week = eod[eod["date"].dt.date <= friday].tail(5)

    latest = eod_week.iloc[-1] if not eod_week.empty else (eod.iloc[-1] if not eod.empty else None)
    base = eod_week.iloc[0] if not eod_week.empty else (eod.iloc[0] if not eod.empty else None)

    # nyckeltal vecka
    port_week = np.nan
    bm_week = np.nan
    rel_week = np.nan
    if latest is not None and base is not None:
        try:
            port_week = (latest["total_value_SEK"]/base["total_value_SEK"] - 1.0) * 100.0
        except Exception: pass
        try:
            if pd.notna(latest.get("benchmark_SEK")) and pd.notna(base.get("benchmark_SEK")) and base["benchmark_SEK"] != 0:
                bm_week = (latest["benchmark_SEK"]/base["benchmark_SEK"] - 1.0) * 100.0
        except Exception: pass
        if not (np.isnan(port_week) or np.isnan(bm_week)):
            rel_week = port_week - bm_week

    # drawdown & sharpe/sortino på veckans data (fallback till hela perioden om för få punkter)
    sref = eod_week if len(eod_week) >= 3 else eod
    dd = _drawdown(sref["total_value_SEK"]) if not sref.empty else float("nan")
    sharpe, sortino = _sharpe_sortino(sref["total_value_SEK"])

    # 2) Förändringar i holdings (jämför måndag vs fredag)
    hold_mon_file = _holdings_file_for(monday)
    hold_fri_file = _holdings_file_for(friday)
    # fallback: närmaste snapshot <= måndag resp <= fredag
    if hold_mon_file is None:
        snaps = sorted(glob.glob(os.path.join(HOLD_SNAP_DIR, "holdings_*.csv")))
        if snaps:
            for f in snaps:
                d = datetime.strptime(os.path.basename(f)[10:18], "%Y%m%d").date()
                if d <= monday: hold_mon_file = f
    if hold_fri_file is None:
        snaps = sorted(glob.glob(os.path.join(HOLD_SNAP_DIR, "holdings_*.csv")))
        if snaps:
            for f in snaps:
                d = datetime.strptime(os.path.basename(f)[10:18], "%Y%m%d").date()
                if d <= friday: hold_fri_file = f

    changes_new, changes_closed, changes_changed = [], [], []
    if hold_mon_file and hold_fri_file:
        h1 = _read_csv_safe(hold_mon_file)
        h2 = _read_csv_safe(hold_fri_file)
        if not h1.empty and not h2.empty and "Ticker" in h1.columns and "Ticker" in h2.columns:
            h1["Ticker"] = h1["Ticker"].astype(str).str.strip()
            h2["Ticker"] = h2["Ticker"].astype(str).str.strip()
            def qty(df): 
                return df.set_index("Ticker")["Antal"].apply(lambda x: float(str(x).replace(" ","").replace(",","."))).to_dict()
            q1 = qty(h1); q2 = qty(h2)
            s1 = set(q1.keys()); s2 = set(q2.keys())
            for t in sorted(s2 - s1):
                if t.upper() != "CASH":
                    changes_new.append(f"{t} ({q2[t]:g})")
            for t in sorted(s1 - s2):
                if t.upper() != "CASH":
                    changes_closed.append(f"{t} (0 från {q1[t]:g})")
            for t in sorted(s1 & s2):
                if t.upper() == "CASH": continue
                if abs(q1[t] - q2[t]) > 1e-9:
                    changes_changed.append(f"{t} ({q1[t]:g} → {q2[t]:g})")

    # 3) Top movers i watchlist för veckan
    wl = _read_csv_safe(WL_LOG_FILE, parse_dates=["date"])
    top_winners, top_losers = [], []
    if not wl.empty:
        wl.columns = [c.strip().lower() for c in wl.columns]
        wl = wl.dropna(subset=["ticker"])
        wweek = wl[(wl["date"].dt.date >= monday) & (wl["date"].dt.date <= friday)].copy()
        if wweek.empty:
            # fallback: närmaste <= fredag, senaste 7 dagar
            wweek = wl[wl["date"].dt.date <= friday].tail(7).copy()
        # ta första och sista pris per ticker i veckan
        movers = []
        for t, grp in wweek.groupby("ticker"):
            grp = grp.sort_values("date")
            p0 = grp["close_sek"].dropna().iloc[0] if "close_sek" in grp.columns and not grp["close_sek"].dropna().empty else np.nan
            p1 = grp["close_sek"].dropna().iloc[-1] if "close_sek" in grp.columns and not grp["close_sek"].dropna().empty else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 != 0:
                pct = (p1/p0 - 1.0) * 100.0
                movers.append((t, pct))
        have = [m for m in movers if pd.notna(m[1])]
        top_winners = sorted(have, key=lambda x: x[1], reverse=True)[:5]
        top_losers  = sorted(have, key=lambda x: x[1])[:5]

    # 4) Bygg graf
    y, w = _isocal(friday)
    title = f"Vecka {w} {y} – Portfölj vs Benchmark"
    chart_path = os.path.join(REPORTS_DIR, f"weekly_chart_{y}-{w:02d}.png")
    _make_chart(eod_week if not eod_week.empty else eod.tail(7), chart_path, title)

    # 5) Skriv HTML-rapport
    html = []
    html.append(f"<h2>AI Portfölj 1 – Veckorapport (v.{w} {y})</h2>")

    # sammanfattning
    html.append("<h3>Sammanfattning</h3><ul>")
    if latest is not None:
        html.append(f"<li><b>Portföljvärde (sista dagen):</b> {_fmt_money(latest.get('total_value_SEK'))} SEK</li>")
        html.append(f"<li><b>Cash:</b> {_fmt_money(latest.get('cash_SEK'))} SEK</li>")
    html.append(f"<li><b>Veckans avkastning:</b> {_fmt_pct(port_week)}</li>")
    html.append(f"<li><b>Benchmark (ACWI, SEK):</b> {_fmt_pct(bm_week)}</li>")
    html.append(f"<li><b>Relativt BM:</b> {_fmt_pct(rel_week)}</li>")
    html.append(f"<li><b>Sharpe (annual.):</b> {'—' if np.isnan(sharpe) else f'{sharpe:.2f}'}</li>")
    html.append(f"<li><b>Sortino (annual.):</b> {'—' if np.isnan(sortino) else f'{sortino:.2f}'}</li>")
    html.append(f"<li><b>Max drawdown (i perioden):</b> {_fmt_pct(dd)}</li>")
    html.append("</ul>")

    if os.path.exists(chart_path):
        html.append(f'<p><img src="cid:weeklychart" alt="Weekly chart" style="max-width: 900px;"/></p>')

    # förändringar
    html.append("<h3>Förändringar i holdings under veckan</h3><ul>")
    html.append(f"<li>Nya innehav: {', '.join(changes_new) if changes_new else 'Inga'}</li>")
    html.append(f"<li>Stängda innehav: {', '.join(changes_closed) if changes_closed else 'Inga'}</li>")
    html.append(f"<li>Ändrade antal: {', '.join(changes_changed) if changes_changed else 'Inga'}</li>")
    html.append("</ul>")

    # movers
    def _fmt_mv(lst):
        return "—" if not lst else ", ".join([f"{t} {p:+.1f}%" for t,p in lst])

    html.append("<h3>Top movers (watchlist, vecka)</h3>")
    html.append(f"<p><b>Winners:</b> {_fmt_mv(top_winners)}<br/>"
                f"<b>Losers:</b>  {_fmt_mv(top_losers)}</p>")

    # anteckningshook (för framtida AI-kommentar)
    html.append("<h3>Kommentar</h3>")
    html.append("<p>Kort veckokommentar: marknadsläge, risk, och fokus till nästa vecka.</p>")

    out_html = os.path.join(REPORTS_DIR, f"weekly_report_{y}-{w:02d}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"✓ Veckorapport: {out_html}")
    if os.path.exists(chart_path):
        print(f"✓ Graf: {chart_path}")

if __name__ == "__main__":
    main()
