#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily report builder for AI Portfölj 1 (final)

Producerar:
- reports/daily_report_YYYYMMDD.html  (mailvänlig HTML med inline PNG)
- reports/daily_chart_YYYYMMDD.png    (graf)
- eod/holdings/holdings_YYYYMMDD.csv  (daglig holdings-snapshot för diff)

Indata (från workflows):
- eod/eod_log.csv
- holdings/holdings_repo.csv
- eod/watchlist/watchlist_eod_YYYYMMDD.csv   (från watchlist_repo.py)
- eod/watchlist/watchlist_log.csv            (valfri, för movers)

Dessutom:
- data/daily/prices_raw_YYYYMMDD.csv (senaste filen, letar bakåt 7 dagar)
"""

from __future__ import annotations
import os, glob, io, base64
from datetime import date, timedelta, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

# --- Paths ---
EOD_FILE        = "eod/eod_log.csv"
HOLDINGS_FILE   = "holdings/holdings_repo.csv"
PRICES_DIR      = "data/daily"
WL_DIR          = "eod/watchlist"
REPORTS_DIR     = "reports"
HOLD_SNAP_DIR   = "eod/holdings"

# --- Utils ---
def _read_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception:
        return pd.DataFrame()

def _as_float(x):
    try: return float(str(x).replace(" ", "").replace(",", "."))
    except: return np.nan

def _fmt_pct(x):
    return "—" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.2f}%"

def _fmt_money(x):
    try: return f"{float(x):,.0f}".replace(",", " ").replace("."," ")
    except: return str(x)

def _last_fx(symbol: str, tries: int = 2) -> float | None:
    """Senaste FX (daglig Close) med enkel retry."""
    for i in range(tries):
        try:
            h = yf.download(symbol, period="2mo", interval="1d", auto_adjust=False, progress=False)
            col = "Close" if "Close" in h.columns else ("Adj Close" if "Adj Close" in h.columns else None)
            s = h[col].dropna() if col else None
            if s is not None and not s.empty:
                return float(s.iloc[-1])
        except Exception:
            if i + 1 == tries:
                return None
    return None

def _ensure_close_sek(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    if "currency" not in out.columns:
        out["currency"] = "SEK"
    out["currency"] = out["currency"].fillna("SEK").astype(str).str.upper().str.strip()

    if "close_sek" in out.columns and out["close_sek"].notna().any():
        need = out["close_sek"].isna()
        if need.any():
            usdsek = _last_fx("SEK=X"); eursek = _last_fx("EURSEK=X")
            fx = {"USD": usdsek, "EUR": eursek}
            out.loc[need, "close_sek"] = out.loc[need].apply(
                lambda r: (r.get("close") * fx.get(r.get("currency"), 1.0)) if pd.notna(r.get("close")) else np.nan,
                axis=1
            )
        return out

    if "close" not in out.columns:
        raise RuntimeError("prices_raw saknar 'close' och 'close_sek'.")

    usdsek = _last_fx("SEK=X"); eursek = _last_fx("EURSEK=X")
    fx = {"USD": usdsek, "EUR": eursek}
    out["close_sek"] = out.apply(
        lambda r: (r.get("close") * fx.get(r.get("currency"), 1.0)) if pd.notna(r.get("close")) else np.nan,
        axis=1
    )
    return out

def _load_latest_prices(max_back_days: int = 7) -> tuple[pd.DataFrame, date]:
    for k in range(max_back_days):
        d = date.today() - timedelta(days=k)
        p = os.path.join(PRICES_DIR, f"prices_raw_{d.strftime('%Y%m%d')}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = _ensure_close_sek(df)
            return df, d
    # no prices, return empty with columns
    return pd.DataFrame(columns=["ticker","close","currency","close_sek"]), date.today()

def _stop_hit(price_sek, buy, sl_str) -> bool:
    sl = str(sl_str or "").strip()
    if pd.isna(price_sek) or pd.isna(buy): return False
    if sl.endswith("%"):
        try: p = float(sl[:-1]) / 100.0
        except: return False
        return price_sek <= (1 - p) * buy
    try:
        return price_sek <= float(sl)
    except:
        return False

def _sl_distance_pct(price_sek, buy, sl_str) -> float | None:
    sl = str(sl_str or "").strip()
    if pd.isna(price_sek) or pd.isna(buy): return np.nan
    try:
        if sl.endswith("%"):
            p = float(sl[:-1]) / 100.0
            lvl = (1 - p) * buy
            return (price_sek / lvl - 1.0) * 100.0
        lvl = float(sl)
        return (price_sek / lvl - 1.0) * 100.0
    except:
        return np.nan

def _build_chart(eod: pd.DataFrame, out_png: str):
    if eod.empty: return
    df = eod.sort_values("date")
    plt.figure(figsize=(10,5))
    plt.plot(df["date"], df["total_value_SEK"], marker="o", label="Portfölj (SEK)")
    if "benchmark_SEK" in df.columns and df["benchmark_SEK"].notna().any():
        plt.plot(df["date"], df["benchmark_SEK"], linestyle="--", label="Benchmark ACWI (SEK)")
    plt.title("EOD – AI Portfölj 1")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def _holdings_snapshot_path(d: date) -> str:
    return os.path.join(HOLD_SNAP_DIR, f"holdings_{d.strftime('%Y%m%d')}.csv")

def _prev_snapshot_path(asof: date) -> str | None:
    """Hitta senaste snapshot < asof."""
    if not os.path.exists(HOLD_SNAP_DIR):
        return None
    files = sorted(glob.glob(os.path.join(HOLD_SNAP_DIR, "holdings_*.csv")))
    if not files: return None
    ymd_asof = int(asof.strftime("%Y%m%d"))
    prev = [f for f in files if int(os.path.basename(f)[10:18]) < ymd_asof]
    return prev[-1] if prev else None

def main():
    today = date.today()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(HOLD_SNAP_DIR, exist_ok=True)

    # --- Läs EOD & holdings ---
    eod = _read_csv_safe(EOD_FILE, parse_dates=["date"]).sort_values("date")
    latest = eod.iloc[-1] if not eod.empty else None

    holdings = _read_csv_safe(HOLDINGS_FILE)
    if not holdings.empty:
        for c in ["Antal","Inköpskurs"]:
            if c in holdings.columns:
                holdings[c] = holdings[c].apply(_as_float)
        holdings["Ticker"] = holdings["Ticker"].astype(str).str.strip()
    else:
        holdings = pd.DataFrame(columns=["Ticker","Antal","Inköpskurs","Stop-loss"])

    # --- Priser (för merge & SL) ---
    prices_raw, prices_asof = _load_latest_prices(max_back_days=7)
    px = prices_raw.copy()
    px.columns = [c.strip().lower() for c in px.columns]
    if "ticker" not in px.columns:
        px["ticker"] = ""
    px["ticker"] = px["ticker"].astype(str).str.strip()

    # Merge holdings ↔ prices to get close_sek
    merged = holdings.copy()
    if "close_sek" in merged.columns:
        merged = merged.drop(columns=["close_sek"])
    merged = merged.merge(
        px[["ticker","close_sek"]].rename(columns={"ticker":"Ticker"}),
        on="Ticker", how="left"
    )
    # CASH = 1 SEK
    is_cash = merged["Ticker"].astype(str).str.upper().eq("CASH")
    if is_cash.any():
        merged.loc[is_cash, "close_sek"] = 1.0
    merged["Marknadsvärde"] = (merged.get("Antal",0) * merged.get("close_sek",np.nan)).fillna(0.0)

    # SL-status & distans
    merged["StopLossTriggad"] = merged.apply(
        lambda r: _stop_hit(r.get("close_sek"), r.get("Inköpskurs"), r.get("Stop-loss")), axis=1
    )
    merged["SL_dist_pct"] = merged.apply(
        lambda r: _sl_distance_pct(r.get("close_sek"), r.get("Inköpskurs"), r.get("Stop-loss")), axis=1
    )

    # --- Spara holdings snapshot på PRISDAGEN & ladda föregående snapshot ---
    snap_today_path = _holdings_snapshot_path(prices_asof)
    merged[["Ticker","Antal","Inköpskurs","Stop-loss","close_sek","Marknadsvärde"]].to_csv(snap_today_path, index=False)
    snap_prev_path = _prev_snapshot_path(prices_asof)
    prev = _read_csv_safe(snap_prev_path) if snap_prev_path else pd.DataFrame()

    # --- Förändringar mot föregående snapshot ---
    changes_new, changes_closed, changes_changed = [], [], []
    if not prev.empty:
        prev["Ticker"] = prev["Ticker"].astype(str).str.strip()
        cur = merged[["Ticker","Antal"]].copy()
        cur["Antal"] = cur["Antal"].apply(_as_float)
        prev2 = prev[["Ticker","Antal"]].copy()
        prev2["Antal"] = prev2["Antal"].apply(_as_float)

        prev_set = set(prev2["Ticker"].tolist())
        cur_set  = set(cur["Ticker"].tolist())

        # nya (ej CASH)
        for t in sorted(cur_set - prev_set):
            if t.upper() != "CASH":
                qty = float(cur.loc[cur["Ticker"]==t,"Antal"].iloc[0])
                changes_new.append(f"{t} ({qty:g})")

        # stängda (ej CASH)
        for t in sorted(prev_set - cur_set):
            if t.upper() != "CASH":
                qty = float(prev2.loc[prev2["Ticker"]==t,"Antal"].iloc[0])
                changes_closed.append(f"{t} (0 från {qty:g})")

        # ändringar
        both = cur_set & prev_set
        for t in sorted(both):
            if t.upper() == "CASH": continue
            q1 = float(prev2.loc[prev2["Ticker"]==t,"Antal"].iloc[0])
            q2 = float(cur.loc[cur["Ticker"]==t,"Antal"].iloc[0])
            if abs(q1 - q2) > 1e-9:
                changes_changed.append(f"{t} ({q1:g} → {q2:g})")

    # --- Watchlist: snapshot & movers ---
    wl_snap_path = os.path.join(WL_DIR, f"watchlist_eod_{prices_asof.strftime('%Y%m%d')}.csv")
    wl_snap = _read_csv_safe(wl_snap_path)
    wl_log  = _read_csv_safe(os.path.join(WL_DIR, "watchlist_log.csv"), parse_dates=["date"])

    movers = []
    top_winners, top_losers = [], []
    risk_flags = []
    if not wl_snap.empty:
        wl_snap.columns = [c.strip().lower() for c in wl_snap.columns]
        wl_snap["ticker"] = wl_snap["ticker"].astype(str).str.strip()
        wl_snap["close_sek"] = wl_snap.get("close_sek")

        # bygg prev price per ticker från loggen (< prices_asof)
        prev_map = {}
        if not wl_log.empty:
            wf = wl_log.copy()
            wf.columns = [c.strip().lower() for c in wf.columns]
            wf["ticker"] = wf["ticker"].astype(str).str.strip()
            for t in wl_snap["ticker"]:
                sub = wf[(wf["ticker"]==t) & (wf["date"].dt.date < prices_asof)]
                if not sub.empty and "close_sek" in sub.columns:
                    prev_map[t] = float(sub.sort_values("date").iloc[-1]["close_sek"])

        # beräkna daglig %förändring
        for _, r in wl_snap.iterrows():
            t = r["ticker"]
            curp = r.get("close_sek", np.nan)
            prevp = prev_map.get(t, np.nan)
            pct = np.nan
            if pd.notna(curp) and pd.notna(prevp) and prevp != 0:
                pct = (curp/prevp - 1.0) * 100.0
            movers.append((t, pct, curp, bool(r.get("in_portfolio", False))))

        have_pct = [(t,p,c,inp) for (t,p,c,inp) in movers if pd.notna(p)]
        top_winners = sorted(have_pct, key=lambda x: x[1], reverse=True)[:3]
        top_losers  = sorted(have_pct, key=lambda x: x[1])[:3]

        # riskflaggor: ±8% + SL-nära
        for (t,p,c,inp) in have_pct:
            if p >= 8.0 or p <= -8.0:
                risk_flags.append(f"{t} ({p:+.1f}%)")
        near = merged[(~merged["Ticker"].str.upper().eq("CASH")) & (merged["SL_dist_pct"].notna())]
        near = near[near["SL_dist_pct"] <= 3.0]
        for _, rr in near.iterrows():
            risk_flags.append(f"{rr['Ticker']} (SL nära: {rr['SL_dist_pct']:.1f}%)")

    # --- Global drawdown (från Dag 0) ---
    global_dd = np.nan
    if not eod.empty:
        eod_sorted = eod.sort_values("date")
        base = eod_sorted["total_value_SEK"].iloc[0]
        last = eod_sorted["total_value_SEK"].iloc[-1]
        if base and not pd.isna(base):
            global_dd = (last/base - 1.0) * 100.0

    # --- Orderförslag (max 3) ---
    orders = []
    # 1) SELL – SL triggade
    for _, r in merged.iterrows():
        if bool(r.get("StopLossTriggad", False)) and str(r["Ticker"]).upper() != "CASH":
            entry = r.get("close_sek")
            slval = r.get("Stop-loss")
            if pd.notna(entry):
                orders.append(
                    f"[ACTION=SELL] TICKER={r['Ticker']} SIZE_%=ALL ENTRY={entry:.2f} SL={slval} TP=— ORDER_TYPE=MOO REASON=Stop-loss triggad"
                )

    # 2) BUY – top winners ej i portföljen, om drawdown ≤ 15% och tillräcklig cash
    can_buy = (pd.isna(global_dd) or global_dd >= -15.0)
    if can_buy and top_winners:
        owned = set(merged[~merged["Ticker"].str.upper().eq("CASH")]["Ticker"].astype(str).str.upper().tolist())
        cash_val = float(merged.loc[merged["Ticker"].str.upper()=="CASH","Marknadsvärde"].sum()) if "Marknadsvärde" in merged.columns else 0.0
        total_val = float(merged["Marknadsvärde"].sum()) if "Marknadsvärde" in merged.columns else 0.0
        remaining_slots = max(0, 3 - len(orders))
        for (t,pct,cur_price,inp) in top_winners:
            if remaining_slots <= 0: break
            if t.upper() in owned: continue
            if not pd.notna(cur_price) or cur_price <= 0: continue
            if total_val <= 0: continue

            size_pct = 5.0
            size_pct = min(10.0, max(2.0, size_pct))
            alloc = total_val * size_pct/100.0

            # Min-cash guard: kräv att cash ≥ 2× allokering (marginal för nästa trade & friktion)
            if alloc * 2.0 > cash_val:
                continue

            sl_val = round(cur_price * 0.92, 2)
            orders.append(
                f"[ACTION=BUY] TICKER={t} SIZE_%={size_pct:.1f} ENTRY={cur_price:.2f} SL={sl_val} TP=— ORDER_TYPE=MOO REASON=Momentum & ej i portföljen"
            )
            remaining_slots -= 1
    elif not can_buy:
        orders.append("[ACTION=HOLD] TICKER=— SIZE_%=— ENTRY=— SL=— TP=— ORDER_TYPE=— REASON=Global drawdown > 15%, inga nya köp")

    orders = orders[:3]

    # --- Sammanfattning ---
    day_tag = latest["day_tag"] if (latest is not None and "day_tag" in latest) else f"Dag – {today.isoformat()}"
    total_value = latest.get("total_value_SEK") if latest is not None else np.nan
    cash_value  = latest.get("cash_SEK") if latest is not None else np.nan
    ret_total   = latest.get("return_total_pct") if latest is not None else np.nan
    ret_vs_bm   = latest.get("return_vs_bm_pct") if latest is not None else np.nan

    # --- Graf (+ inline i HTML) ---
    chart_path = os.path.join(REPORTS_DIR, f"daily_chart_{today.strftime('%Y%m%d')}.png")
    if not eod.empty:
        _build_chart(eod, chart_path)

    # --- HTML-rapport ---
    html = []
    # Preheader / banner om data saknas
    if px.empty or wl_snap.empty:
        miss = []
        if px.empty: miss.append("prices_raw")
        if wl_snap.empty: miss.append("watchlist-snapshot")
        html.append(f"<p style='color:#b7410e'><b>⚠️ Begränsad data idag:</b> {', '.join(miss)} saknas.</p>")

    html.append(f"<h2>AI Portfölj 1 – {day_tag}</h2>")
    html.append("<h3>Sammanfattning</h3>")
    html.append("<ul>")
    html.append(f"<li><b>Portföljvärde:</b> {_fmt_money(total_value)} SEK</li>")
    html.append(f"<li><b>Cash:</b> {_fmt_money(cash_value)} SEK</li>")
    html.append(f"<li><b>Avkastning (totalt):</b> {_fmt_pct(ret_total)}</li>")
    html.append(f"<li><b>Relativt benchmark:</b> {_fmt_pct(ret_vs_bm)}</li>")
    if not pd.isna(global_dd):
        html.append(f"<li><b>Drawdown från Dag 0:</b> {_fmt_pct(global_dd)}</li>")
    html.append("</ul>")

    # Inline PNG (base64) istället för CID
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        html.append(f'<p><img alt="Daily chart" style="max-width: 900px;" src="data:image/png;base64,{b64}"/></p>')

    # Förändringar
    html.append("<h3>Förändringar sedan förra handelsdagen</h3><ul>")
    html.append(f"<li>Nya innehav: {', '.join(changes_new) if changes_new else 'Inga'}</li>")
    html.append(f"<li>Stängda innehav: {', '.join(changes_closed) if changes_closed else 'Inga'}</li>")
    html.append(f"<li>Ändrade antal: {', '.join(changes_changed) if changes_changed else 'Inga'}</li>")
    sls = merged[(merged["StopLossTriggad"]) & (~merged["Ticker"].str.upper().eq("CASH"))]["Ticker"].tolist()
    html.append(f"<li>SL triggade: {', '.join(sls) if sls else 'Inga'}</li>")
    html.append("</ul>")

    # Top movers & risk
    def _fmt_mv(lst):
        if not lst: return "—"
        return ", ".join([f"{t} {p:+.1f}%" for (t,p,_,_) in lst])

    html.append("<h3>Top movers (watchlist, SEK)</h3>")
    html.append(f"<p><b>Winners:</b> {_fmt_mv(top_winners)}<br/>"
                f"<b>Losers:</b> {_fmt_mv(top_losers)}<br/>"
                f"<b>Riskflaggor:</b> {', '.join(risk_flags) if risk_flags else '—'}</p>")

    # Orders
    html.append("<h3>Rekommenderade orders (max 3)</h3>")
    if orders:
        html.append("<pre style='background:#f6f8fa;padding:10px;border-radius:6px;'>")
        for o in orders:
            html.append(o)
        html.append("</pre>")
    else:
        html.append("<p>Inga orderförslag idag.</p>")

    # Holdings
    if not merged.empty:
        view = merged.copy()
        view = view[["Ticker","Antal","Inköpskurs","close_sek","Stop-loss","Marknadsvärde"]]
        view["__ord__"] = view["Ticker"].astype(str).str.upper().eq("CASH").astype(int)
        view = view.sort_values(["__ord__","Ticker"]).drop(columns="__ord__")
        html.append("<h3>Holdings</h3>")
        html.append(view.to_html(index=False))

    # Noteringar
    html.append("<h3>Noteringar</h3><ul>")
    if latest is not None and isinstance(latest.get("notes",""), str) and latest["notes"]:
        html.append(f"<li>{latest['notes']}</li>")
    dq = []
    if px.empty: dq.append("Saknade prices_raw (movers/SL kan bli ofullständigt).")
    if wl_snap.empty: dq.append("Saknade dagens watchlist-snapshot.")
    if dq:
        for d in dq: html.append(f"<li>{d}</li>")
    else:
        html.append("<li>Inga anmärkningar.</li>")
    html.append("</ul>")

    # Skriv ut filer
    out_html = os.path.join(REPORTS_DIR, f"daily_report_{today.strftime('%Y%m%d')}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"✓ Rapport: {out_html}")
    if os.path.exists(chart_path):
        print(f"✓ Graf: {chart_path}")
    print(f"✓ Holdings-snapshot sparad: {snap_today_path}")

if __name__ == "__main__":
    main()
