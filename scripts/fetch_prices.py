import os
import sys
import datetime as dt
import pandas as pd
import yfinance as yf

WATCH = "holdings/watchlist.csv"
OUTDIR = "data/daily"
os.makedirs(OUTDIR, exist_ok=True)

today = dt.date.today()
outpath = f"{OUTDIR}/prices_raw_{today.strftime('%Y%m%d')}.csv"

# --- Helpers -----------------------------------------------------------------

def pick_close(df):
    """Return (close_value, date) from a yfinance DataFrame or None if unavailable."""
    if df is None or df.empty:
        return None
    df = df.sort_index()
    col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if col is None:
        return None
    clean = df.dropna(subset=[col])
    if clean.empty:
        return None
    last = clean.tail(1)
    return float(last[col].iloc[0]), last.index[-1].date()

def fetch_last_close(symbol, periods=("10d", "1mo", "2mo")):
    """Try multiple periods until a last close is found."""
    for per in periods:
        try:
            hist = yf.download(symbol, period=per, interval="1d", progress=False, auto_adjust=False)
            got = pick_close(hist)
            if got:
                return got
        except Exception as e:
            print(f"VARNING: nedladdning misslyckades för {symbol} ({per}): {e}")
    return None

# --- Read watchlist ----------------------------------------------------------

try:
    wl = pd.read_csv(WATCH)
except Exception as e:
    print("ERROR: kunde inte läsa", WATCH, e)
    sys.exit(0)

if "ticker" not in wl.columns:
    print("ERROR: 'ticker' kolumn saknas i watchlist.csv")
    sys.exit(0)

wl["currency"] = wl.get("currency", pd.Series(["SEK"] * len(wl))).fillna("SEK")

tickers = [t for t in wl["ticker"].dropna().astype(str).str.strip().tolist() if t]
rows, fails = [], []

# --- Fetch equities ----------------------------------------------------------

for t in tickers:
    got = fetch_last_close(t)
    if got:
        px, d = got
        rows.append({"ticker": t, "close": px, "asof_date": d.isoformat()})
    else:
        print("VARNING: ingen close-data för", t)
        fails.append(t)

print(f"DEBUG: {len(rows)} tickers lyckades, {len(fails)} tickers misslyckades")
if fails:
    print("DEBUG – misslyckade tickers:", ", ".join(fails))

# Bygg df – även om allt fallerade skapar vi ett skelett-df så merge inte kraschar
if rows:
    df = pd.DataFrame(rows)
else:
    df = pd.DataFrame(
        {"ticker": tickers, "close": [pd.NA] * len(tickers), "asof_date": [today.isoformat()] * len(tickers)}
    )

# --- FX (USD/SEK & EUR/SEK) --------------------------------------------------

usdsek_close = fetch_last_close("SEK=X")   # USD/SEK
usdsek = usdsek_close[0] if usdsek_close else None

eursek_close = fetch_last_close("EURSEK=X")
eursek = eursek_close[0] if eursek_close else None

# --- Merge currency & convert to SEK -----------------------------------------

df = df.merge(wl[["ticker", "currency"]], on="ticker", how="left")

def to_sek(px, ccy):
    if pd.isna(px):
        return None
    if ccy == "USD" and usdsek:
        return px * usdsek
    if ccy == "EUR" and eursek:
        return px * eursek
    # SEK eller okänd -> lämna som är
    return px

df["close_sek"] = [to_sek(px, c) for px, c in zip(df["close"], df["currency"])]
df["source"] = "yfinance"

# --- Write -------------------------------------------------------------------

df.to_csv(outpath, index=False, encoding="utf-8")
print("OK:", outpath, "rader:", len(df))
