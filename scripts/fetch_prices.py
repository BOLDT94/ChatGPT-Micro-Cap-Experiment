import pandas as pd, yfinance as yf, datetime as dt, os, sys

WATCH = "holdings/watchlist.csv"
OUTDIR = "data/daily"
os.makedirs(OUTDIR, exist_ok=True)

today = dt.date.today()
outpath = f"{OUTDIR}/prices_raw_{today.strftime('%Y%m%d')}.csv"

try:
    wl = pd.read_csv(WATCH)
except Exception as e:
    print("ERROR: kunde inte läsa", WATCH, e); sys.exit(0)

if "ticker" not in wl.columns:
    print("ERROR: 'ticker' kolumn saknas i watchlist.csv"); sys.exit(0)

tickers = [t for t in wl["ticker"].dropna().unique().tolist() if isinstance(t, str) and t.strip()]
rows = []

def last_close(df):
    if df is None or df.empty: return None
    df = df.sort_index()
    last = df.dropna(subset=["Close"]).tail(1)
    return None if last.empty else (float(last["Close"].iloc[0]), last.index[-1].date())

for t in tickers:
    try:
        hist = yf.download(t, period="10d", interval="1d", progress=False)
        lc = last_close(hist)
        if lc:
            px, d = lc
            rows.append({"ticker": t, "close": px, "asof_date": d.isoformat()})
        else:
            print("VARNING: tom historik för", t)
    except Exception as e:
        print("Misslyckades:", t, e)

df = pd.DataFrame(rows)

def safe_close(symbol):
    try:
        h = yf.download(symbol, period="10d", interval="1d", progress=False)
        lc = last_close(h)
        return lc[0] if lc else None
    except: return None

usdsek = safe_close("SEK=X")
eursek = safe_close("EURSEK=X")

cur = wl[["ticker","currency"]].copy()
cur["currency"] = cur["currency"].fillna("SEK")
df = df.merge(cur, on="ticker", how="left")

def to_sek(px, ccy):
    if px is None: return None
    if ccy == "USD" and usdsek: return px * usdsek
    if ccy == "EUR" and eursek: return px * eursek
    return px

df["close_sek"] = [to_sek(px, c) for px, c in zip(df["close"], df["currency"])]
df["source"] = "yfinance"
df.to_csv(outpath, index=False, encoding="utf-8")
print("OK:", outpath, "rader:", len(df))
