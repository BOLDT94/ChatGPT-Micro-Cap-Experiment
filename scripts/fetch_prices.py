import pandas as pd, yfinance as yf, datetime as dt, os, sys

WATCH = "holdings/watchlist.csv"
OUTDIR = "data/daily"
os.makedirs(OUTDIR, exist_ok=True)

today = dt.date.today()
outpath = f"{OUTDIR}/prices_raw_{today.strftime('%Y%m%d')}.csv"

# Läs watchlist
try:
    wl = pd.read_csv(WATCH)
except Exception as e:
    print("ERROR: kunde inte läsa", WATCH, e); sys.exit(0)

if "ticker" not in wl.columns:
    print("ERROR: 'ticker' kolumn saknas i watchlist.csv"); sys.exit(0)

tickers = [t for t in wl["ticker"].dropna().unique().tolist() if isinstance(t, str) and t.strip()]
rows = []

def pick_close(df):
    """Returnera (close_value, date) eller None om det inte går."""
    if df is None or df.empty:
        return None
    df = df.sort_index()
    # yfinance kan returnera 'Close' eller 'Adj Close' beroende på auto_adjust
    col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if col is None:
        return None
    clean = df.dropna(subset=[col])
    if clean.empty:
        return None
    last = clean.tail(1)
    return float(last[col].iloc[0]), last.index[-1].date()

for t in tickers:
    try:
        # Sätt auto_adjust explicit så vi vet vad vi får.
        hist = yf.download(t, period="10d", interval="1d", progress=False, auto_adjust=False)
        lc = pick_close(hist)
        if lc:
            px, d = lc
            rows.append({"ticker": t, "close": px, "asof_date": d.isoformat()})
        else:
            print("VARNING: ingen close-data för", t)
    except Exception as e:
        print("Misslyckades:", t, e)

# Bygg df – även om allt misslyckade skapar vi ett tomt skelett med tickers
if rows:
    df = pd.DataFrame(rows)
else:
    print("VARNING: inga rader hämtades – skapar tomt df med tickers från watchlist.")
    df = pd.DataFrame({"ticker": tickers, "close": [pd.NA]*len(tickers), "asof_date": [today.isoformat()]*len(tickers)})

def safe_fx(symbol):
    try:
        h = yf.download(symbol, period="10d", interval="1d", progress=False, auto_adjust=False)
        got = pick_close(h)
        return got[0] if got else None
    except Exception:
        return None

usdsek = safe_fx("SEK=X")       # USD/SEK
eursek = safe_fx("EURSEK=X")    # EUR/SEK

# Mappa valuta
cur = wl[["ticker","currency"]].copy()
cur["currency"] = cur["currency"].fillna("SEK")
df = df.merge(cur, on="ticker", how="left")

def to_sek(px, ccy):
    if px is None or (isinstance(px, float) and pd.isna(px)) or px == pd.NA:
        return None
    if ccy == "USD" and usdsek: return px * usdsek
    if ccy == "EUR" and eursek: return px * eursek
    return px  # anta SEK annars

df["close_sek"] = [to_sek(px, c) for px, c in zip(df["close"], df["currency"])]
df["source"] = "yfinance"
df.to_csv(outpath, index=False, encoding="utf-8")
print("OK:", outpath, "rader:", len(df))
