import os, json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT"]
RSI_LEN, SMA_LEN = 20, 14
FOREX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
         "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURCHF"]
CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
STATE_FILE = "state.json"


def tg(text):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  json={"chat_id": CHAT, "text": text}, timeout=20)


def get_crypto(sym):
    for host in ("data-api.binance.vision", "api.binance.com"):
        try:
            r = requests.get(f"https://{host}/api/v3/klines",
                             params={"symbol": sym, "interval": "4h", "limit": 300},
                             timeout=20)
            r.raise_for_status()
            d = r.json()
            idx = pd.to_datetime([k[0] for k in d], unit="ms", utc=True)
            return pd.Series([float(k[4]) for k in d], index=idx)
        except Exception:
            continue
    raise RuntimeError("Binance недоступен")


def get_forex(sym):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}=X",
                     params={"interval": "1h", "range": "60d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=idx,
                  dtype="float64").dropna()
    return s.resample("4h").last().dropna()


def analyze(close):
    d = close.diff()
    au = d.clip(lower=0).ewm(alpha=1 / RSI_LEN, adjust=False).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / RSI_LEN, adjust=False).mean()
    rsi = 100 - 100 / (1 + au / ad)
    return rsi, rsi.rolling(SMA_LEN).mean()


def cross(r, m, i):
    a = r.iloc[i] - m.iloc[i]
    b = r.iloc[i - 1] - m.iloc[i - 1]
    if b <= 0 < a:
        return "up"
    if b >= 0 > a:
        return "down"
    return None


def text(name, side, kind, rsi_val, bar):
    icon = "🟢" if side == "up" else "🔴"
    way = "ВВЕРХ" if side == "up" else "ВНИЗ"
    t = bar.astimezone(ZoneInfo("Europe/Berlin")).strftime("%d.%m %H:%M")
    return (f"{icon} {name} 4H: RSI пересёк SMA {way}\n{kind}\n"
            f"RSI {rsi_val:.1f} | свеча с {t} (Штутгарт)")


def check(name, close, st):
    rsi, sma = analyze(close)
    print(name, close.index[-1], round(rsi.iloc[-1], 2), round(sma.iloc[-1], 2), round(rsi.iloc[-2], 2), round(sma.iloc[-2], 2))
    now = datetime.now(timezone.utc)
    forming = close.index[-1] + pd.Timedelta(hours=4) > now
    msgs = []
    if forming:
        side = cross(rsi, sma, -1)
        key = f"{close.index[-1]}:{side}"
        if side and st.get("pre") != key:
            msgs.append(text(name, side, "⚠️ ПРЕДВАРИТЕЛЬНЫЙ (свеча ещё идёт)",
                             rsi.iloc[-1], close.index[-1]))
            st["pre"] = key
    ci = -2 if forming else -1
    side = cross(rsi, sma, ci)
    key = f"{close.index[ci]}:{side}"
    if side and st.get("conf") != key:
        msgs.append(text(name, side, "✅ ПОДТВЕРЖДЁННЫЙ (свеча закрылась)",
                         rsi.iloc[ci], close.index[ci]))
        st["conf"] = key
    return msgs


def main():
    first = not os.path.exists(STATE_FILE)
    state = {} if first else json.load(open(STATE_FILE))
    if first:
        tg("✅ Бот запущен. Сигналы RSI/SMA будут приходить сюда.")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    jobs = [(s, get_forex, s) for s in FOREX] + [(s, get_crypto, s) for s in CRYPTO]
    for name, fn, sym in jobs:
        st = state.setdefault(name, {})
        try:
            for m in check(name, fn(sym), st):
                tg(m)
        except Exception as e:
            if st.get("err") != today:
                tg(f"❗ Ошибка данных {name}: {str(e)[:150]}")
                st["err"] = today
    json.dump(state, open(STATE_FILE, "w"), indent=1)


main()
