import os, json, math
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

# --- настройки сигналов ---
UPPER, LOWER = 70, 30      # зоны RSI
TOUCH = 3               # насколько близко RSI должен подойти к SMA (пункты)
ANGLE_MIN = 5             # минимальный угол отскока (градусы)
ANGLE_SCALE = 2.0          # на сколько пунктов RSI за свечу угол = 45 градусов
ANGLE_FOR_CROSS = False    # True = фильтр угла и для пересечений

# --- дневная сводка ---
SUMMARY_HOUR = 12          # час отправки по времени Штутгарта


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


def angle(r, i):
    # угол движения RSI за последнюю свечу относительно горизонтали
    return math.degrees(math.atan(abs(r.iloc[i] - r.iloc[i - 1]) / ANGLE_SCALE))


def bounce(r, m, i):
    # RSI подошёл к SMA, не пересёк её и развернулся
    d1 = r.iloc[i - 1] - m.iloc[i - 1]
    d0 = r.iloc[i] - m.iloc[i]
    if (0 <= d1 <= TOUCH and d0 > 0
            and r.iloc[i - 1] < r.iloc[i - 2] and r.iloc[i] > r.iloc[i - 1]):
        return "up"
    if (-TOUCH <= d1 <= 0 and d0 < 0
            and r.iloc[i - 1] > r.iloc[i - 2] and r.iloc[i] < r.iloc[i - 1]):
        return "down"
    return None


def stamp(bar):
    return bar.astimezone(ZoneInfo("Europe/Berlin")).strftime("%d.%m %H:%M")


def text(name, side, kind, rsi_val, bar):
    icon = "🟢" if side == "up" else "🔴"
    way = "ВВЕРХ" if side == "up" else "ВНИЗ"
    return (f"{icon} {name} 4H: RSI пересёк SMA {way}\n{kind}\n"
            f"RSI {rsi_val:.1f} | свеча с {stamp(bar)} (Штутгарт)")


def note(name, title, rsi_val, bar):
    return f"{title}\n{name} 4H\nRSI {rsi_val:.1f} | свеча с {stamp(bar)} (Штутгарт)"


def fmt_price(p):
    if p >= 1000:
        return f"{p:,.0f}".replace(",", " ")
    if p >= 50:
        return f"{p:.2f}"
    return f"{p:.4f}"


def summary_line(name, close):
    rsi, sma = analyze(close)
    p = close.iloc[-1]
    old = close[close.index <= close.index[-1] - pd.Timedelta(hours=24)]
    ch = (p / old.iloc[-1] - 1) * 100 if len(old) else 0.0
    pos = "выше SMA" if rsi.iloc[-1] > sma.iloc[-1] else "ниже SMA"
    arrow = "↑" if rsi.iloc[-1] > rsi.iloc[-2] else "↓"
    zone = ""
    if rsi.iloc[-1] > UPPER:
        zone = " 🔥"
    elif rsi.iloc[-1] < LOWER:
        zone = " 🧊"
    return (f"{name}  {fmt_price(p)}  ({ch:+.1f}%)  "
            f"RSI {rsi.iloc[-1]:.1f} {arrow} {pos}{zone}")


def check(name, close, st):
    rsi, sma = analyze(close)
    now = datetime.now(timezone.utc)
    forming = close.index[-1] + pd.Timedelta(hours=4) > now
    msgs = []

    # 1. Пересечение RSI и SMA, предварительное (свеча ещё идёт)
    if forming:
        side = cross(rsi, sma, -1)
        if side and ANGLE_FOR_CROSS and angle(rsi, -1) < ANGLE_MIN:
            side = None
        key = f"{close.index[-1]}:{side}"
        if side and st.get("pre") != key:
            msgs.append(text(name, side, "⚠️ ПРЕДВАРИТЕЛЬНЫЙ (свеча ещё идёт)",
                             rsi.iloc[-1], close.index[-1]))
            st["pre"] = key

    ci = -2 if forming else -1
    bar = close.index[ci]

    # 2. Пересечение RSI и SMA, подтверждённое
    side = cross(rsi, sma, ci)
    if side and ANGLE_FOR_CROSS and angle(rsi, ci) < ANGLE_MIN:
        side = None
    key = f"{bar}:{side}"
    if side and st.get("conf") != key:
        msgs.append(text(name, side, "✅ ПОДТВЕРЖДЁННЫЙ (свеча закрылась)",
                         rsi.iloc[ci], bar))
        st["conf"] = key

    # 3. RSI вошёл в зону выше 70 или ниже 30
    zone = None
    if rsi.iloc[ci - 1] <= UPPER < rsi.iloc[ci]:
        zone = "high"
    elif rsi.iloc[ci - 1] >= LOWER > rsi.iloc[ci]:
        zone = "low"
    key = f"{bar}:{zone}"
    if zone and st.get("zone") != key:
        title = (f"🔥 RSI выше {UPPER} (перекупленность)" if zone == "high"
                 else f"🧊 RSI ниже {LOWER} (перепроданность)")
        msgs.append(note(name, title, rsi.iloc[ci], bar))
        st["zone"] = key

    # 4. Отскок RSI от SMA, только если угол не меньше ANGLE_MIN
    b = bounce(rsi, sma, ci)
    key = f"{bar}:{b}"
    if b and st.get("bounce") != key:
        ang = angle(rsi, ci)
        if ang >= ANGLE_MIN:
            way = "вверх" if b == "up" else "вниз"
            icon = "↗️" if b == "up" else "↘️"
            msgs.append(note(name,
                             f"{icon} RSI отскочил от SMA {way} (угол {ang:.0f}°)",
                             rsi.iloc[ci], bar))
        st["bounce"] = key
    return msgs


def main():
    first = not os.path.exists(STATE_FILE)
    state = {} if first else json.load(open(STATE_FILE))
    if first:
        tg("✅ Бот запущен. Сигналы RSI/SMA будут приходить сюда.")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    jobs = [(s, get_forex, s) for s in FOREX] + [(s, get_crypto, s) for s in CRYPTO]
    rows = []
    for name, fn, sym in jobs:
        if name == CRYPTO[0]:
            rows.append("")
        st = state.setdefault(name, {})
        try:
            close = fn(sym)
            for m in check(name, close, st):
                tg(m)
            rows.append(summary_line(name, close))
        except Exception as e:
            rows.append(f"{name}  нет данных")
            if st.get("err") != today:
                tg(f"❗ Ошибка данных {name}: {str(e)[:150]}")
                st["err"] = today

    # Дневная сводка: один раз в день, в час SUMMARY_HOUR по Штутгарту
    berlin = datetime.now(ZoneInfo("Europe/Berlin"))
    day = berlin.strftime("%Y-%m-%d")
    if berlin.hour == SUMMARY_HOUR and state.get("_summary") != day:
        head = f"📊 Сводка 4H, {berlin.strftime('%d.%m %H:%M')} (Штутгарт)\n"
        tg(head + "\n" + "\n".join(rows))
        state["_summary"] = day

    json.dump(state, open(STATE_FILE, "w"), indent=1)


main()
