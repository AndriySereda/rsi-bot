import os, sys, json, math, time, fcntl
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd

# ---------------------------------------------------------------
# Папка с данными бота на сервере: токен (.env), состояние, журнал
# ---------------------------------------------------------------
DATA_DIR = os.path.expanduser(os.environ.get("RSI_DATA", "~/rsi-data"))
os.makedirs(DATA_DIR, exist_ok=True)


def load_env():
    path = os.path.join(DATA_DIR, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()
TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT"]
STATE_FILE = os.path.join(DATA_DIR, "state.json")
LOCK_FILE = os.path.join(DATA_DIR, "bot.lock")
LOG_FILE = os.path.join(DATA_DIR, "bot.log")

RSI_LEN, SMA_LEN = 20, 14
FOREX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
         "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURCHF"]
CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# --- настройки сигналов ---
UPPER, LOWER = 70, 30      # зоны RSI
TOUCH = 1.5                # насколько близко RSI должен подойти к SMA (пункты)
ANGLE_MIN = 0              # минимальный угол отскока (градусы), 0 = без фильтра
ANGLE_SCALE = 2.0          # на сколько пунктов RSI за свечу угол = 45 градусов
ANGLE_FOR_CROSS = False    # True = фильтр угла и для пересечений

# --- вершины и впадины RSI ---
PIVOT_LEFT = 4             # с какими предыдущими свечами сравниваем
PIVOT_RIGHT = 1            # сколько свечей после вершины нужно для подтверждения
PIVOT_SWING = 5.0          # минимальный размах до вершины (пункты RSI)
PIVOT_BACK = 1.0           # минимальный шаг назад после вершины (пункты RSI)
NEAR_ZONE = 5              # насколько близко к 70 или 30 писать "у зоны"
PIVOT_PRELIMINARY = True   # True = сообщать о развороте сразу, не ждать закрытия свечи

# --- свечи с длинной тенью: молот и падающая звезда (только по закрытой свече) ---
CANDLE_PATTERNS = True     # False = выключить эти сигналы
TAIL_RATIO = 0.6           # длинная тень не меньше 60% размаха свечи
OPP_TAIL_MAX = 0.15        # противоположная тень не больше 15% размаха
RANGE_MIN = 0.8            # размах свечи не меньше 0.8 от среднего за 14 свечей
TREND_LEN = 5              # сколько свечей смотрим назад, чтобы определить тренд
STRONG_LOW, STRONG_HIGH = 40, 60   # "усиленный": молот при RSI < 40, звезда при RSI > 60

# --- отправка ---
COMBINE_MESSAGES = False   # True = все новые сигналы одного запуска одним сообщением
LATE_MINUTES = 25          # после скольких минут опоздания добавлять пометку

# --- тихие часы по времени Штутгарта ---
QUIET_START = (23, 30)     # с 23:30
QUIET_END = (6, 0)         # до 6:00
QUIET_MODE = "drop"        # "drop" = не отправлять, "silent" = отправлять без звука

# --- дневная сводка ---
SUMMARY_HOUR = 8           # час отправки по времени Штутгарта


def tg(text, silent=False):
    for attempt in range(3):
        try:
            r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                              json={"chat_id": CHAT, "text": text,
                                    "disable_notification": silent}, timeout=20)
            if r.status_code == 429:
                time.sleep(int(r.json().get("parameters", {}).get("retry_after", 5)))
                continue
            r.raise_for_status()
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(3)


def is_quiet():
    t = datetime.now(ZoneInfo("Europe/Berlin"))
    now = (t.hour, t.minute)
    return now >= QUIET_START or now < QUIET_END


def get_crypto(sym):
    for host in ("data-api.binance.vision", "api.binance.com"):
        try:
            r = requests.get(f"https://{host}/api/v3/klines",
                             params={"symbol": sym, "interval": "4h", "limit": 300},
                             timeout=20)
            r.raise_for_status()
            d = r.json()
            idx = pd.to_datetime([k[0] for k in d], unit="ms", utc=True)
            return pd.DataFrame({"open": [float(k[1]) for k in d],
                                 "high": [float(k[2]) for k in d],
                                 "low": [float(k[3]) for k in d],
                                 "close": [float(k[4]) for k in d]}, index=idx)
        except Exception:
            continue
    raise RuntimeError("Binance недоступен")


def fx_offset():
    # 4H-свечи форекса в TradingView начинаются в 17:00 по Нью-Йорку
    off = datetime.now(ZoneInfo("America/New_York")).utcoffset().total_seconds() / 3600
    return f"{int((1 - off) % 4)}h"


def get_yahoo(ticker, offset):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"interval": "1h", "range": "60d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"],
                       "low": q["low"], "close": q["close"]},
                      index=idx, dtype="float64").dropna()
    out = df.resample("4h", offset=offset).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


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


def pivot(r, i):
    # Вершина или впадина RSI, подтверждённая PIVOT_RIGHT свечами после неё.
    # Возвращает (направление разворота, значение вершины, размах) или None.
    last = i % len(r)
    j = last - PIVOT_RIGHT
    v = r.iloc[j - PIVOT_LEFT:last + 1].tolist()
    left = v[:PIVOT_LEFT]
    p = v[PIVOT_LEFT]
    right = v[PIVOT_LEFT + 1:]
    if p >= max(left) and p > max(right):
        swing, back = p - min(left), p - right[-1]
        if swing >= PIVOT_SWING and back >= PIVOT_BACK:
            return "down", p, swing
    if p <= min(left) and p < min(right):
        swing, back = max(left) - p, right[-1] - p
        if swing >= PIVOT_SWING and back >= PIVOT_BACK:
            return "up", p, swing
    return None


def candle_pattern(df, rsi, i):
    # Молот (разворот вверх) и падающая звезда (разворот вниз) на закрытой свече i.
    # Возвращает ("hammer" | "star", усиленный?, значение RSI) или None.
    k = i % len(df)
    if k < TREND_LEN + 16:
        return None
    o, h, l, c = (float(df[x].iloc[k]) for x in ("open", "high", "low", "close"))
    rng = h - l
    if rng <= 0:
        return None
    avg = (df["high"].iloc[k - 14:k] - df["low"].iloc[k - 14:k]).mean()
    if pd.isna(avg) or avg <= 0 or rng < RANGE_MIN * avg:
        return None
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    prev = float(df["close"].iloc[k - 1])
    before = float(df["close"].iloc[k - 1 - TREND_LEN])
    r = float(rsi.iloc[k])
    if (prev < before and lower >= TAIL_RATIO * rng and lower >= 2 * body
            and upper <= OPP_TAIL_MAX * rng):
        return "hammer", r < STRONG_LOW, r
    if (prev > before and upper >= TAIL_RATIO * rng and upper >= 2 * body
            and lower <= OPP_TAIL_MAX * rng):
        return "star", r > STRONG_HIGH, r
    return None


def stamp(bar):
    berlin = ZoneInfo("Europe/Berlin")
    b = bar.astimezone(berlin)
    e = (bar + pd.Timedelta(hours=4)).astimezone(berlin)
    return f"{b.strftime('%d.%m %H:%M')}–{e.strftime('%H:%M')}"


def late(bar):
    # пометка, если сигнал найден заметно позже закрытия свечи
    mins = int((datetime.now(timezone.utc) - (bar + pd.Timedelta(hours=4))).total_seconds() / 60)
    if mins < LATE_MINUTES:
        return ""
    return f"\n⏱ найден с опозданием {mins // 60} ч {mins % 60} мин"


def msg(name, bar, line, rsi_val):
    # формат: пара, время свечи, суть сигнала
    return (f"{name} · 4H\n🕐 {stamp(bar)} (Штутгарт)\n{line}\n"
            f"RSI {rsi_val:.1f}" + late(bar))


def text(name, side, kind, rsi_val, bar):
    icon = "🟢" if side == "up" else "🔴"
    way = "ВВЕРХ" if side == "up" else "ВНИЗ"
    return msg(name, bar, f"{icon} RSI пересёк SMA {way}\n{kind}", rsi_val)


def note(name, title, rsi_val, bar):
    return msg(name, bar, title, rsi_val)


def pivot_text(name, pv, rsi_now, bar, tag):
    d, val, swing = pv
    way = "вниз" if d == "down" else "вверх"
    icon = "🔻" if d == "down" else "🔺"
    kind = "вершины" if d == "down" else "впадины"
    near = ""
    if val >= UPPER - NEAR_ZONE:
        near = f", у зоны {UPPER}"
    elif val <= LOWER + NEAR_ZONE:
        near = f", у зоны {LOWER}"
    line = (f"{icon} RSI развернулся {way} от {kind} {val:.1f} "
            f"(ход {swing:.1f} п.{near})\n{tag}")
    return note(name, line, rsi_now, bar)


def fmt_price(p):
    if p >= 1000:
        return f"{p:,.0f}".replace(",", " ")
    if p >= 50:
        return f"{p:.2f}"
    return f"{p:.4f}"


def summary_line(name, df):
    close = df["close"]
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


def check(name, df, st):
    # Возвращает список (время свечи, приоритет, текст) для сортировки по времени
    close = df["close"]
    rsi, sma = analyze(close)
    now = datetime.now(timezone.utc)
    forming = close.index[-1] + pd.Timedelta(hours=4) > now
    msgs = []

    if forming:
        # 1. Пересечение RSI и SMA, предварительное (свеча ещё идёт)
        side = cross(rsi, sma, -1)
        if side and ANGLE_FOR_CROSS and angle(rsi, -1) < ANGLE_MIN:
            side = None
        key = f"{close.index[-1]}:{side}"
        if side and st.get("pre") != key:
            msgs.append((close.index[-1], 5,
                         text(name, side, "⚠️ ПРЕДВАРИТЕЛЬНЫЙ (свеча ещё идёт)",
                              rsi.iloc[-1], close.index[-1])))
            st["pre"] = key

        # 1б. Предварительный разворот RSI (без ожидания закрытия свечи)
        if PIVOT_PRELIMINARY:
            pv = pivot(rsi, -1)
            key = f"{close.index[-1]}:{pv[0] if pv else None}"
            if pv and st.get("pivot_pre") != key:
                msgs.append((close.index[-1], 6,
                             pivot_text(name, pv, rsi.iloc[-1], close.index[-1],
                                        "⚠️ ПРЕДВАРИТЕЛЬНЫЙ (свеча ещё идёт)")))
                st["pivot_pre"] = key

    ci = -2 if forming else -1
    bar = close.index[ci]

    # 2. Пересечение RSI и SMA, подтверждённое
    side = cross(rsi, sma, ci)
    if side and ANGLE_FOR_CROSS and angle(rsi, ci) < ANGLE_MIN:
        side = None
    key = f"{bar}:{side}"
    if side and st.get("conf") != key:
        msgs.append((bar, 0,
                     text(name, side, "✅ ПОДТВЕРЖДЁННЫЙ (свеча закрылась)",
                          rsi.iloc[ci], bar)))
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
        msgs.append((bar, 1, note(name, title, rsi.iloc[ci], bar)))
        st["zone"] = key

    # 4. Отскок RSI от SMA, только если угол не меньше ANGLE_MIN
    b = bounce(rsi, sma, ci)
    key = f"{bar}:{b}"
    if b and st.get("bounce") != key:
        ang = angle(rsi, ci)
        if ang >= ANGLE_MIN:
            way = "вверх" if b == "up" else "вниз"
            icon = "↗️" if b == "up" else "↘️"
            msgs.append((bar, 2,
                         note(name,
                              f"{icon} RSI отскочил от SMA {way} (угол {ang:.0f}°)",
                              rsi.iloc[ci], bar)))
        st["bounce"] = key

    # 5. Свеча с длинной тенью: молот или падающая звезда (закрытая свеча)
    if CANDLE_PATTERNS:
        cp = candle_pattern(df, rsi, ci)
        key = f"{bar}:{cp[0] if cp else None}"
        if cp and st.get("candle") != key:
            kind, strong, r = cp
            if kind == "hammer":
                title = "🔨 Молот: возможный разворот вверх"
                extra = f"\nусиленный: RSI {r:.1f} < {STRONG_LOW}" if strong else ""
            else:
                title = "🌠 Падающая звезда: возможный разворот вниз"
                extra = f"\nусиленный: RSI {r:.1f} > {STRONG_HIGH}" if strong else ""
            msgs.append((bar, 3, note(name, title + extra, rsi.iloc[ci], bar)))
            st["candle"] = key

    # 6. Вершина или впадина RSI, подтверждённая закрытой свечой
    pv = pivot(rsi, ci)
    key = f"{bar}:{pv[0] if pv else None}"
    if pv and st.get("pivot") != key:
        msgs.append((bar, 4,
                     pivot_text(name, pv, rsi.iloc[ci], bar,
                                "✅ ПОДТВЕРЖДЁННЫЙ (свеча закрылась)")))
        st["pivot"] = key
    return msgs


def main():
    init = "--init" in sys.argv        # первый запуск: запомнить текущее состояние, ничего не слать
    verbose = "--verbose" in sys.argv

    # не запускаться, если предыдущий запуск ещё работает
    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Предыдущий запуск ещё работает, пропускаю")
        return

    # журнал не должен расти бесконечно
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 500_000:
        open(LOG_FILE, "w").close()

    state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    quiet = is_quiet()
    can_send = (not quiet) or QUIET_MODE == "silent"
    off = fx_offset()
    jobs = ([(s, lambda s=s: get_yahoo(f"{s}=X", off)) for s in FOREX]
            + [("XAUUSD", lambda: get_yahoo("GC=F", off))]
            + [(s, lambda s=s: get_crypto(s)) for s in CRYPTO])
    rows, pending, failed = [], [], []
    for name, fn in jobs:
        if name == CRYPTO[0]:
            rows.append("")
        st = state.setdefault(name, {})
        try:
            df = fn()
            if verbose:
                rsi, sma = analyze(df["close"])
                print(name, df.index[-1], round(rsi.iloc[-1], 2), round(sma.iloc[-1], 2))
            pending.extend(check(name, df, st))
            rows.append(summary_line(name, df))
        except Exception as e:
            failed.append(name)
            rows.append(f"{name}  нет данных")
            print(f"Ошибка {name}: {e}")
            if st.get("err") != today and can_send and not init:
                tg(f"❗ Ошибка данных {name}: {str(e)[:150]}", silent=quiet)
                st["err"] = today

    stamp_now = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m %H:%M")

    if init:
        ok = len(jobs) - len(failed)
        text_init = (f"✅ Серверный бот запущен ({stamp_now}).\n"
                     f"Данные получены по {ok} из {len(jobs)} инструментов.")
        if failed:
            text_init += "\nНет данных: " + ", ".join(failed)
        tg(text_init, silent=quiet)
        json.dump(state, open(STATE_FILE, "w"), indent=1)
        print(text_init)
        return

    # Отправка: от старых сигналов к новым
    pending.sort(key=lambda x: (x[0], x[1]))
    sent = len(pending)
    if quiet and QUIET_MODE != "silent":
        print(f"Тихие часы: пропущено сообщений: {len(pending)}")
        pending = []
    if COMBINE_MESSAGES and pending:
        body = "\n\n".join(m for _, _, m in pending)
        if len(body) < 3800:
            tg(f"📬 Новые сигналы ({len(pending)}):\n\n" + body, silent=quiet)
            pending = []
    for _, _, m in pending:
        tg(m, silent=quiet)
        time.sleep(1.2)

    # Дневная сводка: один раз в день, в час SUMMARY_HOUR по Штутгарту
    berlin = datetime.now(ZoneInfo("Europe/Berlin"))
    day = berlin.strftime("%Y-%m-%d")
    if berlin.hour == SUMMARY_HOUR and state.get("_summary") != day:
        head = f"📊 Сводка 4H, {berlin.strftime('%d.%m %H:%M')} (Штутгарт)\n"
        tg(head + "\n" + "\n".join(rows))
        state["_summary"] = day

    json.dump(state, open(STATE_FILE, "w"), indent=1)
    print(f"{stamp_now} ok: сигналов {sent}, ошибок данных {len(failed)}")


if __name__ == "__main__":
    main()
