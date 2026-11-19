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
TOKEN = os.environ.get("TG_TOKEN", "")
CHAT = os.environ.get("TG_CHAT", "")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
LOCK_FILE = os.path.join(DATA_DIR, "bot.lock")
LOG_FILE = os.path.join(DATA_DIR, "bot.log")

RSI_LEN, SMA_LEN = 20, 14

# ---------------------------------------------------------------
# ЧТО ОТСЛЕЖИВАЕМ: имя -> (источник, тикер, привязка 4H-свечей)
# привязка: "fx" = как у форекса и золота в TradingView, "utc" = по UTC
# Чтобы добавить инструмент, скопируйте строку из CANDIDATES сюда.
# ---------------------------------------------------------------
INSTRUMENTS = {
    "XAUUSD": ("yahoo", "GC=F", "fx"),
    "BTCUSD": ("binance", "BTCUSDT", "utc"),
    "NAS100": ("yahoo", "NQ=F", "utc"),
}

# Кандидаты для проверки: python server_bot.py --volatility
CANDIDATES = {
    "ETHUSD": ("binance", "ETHUSDT", "utc"),
    "SOLUSD": ("binance", "SOLUSDT", "utc"),
    "OIL (WTI)": ("yahoo", "CL=F", "fx"),
    "SILVER": ("yahoo", "SI=F", "fx"),
    "GBPJPY": ("yahoo", "GBPJPY=X", "fx"),
    "NATGAS": ("yahoo", "NG=F", "fx"),
    "SP500": ("yahoo", "ES=F", "utc"),
}

# ---------------------------------------------------------------
# СТРАТЕГИЯ: сигнал приходит, когда совпадают несколько факторов
#   1. RSI пересёк SMA в нужную сторону (на последних 2 закрытых свечах)
#   2. RSI недавно был в слабой зоне (для покупки ниже 40, для продажи выше 60)
#   3. Разворотная свеча: молот (покупка) или падающая звезда (продажа)
#   4. Тренд: цена выше EMA200 для покупки, ниже EMA200 для продажи
# ---------------------------------------------------------------
MIN_SCORE = 3              # сколько факторов из 4 должно совпасть
RSI_ZONE_BULL = 40         # для покупки RSI недавно был ниже этого значения
RSI_ZONE_BEAR = 60         # для продажи RSI недавно был выше этого значения
LOOKBACK = 3               # "недавно" = последние N закрытых свечей
TREND_EMA = 200            # трендовый фильтр: EMA по закрытиям 4H
COOLDOWN = 3               # после сигнала повтор в ту же сторону не раньше, чем через N свечей

# --- параметры свечей с длинной тенью ---
TAIL_RATIO = 0.6           # длинная тень не меньше 60% размаха свечи
OPP_TAIL_MAX = 0.15        # противоположная тень не больше 15% размаха
RANGE_MIN = 0.8            # размах свечи не меньше 0.8 от среднего за 14 свечей
TREND_LEN = 5              # сколько свечей смотрим назад, чтобы определить движение до свечи

# --- отправка ---
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


# ---------------------------------------------------------------
# Данные
# ---------------------------------------------------------------
def get_crypto(sym):
    for host in ("data-api.binance.vision", "api.binance.com"):
        try:
            r = requests.get(f"https://{host}/api/v3/klines",
                             params={"symbol": sym, "interval": "4h", "limit": 500},
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
    # 4H-свечи форекса и золота в TradingView начинаются в 17:00 по Нью-Йорку
    off = datetime.now(ZoneInfo("America/New_York")).utcoffset().total_seconds() / 3600
    return f"{int((1 - off) % 4)}h"


def get_yahoo(ticker, offset):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"interval": "1h", "range": "90d"},
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


def load(spec):
    kind, ticker, align = spec
    if kind == "binance":
        return get_crypto(ticker)
    return get_yahoo(ticker, fx_offset() if align == "fx" else "0h")


# ---------------------------------------------------------------
# Индикаторы и признаки
# ---------------------------------------------------------------
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


def candle_pattern(df, rsi, i):
    # Молот (разворот вверх) и падающая звезда (разворот вниз) на закрытой свече i.
    # Возвращает "hammer", "star" или None.
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
    if (prev < before and lower >= TAIL_RATIO * rng and lower >= 2 * body
            and upper <= OPP_TAIL_MAX * rng):
        return "hammer"
    if (prev > before and upper >= TAIL_RATIO * rng and upper >= 2 * body
            and lower <= OPP_TAIL_MAX * rng):
        return "star"
    return None


def evaluate(df, k):
    # Оценка закрытой свечи с номером k (положительный индекс).
    # Для покупки и для продажи возвращает очки, список факторов и признак
    # "на последней свече случилось что-то новое".
    close = df["close"]
    rsi, sma = analyze(close)
    ema = close.ewm(span=TREND_EMA, adjust=False).mean()
    crosses = [cross(rsi, sma, k), cross(rsi, sma, k - 1)]
    pats = [candle_pattern(df, rsi, k), candle_pattern(df, rsi, k - 1)]
    window = rsi.iloc[k - LOOKBACK + 1:k + 1]
    price = float(close.iloc[k])
    result = {}
    for side in ("long", "short"):
        buy = side == "long"
        want_cross = "up" if buy else "down"
        want_pat = "hammer" if buy else "star"
        items = []

        # 1. пересечение RSI и SMA
        cross_now = crosses[0] == want_cross
        cross_ok = cross_now or crosses[1] == want_cross
        items.append((cross_ok, f"RSI пересёк SMA {'вверх' if buy else 'вниз'}"
                      if cross_ok else "нет пересечения RSI и SMA"))

        # 2. RSI недавно был в слабой (покупка) или сильной (продажа) зоне
        if buy:
            extreme_rsi = float(window.min())
            zone_ok = extreme_rsi <= RSI_ZONE_BULL
            zone_txt = (f"RSI недавно был ниже {RSI_ZONE_BULL} (мин. {extreme_rsi:.1f})" if zone_ok
                        else f"RSI не опускался ниже {RSI_ZONE_BULL} (мин. {extreme_rsi:.1f})")
        else:
            extreme_rsi = float(window.max())
            zone_ok = extreme_rsi >= RSI_ZONE_BEAR
            zone_txt = (f"RSI недавно был выше {RSI_ZONE_BEAR} (макс. {extreme_rsi:.1f})" if zone_ok
                        else f"RSI не поднимался выше {RSI_ZONE_BEAR} (макс. {extreme_rsi:.1f})")
        items.append((zone_ok, zone_txt))

        # 3. разворотная свеча
        pat_now = pats[0] == want_pat
        pat_ok = pat_now or pats[1] == want_pat
        name_pat = "Молот" if buy else "Падающая звезда"
        items.append((pat_ok, f"{name_pat} (разворотная свеча)" if pat_ok
                      else "нет разворотной свечи (молот или звезда)"))

        # 4. тренд
        ema_v = float(ema.iloc[k])
        trend_ok = price > ema_v if buy else price < ema_v
        if buy:
            trend_txt = (f"тренд вверх: цена выше EMA{TREND_EMA}" if trend_ok
                         else f"против тренда: цена ниже EMA{TREND_EMA}")
        else:
            trend_txt = (f"тренд вниз: цена ниже EMA{TREND_EMA}" if trend_ok
                         else f"против тренда: цена выше EMA{TREND_EMA}")
        items.append((trend_ok, trend_txt))

        if buy:
            extreme = float(min(df["low"].iloc[k], df["low"].iloc[k - 1]))
        else:
            extreme = float(max(df["high"].iloc[k], df["high"].iloc[k - 1]))
        result[side] = {
            "score": sum(1 for ok, _ in items if ok),
            "items": items,
            "trigger": cross_now or pat_now,
            "extreme": extreme,
            "price": price,
            "rsi": float(rsi.iloc[k]),
        }
    return result


# ---------------------------------------------------------------
# Сообщения
# ---------------------------------------------------------------
def stamp(bar):
    berlin = ZoneInfo("Europe/Berlin")
    b = bar.astimezone(berlin)
    e = (bar + pd.Timedelta(hours=4)).astimezone(berlin)
    return f"{b.strftime('%d.%m %H:%M')}–{e.strftime('%H:%M')}"


def late(bar):
    mins = int((datetime.now(timezone.utc) - (bar + pd.Timedelta(hours=4))).total_seconds() / 60)
    if mins < LATE_MINUTES:
        return ""
    return f"\n⏱ найден с опозданием {mins // 60} ч {mins % 60} мин"


def fmt_price(p):
    if p >= 1000:
        return f"{p:,.0f}".replace(",", " ")
    if p >= 50:
        return f"{p:.2f}"
    return f"{p:.4f}"


def signal_text(name, side, ev, bar):
    buy = side == "long"
    icon, word = ("🟢", "ПОКУПКА") if buy else ("🔴", "ПРОДАЖА")
    lines = [f"{icon} Идея: {word}, совпало {ev['score']} из 4"]
    for ok, t in ev["items"]:
        lines.append(("✅ " if ok else "▫️ ") + t)
    what = "Минимум" if buy else "Максимум"
    lines.append(f"{what} двух последних свечей: {fmt_price(ev['extreme'])}")
    return (f"{name} · 4H\n🕐 {stamp(bar)} (Штутгарт)\n" + "\n".join(lines)
            + f"\nЦена {fmt_price(ev['price'])} | RSI {ev['rsi']:.1f}" + late(bar))


def summary_line(name, df, ev):
    close = df["close"]
    rsi, sma = analyze(close)
    p = close.iloc[-1]
    old = close[close.index <= close.index[-1] - pd.Timedelta(hours=24)]
    ch = (p / old.iloc[-1] - 1) * 100 if len(old) else 0.0
    pos = "выше SMA" if rsi.iloc[-1] > sma.iloc[-1] else "ниже SMA"
    return (f"{name}  {fmt_price(p)}  ({ch:+.1f}%)  RSI {rsi.iloc[-1]:.1f} {pos}\n"
            f"    покупка {ev['long']['score']}/4, продажа {ev['short']['score']}/4")


def process(name, df, st):
    # Возвращает (сообщения, оценка). Сообщение создаётся, только если совпало
    # не меньше MIN_SCORE факторов и на последней закрытой свече произошло событие.
    if len(df) < 60:
        raise RuntimeError("мало данных")
    close = df["close"]
    forming = close.index[-1] + pd.Timedelta(hours=4) > datetime.now(timezone.utc)
    k = len(df) - (2 if forming else 1)
    bar = close.index[k]
    ev = evaluate(df, k)
    out = []
    for side in ("long", "short"):
        e = ev[side]
        if e["score"] < MIN_SCORE or not e["trigger"]:
            continue
        last = st.get("last")
        if (last and last["side"] == side
                and bar - pd.Timestamp(last["bar"]) < pd.Timedelta(hours=4 * COOLDOWN)):
            continue
        out.append((bar, 0, signal_text(name, side, e, bar)))
        st["last"] = {"side": side, "bar": str(bar)}
    return out, ev


# ---------------------------------------------------------------
# Режимы для проверки: --status и --volatility
# ---------------------------------------------------------------
def status():
    for name, spec in INSTRUMENTS.items():
        try:
            df = load(spec)
            close = df["close"]
            forming = close.index[-1] + pd.Timedelta(hours=4) > datetime.now(timezone.utc)
            k = len(df) - (2 if forming else 1)
            ev = evaluate(df, k)
            rsi, sma = analyze(close)
            print(f"\n{name}  закрытая свеча {close.index[k]:%d.%m %H:%M} UTC | "
                  f"цена {fmt_price(ev['long']['price'])} | RSI {rsi.iloc[k]:.1f} SMA {sma.iloc[k]:.1f}")
            for side, label in (("long", "покупка"), ("short", "продажа")):
                e = ev[side]
                flag = "  <- было бы сообщение" if e["score"] >= MIN_SCORE and e["trigger"] else ""
                print(f"  {label}: {e['score']}/4{flag}")
                for ok, t in e["items"]:
                    print("    " + ("[+] " if ok else "[ ] ") + t)
        except Exception as e:
            print(f"{name}: нет данных ({e})")


def volatility():
    specs = dict(INSTRUMENTS)
    specs.update(CANDIDATES)
    data = {}
    for name, spec in specs.items():
        try:
            data[name] = load(spec)
        except Exception as e:
            print(f"{name}: нет данных ({e})")
    core = list(INSTRUMENTS)
    daily_ret = {}
    for n, d in data.items():
        daily_ret[n] = d["close"].resample("1D").last().dropna().pct_change().dropna().tail(60)
    rows = []
    for n, d in data.items():
        r4 = ((d["high"] - d["low"]) / d["close"] * 100).tail(180).mean()
        day = d.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna().tail(30)
        rd = ((day["high"] - day["low"]) / day["close"] * 100).mean()
        corr = []
        for c in core:
            if c == n or c not in daily_ret:
                corr.append("-")
                continue
            joined = pd.concat([daily_ret[n], daily_ret[c]], axis=1, join="inner").dropna()
            corr.append(f"{joined.iloc[:, 0].corr(joined.iloc[:, 1]):.2f}" if len(joined) >= 20 else "-")
        rows.append((rd, n, r4, corr))
    rows.sort(reverse=True)
    head = f"{'Инструмент':<12}{'дн.размах %':>12}{'4H-размах %':>13}" + "".join(
        f"{'корр.' + c:>14}" for c in core)
    print("\n" + head)
    for rd, n, r4, corr in rows:
        mark = "*" if n in INSTRUMENTS else " "
        print(f"{mark}{n:<11}{rd:>12.2f}{r4:>13.2f}" + "".join(f"{c:>14}" for c in corr))
    print("\n* = уже отслеживается. Корреляция выше 0.7 значит, что инструмент ходит "
          "почти вместе с основным и даёт дубли сигналов.")


# ---------------------------------------------------------------
def main():
    args = sys.argv[1:]
    if "--status" in args:
        return status()
    if "--volatility" in args:
        return volatility()
    init = "--init" in args            # первый запуск: запомнить текущее состояние, ничего не слать

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
    rows, pending, failed = [], [], []
    for name, spec in INSTRUMENTS.items():
        st = state.setdefault(name, {})
        try:
            df = load(spec)
            out, ev = process(name, df, st)
            pending.extend(out)
            rows.append(summary_line(name, df, ev))
        except Exception as e:
            failed.append(name)
            rows.append(f"{name}  нет данных")
            print(f"Ошибка {name}: {e}")
            if st.get("err") != today and can_send and not init:
                tg(f"❗ Ошибка данных {name}: {str(e)[:150]}", silent=quiet)
                st["err"] = today

    stamp_now = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m %H:%M")

    if init:
        ok = len(INSTRUMENTS) - len(failed)
        text_init = (f"✅ Серверный бот запущен ({stamp_now}).\n"
                     f"Данные получены по {ok} из {len(INSTRUMENTS)} инструментов.")
        if failed:
            text_init += "\nНет данных: " + ", ".join(failed)
        tg(text_init, silent=quiet)
        json.dump(state, open(STATE_FILE, "w"), indent=1)
        print(text_init)
        return

    pending.sort(key=lambda x: (x[0], x[1]))
    sent = len(pending)
    if quiet and QUIET_MODE != "silent":
        print(f"Тихие часы: пропущено сообщений: {len(pending)}")
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
