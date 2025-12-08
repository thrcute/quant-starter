
import time, json
import pandas as pd
from pathlib import Path
from strategy import compute_signals
from data import make_exchange, fetch_ohlcv_df
from config import Config

STATE_FILE = Path("runs/live_state.json")
LOG_FILE = Path("runs/live_log.txt")

def ensure_dirs():
    Path("runs").mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ensure_dirs()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{pd.Timestamp.utcnow()}] {msg}\n")
    print(msg, flush=True)

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"position": 0, "last_signal": 0}

def save_state(state):
    ensure_dirs()
    STATE_FILE.write_text(json.dumps(state, indent=2))

def position_signal(df, cfg: Config) -> int:
    sig = compute_signals(df, cfg)
    return int(sig.long.iloc[-2])

def trade_loop(cfg: Config, *, paper: bool = True, demo: bool = False):
    ensure_dirs()
    ex = make_exchange(cfg, demo=demo)
    state = load_state()
    log(f"Starting live loop: symbol={cfg.symbol} tf={cfg.timeframe} paper={paper} demo={demo}")

    while True:
        try:
            df = fetch_ohlcv_df(ex, cfg.symbol, cfg.timeframe, limit=200)
            target = position_signal(df, cfg)
            current = state.get("position", 0)
            if target > current:
                qty = cfg.base_order_size
                log(f"Signal LONG -> BUY {qty} {cfg.symbol}")
                if not paper: ex.create_order(cfg.symbol, "market", "buy", qty)
            elif target < current:
                qty = cfg.base_order_size
                log(f"Signal FLAT -> SELL {qty} {cfg.symbol}")
                if not paper: ex.create_order(cfg.symbol, "market", "sell", qty)
            else:
                log("No change.")
            state["position"] = target
            state["last_signal"] = target
            save_state(state)
        except Exception as e:
            log(f"ERROR: {e}")

        tf = cfg.timeframe
        sleep_s = 60
        if tf.endswith("m"):
            sleep_s = int(tf[:-1]) * 60
        elif tf.endswith("h"):
            sleep_s = int(tf[:-1]) * 3600
        elif tf.endswith(("d","D")):
            sleep_s = 24 * 3600
        time.sleep(max(30, min(sleep_s, 3600)))
