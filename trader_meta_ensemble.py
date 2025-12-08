
import time
import pandas as pd
from pathlib import Path

from config import Config
from data import make_exchange, fetch_ohlcv_df
from meta_ensemble import ensemble_vote
from meta import class_to_target as class_to_target_rf

STATE_FILE = Path("runs/live_state_meta_ensemble.txt")
LOG_FILE = Path("runs/live_log_meta_ensemble.txt")

def ensure_dirs():
    Path("runs").mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ensure_dirs()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{pd.Timestamp.utcnow()}] {msg}\n")
    print(msg, flush=True)

def load_state():
    if STATE_FILE.exists():
        s = STATE_FILE.read_text().strip()
        if s.isdigit(): return int(s)
    return 0

def save_state(v: int):
    ensure_dirs()
    STATE_FILE.write_text(str(v))

def trade_loop_meta_ensemble(cfg: Config, *, paper: bool = True, demo: bool = False, w_rf: float = 0.5, w_torch: float = 0.5):
    ensure_dirs()
    ex = make_exchange(cfg, demo=demo)
    log(f"Starting Ensemble live: symbol={cfg.symbol} tf={cfg.timeframe} paper={paper} demo={demo} weights rf={w_rf}, torch={w_torch}")
    while True:
        try:
            df = fetch_ohlcv_df(ex, cfg.symbol, cfg.timeframe, limit=500)
            cls_rf, cls_t, cls = ensemble_vote(df, w_rf, w_torch)
            target = class_to_target_rf(df, cfg, cls)
            current = load_state()
            log(f"Vote -> RF={cls_rf}, Torch={cls_t}, Final={cls}, target={target}, current={current}")
            if target > current:
                qty = cfg.base_order_size
                log(f"BUY {qty} {cfg.symbol}")
                if not paper: ex.create_order(cfg.symbol, "market", "buy", qty)
            elif target < current:
                qty = cfg.base_order_size
                log(f"SELL {qty} {cfg.symbol}")
                if not paper: ex.create_order(cfg.symbol, "market", "sell", qty)
            else:
                log("No change")
            save_state(target)
        except Exception as e:
            log(f"ERROR: {e}")
        tf = cfg.timeframe
        sleep_s = 60
        if tf.endswith("m"): sleep_s = int(tf[:-1]) * 60
        elif tf.endswith("h"): sleep_s = int(tf[:-1]) * 3600
        elif tf.endswith(("d","D")): sleep_s = 24 * 3600
        time.sleep(max(30, min(sleep_s, 3600)))
