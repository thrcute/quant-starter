
import argparse, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import Config
from data import make_exchange, fetch_ohlcv_df
from meta import train_meta
from meta_torch import train_meta_torch

def utc_now():
    return datetime.now(timezone.utc)

def main():
    p = argparse.ArgumentParser(description="Rolling retrain (RF + Torch)")
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--timeframe", type=str, default=None)
    p.add_argument("--lookback_days", type=int, default=90)
    p.add_argument("--interval_days", type=int, default=7)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    cfg = Config()
    ex = make_exchange(cfg, demo=False)
    Path("runs").mkdir(parents=True, exist_ok=True)

    def run_once():
        # estimate limit and trim by time
        per_day = 24 if (args.timeframe or cfg.timeframe).endswith("h") else 24*4
        limit = max(1500, args.lookback_days * per_day + 500)
        df = fetch_ohlcv_df(ex, args.symbol or cfg.symbol, args.timeframe or cfg.timeframe, limit=limit)
        cutoff = utc_now() - timedelta(days=args.lookback_days)
        df = df[df.index >= cutoff]
        res_rf = train_meta(df, cfg)
        res_t = train_meta_torch(df, cfg, epochs=40)
        print(f"[{utc_now()}] retrain done -> RF: {res_rf}, Torch: {res_t}")

    if args.once:
        run_once(); return
    while True:
        run_once()
        time.sleep(args.interval_days * 24 * 3600)

if __name__ == "__main__":
    main()
