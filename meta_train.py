
import argparse
from pathlib import Path
from config import Config
from data import make_exchange, fetch_ohlcv_df
from meta import train_meta

def main():
    p = argparse.ArgumentParser(description="Train meta RF selector")
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--timeframe", type=str, default=None)
    p.add_argument("--limit", type=int, default=2000)
    args = p.parse_args()

    cfg = Config()
    ex = make_exchange(cfg, demo=False)
    df = fetch_ohlcv_df(ex, args.symbol or cfg.symbol, args.timeframe or cfg.timeframe, limit=args.limit)
    Path("runs").mkdir(parents=True, exist_ok=True)
    res = train_meta(df, cfg)
    print("Meta RF trained:", res)

if __name__ == "__main__":
    main()
