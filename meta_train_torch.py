
import argparse
from pathlib import Path
from config import Config
from data import make_exchange, fetch_ohlcv_df
from meta_torch import train_meta_torch

def main():
    p = argparse.ArgumentParser(description="Train Torch meta selector")
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--timeframe", type=str, default=None)
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    cfg = Config()
    ex = make_exchange(cfg, demo=False)
    df = fetch_ohlcv_df(ex, args.symbol or cfg.symbol, args.timeframe or cfg.timeframe, limit=args.limit)
    Path("runs").mkdir(parents=True, exist_ok=True)
    res = train_meta_torch(df, cfg, epochs=args.epochs)
    print("Meta Torch trained:", res)

if __name__ == "__main__":
    main()
