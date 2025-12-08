
import argparse
from typing import Tuple
import pandas as pd

from config import Config
from data import make_exchange, fetch_ohlcv_df
from meta import predict_class as predict_class_rf, class_to_target as class_to_target_rf
from meta_torch import predict_class_torch

def ensemble_vote(df: pd.DataFrame, w_rf: float = 0.5, w_torch: float = 0.5) -> Tuple[int, int, int]:
    cls_rf = predict_class_rf(df)
    cls_t = predict_class_torch(df)
    scores = {0:0.0,1:0.0,2:0.0,3:0.0}
    scores[cls_rf]+=w_rf; scores[cls_t]+=w_torch
    order=[1,2,3,0]
    best = max(scores.items(), key=lambda kv: (kv[1], order.index(kv[0]) if kv[0] in order else 99))[0]
    return cls_rf, cls_t, best

def main():
    p = argparse.ArgumentParser(description="Meta Ensemble vote (RF + Torch)")
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--timeframe", type=str, default=None)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--w_rf", type=float, default=0.5)
    p.add_argument("--w_torch", type=float, default=0.5)
    args = p.parse_args()

    cfg = Config()
    ex = make_exchange(cfg, demo=False)
    df = fetch_ohlcv_df(ex, args.symbol or cfg.symbol, args.timeframe or cfg.timeframe, limit=args.limit)
    cls_rf, cls_t, cls = ensemble_vote(df, args.w_rf, args.w_torch)
    target = class_to_target_rf(df, cfg, cls)
    print(f"RF={cls_rf}, Torch={cls_t} -> Ensemble={cls}, target={target} (1=long,0=flat)")

if __name__ == "__main__":
    main()
