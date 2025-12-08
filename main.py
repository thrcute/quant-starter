
import argparse
from pathlib import Path
from config import Config
from data import make_exchange, fetch_ohlcv_df
from backtest import run_backtest
from trader import trade_loop
import os
os.environ["ALL_PROXY"] = "socks5://127.0.0.1:10808"

def do_backtest(args):
    cfg = Config()
    ex = make_exchange(cfg, demo=False)
    df = fetch_ohlcv_df(ex, args.symbol or cfg.symbol, args.timeframe or cfg.timeframe, limit=args.limit)
    res = run_backtest(df, cfg)
    print("=== Backtest Stats ===")
    for k,v in res.stats.items():
        print(f"{k:22s}: {v}")
    Path("runs").mkdir(parents=True, exist_ok=True)
    res.trades.to_csv("runs/trades.csv", index=False)
    res.equity.to_csv("runs/equity.csv", header=["equity"])
    print("Saved -> runs/trades.csv, runs/equity.csv")

def do_live(args):
    cfg = Config()
    paper = str(args.paper).lower() in ["1","true","yes","y"]
    demo = str(args.demo).lower() in ["1","true","yes","y"]
    trade_loop(cfg, paper=paper, demo=demo)

def main():
    p = argparse.ArgumentParser(description="Quant Starter")
    sub = p.add_subparsers(dest="cmd")

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--symbol", type=str, default=None)
    p_bt.add_argument("--timeframe", type=str, default=None)
    p_bt.add_argument("--limit", type=int, default=1000)
    p_bt.set_defaults(func=do_backtest)

    p_live = sub.add_parser("live")
    p_live.add_argument("--paper", type=str, default="true")
    p_live.add_argument("--demo", type=str, default="false")
    p_live.set_defaults(func=do_live)

    args = p.parse_args()
    if not args.cmd:
        p.print_help(); return
    args.func(args)

if __name__ == "__main__":
    main()
