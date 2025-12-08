
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from strategy import compute_signals
from config import Config

@dataclass
class BTResult:
    stats: Dict[str, float]
    trades: pd.DataFrame
    equity: pd.Series

def run_backtest(df: pd.DataFrame, cfg: Config) -> BTResult:
    sig = compute_signals(df, cfg)
    pos = sig.long.shift(1).fillna(0)
    px = df["open"]
    ret = px.pct_change().fillna(0)
    gross = pos * ret

    trade_change = pos.diff().fillna(pos)
    fee = (abs(trade_change) * cfg.fee_rate)
    slippage = abs(trade_change) * cfg.slippage_pct
    net = gross - fee - slippage

    equity = (1 + net).cumprod()

    trades = []
    in_trade=False; entry_px=None; entry_time=None
    for i in range(1,len(pos)):
        if not in_trade and pos.iloc[i-1]==0 and pos.iloc[i]==1:
            in_trade=True; entry_px=px.iloc[i]; entry_time=px.index[i]
        elif in_trade and pos.iloc[i-1]==1 and pos.iloc[i]==0:
            exit_px=px.iloc[i]; exit_time=px.index[i]
            pnl=(exit_px-entry_px)/entry_px - 2*(cfg.fee_rate+cfg.slippage_pct)
            trades.append({"entry_time":entry_time,"entry_px":entry_px,"exit_time":exit_time,"exit_px":exit_px,"pnl_pct":pnl})
            in_trade=False
    trades_df = pd.DataFrame(trades)

    total_return = equity.iloc[-1]-1
    daily_ret = net.resample("1D").sum(min_count=1).fillna(0)
    sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252)
    running_max = equity.cummax()
    dd = equity / running_max - 1
    max_dd = dd.min()
    win_rate = (trades_df["pnl_pct"]>0).mean() if len(trades_df)>0 else 0.0

    stats = {
        "bars": int(len(df)),
        "trades": int(len(trades_df)),
        "total_return_pct": float(total_return*100),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_dd*100),
        "win_rate_pct": float(win_rate*100),
    }
    return BTResult(stats=stats, trades=trades_df, equity=equity)
