
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
    """Execute close signals at the next open; value equity after that execution.

    Positions are long/flat. Fee plus slippage is a proportional equity haircut
    at each entry/exit, preserving the existing simplified cost model. An open
    final position is marked at the last open, without a forced exit or exit cost.
    """
    sig = compute_signals(df, cfg)
    # Position immediately AFTER trading at this row's open.
    pos = sig.long.shift(1).fillna(0)
    px = df["open"]
    ret = px.pct_change().fillna(0)
    # The previous position earned the move ending at this open.
    gross = pos.shift(1).fillna(0) * ret

    trade_change = pos.diff().fillna(pos)
    cost_rate = cfg.fee_rate + cfg.slippage_pct
    cost = abs(trade_change) * cost_rate
    # Mark the existing holding first, then deduct this open's trading cost.
    net = (1 + gross) * (1 - cost) - 1

    equity = (1 + net).cumprod()

    trades = []
    in_trade=False; entry_px=None; entry_time=None
    for i in range(1,len(pos)):
        if not in_trade and pos.iloc[i-1]==0 and pos.iloc[i]==1:
            in_trade=True; entry_px=px.iloc[i]; entry_time=px.index[i]
        elif in_trade and pos.iloc[i-1]==1 and pos.iloc[i]==0:
            exit_px=px.iloc[i]; exit_time=px.index[i]
            pnl=(exit_px/entry_px) * (1-cost_rate)**2 - 1
            trades.append({"entry_time":entry_time,"entry_px":entry_px,"exit_time":exit_time,"exit_px":exit_px,"pnl_pct":pnl})
            in_trade=False
    trades_df = pd.DataFrame(trades)

    total_return = equity.iloc[-1]-1
    daily_ret = (1 + net).resample("1D").prod() - 1
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
