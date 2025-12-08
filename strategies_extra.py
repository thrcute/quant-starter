
import pandas as pd
import numpy as np

def _build_pos(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    pos = pd.Series(0, index=entry.index, dtype=float)
    hold = False
    for i in range(len(entry)):
        if not hold and bool(entry.iloc[i]): hold=True
        elif hold and bool(exit_.iloc[i]): hold=False
        pos.iloc[i] = 1.0 if hold else 0.0
    return pos

def breakout_long(df: pd.DataFrame, lookback: int = 20, exit_lb: int = 10) -> pd.Series:
    hi = df["high"].rolling(lookback).max()
    lo = df["low"].rolling(exit_lb).min()
    entry = df["close"] >= hi.shift(1)
    exit_ = df["close"] <= lo.shift(1)
    return _build_pos(entry, exit_)

def mean_reversion_long(df: pd.DataFrame, bb_len: int = 20, z_entry: float = -1.0, z_exit: float = -0.1) -> pd.Series:
    ma = df["close"].rolling(bb_len).mean()
    std = df["close"].rolling(bb_len).std(ddof=0)
    z = (df["close"] - ma) / (std.replace(0, np.nan))
    entry = z <= z_entry
    exit_ = z >= z_exit
    return _build_pos(entry, exit_)
