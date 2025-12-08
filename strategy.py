
import numpy as np
import pandas as pd
from dataclasses import dataclass
from config import Config

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

@dataclass
class Signals:
    ma_fast: pd.Series
    ma_slow: pd.Series
    rsi: pd.Series
    long: pd.Series

def compute_signals(df: pd.DataFrame, cfg: Config) -> Signals:
    price = df["close"]
    ma_fast = price.rolling(cfg.fast_ma).mean()
    ma_slow = price.rolling(cfg.slow_ma).mean()
    r = rsi(price, cfg.rsi_len)

    enter = (ma_fast > ma_slow) & (r > cfg.rsi_buy)
    exit_ = (ma_fast < ma_slow) | (r < cfg.rsi_sell)

    long_pos = pd.Series(0, index=df.index, dtype=float)
    hold = False
    for i in range(len(df)):
        if not hold and bool(enter.iloc[i]): hold=True
        elif hold and bool(exit_.iloc[i]): hold=False
        long_pos.iloc[i] = 1.0 if hold else 0.0
    return Signals(ma_fast=ma_fast, ma_slow=ma_slow, rsi=r, long=long_pos)
