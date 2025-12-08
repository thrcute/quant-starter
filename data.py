
import ccxt
import pandas as pd
from typing import Optional
from config import Config

import ccxt

def make_exchange(cfg: Config, *, demo: bool = False):
    # 本地代理配置 —— 记得改成你自己的端口
    # v2rayN 默认：10808 = SOCKS5，10809 = HTTP
    socks_proxy = "socks5h://127.0.0.1:10808"

    ex = getattr(ccxt, cfg.exchange_id)({
        "apiKey": cfg.api_key,
        "secret": cfg.api_secret,
        "password": cfg.api_password,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},

        # 关键：让 ccxt 的 requests 走代理（包括 DNS）
        "proxies": {
            "http":  socks_proxy,
            "https": socks_proxy,
            # 如果你更习惯 HTTP 代理，可以写：
            # "http":  "http://127.0.0.1:10809",
            # "https": "http://127.0.0.1:10809",
        },
    })

    # （可选）防止 requests 读系统环境里的 HTTP_PROXY/HTTPS_PROXY 乱入
    # ex.session.trust_env = False

    # OKX 模拟盘 header
    if demo and cfg.exchange_id == "okx":
        ex.headers = ex.headers or {}
        ex.headers["x-simulated-trading"] = "1"

    return ex


def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int = 500, since: Optional[int] = None) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("UTC")
    return df.set_index("datetime")
