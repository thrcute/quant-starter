
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    exchange_id: str = os.getenv("EXCHANGE_ID", "okx")
    api_key: str = os.getenv("OKX_API_KEY", "")
    api_secret: str = os.getenv("OKX_API_SECRET", "")
    api_password: str = os.getenv("OKX_API_PASSWORD", "")
    symbol: str = os.getenv("SYMBOL", "BTC/USDT")
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    base_order_size: float = float(os.getenv("BASE_ORDER_SIZE", "0.001"))
    max_risk_pct: float = float(os.getenv("MAX_RISK_PCT", "0.01"))
    fee_rate: float = float(os.getenv("FEE_RATE", "0.001"))
    slippage_pct: float = float(os.getenv("SLIPPAGE_PCT", "0.0005"))

    fast_ma: int = int(os.getenv("FAST_MA", "20"))
    slow_ma: int = int(os.getenv("SLOW_MA", "50"))
    rsi_len: int = int(os.getenv("RSI_LEN", "14"))
    rsi_buy: float = float(os.getenv("RSI_BUY", "55"))
    rsi_sell: float = float(os.getenv("RSI_SELL", "45"))
