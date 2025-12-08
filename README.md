
# Quant Starter — Backtest + Live + Meta (RF/Torch) + Ensemble + Retrain

开箱即用的量化入门框架：
- ✅ 向量化回测（含手续费/滑点）
- ✅ 实盘循环（OKX/ccxt，纸面/模拟/真盘）
- ✅ 元策略选择（RandomForest & PyTorch-MLP）
- ✅ 集成投票（RF+Torch）
- ✅ 滚动再训练（定期更新模型）

> 强烈建议：先纸面/模拟盘，控制仓位。

## 1) 安装
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 配置
复制 `.env.example` → `.env`，按需填写：交易所、API Key、策略参数等。

## 3) 回测
```bash
python main.py backtest --symbol BTC/USDT --timeframe 1h --limit 1000
```

## 4) 实盘
- 纸面：
```bash
python main.py live --paper true
```
- OKX 模拟盘：
```bash
python main.py live --paper false --demo true
```
- 真盘（谨慎）：
```bash
python main.py live --paper false --demo false
```

## 5) 元策略（RF 版 / Torch 版）
- 训练：
```bash
python meta_train.py --symbol BTC/USDT --timeframe 1h --limit 2000
python meta_train_torch.py --symbol BTC/USDT --timeframe 1h --limit 3000 --epochs 50
```
- 推理：
```bash
python meta_infer.py
python meta_infer_torch.py
```

## 6) 集成投票（Ensemble）
- 一次性查看投票：
```bash
python meta_ensemble.py --symbol BTC/USDT --timeframe 1h --limit 500 --w_rf 0.5 --w_torch 0.5
```
- 实盘投票：
```python
from trader_meta_ensemble import trade_loop_meta_ensemble
from config import Config
trade_loop_meta_ensemble(Config(), paper=True, demo=True, w_rf=0.5, w_torch=0.5)
```

## 7) 滚动再训练（保持模型新鲜）
- 单次：
```bash
python retrain.py --symbol BTC/USDT --timeframe 1h --lookback_days 90 --once
```
- 周期（每 7 天）：
```bash
python retrain.py --symbol BTC/USDT --timeframe 1h --lookback_days 90 --interval_days 7
```

---
**免责声明**：本项目仅用于学习交流，不构成任何投资建议。使用风险自负。
