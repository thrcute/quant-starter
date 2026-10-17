
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

### 回测成交与对账口径

- 第 `t` 根收盘生成信号，在第 `t+1` 根开盘成交；策略信号逻辑不变。
- 净值记录为每根开盘成交后的权益。`open[t-1] -> open[t]` 的涨跌由上一根开盘后的持仓承担，建仓前和平仓后的涨跌不计入收益。
- 沿用全仓做多/空仓的归一化回测，单边成本近似为 `c = fee_rate + slippage_pct`。每次成交从当时权益扣除比例 `c`；持仓期间不重复扣费。滑点仍是比例成本近似，并未模拟成交价或订单簿。
- 净值逐期因子为 `(1 + previous_position * open_return) * (1 - abs(position_change) * c)`。已平仓单笔收益 `pnl_pct = (exit_px / entry_px) * (1-c)**2 - 1`，与净值使用相同复利口径；日收益也按复利汇总。
- 期末空仓时，最终净值等于所有已平仓交易的 `(1+pnl_pct)` 连乘。期末仍持仓时，再乘未平仓部分的 `(last_open / entry_open) * (1-c)`。交易表只记录已平仓交易，不强制平仓，也不提前扣除平仓成本；最后一根收盘信号没有下一根开盘时不执行。

例如开盘价 `[100,100,110,121,121]`、收盘信号 `[0,1,1,0,0]`，在 110 买入、121 卖出，默认 `c=0.0015` 时最终净值为 `1.096702475`，单笔净收益为 `9.6702475%`。直接从毛收益减去双边成本得到的 `9.70%` 是另一种线性近似，不再用作交易对账值。

### 离线回归测试

在仓库根目录执行（仅需以下依赖，无需 API Key、ccxt 或交易所连接）：

```bash
python -m pip install "pandas>=2.2.2" "numpy>=1.26.4" "python-dotenv>=1.0.1"
python -m unittest discover -s tests -v
```

测试使用合成 OHLC 行情和独立的现金/持币数量账本，覆盖建仓/平仓跳空、零成本与双边成本、多笔交易、期末未平仓及最后一根信号，并包含真实 MA/RSI 信号的集成测试。测试中禁用 socket，不运行 `main.py` 或实盘入口。

仓库已有 `runs/*.csv` 是历史输出，未随本次修复重算；缺少对应原始行情和期末仓位时，不能将其净值与已平仓交易的差额全部归因于时间错位。

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
