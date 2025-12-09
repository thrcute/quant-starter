import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time

# ================= 配置页面 =================
st.set_page_config(
    page_title="Quant Starter 监控台",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Quant Starter 实盘监控")

# ================= 侧边栏设置 =================
with st.sidebar:
    st.header("设置")
    refresh_rate = st.slider("刷新频率 (秒)", 1, 60, 5)
    
    st.divider()
    
    # 选择日志文件 (兼容普通模式和集成模式)
    log_options = ["live_log.txt", "live_log_meta_ensemble.txt"]
    selected_log = st.selectbox("选择日志文件", log_options, index=0)
    
    st.divider()
    st.markdown("### 文件路径检查")
    runs_dir = Path("runs")
    if not runs_dir.exists():
        st.error("❌ 找不到 runs/ 目录，请先运行回测或实盘。")
    else:
        st.success(f"✅ 已连接 runs/ 目录")

# ================= 数据加载函数 =================
@st.cache_data(ttl=refresh_rate)
def load_data():
    equity_path = Path("runs/equity.csv")
    trades_path = Path("runs/trades.csv")
    
    eq_df = pd.DataFrame()
    tr_df = pd.DataFrame()
    
    if equity_path.exists():
        eq_df = pd.read_csv(equity_path)
        eq_df["datetime"] = pd.to_datetime(eq_df["datetime"])
        
    if trades_path.exists():
        tr_df = pd.read_csv(trades_path)
        if not tr_df.empty:
            tr_df["entry_time"] = pd.to_datetime(tr_df["entry_time"])
            tr_df["exit_time"] = pd.to_datetime(tr_df["exit_time"])

    return eq_df, tr_df

def load_logs(filename, n_lines=50):
    log_path = Path(f"runs/{filename}")
    if log_path.exists():
        # 读取最后 N 行
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return "".join(lines[-n_lines:])
    return "等待日志生成..."

# ================= 主界面逻辑 =================

# 1. 自动刷新占位符
placeholder = st.empty()

# 循环逻辑 (利用 Streamlit 的 rerun 机制，或者手动刷新)
# 这里我们使用简单的加载展示，配合 st.empty 实现局部刷新感

equity_df, trades_df = load_data()

# --- 第一部分：核心指标 ---
col1, col2, col3, col4 = st.columns(4)

current_equity = 1.0
total_return = 0.0
win_rate = 0.0
trade_count = 0

if not equity_df.empty:
    current_equity = equity_df.iloc[-1]["equity"]
    initial_equity = equity_df.iloc[0]["equity"]
    total_return = (current_equity - initial_equity) / initial_equity * 100

if not trades_df.empty:
    trade_count = len(trades_df)
    win_count = len(trades_df[trades_df["pnl_pct"] > 0])
    win_rate = (win_count / trade_count) * 100

col1.metric("当前净值", f"{current_equity:.4f}")
col2.metric("总收益率", f"{total_return:.2f}%", delta_color="normal")
col3.metric("交易次数", f"{trade_count}")
col4.metric("胜率", f"{win_rate:.1f}%")

# --- 第二部分：图表 ---
col_chart, col_log = st.columns([2, 1])

with col_chart:
    st.subheader("资金曲线 (Equity Curve)")
    if not equity_df.empty:
        fig = px.line(equity_df, x="datetime", y="equity", title=None)
        fig.update_layout(xaxis_title="时间", yaxis_title="净值", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无资金数据，请运行策略生成 runs/equity.csv")

    st.subheader("最近交易记录")
    if not trades_df.empty:
        # 格式化一下显示
        display_df = trades_df.copy()
        display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x*100:.2f}%")
        display_df = display_df.sort_values("entry_time", ascending=False).head(10)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("暂无交易记录")

# --- 第三部分：日志监控 ---
with col_log:
    st.subheader("实时日志")
    log_content = load_logs(selected_log)
    st.text_area("Log Output", log_content, height=500, disabled=True)

# 自动刷新逻辑
time.sleep(refresh_rate)
st.rerun()