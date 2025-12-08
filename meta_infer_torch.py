
from config import Config
from data import make_exchange, fetch_ohlcv_df
from meta_torch import predict_class_torch, class_to_target

def main():
    cfg = Config()
    ex = make_exchange(cfg, demo=False)
    df = fetch_ohlcv_df(ex, cfg.symbol, cfg.timeframe, limit=500)
    cls = predict_class_torch(df)
    target = class_to_target(df, cfg, cls)
    print(f"Meta Torch -> class={cls} target={target} (1=long,0=flat)")

if __name__ == "__main__":
    main()
