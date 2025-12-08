
from config import Config
from data import make_exchange, fetch_ohlcv_df
from meta import predict_class, class_to_target

def main():
    cfg = Config()
    ex = make_exchange(cfg, demo=False)
    df = fetch_ohlcv_df(ex, cfg.symbol, cfg.timeframe, limit=400)
    cls = predict_class(df)
    target = class_to_target(df, cfg, cls)
    print(f"Meta RF -> class={cls} target={target} (1=long,0=flat)")

if __name__ == "__main__":
    main()
