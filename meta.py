
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier

from config import Config
from strategy import compute_signals
from strategies_extra import breakout_long, mean_reversion_long

META_PATH = "runs/meta_model.pkl"
META_INFO = "runs/meta_info.json"

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    px = df["close"]
    feat = pd.DataFrame(index=df.index)
    feat["ret1"] = px.pct_change()
    feat["ret5"] = px.pct_change(5)
    feat["ret20"] = px.pct_change(20)
    feat["vol20"] = px.pct_change().rolling(20).std().fillna(0)
    feat["ma20"] = px.rolling(20).mean()
    feat["ma50"] = px.rolling(50).mean()
    feat["ma_ratio"] = (feat["ma20"] / (feat["ma50"] + 1e-12) - 1).fillna(0)
    roll_hi = df["high"].rolling(20).max()
    roll_lo = df["low"].rolling(20).min()
    feat["pos_in_range"] = ((px - roll_lo) / (roll_hi - roll_lo + 1e-12)).fillna(0.5)
    if "volume" in df:
        v = df["volume"].astype(float)
        feat["v_z"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    else:
        feat["v_z"] = 0.0
    return feat.fillna(0)

def build_labels(df: pd.DataFrame, cfg: Config):
    sig_ma = compute_signals(df, cfg).long
    sig_bo = breakout_long(df, 20, 10)
    sig_mr = mean_reversion_long(df, 20, -1.0, -0.1)
    pos = {1:sig_ma, 2:sig_bo, 3:sig_mr}

    ret_next = df["open"].pct_change().shift(-1).fillna(0)
    cost = 2 * (cfg.fee_rate + cfg.slippage_pct)
    pay = pd.DataFrame({k: v*ret_next - cost*(v.diff().abs().fillna(v)) for k,v in pos.items()})
    pay[0] = 0.0
    y = pay.idxmax(axis=1).astype(int)

    X = make_features(df)
    start = max(60,50)
    X = X.iloc[start:-1]
    y = y.iloc[start:-1]

    id2name = {0:"flat", 1:"ma_rsi", 2:"breakout", 3:"mean_reversion"}
    return X,y,id2name

def train_meta(df: pd.DataFrame, cfg: Config, *, n_estimators: int = 200, random_state: int = 0):
    X,y,id2name = build_labels(df, cfg)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight="balanced")
    clf.fit(X,y)
    joblib.dump(clf, META_PATH)
    with open(META_INFO, "w", encoding="utf-8") as f:
        json.dump({"id2name":id2name, "features":list(X.columns)}, f, indent=2)
    acc = clf.score(X,y)
    return {"samples": int(len(X)), "train_acc": float(acc)}

def load_meta():
    clf = joblib.load(META_PATH)
    with open(META_INFO, "r", encoding="utf-8") as f:
        info = json.load(f)
    return clf, info

def predict_class(df: pd.DataFrame) -> int:
    clf, info = load_meta()
    X = make_features(df).tail(1)[info["features"]]
    pred = int(clf.predict(X)[0])
    return pred

def class_to_target(df: pd.DataFrame, cfg: Config, cls: int) -> int:
    if cls == 0:
        return 0
    elif cls == 1:
        pos = compute_signals(df, cfg).long
    elif cls == 2:
        pos = breakout_long(df, 20, 10)
    else:
        pos = mean_reversion_long(df, 20, -1.0, -0.1)
    return int(pos.iloc[-2])
