
import json, math, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict

from config import Config
from strategies_extra import breakout_long, mean_reversion_long
from strategy import compute_signals

META_TORCH_PATH = "runs/meta_torch.pt"
META_TORCH_INFO = "runs/meta_torch_info.json"

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
    X = X.iloc[60:-1]
    y = y.iloc[60:-1]
    id2name = {0:"flat",1:"ma_rsi",2:"breakout",3:"mean_reversion"}
    return X,y,id2name

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_classes: int = 4, pdrop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x): return self.net(x)

def standardize(train_X, test_X):
    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True) + 1e-8
    return (train_X-mean)/std, (test_X-mean)/std, mean.squeeze().tolist(), std.squeeze().tolist()

def time_split(n:int, frac:float=0.8):
    cut = int(n*frac)
    return list(range(0,cut)), list(range(cut,n))

def train_meta_torch(df: pd.DataFrame, cfg: Config, epochs: int = 50, lr: float = 1e-3, batch: int = 64, hidden: int = 64, pdrop: float = 0.1):
    Xdf, yser, id2name = build_labels(df, cfg)
    feats = list(Xdf.columns)
    X = Xdf.values.astype("float32")
    y = yser.values.astype("int64")

    tr_idx, va_idx = time_split(len(X), 0.8)
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    Xtr, Xva, mean, std = standardize(Xtr, Xva)

    dev = torch.device("cpu")
    model = MLP(in_dim=X.shape[1], hidden=hidden, num_classes=4, pdrop=pdrop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dtr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=batch, shuffle=False)
    dva = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=batch, shuffle=False)

    best=-1.0; best_state=None; patience=8; bad=0
    for ep in range(1,epochs+1):
        model.train()
        for xb,yb in dtr:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            n=0; correct=0
            for xb,yb in dva:
                pred = model(xb).argmax(dim=1)
                correct += int((pred==yb).sum())
                n += len(xb)
            acc = correct/max(1,n)
        if acc>best: best=acc; best_state=model.state_dict(); bad=0
        else: bad+=1
        if bad>=patience: break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs("runs", exist_ok=True)
    torch.save(model.state_dict(), META_TORCH_PATH)
    with open(META_TORCH_INFO, "w", encoding="utf-8") as f:
        json.dump({"id2name":id2name,"features":feats,"mean":mean,"std":std,"hidden":hidden,"pdrop":pdrop}, f, indent=2)
    return {"samples": int(len(X)), "val_acc": float(best)}

def load_meta_torch():
    with open(META_TORCH_INFO, "r", encoding="utf-8") as f:
        info = json.load(f)
    model = MLP(in_dim=len(info["features"]), hidden=info.get("hidden",64), pdrop=info.get("pdrop",0.1))
    state = torch.load(META_TORCH_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, info

def predict_class_torch(df: pd.DataFrame) -> int:
    model, info = load_meta_torch()
    X = make_features(df)[info["features"]].tail(1).values.astype("float32")
    mean = np.array(info["mean"], dtype="float32")
    std = np.array(info["std"], dtype="float32")
    X = (X - mean) / (std + 1e-8)
    with torch.no_grad():
        pred = model(torch.tensor(X)).argmax(dim=1).item()
    return int(pred)

def class_to_target(df: pd.DataFrame, cfg: Config, cls: int) -> int:
    if cls == 0: return 0
    from strategy import compute_signals
    if cls == 1:
        pos = compute_signals(df, cfg).long
    elif cls == 2:
        pos = breakout_long(df, 20, 10)
    else:
        pos = mean_reversion_long(df, 20, -1.0, -0.1)
    return int(pos.iloc[-2])
