"""基礎策略（EWMAC+Carry, pysystemtrade ch15 預設）、影子帳戶、標籤、特徵。"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INST = ["CORN", "SOYBEAN", "WHEAT", "COFFEE", "COCOA", "SUGAR11"]
ZH = {"CORN": "玉米", "SOYBEAN": "黃豆", "WHEAT": "小麥", "COFFEE": "咖啡", "COCOA": "可可", "SUGAR11": "11號糖"}
K = 1_000_000.0; VT = 0.10; H = 5; KAPPA = 1.5
cfg = pd.read_csv(f"{ROOT}/data/raw/instrumentconfig.csv").set_index("Instrument")
sc = pd.read_csv(f"{ROOT}/data/raw/spreadcosts.csv").set_index("Instrument")
RULES = {"ewmac16_64": (16, 64, 3.75), "ewmac32_128": (32, 128, 2.65), "ewmac64_256": (64, 256, 1.87)}
W = {"ewmac16_64": 0.21, "ewmac32_128": 0.08, "ewmac64_256": 0.21, "carry": 0.50}
FEATS = ["ret20_v", "ret60_v", "fc_ewmac16_64", "fc_carry", "vol20_ann", "vol_ratio_20_120", "dist_hi60_v",
         "dist_lo60_v", "dir", "abs_fc", "shadow_ret20", "shadow_dd250", "shadow_losing_streak",
         "days_since_flip", "days_to_roll"]
SHADOW_FEATS = ["shadow_ret20", "shadow_dd250", "shadow_losing_streak"]


def cost_per_contract(i, mult=1.0):
    return mult * (sc.loc[i, "SpreadCost"] * cfg.loc[i, "Pointsize"] + cfg.loc[i, "PerBlock"])


def simulate_positions(d, target, i, cost_mult=1.0):
    """v2：target_t 在 t 收盤後決定、t+1 收盤成交；持有契約當日無報價（valued=0）則不成交、沿用前一持倉。
    換月在 roll_today=1 當日收盤：舊倉 |held_{d-1}| 平倉、新倉 |held_d| 開倉。
    unscorable=1（資料源提前換月、舊倉當日價格不可得）之毛損益為 NaN，由評估端對所有政策一致排除。"""
    ps = cfg.loc[i, "Pointsize"]; cpc = cost_per_contract(i, cost_mult)
    tradeable = d.valued == 1
    unsc = d.unscorable == 1 if "unscorable" in d else pd.Series(False, index=d.index)
    held = target.shift(1).where(tradeable).ffill().fillna(0.0)   # 收盤 d 成交後持有
    prev = held.shift(1).fillna(0.0)
    gross = prev * d.dP * ps                                       # 區間 (d-1, d]；unscorable 日為 NaN
    roll = d.roll_today == 1
    trade = (held - prev).abs().where(~roll, 0.0)
    roll_qty = (prev.abs() + held.abs()).where(roll, 0.0)
    cost = (trade + roll_qty) * cpc
    cost = cost.where(~unsc, 0.0) + cost.where(unsc, 0.0).shift(1).fillna(0.0)   # 不可計分日的換月成本記入下一日
    return pd.DataFrame({"held": held, "gross": gross, "cost": cost, "trade_qty": trade, "roll_qty": roll_qty})


def build_instrument(i):
    d = pd.read_csv(f"{ROOT}/data/processed/{i}_v2.csv", index_col=0, parse_dates=True)   # v2 因果換月鏈
    ps = cfg.loc[i, "Pointsize"]
    obs = (d.valued == 1) & d.dP.notna()
    dPo = d.dP.where(obs)                                          # 只用實際觀測到的價差估波動
    adj = d.dP_feat.fillna(0).cumsum()                            # 不可計分日以新契約同日價差橋接（僅供特徵）
    sig = dPo.ewm(span=35, min_periods=30).std().ffill()
    f = pd.DataFrame(index=d.index)
    for k, (a, b, s) in RULES.items():
        raw = (adj.ewm(span=a, min_periods=a).mean() - adj.ewm(span=b, min_periods=b).mean()) / sig
        f[k] = (raw * s).clip(-20, 20)
    cy = d.carry_yrs.where(d.carry_yrs.abs() > 1e-6)
    rc = ((d.carry_price - d.carry_ref_price) / cy / (sig * 16)).replace([np.inf, -np.inf], np.nan).ffill()
    f["carry"] = (rc.ewm(com=90, min_periods=60).mean() * 30).clip(-20, 20)
    comb = (sum(W[k] * f[k] for k in W) * 1.31).clip(-20, 20)
    valid = f.notna().all(axis=1) & sig.notna()
    n0 = (comb / 10 * (VT * K / 16) / (sig * ps)).where(valid, 0.0).fillna(0.0)
    sim = simulate_positions(d, n0, i)
    r0 = (sim.gross - sim.cost) / K
    gross_feat = sim.held.shift(1).fillna(0.0) * d.dP_feat * ps
    r0_feat = (gross_feat - sim.cost) / K                          # 僅供影子特徵：不可計分日用橋接估值
    out = pd.DataFrame({"price": d.price.ffill(), "dP": d.dP, "contract": d.contract, "roll_today": d.roll_today,
                        "valued": d.valued, "unscorable": d.unscorable,
                        "sigma_price": sig, "fc_comb": comb, "n0": n0, "valid": valid.astype(int),
                        "r0": r0, "r0_gross": sim.gross / K, "r0_cost": sim.cost / K})
    for k in W:
        out["fc_" + k] = f[k]
    # ---- 標籤：決策 j，執行 j+1，區間 r0[j+2 .. j+1+H] ----
    r = r0.values; n = len(r)
    L = np.full(n, np.nan); avail = np.full(n, np.nan)
    for j in range(n - H - 1):
        w = r[j + 2:j + 2 + H]
        if np.isnan(w).any():
            continue                                                # 窗內含不可計分日 → 標籤未知
        c = np.cumsum(w); L[j] = max(0.0, -c.min())
    idx = np.arange(n)
    av = idx + H + 1
    out["label_available_date"] = [d.index[k] if k < n else pd.NaT for k in av]
    out["L0"] = L
    s0 = r0.where(out.valid.shift(1).fillna(0) == 1).ewm(span=60, min_periods=40).std()
    floor = s0.expanding(min_periods=40).median() * 0.25
    out["sigma0"] = np.maximum(s0, floor)
    out["budget"] = KAPPA * out.sigma0 * np.sqrt(H)
    out["y"] = np.where(np.isnan(L), np.nan, (L > out.budget).astype(float))
    # ---- 特徵（截至 t 收盤）----
    out["ret20_v"] = (adj - adj.shift(20)) / (sig * np.sqrt(20))
    out["ret60_v"] = (adj - adj.shift(60)) / (sig * np.sqrt(60))
    pr = dPo / d.price.ffill().shift(1)
    out["vol20_ann"] = pr.rolling(20, min_periods=15).std() * 16
    out["vol_ratio_20_120"] = dPo.rolling(20, min_periods=15).std() / dPo.rolling(120, min_periods=90).std()
    out["dist_hi60_v"] = (adj.rolling(60).max() - adj) / (sig * np.sqrt(60))
    out["dist_lo60_v"] = (adj - adj.rolling(60).min()) / (sig * np.sqrt(60))
    out["dir"] = np.sign(comb)
    out["abs_fc"] = comb.abs()
    out["shadow_ret20"] = r0_feat.rolling(20, min_periods=15).sum()
    eq = r0_feat.fillna(0).cumsum()
    out["shadow_dd250"] = eq.rolling(250, min_periods=60).max() - eq
    neg = (r0_feat < 0).astype(int)
    out["shadow_losing_streak"] = neg.groupby((neg == 0).cumsum()).cumsum()
    sg = np.sign(comb)
    out["days_since_flip"] = sg.groupby((sg != sg.shift()).cumsum()).cumcount().clip(upper=250)
    out["days_to_roll"] = d.days_to_roll.clip(upper=200)          # v2：只用過去換月紀錄排定的計畫換月日
    out["instrument"] = i
    return out


def build_all():
    os.makedirs(f"{ROOT}/outputs/panel", exist_ok=True)
    P = {}
    for i in INST:
        P[i] = build_instrument(i)
        P[i].to_csv(f"{ROOT}/outputs/panel/{i}.csv")
    return P


def load_all():
    return {i: pd.read_csv(f"{ROOT}/outputs/panel/{i}.csv", index_col=0, parse_dates=True,
                           ) .assign(label_available_date=lambda x: pd.to_datetime(x.label_available_date)) for i in INST}


if __name__ == "__main__":
    P = build_all()
    for i, o in P.items():
        o2 = o.loc["2008":]
        ann = o2.r0.mean() * 256; vol = o2.r0.std() * 16
        s = o2[(o2.valid == 1) & o2.y.notna()]
        print(f"{i:8s} B0 2008+ ann={ann:.3%} vol={vol:.3%} SR={ann/vol:.2f} cost/yr={o2.r0_cost.mean()*256:.3%} "
              f"event_rate={s.y.mean():.3f} n={len(s)} first_valid={o.index[o.valid==1][0].date()}")
