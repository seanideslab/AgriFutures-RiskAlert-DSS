"""第17.3節核心驗收：未來資料不影響過去、標籤成熟、決策可重播、帳可對、狀態跨年連續。"""
import os, sys, shutil, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
import core
from experiment import step, NORMAL, REDUCED, PAUSED
R = {}

# 1. 修改 2015-07-01 以後的玉米價格，2015-06-23 以前的特徵／部位／標籤不得改變
orig = core.build_instrument("CORN")
src = f"{ROOT}/data/processed/CORN_v2.csv"; bak = src + ".bak"; shutil.copy(src, bak)
try:
    d = pd.read_csv(src, index_col=0, parse_dates=True)
    m = d.index >= "2015-07-01"; d.loc[m, "dP"] *= 3; d.loc[m, "price"] *= 1.7; d.to_csv(src)  # 價格變動放大三倍、價格水準放大1.7倍
    pert = core.build_instrument("CORN")
finally:
    shutil.move(bak, src)
cut = "2015-06-30"
cols = core.FEATS + ["n0", "fc_comb"]
a, b = orig.loc[:cut, cols], pert.loc[:cut, cols]
R["future_change_features_identical"] = bool(np.allclose(a.fillna(-9e9), b.fillna(-9e9), equal_nan=True))
lab_ok = orig.label_available_date < cut
R["future_change_mature_labels_identical"] = bool((orig.y[lab_ok].fillna(-1) == pert.y[lab_ok].fillna(-1)).all())
R["future_change_affects_after"] = bool(not np.allclose(orig.loc["2015-08":, "n0"], pert.loc["2015-08":, "n0"]))
# 1b. 前綴重建：只用截至 2015-06-30 的原始資料重建換月鏈與特徵，與全樣本版本在截止日前必須完全相同（含 days_to_roll）
import causal_chain
for inst in core.INST:
    full_c, _ = causal_chain.build(inst)
    pre_c, _ = causal_chain.build(inst, end="2015-06-30")
    cols_c = ["contract", "dP", "dP_feat", "roll_today", "days_to_roll", "plan_roll_date", "valued", "unscorable", "next_contract"]
    A, Bp = full_c.loc[:"2015-06-30", cols_c], pre_c.loc[:"2015-06-30", cols_c]
    R[f"prefix_rebuild_identical_{inst}"] = bool(A.fillna(-9e9).equals(Bp.fillna(-9e9)))
# 2. 訓練樣本的標籤在訓練截止前已可得
P = core.load_all()
pred = pd.read_csv(f"{ROOT}/outputs/predictions/pred_logit.csv", parse_dates=["date", "train_end_cut"])
R["predictions_after_train_cut"] = bool((pred.date >= pred.train_end_cut).all())
# 3. 決策重播
dec = pd.read_csv(f"{ROOT}/outputs/decisions/decisions_B3.csv", parse_dates=["decision_date"])
S = {"NORMAL": NORMAL, "REDUCED": REDUCED, "PAUSED": PAUSED}
bad = 0; n = 0
for i, g in dec.groupby("instrument"):
    g = g.sort_values("decision_date"); prev_cnt = 0; prevY = None
    for r in g.itertuples():
        if prevY is not None and r.decision_date.year != prevY: prev_cnt = 0
        prevY = r.decision_date.year
        st, cnt, _ = step(S[r.previous_state], prev_cnt, r.risk_probability, r.threshold_low, r.threshold_reduce,
                          r.threshold_pause, int(r.m), True, r.base_target != 0, r.reason_code != "NO_QUOTE_HOLD")
        bad += (st != S[r.new_state]) or (cnt != r.recovery_count); prev_cnt = r.recovery_count; n += 1
R["replay_mismatch"] = int(bad); R["replay_n"] = n
# 4. 帳務對帳：淨=毛-成本；持倉=前一日目標（a×n0）
dp = pd.read_csv(f"{ROOT}/outputs/daily/daily_pnl.csv.gz", parse_dates=["date"])
x = dp[dp.policy == "B3"]
R["net_equals_gross_minus_cost"] = bool(np.allclose(dp.net, (dp.gross - dp.cost) / core.K, equal_nan=True))
err = 0; cerr = 0
for pol in dp.policy.unique():
    xp = dp[dp.policy == pol]; cm = 2.0 if pol.endswith("_cost2x") else 1.0
    for i in core.INST:
        xi = xp[xp.instrument == i].set_index("date"); o = P[i]
        gross = xi.held.shift(1) * o.dP.reindex(xi.index) * core.cfg.loc[i, "Pointsize"]
        ok = gross.notna() & xi.gross.notna()
        R.setdefault("ledger_nan_mismatch", 0); R["ledger_nan_mismatch"] += int((gross.iloc[1:].isna() != xi.gross.iloc[1:].isna()).sum())
        err = max(err, float((gross[ok].iloc[1:] - xi.gross[ok].iloc[1:]).abs().max()))
        rl = o.roll_today.reindex(xi.index) == 1; prv = xi.held.shift(1)
        tq = (xi.held - prv).abs().where(~rl, 0.0); rq = (prv.abs() + xi.held.abs()).where(rl, 0.0)
        cost = (tq + rq) * core.cost_per_contract(i, cm)
        us = o.unscorable.reindex(xi.index) == 1
        cost = cost.where(~us, 0.0) + cost.where(us, 0.0).shift(1).fillna(0.0)
        cerr = max(cerr, float((cost.iloc[1:] - xi.cost.iloc[1:]).abs().max()))
        if not pol.endswith("_int"):
            tgt = (xi.a * o.n0.reindex(xi.index)).shift(1).where(o.valued.reindex(xi.index) == 1)
            okk = tgt.notna()
            err = max(err, float((xi.held[okk].iloc[1:] - tgt[okk].iloc[1:]).abs().max()))
R["ledger_policies_checked"] = int(dp.policy.nunique())
R["ledger_max_abs_error_usd"] = err; R["cost_max_abs_error_usd"] = cerr
# 5. 影子帳戶不受政策影響：B0 淨值 = 面板 r0
b0 = dp[dp.policy == "B0"]
R["shadow_equals_B0"] = bool(all(np.allclose(b0[b0.instrument == i].set_index("date").net, P[i].r0.loc["2008":], equal_nan=True) for i in core.INST))
# 6. 跨年度狀態延續（未重設為 NORMAL）
dec["y"] = dec.decision_date.dt.year
first = dec.sort_values("decision_date").groupby(["instrument", "y"]).head(1)
R["year_start_nonNORMAL_previous_state"] = int((first.previous_state != "NORMAL").sum())
# 7. 換月成本：換月日有持倉時必有 roll_qty
rr = x.merge(pd.concat([P[i][["roll_today"]].assign(instrument=i) for i in core.INST]).reset_index().rename(columns={"index": "date"}), on=["date", "instrument"])
R["roll_days_with_position_not_charged"] = int(((rr.roll_today == 1) & (rr.held.abs() > 0) & (rr.roll_qty == 0)).sum())
json.dump(R, open(f"{ROOT}/outputs/acceptance.json", "w"), indent=1)
print(json.dumps(R, indent=1))
