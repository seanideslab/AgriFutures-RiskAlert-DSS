"""走步風險模型、政策狀態機（B0–B4）、驗證期參數選擇、測試期連續模擬、決策紀錄。"""
import itertools, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(__file__))
from core import INST, FEATS, SHADOW_FEATS, K, load_all, simulate_positions, cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs"
FIRST_PRED, FIRST_TEST, LAST = 2005, 2008, 2024
ALPHA = 0.5
NORMAL, REDUCED, PAUSED = 0, 1, 2
MULT = {NORMAL: 1.0, REDUCED: ALPHA, PAUSED: 0.0}
SNAME = {0: "NORMAL", 1: "REDUCED", 2: "PAUSED"}


# ------------------------------------------------------------------ 模型
def fit_predict(P, feats, model="logit", tag=""):
    pan = pd.concat([P[i].assign(date=P[i].index) for i in INST], ignore_index=True)
    pan = pan[pan.valid == 1].copy()
    preds = []
    for Y in range(FIRST_PRED, LAST + 1):
        cut = pd.Timestamp(f"{Y}-01-01")
        tr = pan[(pan.label_available_date < cut) & pan.y.notna() & (pan.n0.abs() > 0)].dropna(subset=feats)
        te = pan[(pan.date >= cut) & (pan.date < pd.Timestamp(f"{Y+1}-01-01"))].copy()
        te = te[te[feats].notna().all(axis=1)]
        lo, hi = tr[feats].quantile(0.01), tr[feats].quantile(0.99)
        Xtr, Xte = tr[feats].clip(lo, hi, axis=1), te[feats].clip(lo, hi, axis=1)
        rate = tr.y.mean()
        if model == "clim" or tr.y.nunique() < 2:
            p = np.full(len(te), rate); status = "climatology"
        elif model == "logit":
            s = StandardScaler().fit(Xtr)
            m = LogisticRegression(C=1.0, max_iter=2000).fit(s.transform(Xtr), tr.y)
            p = m.predict_proba(s.transform(Xte))[:, 1]; status = "logit"
            if tag == "main":
                coefs.append({"year": Y, **dict(zip(feats, m.coef_[0])), "intercept": m.intercept_[0],
                              "n_train": len(tr), "events": int(tr.y.sum())})
        elif model == "tree":
            m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=0).fit(Xtr, tr.y)
            p = m.predict_proba(Xte)[:, 1]; status = "tree"
        elif model == "volsig":  # 單一波動訊號（B1 用），只是分數不是機率
            p = te.vol_ratio_20_120.values; status = "vol_signal"
        preds.append(pd.DataFrame({"date": te.date.values, "instrument": te.instrument.values, "p": p,
                                   "y": te.y.values, "model_version": f"{model}{tag}_{Y}", "status": status,
                                   "train_rate": rate, "n_train": len(tr), "train_end_cut": cut}))
    return pd.concat(preds, ignore_index=True)


coefs = []


# ------------------------------------------------------------------ 狀態機
def step(state, cnt, p, thL, thR, thP, m, memory, has_pos, quoted=True):
    """回傳 (new_state, new_cnt, reason)。持有契約無報價 → 不交易、狀態與計數凍結；p 為 NaN → 覆核：保留狀態、清除計數。"""
    if not quoted:
        return state, cnt, "NO_QUOTE_HOLD"
    if p is None or np.isnan(p):
        return state, 0, "REVIEW_NO_SCORE"
    if not memory:
        if p >= thP: return PAUSED, 0, "p>=thP"
        if p >= thR: return REDUCED, 0, "thR<=p<thP"
        return NORMAL, 0, "p<thR"
    if p >= thP:
        return PAUSED, 0, "p>=thP"
    if state == NORMAL:
        return (REDUCED, 0, "thR<=p<thP") if p >= thR else (NORMAL, 0, "p<thR")
    if p <= thL:
        if not has_pos:
            return state, cnt, "low_p_but_flat_no_count"
        cnt += 1
        if cnt >= m:
            return (NORMAL if state == REDUCED else REDUCED), 0, f"recover_after_{m}"
        return state, cnt, f"recovering_{cnt}/{m}"
    return state, 0, "hold_mid_p"


def run_policy(P, pmap, params_by_year, memory, years, fixed_mult=None, log=False, cost_mult=1.0, integer=False):
    """pmap: inst -> Series(date->score)。params_by_year: Y -> dict(thL,thR,thP,m) 或 None（不介入，a=1）。"""
    res, logs = {}, []
    for i in INST:
        o = P[i]
        msk = (o.index.year >= years[0]) & (o.index.year <= years[1])
        idx = o.index[msk]
        sc = pmap[i].reindex(idx).values.astype(float)
        n0 = o.n0.values[msk]; yrs = idx.year.values; qv = o.valued.values[msk] == 1
        a = np.ones(len(idx)); st = np.zeros(len(idx), int)
        if fixed_mult is not None:
            a = np.array([fixed_mult.get(int(y), 1.0) for y in yrs])
        else:
            state, cnt, prevY = NORMAL, 0, None
            for k in range(len(idx)):
                Y = int(yrs[k])
                if prevY is not None and Y != prevY:
                    cnt = 0                                # 年度更新：保留狀態、清除恢復計數
                prevY = Y
                pr = params_by_year.get(Y)
                if pr is None:
                    state, cnt, reason = NORMAL, 0, "no_feasible_params_default_normal"
                else:
                    state, cnt, reason = step(state, cnt, sc[k], pr["thL"], pr["thR"], pr["thP"], pr["m"], memory, n0[k] != 0, bool(qv[k]))
                a[k] = MULT[state]; st[k] = state
                if log:
                    logs.append((i, idx[k], sc[k], state, cnt, reason, a[k]))
        full = o.n0.copy()                                 # 區段前沿用原策略部位，避免第一天假性開倉成本
        full.loc[idx] = a * n0
        if integer:
            full = full.round()
        sim = simulate_positions(o[["dP", "roll_today", "valued", "unscorable"]], full, i, cost_mult)
        sim["a"] = pd.Series(a, index=idx).reindex(o.index)
        sim["state"] = pd.Series(st, index=idx).reindex(o.index)
        sim["notional"] = sim.held.abs() * o.price * cfg.loc[i, "Pointsize"] / K
        sim["unscorable"] = o.unscorable
        r = sim.loc[idx].copy()
        r["net"] = (r.gross - r.cost) / K
        res[i] = r
    return res, logs


def portfolio(res):
    """六商品等權組合。商品未交易日貢獻 0；任一商品不可計分（unscorable）的日期整日排除，所有政策一致。"""
    net = pd.concat({i: r.net for i, r in res.items()}, axis=1)
    bad = pd.concat({i: r.unscorable for i, r in res.items()}, axis=1).fillna(0).max(axis=1) > 0
    return (net.fillna(0.0).sum(axis=1) / len(INST))[~bad]


def es95(x):
    loss = -x.values; q = np.quantile(loss, 0.95); return loss[loss >= q].mean()


def summarize(res, base=None):
    pr = portfolio(res)
    a = pd.concat({i: r.a for i, r in res.items()}, axis=1)
    return {"ann": pr.mean() * 256, "vol": pr.std() * 16, "es95": es95(pr), "a_mean": np.nanmean(a.values)}


# ------------------------------------------------------------------ 參數選擇
def candidates(scores_val, memory):
    q = lambda x: float(np.nanquantile(scores_val, x))
    out = []
    for qr, qp in itertools.product([0.70, 0.80], [0.90, 0.95]):
        for m in ([1, 3] if memory else [1]):
            out.append({"thL": q(0.50), "thR": q(qr), "thP": q(qp), "m": m, "qR": qr, "qP": qp})
    return out


def select(P, pmap, Y, memory, b0_val, trials, label, max_drop=0.01, min_a=0.5):
    vy = (FIRST_PRED, Y - 1)
    allsc = np.concatenate([pmap[i][(pmap[i].index.year >= vy[0]) & (pmap[i].index.year <= vy[1])].values for i in INST])
    best = None
    for c in candidates(allsc, memory):
        res, _ = run_policy(P, pmap, {yy: c for yy in range(vy[0], vy[1] + 1)}, memory, vy)
        s = summarize(res)
        feas = (b0_val["ann"] - s["ann"] <= max_drop) and (s["a_mean"] >= min_a)
        trials.append({"policy": label, "test_year": Y, **c, **s, "feasible": feas})
        if feas and (best is None or s["es95"] < best[1]["es95"]):
            best = (c, s)
    return best


def main():
    t0 = time.time()
    P = load_all()
    os.makedirs(f"{OUT}/predictions", exist_ok=True)
    pr_main = fit_predict(P, FEATS, "logit", "main")
    pr_clim = fit_predict(P, FEATS, "clim")
    pr_tree = fit_predict(P, FEATS, "tree")
    noshadow = [f for f in FEATS if f not in SHADOW_FEATS]
    pr_a2 = fit_predict(P, noshadow, "logit", "_noshadow")
    pr_vol = fit_predict(P, FEATS, "volsig")
    for nm, df in [("logit", pr_main), ("clim", pr_clim), ("tree", pr_tree), ("logit_noshadow", pr_a2), ("volsig", pr_vol)]:
        df.to_csv(f"{OUT}/predictions/pred_{nm}.csv", index=False)
    pd.DataFrame(coefs).to_csv(f"{OUT}/predictions/logit_coefs.csv", index=False)
    print("models done", round(time.time() - t0))

    def pm(df):
        return {i: df[df.instrument == i].set_index("date").p for i in INST}
    maps = {"AI": pm(pr_main), "VOL": pm(pr_vol), "A2": pm(pr_a2), "TREE": pm(pr_tree)}
    ones = {Y: 1.0 for Y in range(FIRST_TEST, LAST + 1)}
    trials, chosen = [], []
    params = {k: {} for k in ["B1", "B2", "B3", "A2", "TREE", "B1mem"]}
    b4mult = {}; b5mult = {}
    spec = {"B1": ("VOL", False), "B2": ("AI", False), "B3": ("AI", True), "A2": ("A2", True),
            "TREE": ("TREE", True), "B1mem": ("VOL", True)}
    for Y in range(FIRST_TEST, LAST + 1):
        vy = (FIRST_PRED, Y - 1)
        b0v, _ = run_policy(P, maps["AI"], {}, False, vy, fixed_mult={yy: 1.0 for yy in range(vy[0], vy[1] + 1)})
        b0s = summarize(b0v)
        for lab, (mk, mem) in spec.items():
            best = select(P, maps[mk], Y, mem, b0s, trials, lab)
            params[lab][Y] = best[0] if best else None
            chosen.append({"policy": lab, "test_year": Y, "feasible": best is not None,
                           **(best[0] if best else {}), **({"val_" + k: v for k, v in best[1].items()} if best else {})})
            if lab == "B3":
                b4mult[Y] = best[1]["a_mean"] if best else 1.0
                b5mult[Y] = best[1]["vol"] / b0s["vol"] if best else 1.0   # v2 新增：事前風險配對（驗證期實現波動比）
        print("year", Y, "selected", {k: (params[k][Y]["qR"], params[k][Y]["qP"], params[k][Y]["m"]) if params[k][Y] else None for k in ["B1", "B2", "B3"]}, round(time.time() - t0))
    pd.DataFrame(trials).to_csv(f"{OUT}/experiment_log.csv", index=False)
    pd.DataFrame(chosen).to_csv(f"{OUT}/selected_params.csv", index=False)
    json.dump({str(k): v for k, v in b4mult.items()}, open(f"{OUT}/b4_multiplier.json", "w"), indent=1)
    json.dump({str(k): v for k, v in b5mult.items()}, open(f"{OUT}/b5_multiplier.json", "w"), indent=1)

    ty = (FIRST_TEST, LAST)
    runs = {}
    for cm, suf in [(1.0, ""), (2.0, "_cost2x")]:
        runs["B0" + suf], _ = run_policy(P, maps["AI"], {}, False, ty, fixed_mult=ones, cost_mult=cm)
        runs["B4" + suf], _ = run_policy(P, maps["AI"], {}, False, ty, fixed_mult=b4mult, cost_mult=cm)
        runs["B5" + suf], _ = run_policy(P, maps["AI"], {}, False, ty, fixed_mult=b5mult, cost_mult=cm)
        for lab, (mk, mem) in spec.items():
            runs[lab + suf], lg = run_policy(P, maps[mk], params[lab], mem, ty, log=(lab == "B3" and cm == 1.0), cost_mult=cm)
            if lab == "B3" and cm == 1.0:
                logs = lg
    for lab in ["B0", "B3", "B4"]:
        mk = {"B0": None, "B3": "AI", "B4": None}[lab]
        if lab == "B3":
            runs[lab + "_int"], _ = run_policy(P, maps["AI"], params["B3"], True, ty, integer=True)
        else:
            runs[lab + "_int"], _ = run_policy(P, maps["AI"], {}, False, ty, fixed_mult=ones if lab == "B0" else b4mult, integer=True)
    os.makedirs(f"{OUT}/daily", exist_ok=True)
    rows = []
    for nm, res in runs.items():
        for i, r in res.items():
            rr = r[["held", "gross", "cost", "trade_qty", "roll_qty", "a", "state", "notional", "net", "unscorable"]].copy()
            rr["instrument"] = i; rr["policy"] = nm; rows.append(rr.reset_index().rename(columns={"index": "date"}))
    pd.concat(rows).to_csv(f"{OUT}/daily/daily_pnl.csv.gz", index=False)
    # 決策紀錄（B3）
    lg = pd.DataFrame(logs, columns=["instrument", "decision_date", "risk_probability", "new_state", "recovery_count", "reason_code", "exposure_multiplier"])
    dec = []
    for i in INST:
        g = lg[lg.instrument == i].copy().set_index("decision_date")
        o = P[i].loc[g.index]
        g["previous_state"] = g.new_state.shift().fillna(NORMAL).astype(int)
        g["base_signal"] = o.fc_comb; g["base_target"] = o.n0; g["contract_id"] = o.contract
        g["target_position"] = g.exposure_multiplier * o.n0
        pr = params["B3"]
        g["threshold_low"] = [pr[t.year]["thL"] if pr[t.year] else np.nan for t in g.index]
        g["threshold_reduce"] = [pr[t.year]["thR"] if pr[t.year] else np.nan for t in g.index]
        g["threshold_pause"] = [pr[t.year]["thP"] if pr[t.year] else np.nan for t in g.index]
        g["m"] = [pr[t.year]["m"] if pr[t.year] else np.nan for t in g.index]
        g["intended_execution_date"] = list(P[i].index.to_series().shift(-1).loc[g.index])
        g["label_available_date"] = o.label_available_date
        g["model_version"] = [f"logitmain_{t.year}" for t in g.index]
        g["policy_version"] = [f"B3_{t.year}" for t in g.index]
        g["data_snapshot_id"] = "pst-master-20261105"
        g["data_quality_status"] = np.select([o.unscorable.values == 1, g.reason_code.values == "NO_QUOTE_HOLD", g.reason_code.values == "REVIEW_NO_SCORE"],
                                             ["UNSCORABLE_FORCED_ROLL", "NO_QUOTE_HOLD", "REVIEW_NO_SCORE"], "OK")
        dec.append(g.reset_index())
    dec = pd.concat(dec)
    dec["run_id"] = "spec-v2"; dec["decision_id"] = range(len(dec))
    for c in ["new_state", "previous_state"]:
        dec[c] = dec[c].map(SNAME)
    os.makedirs(f"{OUT}/decisions", exist_ok=True)
    dec.to_csv(f"{OUT}/decisions/decisions_B3.csv", index=False)
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
