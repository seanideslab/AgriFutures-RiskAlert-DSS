"""補充分析（論文 5.2 節同門檻比較、5.4 節警示計分）。
(1) 同門檻：以 B3 逐年選定的 θL/θR/θP 重跑無記憶政策（B2_sameθ），比較切換與成本。
(2) 警示：生效日＝決策日後第一個持有契約有報價、可依交易引擎調整持倉的收盤日；
    及時＝生效日落在 τ−10…τ−1；事後＝τ…τ+6 內有生效警示；隨機基準自有報價且分數有效的日期抽取等量警示，重抽 300 次。
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from core import INST, H, load_all
from experiment import run_policy, FIRST_TEST, LAST
from evaluation import pol_metrics, breach_days

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs"; LEAD_MAX = 10; LATE = 6; SEED = 20261105


def same_threshold(P):
    sel = pd.read_csv(f"{OUT}/selected_params.csv")
    b3 = {int(r.test_year): {"thL": r.thL, "thR": r.thR, "thP": r.thP, "m": int(r.m)} for r in sel[sel.policy == "B3"].itertuples()}
    pr = pd.read_csv(f"{OUT}/predictions/pred_logit.csv", parse_dates=["date"])
    pmap = {i: pr[pr.instrument == i].set_index("date").p for i in INST}
    res, _ = run_policy(P, pmap, b3, False, (FIRST_TEST, LAST))
    rows = []
    for i, r in res.items():
        rr = r[["held", "gross", "cost", "trade_qty", "roll_qty", "a", "state", "notional", "net", "unscorable"]].copy()
        rr["instrument"] = i; rr["policy"] = "B2_sameθ"; rows.append(rr.reset_index().rename(columns={"index": "date"}))
    d = pd.concat(rows)
    d0 = pd.read_csv(f"{OUT}/daily/daily_pnl.csv.gz", parse_dates=["date"])
    m = pol_metrics(pd.concat([d, d0[d0.policy.isin(["B2", "B3"])]]), ["B2_sameθ", "B2", "B3"])
    m = m[m.group == "PORT"].set_index("policy")
    out = {"switches_B2_same": m.loc["B2_sameθ", "switches_per_yr"], "switches_B3": m.loc["B3", "switches_per_yr"],
           "cost_B2_same": m.loc["B2_sameθ", "cost_ann"], "cost_B3": m.loc["B3", "cost_ann"],
           "ann_B2_same": m.loc["B2_sameθ", "ann"], "ann_B3": m.loc["B3", "ann"]}
    out["switch_reduction"] = 1 - out["switches_B3"] / out["switches_B2_same"]
    out["cost_reduction"] = 1 - out["cost_B3"] / out["cost_B2_same"]
    return out


def score(alert_eff, mult_exec, y, br, first, last):
    ev = []; k = first
    while k <= last:
        if y[k] == 1:
            s0 = k
            while k <= last and y[k] == 1: k += 1
            taus = [br[j] for j in range(s0, k) if br[j] >= 0]
            if taus: ev.append(min(taus))
        k += 1
    used = set(); r = {"events": len(ev), "timely": 0, "already": 0, "late": 0, "missed": 0}; leads = []
    eff = sorted(alert_eff)
    for tau in ev:
        cand = [e for e in eff if e not in used and tau - LEAD_MAX <= e <= tau - 1]
        if cand:
            used.add(cand[0]); r["timely"] += 1; leads.append(tau - cand[0]); continue
        if mult_exec[tau - 1] < 1:
            r["already"] += 1; continue
        if any(tau <= e <= tau + LATE for e in eff):
            r["late"] += 1; continue
        r["missed"] += 1
    sc = [e for e in alert_eff if e + LEAD_MAX <= last + H + 1]
    r["alerts_scorable"] = len(sc); r["matched"] = len(used & set(sc)); r["leads"] = leads
    return r


def alerts(P, n_rand=300):
    dec = pd.read_csv(f"{OUT}/decisions/decisions_B3.csv", parse_dates=["decision_date"])
    rng = np.random.default_rng(SEED); ctx = {}; tot = {}
    per = []
    for i in INST:
        o = P[i]; g = dec[dec.instrument == i].set_index("decision_date").sort_index()
        pos = o.index.get_indexer(g.index); n = len(o)
        valued = o.valued.values == 1
        nxt = np.full(n + 1, -1); k = -1
        for j in range(n - 1, -1, -1):                        # 第 j 日之後第一個有報價日
            nxt[j] = k
            if valued[j]: k = j
        st = g.new_state.values
        al = [pos[k] for k in range(len(st)) if (k == 0 and st[0] != "NORMAL") or (k > 0 and st[k - 1] == "NORMAL" and st[k] != "NORMAL")]
        eff = [nxt[a] for a in al if nxt[a] >= 0]
        a_dec = pd.Series(g.exposure_multiplier.values, index=pos).reindex(range(n)).ffill().fillna(1.0).values
        held_mult = np.ones(n); cur = 1.0                      # 已生效倍率：只在有報價日更新
        for j in range(1, n):
            if valued[j]: cur = a_dec[j - 1]
            held_mult[j] = cur
        y = o.y.values; br = breach_days(o)
        first, last = pos[0], int(np.max(np.where(~np.isnan(y))[0]))
        r = score(eff, held_mult, y, br, first, last)
        r["instrument"] = i; r["active_days"] = int((o.n0.iloc[first:] != 0).sum())
        r["lead_median"] = float(np.median(r["leads"])) if r["leads"] else np.nan
        per.append(r)
        okday = np.where(valued & ~np.isnan(pd.read_csv(f"{OUT}/predictions/pred_logit.csv", usecols=["date", "instrument", "p"], parse_dates=["date"])
                                            .query("instrument == @i").set_index("date").p.reindex(o.index).values))[0]
        okday = okday[(okday >= first) & (okday <= last)]
        ctx[i] = (held_mult, y, br, first, last, len(al), okday, nxt)
    df = pd.DataFrame(per).drop(columns=["leads"])
    s = df.drop(columns=["instrument", "lead_median"]).sum()
    prec_r, rec_r = [], []
    for _ in range(n_rand):
        m = a = e = 0
        for i, (hm, y, br, first, last, na, okday, nxt) in ctx.items():
            ra = rng.choice(okday, na, replace=False)
            eff = [nxt[x] for x in ra if nxt[x] >= 0]
            r = score(eff, hm, y, br, first, last)
            m += r["matched"]; a += r["alerts_scorable"]; e += r["timely"]
        prec_r.append(m / a); rec_r.append(e / s["events"])
    summ = {"events": int(s["events"]), "alerts_scorable": int(s["alerts_scorable"]), "timely": int(s["timely"]),
            "already": int(s["already"]), "late": int(s["late"]), "missed": int(s["missed"]),
            "timely_recall": s["timely"] / s["events"], "precision": s["matched"] / s["alerts_scorable"],
            "alerts_per_100_active": 100 * s["alerts_scorable"] / s["active_days"],
            "lead_median_range": [float(df.lead_median.min()), float(df.lead_median.max())],
            "rand_precision": [float(np.mean(prec_r)), float(np.quantile(prec_r, .025)), float(np.quantile(prec_r, .975))],
            "rand_recall": [float(np.mean(rec_r)), float(np.quantile(rec_r, .025)), float(np.quantile(rec_r, .975))]}
    return df, summ


if __name__ == "__main__":
    P = load_all()
    st = same_threshold(P)
    df, summ = alerts(P)
    os.makedirs(f"{OUT}/metrics", exist_ok=True)
    df.to_csv(f"{OUT}/metrics/alert_events_effective.csv", index=False)
    json.dump({"same_threshold": st, "alerts_effective_date": summ}, open(f"{OUT}/metrics/supplement.json", "w"), indent=1, default=float)
    print(json.dumps({"same_threshold": st, "alerts_effective_date": summ}, indent=1, default=float))
