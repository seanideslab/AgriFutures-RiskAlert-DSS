"""評估：機率品質、政策績效、曝險、切換、錯失報酬分解、警示事件、區塊 bootstrap + Holm。"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from core import INST, ZH, H, load_all

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs"; os.makedirs(f"{OUT}/metrics", exist_ok=True)
T0 = "2008-01-01"
rng_seed = 20261105


def es95(x):
    loss = -np.asarray(x); q = np.quantile(loss, 0.95); return loss[loss >= q].mean()


def mdd(x):
    eq = np.cumsum(x); return np.max(np.maximum.accumulate(eq) - eq)


def prob_metrics():
    rows = []
    for nm in ["clim", "logit", "tree", "logit_noshadow"]:
        d = pd.read_csv(f"{OUT}/predictions/pred_{nm}.csv", parse_dates=["date"])
        d = d[(d.date >= T0) & d.y.notna()]
        for g, x in [("ALL", d)] + [(i, d[d.instrument == i]) for i in INST]:
            rows.append({"model": nm, "instrument": g, "n": len(x), "event_rate": x.y.mean(),
                         "brier": np.mean((x.p - x.y) ** 2), "pr_auc": average_precision_score(x.y, x.p),
                         "roc_auc": roc_auc_score(x.y, x.p) if nm != "clim" else 0.5})
    m = pd.DataFrame(rows)
    c = m[m.model == "clim"].set_index("instrument").brier
    m["brier_skill_vs_clim"] = 1 - m.brier / m.instrument.map(c)
    m.to_csv(f"{OUT}/metrics/prob_metrics.csv", index=False)
    # 校準（十分位）
    d = pd.read_csv(f"{OUT}/predictions/pred_logit.csv", parse_dates=["date"])
    d = d[(d.date >= T0) & d.y.notna()]
    d["bin"] = pd.qcut(d.p, 10, labels=False, duplicates="drop")
    d.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size")).to_csv(f"{OUT}/metrics/calibration.csv")
    return m, d


def load_daily():
    d = pd.read_csv(f"{OUT}/daily/daily_pnl.csv.gz", parse_dates=["date"])
    return d


def port_series(x, i=None):
    """組合（或單一商品）日淨報酬。商品未交易日貢獻 0；不可計分日（unscorable）整日排除，對所有政策一致。"""
    if i is not None:
        x = x[x.instrument == i]
    net = x.pivot(index="date", columns="instrument", values="net")
    bad = x.pivot(index="date", columns="instrument", values="unscorable").fillna(0).max(axis=1) > 0
    return (net.fillna(0).sum(axis=1) / (len(INST) if i is None else 1))[~bad]


def pol_metrics(d, pols, per=None):
    rows = []
    for pol in pols:
        x = d[d.policy == pol]
        groups = [("PORT", None)] + [(i, i) for i in INST]
        for g, i in groups:
            xx = x if i is None else x[x.instrument == i]
            ret = port_series(x, i)
            if per:
                ret = ret.loc[per[0]:per[1]]; xx = xx[(xx.date >= per[0]) & (xx.date <= per[1])]
            sw = xx.sort_values("date").groupby("instrument").state.apply(lambda s: (s.diff().fillna(0) != 0).sum()).sum()
            yrs = ret.index.nunique() / 256
            rows.append({"policy": pol, "group": g, "ann": ret.mean() * 256, "vol": ret.std() * 16,
                         "sharpe": ret.mean() / ret.std() * 16, "es95": es95(ret), "mdd": mdd(ret.values),
                         "worst_month": ret.groupby(ret.index.to_period("M")).sum().min(),
                         "cost_ann": xx.cost.sum() / 1e6 / yrs / (len(INST) if i is None else 1),
                         "turnover_contracts_yr": (xx.trade_qty.sum() + xx.roll_qty.sum()) / yrs,
                         "notional_mean": xx.groupby("date").notional.sum().mean() / (len(INST) if i is None else 1),
                         "a_mean": xx.a.mean(), "active_full_share": (xx.a == 1).mean(),
                         "paused_share": (xx.a == 0).mean(), "switches_per_yr": sw / yrs / (len(INST) if i is None else 1),
                         "n_days": len(ret)})
    return pd.DataFrame(rows)


def decompose(d, pols):
    """相對 B0：毛損益差拆成錯失獲利（B0 毛>0 的日子）與避免損失（B0 毛<0），另計成本差；不可計分日排除。"""
    keep = port_series(d[d.policy == "B0"]).index                  # 與組合一致：不可計分日期整日排除
    b0 = d[(d.policy == "B0") & d.date.isin(keep)].set_index(["date", "instrument"])
    rows = []
    yrs = port_series(d[d.policy == "B0"]).index.nunique() / 256
    for pol in pols:
        x = d[d.policy == pol].set_index(["date", "instrument"])
        g0, g1 = b0.gross, x.gross.reindex(b0.index)
        diff = g1 - g0
        rows.append({"policy": pol,
                     "missed_gain_ann": -diff[g0 > 0].sum() / 6e6 / yrs,
                     "avoided_loss_ann": diff[g0 < 0].sum() / 6e6 / yrs,
                     "cost_saving_ann": (b0.cost.sum() - x.cost.reindex(b0.index).sum()) / 6e6 / yrs,
                     "net_diff_ann": (x.net.reindex(b0.index) - b0.net).sum() / 6 / yrs})
    return pd.DataFrame(rows)


LEAD_MAX = 10   # 警示可執行日最早可在突破日前 10 個交易日


def breach_days(o):
    """每個決策日 j 的實際突破日（影子帳戶自 j+1 收盤起累積損失首次超過預算 b_j 的交易日位置）；無突破或未成熟為 -1。"""
    r = o.r0.values; b = o.budget.values; n = len(r); out = np.full(n, -1)
    for j in range(n - H - 1):
        w = r[j + 2:j + 2 + H]
        if np.isnan(w).any() or np.isnan(b[j]): continue
        c = -np.cumsum(w); k = np.where(c > b[j])[0]
        if len(k): out[j] = j + 2 + k[0]
    return out


def score_alerts(i, o, alert_pos, mult_exec, y, br, first, last_mature):
    """事件＝相連正標籤群；事件時點 τ＝群內最早實際突破日。警示於決策日 a，a+1 收盤生效。
    一對一匹配：依時間處理事件，取 a+1 ∈ [τ−LEAD_MAX, τ−1] 中最早未用之警示。"""
    ev = []; k = first
    while k <= last_mature:
        if y[k] == 1:
            s0 = k
            while k <= last_mature and y[k] == 1: k += 1
            taus = [br[j] for j in range(s0, k) if br[j] >= 0]
            if taus: ev.append(min(taus))
        k += 1
    used = set(); res = {"events": len(ev), "timely_new_alert": 0, "already_reduced": 0, "late_alert": 0, "missed": 0}
    leads = []
    al_sorted = sorted(alert_pos)
    for tau in ev:
        cand = [a for a in al_sorted if a not in used and tau - LEAD_MAX <= a + 1 <= tau - 1]
        if cand:
            used.add(cand[0]); res["timely_new_alert"] += 1; leads.append(tau - (cand[0] + 1)); continue
        if mult_exec[tau - 1] < 1:                       # 突破前一日收盤持倉已在減碼／暫停
            res["already_reduced"] += 1; continue
        if any(tau - 1 <= a <= tau + 5 for a in al_sorted):
            res["late_alert"] += 1; continue
        res["missed"] += 1
    scor_al = [a for a in alert_pos if a + 1 + LEAD_MAX <= last_mature + H + 1]   # 前瞻窗已成熟的警示
    res["alerts_scorable"] = len(scor_al); res["alerts_matched"] = len([a for a in used if a in set(scor_al)])
    res["lead_median"] = float(np.median(leads)) if leads else np.nan
    return res


def alert_events(P, n_rand=300):
    dec = pd.read_csv(f"{OUT}/decisions/decisions_B3.csv", parse_dates=["decision_date"])
    rng = np.random.default_rng(rng_seed)
    rows = []; rand_prec = []; rand_rec = []
    ctx = {}
    for i in INST:
        o = P[i]; g = dec[dec.instrument == i].set_index("decision_date").sort_index()
        pos = o.index.get_indexer(g.index)
        y = o.y.values; br = breach_days(o)
        st = g.new_state.values
        alert_pos = [pos[k] for k in range(len(st)) if (k == 0 and st[0] != "NORMAL") or (k > 0 and st[k - 1] == "NORMAL" and st[k] != "NORMAL")]
        a_dec = pd.Series(g.exposure_multiplier.values, index=pos).reindex(range(len(o))).ffill().fillna(1.0).values
        mult_exec = np.r_[1.0, a_dec[:-1]]                # a+1 收盤生效：位置 k 的持倉倍率＝決策 k-1 的倍率
        first, last = pos[0], int(np.max(np.where(~np.isnan(y))[0]))
        res = score_alerts(i, o, alert_pos, mult_exec, y, br, first, last)
        res["instrument"] = i; res["active_days"] = int((o.n0.iloc[first:] != 0).sum()); rows.append(res)
        ctx[i] = (o, mult_exec, y, br, first, last, len(alert_pos), pos)
    df = pd.DataFrame(rows)
    tot = df.drop(columns=["instrument", "lead_median"]).sum(); tot["instrument"] = "ALL"
    tot["lead_median"] = np.nan; df.loc[len(df)] = tot
    for _ in range(n_rand):
        m = e = al = 0
        for i, (o, mult_exec, y, br, first, last, na, pos) in ctx.items():
            ra = sorted(rng.choice(np.arange(first, last + 1), na, replace=False).tolist())
            r = score_alerts(i, o, ra, mult_exec, y, br, first, last)
            m += r["alerts_matched"]; al += r["alerts_scorable"]; e += r["timely_new_alert"]
        rand_prec.append(m / al); rand_rec.append(e / df.loc[df.instrument == "ALL", "events"].iloc[0])
    df["timely_recall"] = df.timely_new_alert / df.events
    df["alert_precision_1to1"] = df.alerts_matched / df.alerts_scorable
    df["alerts_per_100_active_days"] = 100 * df.alerts_scorable / df.active_days
    df.to_csv(f"{OUT}/metrics/alert_events.csv", index=False)
    pd.Series({"precision_mean": np.mean(rand_prec), "precision_lo": np.quantile(rand_prec, .025), "precision_hi": np.quantile(rand_prec, .975),
               "recall_mean": np.mean(rand_rec), "recall_lo": np.quantile(rand_rec, .025), "recall_hi": np.quantile(rand_rec, .975)}
              ).to_csv(f"{OUT}/metrics/alert_random_baseline.csv")
    return df


def centered_p(v, obs, B):
    """虛無中心化的雙尾 bootstrap p：以 T*−T̂ 近似虛無分布，p=(1+#{|T*−T̂|≥|T̂|})/(B+1)。"""
    v = np.asarray(v)
    return (1 + np.sum(np.abs(v - obs) >= abs(obs))) / (B + 1)


def bootstrap(d, contrasts, block=20, B=2000, seed=rng_seed):
    piv = {p: port_series(d[d.policy == p]) for p in set(sum(contrasts, ()))}
    idx = piv["B0"].index
    R = pd.DataFrame({p: s.reindex(idx) for p, s in piv.items()})
    assert not R.isna().any().any()
    cols = list(piv.keys()); R = R.values; n = len(idx)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    stats = {c: {"es": [], "ann": [], "sh": []} for c in contrasts}
    obs = {}
    for a, b in contrasts:
        assert a != b
        ra, rb = R[:, cols.index(a)], R[:, cols.index(b)]
        obs[(a, b)] = (es95(ra) - es95(rb), (ra.mean() - rb.mean()) * 256, ra.mean() / ra.std() * 16 - rb.mean() / rb.std() * 16)
    for _ in range(B):
        st = rng.integers(0, n - block + 1, nb)
        ii = (st[:, None] + np.arange(block)).ravel()[:n]
        S = R[ii]
        for a, b in contrasts:
            ra, rb = S[:, cols.index(a)], S[:, cols.index(b)]
            stats[(a, b)]["es"].append(es95(ra) - es95(rb))
            stats[(a, b)]["ann"].append((ra.mean() - rb.mean()) * 256)
            stats[(a, b)]["sh"].append(ra.mean() / ra.std() * 16 - rb.mean() / rb.std() * 16)
    rows = []
    for c in contrasts:
        o = obs[c]
        for k, j in [("es", 0), ("ann", 1), ("sh", 2)]:
            v = np.array(stats[c][k])
            rows.append({"contrast": f"{c[0]}-{c[1]}", "metric": k, "obs": o[j], "lo": np.quantile(v, 0.025),
                         "hi": np.quantile(v, 0.975), "p_boot": centered_p(v, o[j], B), "block": block, "B": B})
    return pd.DataFrame(rows)


def holm(df):
    x = df.sort_values("p_boot").copy(); m = len(x)
    adj = np.maximum.accumulate([(m - k) * p for k, p in enumerate(x.p_boot)])
    x["p_holm"] = np.minimum(adj, 1); return x.sort_index()


def brier_boot(B=2000, block=20):
    a = pd.read_csv(f"{OUT}/predictions/pred_logit.csv", parse_dates=["date"])
    c = pd.read_csv(f"{OUT}/predictions/pred_clim.csv", parse_dates=["date"])
    m = a.merge(c[["date", "instrument", "p"]], on=["date", "instrument"], suffixes=("", "_c"))
    m = m[(m.date >= T0) & m.y.notna()]
    m["dl"] = (m.p - m.y) ** 2 - (m.p_c - m.y) ** 2
    s = m.groupby("date").dl.mean(); v = s.values; n = len(v)
    rng = np.random.default_rng(rng_seed); nb = int(np.ceil(n / block)); bs = []
    for _ in range(B):
        st = rng.integers(0, n - block + 1, nb); ii = (st[:, None] + np.arange(block)).ravel()[:n]; bs.append(v[ii].mean())
    bs = np.array(bs)
    return {"brier_diff_logit_minus_clim": float(v.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)),
            "p_boot": float(centered_p(bs, v.mean(), B))}


if __name__ == "__main__":
    P = load_all()
    pm, _ = prob_metrics()
    print(pm[pm.instrument == "ALL"].round(4).to_string(index=False))
    print("brier boot", brier_boot())
    d = load_daily()
    main = ["B0", "B1", "B2", "B3", "B4", "B5", "A2", "TREE", "B1mem"]
    met = pol_metrics(d, main + [p + "_cost2x" for p in main] + ["B0_int", "B3_int", "B4_int"])
    met.to_csv(f"{OUT}/metrics/policy_metrics.csv", index=False)
    print(met[met.group == "PORT"].round(4).to_string(index=False))
    sub = pd.concat([pol_metrics(d, main, per).assign(period=f"{per[0][:4]}-{per[1][:4]}")
                     for per in [("2008-01-01", "2015-12-31"), ("2016-01-01", "2024-12-31")]])
    sub.to_csv(f"{OUT}/metrics/policy_metrics_subperiod.csv", index=False)
    dc = decompose(d, ["B1", "B2", "B3", "B4", "B5"]); dc.to_csv(f"{OUT}/metrics/decomposition.csv", index=False)
    print(dc.round(4).to_string(index=False))
    print(alert_events(P).round(3).to_string(index=False))
    C = [("B3", "B2"), ("B3", "B1"), ("B3", "B4")]
    bt = pd.concat([bootstrap(d, C + [("B3", "B5"), ("B3", "B0"), ("B2", "B1"), ("B1", "B0"), ("A2", "B5")], block=bl) for bl in [20, 10, 40]])
    prim = holm(bt[(bt.metric == "es") & (bt.block == 20) & bt.contrast.isin(["B3-B2", "B3-B1", "B3-B4"])])
    bt = bt.merge(prim[["contrast", "metric", "block", "p_holm"]], how="left", on=["contrast", "metric", "block"])
    bt.to_csv(f"{OUT}/metrics/bootstrap.csv", index=False)
    print(bt[bt.block == 20].round(4).to_string(index=False))
    json.dump(brier_boot(), open(f"{OUT}/metrics/brier_boot.json", "w"), indent=1)
