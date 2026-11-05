"""事後診斷（非事前規格）：等波動縮放 B0、隨機警示基準、分商品／分期表。"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from evaluation import es95, mdd, port_series, centered_p
from core import INST
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); M = f"{ROOT}/outputs/metrics"
d = pd.read_csv(f"{ROOT}/outputs/daily/daily_pnl.csv.gz", parse_dates=["date"])
POL = ["B0", "B1", "B2", "B3", "B4", "B5", "A2", "TREE", "B1mem"]
port = {p: port_series(d[d.policy == p]) for p in POL}
b0 = port["B0"]; rows = []; rng = np.random.default_rng(20261105); n = len(b0); blk = 20; nb = int(np.ceil(n / blk))
for p, r in port.items():
    sc = b0 * (r.std() / b0.std()); rr = r.reindex(b0.index).fillna(0).values; v = []
    for _ in range(2000):
        st = rng.integers(0, n - blk + 1, nb); ii = (st[:, None] + np.arange(blk)).ravel()[:n]; v.append((rr[ii].mean() - sc.values[ii].mean()) * 256)
    v = np.array(v)
    rows.append(dict(policy=p, vol=r.std() * 16, es_over_dailysd=es95(r) / r.std(), ann=r.mean() * 256,
                     b0_volmatched_ann=sc.mean() * 256, b0_volmatched_es=es95(sc), diff_vs_volmatched=(r.mean() - sc.mean()) * 256,
                     lo=np.quantile(v, .025), hi=np.quantile(v, .975), p_boot=(centered_p(v, (r.mean() - sc.mean()) * 256, 2000) if p != "B0" else np.nan),
                     mdd=mdd(r.values), b0vm_mdd=mdd(sc.values)))
pd.DataFrame(rows).to_csv(f"{M}/vol_matched_diagnostic.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
