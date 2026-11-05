"""圖4（v2）：失效案例，沿用核查版版面——灰底為事先規則選出的 60 日窗、(c) 分開決策日倍率與延後成交倍率。"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": ["Noto Sans CJK TC"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#dde1e6", "grid.linewidth": 0.6, "axes.unicode_minus": False})
d = pd.read_csv(f"{ROOT}/outputs/daily/daily_pnl.csv.gz", parse_dates=["date"])
b0 = d[d.policy == "B0"].set_index(["instrument", "date"]); b3 = d[d.policy == "B3"].set_index(["instrument", "date"])
miss = ((b0.gross - b3.gross.reindex(b0.index)).where(b0.gross > 0, 0)).groupby(level=0).apply(lambda s: s.droplevel(0).rolling(60).sum())
(i, end) = miss.idxmax()
s0 = b0.loc[i]; s3 = b3.loc[i]
e = s0.index.get_loc(end); w0, w1 = s0.index[e - 59], end            # 事先規則：60 交易日窗
idx = s0.index[e - 80:min(e + 20, len(s0))]                           # 展示：選取窗前 20、後 20 交易日，共 100 日
dec = pd.read_csv(f"{ROOT}/outputs/decisions/decisions_B3.csv", parse_dates=["decision_date"])
g = dec[dec.instrument == i].set_index("decision_date").reindex(idx)
c0 = s0.net.loc[idx].fillna(0).cumsum() * 100; c3 = s3.net.reindex(idx).fillna(0).cumsum() * 100
fig, ax = plt.subplots(3, 1, figsize=(6.4, 5.1), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.1, 0.8]})
for a in ax:
    a.axvspan(w0, w1, color="#eef0f3", zorder=0); a.axvline(w0, color="#9aa5b1", lw=0.6, ls=":"); a.axvline(w1, color="#9aa5b1", lw=0.6, ls=":")
ax[0].plot(idx, c0, color="#1f2328", lw=1.5, label="B0 原策略"); ax[0].plot(idx, c3, color="#1f5f9e", lw=1.5, label="B3 有恢復記憶")
ax[0].annotate(f"{c0.iloc[-1]:.1f}%", (idx[-1], c0.iloc[-1]), xytext=(-2, 6), textcoords="offset points", ha="right", fontsize=8, color="#1f2328")
ax[0].annotate(f"{c3.iloc[-1]:.1f}%", (idx[-1], c3.iloc[-1]), xytext=(-2, 6), textcoords="offset points", ha="right", fontsize=8, color="#1f5f9e")
ax[0].set_ylabel("累積淨損益／\n參考資本（%）"); ax[0].legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
ax[0].set_title("(a) 個案期間損益", loc="left", fontsize=9)
ax[1].plot(idx, g.risk_probability, color="#1f5f9e", lw=1.3, label="損失機率 p")
ax[1].plot(idx, g.threshold_reduce, color="#c06a00", lw=0.9, ls="--", label="減碼 θR")
ax[1].plot(idx, g.threshold_low, color="#2e7d5b", lw=0.9, ls=":", label="恢復 θL")
ax[1].plot(idx, g.threshold_pause, color="#a23b3b", lw=0.9, ls=(0, (6, 3)), label="暫停 θP")
ax[1].set_ylabel("風險機率"); ax[1].legend(frameon=False, ncol=2, fontsize=7.5, loc="upper right")
ax[1].set_title("(b) 決策當日機率與有效門檻", loc="left", fontsize=9)
dm = g.exposure_multiplier
ax[2].step(idx, dm, where="post", color="#c06a00", lw=1.2, ls="--", label="當日決策倍率")
ax[2].step(idx, dm.shift(1), where="post", color="#1f5f9e", lw=1.4, label="延後成交倍率")
ax[2].set_yticks([0, 0.5, 1]); ax[2].set_ylim(-0.08, 1.25); ax[2].set_ylabel("曝險倍率")
ax[2].legend(frameon=False, ncol=2, fontsize=7.5, loc="upper left"); ax[2].set_title("(c) 決策與成交時點分開呈現", loc="left", fontsize=9)
ax[2].set_xlabel("交易日期")
fig.tight_layout(); fig.savefig(f"{ROOT}/outputs/figures/fig5_case.png", dpi=250, bbox_inches="tight")
fig.savefig(f"{ROOT}/outputs/figures/fig5_case.svg", bbox_inches="tight")
meta = dict(instrument=i, window_start=str(w0.date()), window_end=str(w1.date()), shown_start=str(idx[0].date()), shown_end=str(idx[-1].date()),
            shown_days=len(idx), b0_shown=float(c0.iloc[-1]), b3_shown=float(c3.iloc[-1]), p_max=float(g.risk_probability.max()),
            b0_window=float(s0.net.loc[w0:w1].sum() * 100), b3_window=float(s3.net.loc[w0:w1].sum() * 100))
pd.Series(meta).to_json(f"{ROOT}/outputs/figures/fig5_case_meta.json", force_ascii=False)
print(meta)
