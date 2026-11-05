import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0, os.path.dirname(__file__))
from core import INST, ZH, load_all
from evaluation import port_series

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs/figures"; os.makedirs(OUT, exist_ok=True)
for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Noto Sans CJK TC", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "figure.dpi": 200, "savefig.bbox": "tight", "axes.unicode_minus": False})
C = {"B5": "#8f5a2a", "B0": "#0b0b0b", "B1": "#eda100", "B2": "#eb6834", "B3": "#2a78d6", "B4": "#1baf7a", "A2": "#6250d6",
     "TREE": "#e87ba4", "B1mem": "#8a8984"}
LAB = {"B5": "B5 事前風險配對", "B0": "B0 原策略", "B1": "B1 波動風控", "B2": "B2 AI無記憶", "B3": "B3 AI有記憶", "B4": "B4 固定減碼",
       "A2": "A2 去策略特徵", "TREE": "淺層樹(有記憶)", "B1mem": "B1+記憶"}
LS = {"B0": "-", "B2": "--", "B3": "-", "B4": ":"}

d = pd.read_csv(f"{ROOT}/outputs/daily/daily_pnl.csv.gz", parse_dates=["date"])
port = {p: port_series(d[d.policy == p]) for p in C}

# 圖3：淨值與回撤
fig, ax = plt.subplots(2, 1, figsize=(6.3, 4.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for p in ["B0", "B2", "B3", "B4"]:
    eq = port[p].cumsum() * 100
    ax[0].plot(eq.index, eq, color=C[p], lw=1.4, ls=LS[p], label=LAB[p])
    dd = (eq.cummax() - eq)
    ax[1].plot(dd.index, -dd, color=C[p], lw=1.1, ls=LS[p])
ax[0].set_ylabel("累積淨報酬（%，對參考資本）"); ax[1].set_ylabel("回撤（%）")
ax[0].legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
fig.savefig(f"{OUT}/fig3_equity.png"); plt.close(fig)

# 圖4：(a) 校準 (b) 風險-報酬與等風險縮放線
cal = pd.read_csv(f"{ROOT}/outputs/metrics/calibration.csv")
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
ax[0].plot([0, 0.35], [0, 0.35], color="#b5b4ae", lw=1)
ax[0].plot(cal.p_mean, cal.y_rate, "o-", color=C["B3"], ms=5, lw=1.5)
ax[0].set_xlabel("預測機率（十分位平均）"); ax[0].set_ylabel("實際事件率"); ax[0].set_title("(a) 邏輯迴歸校準，2008–2024", fontsize=9)
b0 = port["B0"]; vmax = b0.std() * 16 * 100
xs = np.linspace(0, vmax * 1.05, 20)
ax[1].plot(xs, xs * (b0.mean() * 256) / (b0.std() * 16), color="#b5b4ae", lw=1.2, ls="--", label="B0 等比例縮放")
off = {"B5": (-16, 5), "B0": (4, -9), "B1": (4, 3), "B2": (4, 3), "B3": (4, -10), "B4": (4, 3), "A2": (4, 3), "TREE": (4, 3), "B1mem": (4, -9)}
for p in C:
    r = port[p]; x, y = r.std() * 16 * 100, r.mean() * 256 * 100
    ax[1].plot(x, y, "o", color=C[p], ms=7, mec="white", mew=1.2)
    ax[1].annotate({"B1mem": "A3"}.get(p, p), (x, y), xytext=off[p], textcoords="offset points", fontsize=8, color="#0b0b0b")
ax[1].set_xlabel("年化波動（%）"); ax[1].set_ylabel("年化淨報酬（%）"); ax[1].set_title("(b) 風險與報酬（六商品等權）", fontsize=9)
ax[1].legend(frameon=False, fontsize=8, loc="upper left"); ax[1].set_xlim(0, vmax * 1.12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_calib_frontier.png"); plt.close(fig)

# 圖5：失效案例：事先規則＝B3 相對 B0 錯失毛獲利最大的 60 交易日窗
P = load_all()
b3 = d[d.policy == "B3"].set_index(["instrument", "date"]); b0d = d[d.policy == "B0"].set_index(["instrument", "date"])
miss = ((b0d.gross - b3.gross.reindex(b0d.index)).where(b0d.gross > 0, 0)).groupby(level=0).apply(lambda s: s.droplevel(0).rolling(60).sum())
inst, end = miss.idxmax()
i = inst if isinstance(inst, str) else inst[0]
s = b0d.loc[i]; t = b3.loc[i]
w0 = s.index.get_loc(end) - 80; w1 = s.index.get_loc(end) + 20
idx = s.index[w0:w1]
dec = pd.read_csv(f"{ROOT}/outputs/decisions/decisions_B3.csv", parse_dates=["decision_date"])
g = dec[dec.instrument == i].set_index("decision_date").reindex(idx)
fig, ax = plt.subplots(3, 1, figsize=(6.3, 3.9), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 0.6]})
ax[0].plot(idx, (s.net.loc[idx].cumsum()) * 100, color=C["B0"], lw=1.4, label="B0")
ax[0].plot(idx, (t.net.reindex(idx).cumsum()) * 100, color=C["B3"], lw=1.4, label="B3")
ax[0].set_ylabel("累積淨報酬 %"); ax[0].legend(frameon=False, fontsize=8, ncol=2)
ax[1].plot(idx, g.risk_probability, color=C["B3"], lw=1.2)
for th, nm in [("threshold_low", "θL"), ("threshold_reduce", "θR"), ("threshold_pause", "θP")]:
    ax[1].plot(idx, g[th], color="#8a8984", lw=0.8, ls=":")
    ax[1].annotate(nm, (idx[-1], g[th].iloc[-1]), xytext=(3, -3), textcoords="offset points", fontsize=7, color="#52514e")
ax[1].set_ylabel("p")
ax[2].step(idx, g.exposure_multiplier, where="post", color=C["B3"], lw=1.4)
ax[2].set_ylabel("倍率 a"); ax[2].set_yticks([0, 0.5, 1])
fig.savefig(f"{OUT}/fig5_case.png"); plt.close(fig)
open(f"{OUT}/fig5_case_meta.txt", "w").write(f"{i},{idx[0].date()},{idx[-1].date()},{end}")
print("case", i, idx[0].date(), idx[-1].date())
