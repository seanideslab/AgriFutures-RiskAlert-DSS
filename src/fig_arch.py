import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = ["Noto Sans CJK TC"]
fig, ax = plt.subplots(figsize=(6.6, 2.2)); ax.set_xlim(0, 100); ax.set_ylim(6, 41); ax.axis("off")
B = {}
def box(k, x, y, w, t, fc="#eef4fc", ec="#2a78d6"):
    ax.add_patch(FancyBboxPatch((x, y), w, 10, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + 5, t, ha="center", va="center", fontsize=6.4); B[k] = (x, y, w)
def arr(a, b, side="h", label=None):
    xa, ya, wa = B[a]; xb, yb, wb = B[b]
    if side == "h": p0, p1 = (xa + wa + 0.4, ya + 5), (xb - 0.4, yb + 5)
    else: p0, p1 = (xa + wa / 2, ya - 0.4), (xb + wb / 2, yb + 10.4)
    ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9))
    if label: ax.text((p0[0] + p1[0]) / 2 + 1, (p0[1] + p1[1]) / 2 + 1, label, fontsize=7, color="#52514e")
box("A", 1, 27, 17, "逐契約收盤資料\n換月對應")
box("Q", 23, 27, 15, "資料品質與\n特徵有效？", fc="#fff6e0", ec="#c98500")
box("S", 43, 27, 20, "基礎策略\nEWMAC+Carry\nq0、影子帳戶")
box("F", 68, 27, 14, "15 個特徵\n市場／策略／契約")
box("M", 68, 8, 14, "邏輯迴歸\n損失事件機率 p")
box("SM", 43, 8, 20, "狀態機（恢復記憶）\nNORMAL／REDUCED\n／PAUSED")
box("X", 23, 8, 15, "q = a × q0\nt+1 收盤成交\n成本與換月")
box("L", 1, 8, 17, "決策紀錄\n版本、門檻、理由")
box("RV", 85.5, 27, 13.5, "覆核：保留狀態\n清除恢復計數", fc="#f3f2ee", ec="#8a8984")
arr("A", "Q"); arr("Q", "S", label=None); arr("S", "F"); arr("F", "M", "v")
for a, b in [("M", "SM"), ("SM", "X"), ("X", "L")]:
    xa, ya, wa = B[a]; xb, yb, wb = B[b]
    ax.annotate("", (xb + wb + 0.4, yb + 5), (xa - 0.4, ya + 5), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9))
ax.annotate("", (92, 37.8), (30.5, 37.6), arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.8, ls="--", connectionstyle="arc3,rad=-0.12"))
ax.text(39.5, 33.5, "是", fontsize=7, color="#52514e"); ax.text(60, 39.3, "否", fontsize=7, color="#52514e")
ax.annotate("", (55, 18.4), (92, 26.6), arrowprops=dict(arrowstyle="-|>", color="#8a8984", lw=0.8, ls="--"))
fig.savefig("outputs/figures/fig1_arch.png", dpi=250, bbox_inches="tight"); print("ok")
