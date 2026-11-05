"""v2 因果化換月：逐契約價格面板＋只用過去換月紀錄排定的事前換月日。

- 價格面板：每日把 PRICE/FORWARD/CARRY 三欄展開為 (日期, 契約) → 收盤價。
- 事前排程：持有契約 c 的計畫換月日 = c 的契約月起點 + 「最近三次已發生、同月份契約換月的最早相對位移」 − 10 日緩衝。
  只使用 t 日以前已發生的換月；同月份契約尚無任何換月歷史時（僅咖啡 2007–2008 年的前幾個契約），固定以契約月起點 − 45 日 − 10 日排定並標記 default。
- 每日資料：每個（價格, 契約代碼）欄位對各自取當日最後一筆有值的列，價格與代碼必來自同一列；週六、週日列捨棄（其報價併入下一交易日）。
- 執行：在計畫日當日或之後，第一個「舊、新契約當日都有價格」的交易日收盤換月；新契約依 rollconfig 的 HoldRollCycle 規則決定。
- 資料源提前換月使持有契約消失：當日舊倉損益未知，標記 unscorable（所有政策一致排除），並於當日收盤轉入新契約。
- 估值：持有契約當日無價格（假日／缺口）→ 該日不交易、價差不更新（deferred），下一個有價日一次認列完整價差。
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw"; OUT = f"{ROOT}/data/processed"; os.makedirs(OUT, exist_ok=True)
START, END = "2000-01-01", "2024-03-28"
BUFFER_DAYS = 10


def cdate(c):
    c = int(c); return pd.Timestamp(year=c // 10000, month=(c // 100) % 100, day=1)


MON = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
RC = pd.read_csv(f"{RAW}/rollconfig.csv").set_index("Instrument")


def next_in_cycle(c, cycle):
    """依持有循環（事前契約規則）決定下一個持有契約。"""
    y, mo = c // 10000, (c // 100) % 100
    ms = sorted(MON[x] for x in cycle)
    later = [x for x in ms if x > mo]
    return (y * 10000 + later[0] * 100) if later else ((y + 1) * 10000 + ms[0] * 100)


def build(i, end=END):
    m = pd.read_csv(f"{RAW}/multiple_prices_csv/{i}.csv", parse_dates=["DATETIME"])
    m["date"] = m.DATETIME.dt.normalize()
    m = m.sort_values("DATETIME")
    m = m[m.date.dt.dayofweek < 5]                              # 週末列（夜盤記錄）併入下一交易日
    parts = []
    for col, cc in [("PRICE", "PRICE_CONTRACT"), ("FORWARD", "FORWARD_CONTRACT"), ("CARRY", "CARRY_CONTRACT")]:
        x = m[m[col].notna() & m[cc].notna()].groupby("date")[[col, cc]].last()   # 同一列的價格與代碼配對
        parts.append(x)
    m = pd.concat(parts, axis=1).sort_index()
    m = m.loc[:end]                                             # 前綴重建檢查可傳入較早的 end
    full = m.copy()
    m = m.loc[START:end]
    cycle = RC.loc[i, "HoldRollCycle"]
    # ---- 價格面板 ----
    rows = []
    for col, cc in [("PRICE", "PRICE_CONTRACT"), ("FORWARD", "FORWARD_CONTRACT"), ("CARRY", "CARRY_CONTRACT")]:
        x = m[[col, cc]].dropna()
        x = x[x[col] > 0]
        rows.append(pd.DataFrame({"date": x.index, "contract": x[cc].astype(int).values, "px": x[col].values}))
    panel = pd.concat(rows).drop_duplicates(["date", "contract"]).set_index(["date", "contract"]).px.sort_index()
    # ---- 資料源過去的換月位移（只用 switch < t 者）----
    hold = full.PRICE_CONTRACT.dropna().astype(int)
    sw = hold[hold != hold.shift()].iloc[1:]
    src_rolls = pd.DataFrame({"switch": sw.index, "old": hold.shift().loc[sw.index].astype(int).values})
    src_rolls["offset"] = [(s - cdate(o)).days for s, o in zip(src_rolls.switch, src_rolls.old)]
    src_rolls["month"] = [(o // 100) % 100 for o in src_rolls.old]
    dates = m.index
    held = int(hold.loc[:dates[0]].iloc[-1])
    rec = []; prev_px = None; d_prev_date = None
    plan_cache = {}

    def plan(c, t):
        if c in plan_cache: return plan_cache[c]
        mo = (c // 100) % 100
        past = src_rolls[(src_rolls.month == mo) & (src_rolls.switch < t)]
        if len(past):
            off = int(past.offset.iloc[-3:].min()); src = "history"
        else:
            off = -45; src = "default"
        plan_cache[c] = (cdate(c) + pd.Timedelta(days=off - BUFFER_DAYS), src)
        return plan_cache[c]

    for t in dates:
        px = panel.get((t, held), np.nan)
        pdate, psrc = plan(held, t)
        src_c = m.PRICE_CONTRACT.get(t, np.nan)
        dP = np.nan; valued = 0; unscorable = 0; forced = 0; roll = 0; dP_feat = np.nan
        new = next_in_cycle(held, cycle)
        if not np.isnan(px):
            valued = 1
            if prev_px is not None:
                dP = px - prev_px
            prev_px = px
            if t >= pdate and not np.isnan(panel.get((t, new), np.nan)):
                roll = 1
        elif pd.notna(src_c) and int(src_c) > held:
            # 資料源已提前換到更遠契約、持有契約不再有價：當日舊倉損益未知 → 標記不可計分，於當日收盤強制轉入可得之新契約
            cand = new if not np.isnan(panel.get((t, new), np.nan)) else int(src_c)
            if not np.isnan(panel.get((t, cand), np.nan)):
                new = cand; roll = 1; forced = 1; unscorable = 1
                # 僅供特徵：以新契約同日價差橋接（損益仍為未知、不計分）
                prv = d_prev_date and panel.get((d_prev_date, cand), np.nan)
                if prv is not None and not np.isnan(prv):
                    dP_feat = panel.get((t, cand)) - prv
        if valued: dP_feat = dP
        d_prev_date = t
        rec.append({"date": t, "contract": held, "price": px, "dP": dP, "dP_feat": dP_feat, "valued": valued, "roll_today": roll,
                    "roll_forced": forced, "unscorable": unscorable,
                    "plan_roll_date": pdate, "plan_source": psrc, "days_to_roll": max((pdate - t).days, 0),
                    "next_contract": new,
                    "carry_price": m.CARRY.get(t, np.nan), "carry_contract": m.CARRY_CONTRACT.get(t, np.nan),
                    "src_price_contract": src_c})
        if roll:
            held = new; prev_px = panel.get((t, new))
    d = pd.DataFrame(rec).set_index("date")
    # carry 用的「較近契約」價格：資料源 CARRY 欄位（與持有鏈無關的市場資訊，當日可得）
    d["carry_yrs"] = [((cdate(c) - cdate(cc)).days / 365.25) if (pd.notna(cc) and pd.notna(c)) else np.nan
                      for c, cc in zip(d.src_price_contract, d.carry_contract)]
    d["carry_ref_price"] = m.PRICE.reindex(d.index)
    # 未估值但非強制換月之日：持有契約當日無報價（假日或資料缺口）；不交易，下一個有價日一次認列價差
    d.loc[(d.valued == 0) & (d.unscorable == 0), "dP"] = 0.0
    d.loc[(d.valued == 0) & (d.unscorable == 0), "dP_feat"] = 0.0
    q = {"days": len(d), "deferred_days": int(((d.valued == 0) & (d.unscorable == 0)).sum()),
         "forced_rolls_unscorable": int(d.roll_forced.sum()), "rolls": int(d.roll_today.sum()),
         "max_deferred_run": int((d.valued == 0).astype(int).groupby((d.valued == 1).cumsum()).sum().max()),
         "test_deferred": int(((d.valued == 0) & (d.unscorable == 0) & (d.index >= "2008-01-01")).sum()),
         "test_forced": int(((d.roll_forced == 1) & (d.index >= "2008-01-01")).sum()),
         "forced_dates": [str(x.date()) for x in d.index[d.roll_forced == 1]],
         "default_plan_days_test": int(((d.plan_source == "default") & (d.index >= "2008-01-01")).sum()),
         "default_plan_contracts": int(d[d.plan_source == "default"].contract.nunique())}
    return d, q


if __name__ == "__main__":
    import json
    allq = {}
    for i in ["CORN", "SOYBEAN", "WHEAT", "COFFEE", "COCOA", "SUGAR11"]:
        d, q = build(i)
        d.to_csv(f"{OUT}/{i}_v2.csv")
        allq[i] = q; print(i, q)
    os.makedirs(f"{ROOT}/outputs/manifests", exist_ok=True)
    json.dump(allq, open(f"{ROOT}/outputs/manifests/data_quality.json", "w"), indent=1)
    # 資料清單：來源、期間、契約設定、雜湊
    import hashlib
    cfg = pd.read_csv(f"{RAW}/instrumentconfig.csv").set_index("Instrument")
    sc = pd.read_csv(f"{RAW}/spreadcosts.csv").set_index("Instrument")
    man = []
    for i, q in allq.items():
        d = pd.read_csv(f"{OUT}/{i}_v2.csv", index_col=0, parse_dates=True)
        man.append({"instrument": i, "description": cfg.loc[i, "Description"], "pointsize": cfg.loc[i, "Pointsize"],
                    "commission_per_contract_usd": cfg.loc[i, "PerBlock"], "spread_cost_price": sc.loc[i, "SpreadCost"],
                    "start": d.index.min().date(), "end": d.index.max().date(), "days": q["days"], "rolls": q["rolls"],
                    "no_quote_days": q["deferred_days"], "unscorable_days": q["forced_rolls_unscorable"],
                    "source": "pysystemtrade data/futures @ a62269d27707806ba61f516791b71a9cd14cf268 (GPL-3.0)",
                    "sha256_multiple_prices": hashlib.sha256(open(f"{RAW}/multiple_prices_csv/{i}.csv", "rb").read()).hexdigest()})
    pd.DataFrame(man).to_csv(f"{ROOT}/config/data_manifest.csv", index=False)
