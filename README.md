# AgriFutures-RiskAlert-DSS

農產品期貨風險警示的行動轉換：具狀態記憶之決策支援流程與樣本外評估（IMP 2026 研討會論文之重現包）

*Turning risk alerts into actions in agricultural futures: a stateful decision-support workflow and out-of-sample evaluation — replication package.*

本 repo 只收錄論文採用的現行版本（spec-v2.1）。執行下列步驟可逐位元重現 `outputs/` 內所有結果。

## 內容

| 路徑 | 說明 |
|---|---|
| `config/study_spec.yaml` | 研究規格與所有修訂紀錄（amendments） |
| `config/*_manifest.csv` | 資料、策略與走步切分清單 |
| `data/raw/` | 原始資料（pysystemtrade commit `a62269d`，未修改；見 `data/raw/NOTICE.md`） |
| `src/` | 資料處理、基礎策略、風險模型、政策狀態機、評估與繪圖 |
| `tests/acceptance.py` | 前視資訊、前綴重建、決策重播、帳務對帳等驗收 |
| `outputs/predictions/` | 每一筆樣本外風險機率與模型版本 |
| `outputs/decisions/decisions_B3.csv` | B3 逐日決策紀錄（狀態、門檻、理由碼、資料品質） |
| `outputs/daily/daily_pnl.csv.gz` | 全部政策逐日持倉、毛損益、成本、倍率 |
| `outputs/metrics/` | 論文表格與文字中的所有指標 |
| `outputs/experiment_log.csv` | 全部政策候選試驗紀錄 |
| `outputs/manifests/` | 資料品質、環境與檔案雜湊 |

## 重跑（Python 3.10 以上）

    pip install -r requirements.txt
    python src/fetch_data.py      # 選用：自 pysystemtrade 固定 commit 重新下載原始資料
    python src/causal_chain.py    # 因果換月鏈、無報價處理、資料清單 → data/processed/, config/data_manifest.csv
    python src/core.py            # 基礎策略、影子帳戶、標籤、特徵 → outputs/panel/
    python src/experiment.py      # 走步模型、政策選擇、B0–B5 與消融、決策紀錄
    python src/evaluation.py      # 指標、bootstrap、Holm、警示事件、損益分解
    python src/diagnostics.py     # 事後等波動診斷
    python src/figures.py && python src/fig_arch.py && python src/fig_case.py
    python tests/acceptance.py    # → outputs/acceptance.json

在 2 核心 CPU 上全部約 2–3 分鐘。bootstrap 隨機種子固定為 20261105。

## 研究設計摘要

- **商品：** 玉米、黃豆、小麥、咖啡、可可、11號糖，逐契約收盤價，2000-01 至 2024-03-28。
- **換月：** 只依過去換月紀錄事前排定，不使用未來實際換月日。
- **基礎策略：** EWMAC＋Carry（pysystemtrade 第 15 章公開參數，自行重建）。每商品 10% 波動目標，t 日收盤決策、t+1 日收盤成交。
- **風險模型：** 邏輯迴歸，預測五日內超過風險預算的損失事件，每年走步重訓。
- **政策：**
  - B0：原策略。
  - B1：波動風控。
  - B2：AI 無記憶。
  - B3：AI 有記憶。
  - B4：平均倍率配對。
  - B5：事前風險配對。
- **測試期：** 2008-01 至 2024-03。

## 授權

程式碼以 GNU GPL v3 釋出（見 `LICENSE`）。`data/raw/` 的資料取自 pysystemtrade（GPL v3），來源與條件見 `data/raw/NOTICE.md`。

本 repo 僅供研究重現，不構成投資建議。
