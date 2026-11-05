# 原始資料來源

本資料夾的檔案未經修改，取自 pysystemtrade 專案：

- Repository：https://github.com/robcarver17/pysystemtrade
- Commit：`a62269d27707806ba61f516791b71a9cd14cf268`（2026-09-29）
- 路徑：`data/futures/multiple_prices_csv/`、`adjusted_prices_csv/`、`roll_calendars_csv/`、`csvconfig/{instrumentconfig,spreadcosts,rollconfig}.csv`
- 商品：CORN、SOYBEAN、WHEAT、COFFEE、COCOA、SUGAR11
- 授權：該專案以 GNU GPL v3 發布（見本 repo 根目錄 LICENSE）。價格資料的原始來源與使用條件以 pysystemtrade 專案說明為準。

21 個檔案的 git blob 雜湊與上述 commit 逐檔一致；SHA-256 見 `outputs/manifests/reproduction_manifest.json`。
可用 `python src/fetch_data.py` 從同一 commit 重新下載並比對。
