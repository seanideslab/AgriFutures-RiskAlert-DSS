import os, urllib.request
B = "https://raw.githubusercontent.com/robcarver17/pysystemtrade/a62269d27707806ba61f516791b71a9cd14cf268/data/futures"   # 固定 commit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RAW = f"{ROOT}/data/raw"
for f in ["csvconfig/instrumentconfig.csv", "csvconfig/spreadcosts.csv", "csvconfig/rollconfig.csv"]:
    os.makedirs(RAW, exist_ok=True); urllib.request.urlretrieve(f"{B}/{f}", f"{RAW}/{os.path.basename(f)}")
for i in ["CORN", "SOYBEAN", "WHEAT", "COFFEE", "COCOA", "SUGAR11"]:
    for d in ["multiple_prices_csv", "adjusted_prices_csv", "roll_calendars_csv"]:
        os.makedirs(f"{RAW}/{d}", exist_ok=True); urllib.request.urlretrieve(f"{B}/{d}/{i}.csv", f"{RAW}/{d}/{i}.csv")
print("done")
