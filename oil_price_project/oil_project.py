from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"C:\users\ksh62\OneDrive\Desktop\oil_price_project\data")

INPUT_DIR = BASE_DIR / "input"
PROCESSED_DIR = BASE_DIR / "processed"

domestic = pd.read_csv(INPUT_DIR / "domestic_oil_manual.csv")#국내 유가
international = pd.read_csv(INPUT_DIR / "international_oil_manual.csv")#국제 유가
exchange = pd.read_csv(INPUT_DIR / "exchange_rate_manual.csv")#환율

print("국내 유가 데이터 크기 :", domestic.shape)
print("국제 유가 데이터 크기 :", international.shape)
print("환율 데이터 크기 :", exchange.shape)

print(domestic.head())
print(international.head())
print(exchange.head())

domestic.info()
international.info()
exchange.info()

domestic["date"] = pd.to_datetime(domestic["date"])
international["date"] = pd.to_datetime(international["date"])
exchange["date"] = pd.to_datetime(exchange["date"])

domestic = domestic.sort_values("date")
international = international.sort_values("date")
exchange = exchange.sort_values("date")

domestic = domestic.drop_duplicates("date", keep = "last")
international = international.drop_duplicates("date", keep = "last")
exchange = exchange.drop_duplicates("date", keep = "last")

domestic = domestic[[
    "date",
    "gasoline_price_krw",
    "diesel_price_krw",
    "lpg_price_krw",
]]

international = international[[
    "date",
    "wti_close",
    "brent_close",
]]

exchange = exchange[[
    "date",
    "usd_krw",
]]

exchange_error = exchange["usd_krw"] < 500

exchange.loc[exchange_error,"usd_krw"]=(
    exchange.loc[exchange_error, "usd_krw"] * 10000
)

print("환율 보정 행 수: ", exchange_error.sum())

merged = domestic.merge(international, on = "date", how = "left")
merged = merged.merge(exchange, on = "date", how = "left")

print("병합된 데이터 크기 : " , merged.shape)
print(merged.head())
print(merged.tail())

print(merged.isna().sum())

merged[["wti_close", "brent_close","usd_krw"]] = merged[["wti_close"
                            , "brent_close","usd_krw"]].ffill()

print("결측치 처리 후")
print(merged.isna().sum())


merged["gasoline_change_rate"] = merged["gasoline_price_krw"].pct_change()
merged["diesel_change_rate"] = merged["diesel_price_krw"].pct_change()
merged["lpg_change_rate"] = merged["lpg_price_krw"].pct_change()

merged["wti_change_rate"] = merged["wti_close"].pct_change()
merged["brent_change_rate"] = merged["brent_close"].pct_change()
merged["usd_krw_change_rate"] = merged["usd_krw"].pct_change()

merged["gasoline_ma7"] = merged["gasoline_price_krw"].rolling(window=7).mean()
merged["diesel_ma7"] = merged["diesel_price_krw"].rolling(window=7).mean()
merged["lpg_ma7"] = merged["lpg_price_krw"].rolling(window=7).mean()

merged["gasoline_ma30"] = merged["gasoline_price_krw"].rolling(window=30).mean()
merged["diesel_ma30"] = merged["diesel_price_krw"].rolling(window=30).mean()
merged["lpg_ma30"] = merged["lpg_price_krw"].rolling(window=30).mean()


analysis_columns = [
    "gasoline_price_krw",
    "diesel_price_krw",
    "lpg_price_krw",
    "wti_close",
    "brent_close",
    "usd_krw",
]

print("기초 통계")
print(merged[analysis_columns].describe())

correlation = merged[analysis_columns].corr()

print("상관관계")
print(correlation)

PROCESSED_DIR.mkdir(parents=True, exist_ok = True)

merged.to_csv(PROCESSED_DIR / "preprocessed_oil_dataset.csv",
              index=False,encoding="utf-8-sig")
correlation.to_csv(PROCESSED_DIR / "oil_correlation.csv",
                   encoding="utf-8-sig")

summary = pd.DataFrame({
    "item":[
        "domestic_rows",
        "international_rows",
        "exchange_rows",
        "merged_rows",
        "start_date",
        "end_date",
        "exchange_fixed_rows",
        "lpg_missing_rows"
    ],
    "value":[
        len(domestic),
        len(international),
        len(exchange),
        len(merged),
        merged["date"].min().strftime("%Y-%m-%d"),
        merged["date"].max().strftime("%Y-%m-%d"),
        int(exchange_error.sum()),
        int(merged["lpg_price_krw"].isna().sum()),
    ],
})

summary.to_csv(PROCESSED_DIR / "preprocessing_summary.csv", index = False, encoding = "utf-8-sig")