import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\users\ksh62\OneDrive\Desktop\oil_price_project\data"

PROCESSED_DIR = BASE_DIR + r"\processed"
FIGURE_DIR = BASE_DIR + r"\data\processed\figures"

df = pd.read_csv(PROCESSED_DIR + r"\preprocessed_oil_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

import os

if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

print("EDA 데이터 크기:", df.shape)
print(df.head())
print(df.info())

plt.figure(figsize=(12, 6))

plt.plot(df["date"], df["gasoline_price_krw"], label="Gasoline")
plt.plot(df["date"], df["diesel_price_krw"], label="Diesel")
plt.plot(df["date"], df["lpg_price_krw"], label="LPG")

plt.title("Domestic Oil Product Price Trend")
plt.xlabel("Date")
plt.ylabel("Price (KRW)")
plt.legend()
plt.grid(True)

plt.savefig(FIGURE_DIR + r"\domestic_oil_price_trend.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(df["date"], df["wti_close"], label="WTI")
plt.plot(df["date"], df["brent_close"], label="Brent")

plt.title("International Oil Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.savefig(FIGURE_DIR + r"\international_oil_price_trend.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(df["date"], df["usd_krw"], label="USD/KRW", color="green")

plt.title("USD/KRW Exchange Rate Trend")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(True)

plt.savefig(FIGURE_DIR + r"\exchange_rate_trend.png", dpi=300)
plt.show()

plt.figure(figsize=(12,6))

ax1 = plt.gca()

ax1.plot(
    df["date"],
    df["gasoline_price_krw"],
    label="Gasoline",
    color="blue"
)
ax1.plot(
    df["date"],
    df["diesel_price_krw"],
    label="Diesel",
    color="orange")

ax1.set_xlabel("Date")
ax1.set_ylabel("Domestic Oil Price(KRW/L)")
ax1.grid(True)

ax2 = ax1.twinx()

ax2.plot(
    df["date"],
    df["usd_krw"],
    label="USD/KRW",
    color="green",
    alpha=0.8)

ax2.set_ylabel("USD/KRW Exchange Rage")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc = "upper left")

plt.title("Domestic Oil Prices and USD/KRW Exchange Rate")
plt.tight_layout()

plt.savefig(FIGURE_DIR+r"\domestic_oil_exchange_rate_comparison.png",dpi=300)
plt.show()


analysis_columns = [
    "gasoline_price_krw",
    "diesel_price_krw",
    "lpg_price_krw",
    "wti_close",
    "brent_close",
    "usd_krw",
]

correlation = df[analysis_columns].corr()

print("상관관계")
print(correlation)

plt.figure(figsize=(8, 6))

plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()

plt.xticks(
    range(len(analysis_columns)),
    analysis_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(analysis_columns)),
    analysis_columns
)

plt.title("Correlation Matrix")
plt.tight_layout()

plt.savefig(FIGURE_DIR + r"\correlation_heatmap.png", dpi=300)
plt.show()

lag_results = []

for lag in range(0, 31):
    result = {
        "lag_days": lag,
        "wti_gasoline_return_corr": df["wti_change_rate"].shift(lag).corr(df["gasoline_change_rate"]),
        "brent_gasoline_return_corr": df["brent_change_rate"].shift(lag).corr(df["gasoline_change_rate"]),
        "usdkrw_gasoline_return_corr": df["usd_krw_change_rate"].shift(lag).corr(df["gasoline_change_rate"]),
    }
    lag_results.append(result)

lag_df = pd.DataFrame(lag_results)

print("변화율 기준 시차 상관관계")
print(lag_df)

plt.figure(figsize=(12, 6))

plt.plot(
    lag_df["lag_days"],
    lag_df["wti_gasoline_return_corr"],
    label="WTI Change -> Gasoline Change"
)

plt.plot(
    lag_df["lag_days"],
    lag_df["brent_gasoline_return_corr"],
    label="Brent Change -> Gasoline Change"
)

plt.plot(
    lag_df["lag_days"],
    lag_df["usdkrw_gasoline_return_corr"],
    label="USD/KRW Change -> Gasoline Change"
)

plt.title("Lag Correlation of Daily Change Rates")
plt.xlabel("Lag Days")
plt.ylabel("Correlation")
plt.legend()
plt.grid(True)

plt.savefig(FIGURE_DIR + r"\lag_correlation_change_rate.png", dpi=300)
plt.show()

lag_df.to_csv(
    PROCESSED_DIR + r"\lag_correlation_change_rate.csv",
    index=False,
    encoding="utf-8-sig"
)

correlation.to_csv(
    PROCESSED_DIR + r"\eda_correlation.csv",
    encoding="utf-8-sig"
)

summary_stats = df[analysis_columns].describe()

summary_stats.to_csv(
    PROCESSED_DIR + r"\eda_summary_statistics.csv",
    encoding="utf-8-sig"
)


print("EDA 완료")
print("국내 유가 그래프:", FIGURE_DIR + r"\domestic_oil_price_trend.png")
print("국제 유가 그래프:", FIGURE_DIR + r"\international_oil_price_trend.png")
print("환율 그래프:", FIGURE_DIR + r"\exchange_rate_trend.png")
print("상관관계 히트맵:", FIGURE_DIR + r"\correlation_heatmap.png")
print("상관관계 CSV:", PROCESSED_DIR + r"\eda_correlation.csv")
print("요약 통계 CSV:", PROCESSED_DIR + r"\eda_summary_statistics.csv")
