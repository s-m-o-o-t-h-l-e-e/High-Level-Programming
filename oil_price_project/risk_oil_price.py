import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\ksh62\OneDrive\Desktop\oil_price_project\data"

PROCESSED_DIR = BASE_DIR + r"\processed"
EVENT_FIGURE_DIR = PROCESSED_DIR + r"\figures\events"
SUMMARY_FIGURE_DIR = EVENT_FIGURE_DIR + r"\summary"

if not os.path.exists(EVENT_FIGURE_DIR):
    os.makedirs(EVENT_FIGURE_DIR)

if not os.path.exists(SUMMARY_FIGURE_DIR):
    os.makedirs(SUMMARY_FIGURE_DIR)

oil = pd.read_csv(PROCESSED_DIR + r"\preprocessed_oil_dataset.csv")

if os.path.exists(PROCESSED_DIR + r"\news_pressure_index.csv"):
    news = pd.read_csv(PROCESSED_DIR + r"\news_pressure_index.csv")
else:
    news = pd.read_csv(PROCESSED_DIR + r"\news_risk_index.csv")
if os.path.exists(PROCESSED_DIR + r"\kor_news_pressure_index.csv"):
    kor_news = pd.read_csv(PROCESSED_DIR + r"\kor_news_pressure_index.csv")
else:
    kor_news = pd.read_csv(PROCESSED_DIR + r"\kor_news_risk_index.csv")

oil["date"] = pd.to_datetime(oil["date"])
news["date"] = pd.to_datetime(news["date"])
kor_news["date"] = pd.to_datetime(kor_news["date"])

oil = oil.sort_values("date")
news = news.sort_values("date")
kor_news = kor_news.sort_values("date")

oil = oil.drop_duplicates("date", keep="last")
news = news.drop_duplicates("date", keep="last")
kor_news = kor_news.drop_duplicates("date", keep="last")

if "article_count" not in news.columns:
    if "intl_article_count" in news.columns:
        news["article_count"] = news["intl_article_count"]
    else:
        news["article_count"] = 0

if "relevant_article_count" not in news.columns:
    if "intl_relevant_article_count" in news.columns:
        news["relevant_article_count"] = news["intl_relevant_article_count"]
    else:
        news["relevant_article_count"] = news["article_count"]

if "average_pressure_score" not in news.columns:
    if "intl_average_pressure_score" in news.columns:
        news["average_pressure_score"] = news["intl_average_pressure_score"]
    elif "average_risk_score" in news.columns:
        news["average_pressure_score"] = news["average_risk_score"]
    else:
        news["average_pressure_score"] = 0

if "max_pressure_score" not in news.columns:
    if "intl_max_pressure_score" in news.columns:
        news["max_pressure_score"] = news["intl_max_pressure_score"]
    elif "max_risk_score" in news.columns:
        news["max_pressure_score"] = news["max_risk_score"]
    else:
        news["max_pressure_score"] = 0

if "kor_article_count" not in kor_news.columns:
    kor_news["kor_article_count"] = 0

if "kor_relevant_article_count" not in kor_news.columns:
    kor_news["kor_relevant_article_count"] = kor_news["kor_article_count"]

if "kor_average_pressure_score" not in kor_news.columns:
    if "kor_average_risk_score" in kor_news.columns:
        kor_news["kor_average_pressure_score"] = kor_news["kor_average_risk_score"]
    else:
        kor_news["kor_average_pressure_score"] = 0

if "kor_max_pressure_score" not in kor_news.columns:
    if "kor_max_risk_score" in kor_news.columns:
        kor_news["kor_max_pressure_score"] = kor_news["kor_max_risk_score"]
    else:
        kor_news["kor_max_pressure_score"] = 0

news = news[
    [
        "date",
        "article_count",
        "relevant_article_count",
        "average_pressure_score",
        "max_pressure_score",
    ]
]

kor_news = kor_news[
    [
        "date",
        "kor_article_count",
        "kor_relevant_article_count",
        "kor_average_pressure_score",
        "kor_max_pressure_score",
    ]
]

df = oil.merge(news, on="date", how="left")
df = df.merge(kor_news, on="date", how="left")

fill_columns = [
    "article_count",
    "relevant_article_count",
    "average_pressure_score",
    "max_pressure_score",
    "kor_article_count",
    "kor_relevant_article_count",
    "kor_average_pressure_score",
    "kor_max_pressure_score",
]

df[fill_columns] = df[fill_columns].fillna(0)

events = [
    {
        "event_name": "global_financial_crisis",
        "event_type": "international_economy",
        "event_date": "2008-09-15",
        "start_date": "2008-06-01",
        "end_date": "2009-06-30",
    },
    {
        "event_name": "covid19_pandemic",
        "event_type": "international_economy",
        "event_date": "2020-03-11",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
    },
    {
        "event_name": "russia_ukraine_war",
        "event_type": "international_geopolitics",
        "event_date": "2022-02-24",
        "start_date": "2021-11-01",
        "end_date": "2022-12-31",
    },
    {
        "event_name": "israel_hamas_middle_east_conflict",
        "event_type": "international_geopolitics",
        "event_date": "2023-10-07",
        "start_date": "2023-07-01",
        "end_date": "2024-06-30",
    },
    {
        "event_name": "fuel_tax_cut_start",
        "event_type": "domestic_policy",
        "event_date": "2021-11-12",
        "start_date": "2021-10-01",
        "end_date": "2022-01-31",
    },
    {
        "event_name": "fuel_tax_cut_expansion",
        "event_type": "domestic_policy",
        "event_date": "2022-07-01",
        "start_date": "2022-05-01",
        "end_date": "2022-09-30",
    },
    {
        "event_name": "domestic_oil_price_peak",
        "event_type": "domestic_price",
        "event_date": "2022-06-30",
        "start_date": "2022-04-01",
        "end_date": "2022-08-31",
    },
    {
        "event_name": "domestic_oil_price_low_after_covid",
        "event_type": "domestic_price",
        "event_date": "2020-05-15",
        "start_date": "2020-03-01",
        "end_date": "2020-07-31",
    },
    {
        "event_name": "truckers_strike_supply_risk",
        "event_type": "domestic_supply_risk",
        "event_date": "2022-11-24",
        "start_date": "2022-11-01",
        "end_date": "2022-12-31",
    },
    {
        "event_name": "urea_solution_crisis",
        "event_type": "domestic_supply_risk",
        "event_date": "2021-11-01",
        "start_date": "2021-10-01",
        "end_date": "2022-02-28",
    },

    {   "event_name": "iran_hormuz_strait_risk",
        "event_type": "international_geopolitics",
        "event_date": "2026-04-19",
        "start_date": "2026-03-15",
        "end_date" : "2026-05-24",
},
]
for event in events:
    event["event_date"] = pd.to_datetime(event["event_date"])
    event["start_date"] = pd.to_datetime(event["start_date"])
    event["end_date"] = pd.to_datetime(event["end_date"])

def change_rate(before_value, after_value):
    if pd.isna(before_value):
        return None
    if before_value == 0:
        return None
    return ((after_value - before_value) / before_value) * 100

summary_list = []
window_list = []

for event in events:
    event_name = event["event_name"]
    event_type = event["event_type"]
    event_date = event["event_date"]
    start_date = event["start_date"]
    end_date = event["end_date"]

    event_data = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].copy()

    event_data["event_name"] = event_name
    event_data["event_type"] = event_type
    event_data["event_date"] = event_date

    event_data["event_phase"] = "before"
    event_data.loc[event_data["date"] >= event_date, "event_phase"] = "after"

    before = event_data[event_data["date"] < event_date]
    after = event_data[event_data["date"] >= event_date]

    gasoline_before_mean = before["gasoline_price_krw"].mean()
    gasoline_after_mean = after["gasoline_price_krw"].mean()

    diesel_before_mean = before["diesel_price_krw"].mean()
    diesel_after_mean = after["diesel_price_krw"].mean()

    lpg_before_mean = before["lpg_price_krw"].mean()
    lpg_after_mean = after["lpg_price_krw"].mean()

    wti_before_mean = before["wti_close"].mean()
    wti_after_mean = after["wti_close"].mean()

    brent_before_mean = before["brent_close"].mean()
    brent_after_mean = after["brent_close"].mean()

    usd_krw_before_mean = before["usd_krw"].mean()
    usd_krw_after_mean = after["usd_krw"].mean()

    intl_pressure_before_mean = before["average_pressure_score"].mean()
    intl_pressure_after_mean = after["average_pressure_score"].mean()

    kor_pressure_before_mean = before["kor_average_pressure_score"].mean()
    kor_pressure_after_mean = after["kor_average_pressure_score"].mean()

    summary_list.append(
        {
            "event_name": event_name,
            "event_type": event_type,
            "event_date": event_date.strftime("%Y-%m-%d"),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "before_days": len(before),
            "after_days": len(after),

            "gasoline_before_mean": gasoline_before_mean,
            "gasoline_after_mean": gasoline_after_mean,
            "gasoline_change_rate": change_rate(gasoline_before_mean, gasoline_after_mean),

            "diesel_before_mean": diesel_before_mean,
            "diesel_after_mean": diesel_after_mean,
            "diesel_change_rate": change_rate(diesel_before_mean, diesel_after_mean),

            "lpg_before_mean": lpg_before_mean,
            "lpg_after_mean": lpg_after_mean,
            "lpg_change_rate": change_rate(lpg_before_mean, lpg_after_mean),

            "wti_before_mean": wti_before_mean,
            "wti_after_mean": wti_after_mean,
            "wti_change_rate": change_rate(wti_before_mean, wti_after_mean),

            "brent_before_mean": brent_before_mean,
            "brent_after_mean": brent_after_mean,
            "brent_change_rate": change_rate(brent_before_mean, brent_after_mean),

            "usd_krw_before_mean": usd_krw_before_mean,
            "usd_krw_after_mean": usd_krw_after_mean,
            "usd_krw_change_rate": change_rate(usd_krw_before_mean, usd_krw_after_mean),

            "intl_pressure_before_mean": intl_pressure_before_mean,
            "intl_pressure_after_mean": intl_pressure_after_mean,

            "kor_pressure_before_mean": kor_pressure_before_mean,
            "kor_pressure_after_mean": kor_pressure_after_mean,
        }
    )

    window_list.append(event_data)

    plt.figure(figsize=(14, 12))

    plt.subplot(3, 1, 1)
    plt.plot(event_data["date"], event_data["gasoline_price_krw"], label="Gasoline")
    plt.plot(event_data["date"], event_data["diesel_price_krw"], label="Diesel")
    plt.plot(event_data["date"], event_data["lpg_price_krw"], label="LPG")
    plt.axvline(event_date, color="red", linestyle="--", label="Event Date")
    plt.title(event_name + " - Domestic Oil Prices")
    plt.ylabel("KRW")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(event_data["date"], event_data["wti_close"], label="WTI")
    plt.plot(event_data["date"], event_data["brent_close"], label="Brent")
    plt.axvline(event_date, color="red", linestyle="--", label="Event Date")
    plt.title(event_name + " - International Oil Prices")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(event_data["date"], event_data["average_pressure_score"], label="International News Pressure")
    plt.plot(event_data["date"], event_data["kor_average_pressure_score"], label="Korean News Pressure")
    plt.axvline(event_date, color="red", linestyle="--", label="Event Date")
    plt.title(event_name + " - News Pressure")
    plt.xlabel("Date")
    plt.ylabel("Pressure Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(EVENT_FIGURE_DIR + "\\" + event_name + "_event_trend.png", dpi=300)
    plt.close()

summary = pd.DataFrame(summary_list)
event_window = pd.concat(window_list, ignore_index=True)

summary.to_csv(
    PROCESSED_DIR + r"\event_analysis_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

event_window.to_csv(
    PROCESSED_DIR + r"\event_window_dataset_final.csv",
    index=False,
    encoding="utf-8-sig"
)

report_table = summary[
    [
        "event_name",
        "event_type",
        "event_date",
        "gasoline_change_rate",
        "diesel_change_rate",
        "lpg_change_rate",
        "wti_change_rate",
        "brent_change_rate",
        "usd_krw_change_rate",
        "intl_pressure_before_mean",
        "intl_pressure_after_mean",
        "kor_pressure_before_mean",
        "kor_pressure_after_mean",
    ]
]

report_table.to_csv(
    PROCESSED_DIR + r"\event_analysis_report_table.csv",
    index=False,
    encoding="utf-8-sig"
)

x = list(range(len(report_table)))

plt.figure(figsize=(14, 6))
plt.bar([i - 0.25 for i in x], report_table["gasoline_change_rate"], width=0.25, label="Gasoline")
plt.bar(x, report_table["diesel_change_rate"], width=0.25, label="Diesel")
plt.bar([i + 0.25 for i in x], report_table["lpg_change_rate"], width=0.25, label="LPG")
plt.axhline(0, color="black")
plt.xticks(x, report_table["event_name"], rotation=45, ha="right")
plt.ylabel("Change Rate (%)")
plt.title("Domestic Oil Price Change Rate by Event")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\domestic_price_change_by_event.png", dpi=300)
plt.close()

plt.figure(figsize=(14, 6))
plt.bar([i - 0.2 for i in x], report_table["wti_change_rate"], width=0.4, label="WTI")
plt.bar([i + 0.2 for i in x], report_table["brent_change_rate"], width=0.4, label="Brent")
plt.axhline(0, color="black")
plt.xticks(x, report_table["event_name"], rotation=45, ha="right")
plt.ylabel("Change Rate (%)")
plt.title("International Oil Price Change Rate by Event")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\international_price_change_by_event.png", dpi=300)
plt.close()

plt.figure(figsize=(14, 6))
plt.bar(x, report_table["usd_krw_change_rate"], width=0.5, label="USD/KRW")
plt.axhline(0, color="black")
plt.xticks(x, report_table["event_name"], rotation=45, ha="right")
plt.ylabel("Change Rate (%)")
plt.title("USD/KRW Exchange Rate Change Rate by Event")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\exchange_rate_change_by_event.png", dpi=300)
plt.close()

plt.figure(figsize=(14, 6))
plt.bar([i - 0.3 for i in x], report_table["intl_pressure_before_mean"], width=0.2, label="International Before")
plt.bar([i - 0.1 for i in x], report_table["intl_pressure_after_mean"], width=0.2, label="International After")
plt.bar([i + 0.1 for i in x], report_table["kor_pressure_before_mean"], width=0.2, label="Korean Before")
plt.bar([i + 0.3 for i in x], report_table["kor_pressure_after_mean"], width=0.2, label="Korean After")
plt.axhline(0, color="black")
plt.xticks(x, report_table["event_name"], rotation=45, ha="right")
plt.ylabel("News Pressure Score")
plt.title("News Pressure Before and After Events")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\news_pressure_before_after_by_event.png", dpi=300)
plt.close()

corr_columns = [
    "gasoline_price_krw",
    "diesel_price_krw",
    "lpg_price_krw",
    "wti_close",
    "brent_close",
    "usd_krw",
    "average_pressure_score",
    "kor_average_pressure_score",
]

event_corr = event_window[corr_columns].corr()

plt.figure(figsize=(10, 8))
plt.imshow(event_corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr_columns)), corr_columns, rotation=45, ha="right")
plt.yticks(range(len(corr_columns)), corr_columns)

for i in range(len(corr_columns)):
    for j in range(len(corr_columns)):
        plt.text(j, i, round(event_corr.iloc[i, j], 2), ha="center", va="center", fontsize=8)

plt.title("Correlation Heatmap in Event Periods")
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\event_period_correlation_heatmap.png", dpi=300)
plt.close()

event_corr.to_csv(
    PROCESSED_DIR + r"\event_period_correlation.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(8, 6))
plt.scatter(event_window["wti_close"], event_window["gasoline_price_krw"], alpha=0.5)
plt.xlabel("WTI")
plt.ylabel("Gasoline Price KRW")
plt.title("WTI and Domestic Gasoline Price")
plt.grid(True)
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\scatter_wti_gasoline.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(event_window["brent_close"], event_window["diesel_price_krw"], alpha=0.5, color="orange")
plt.xlabel("Brent")
plt.ylabel("Diesel Price KRW")
plt.title("Brent and Domestic Diesel Price")
plt.grid(True)
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\scatter_brent_diesel.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(event_window["usd_krw"], event_window["gasoline_price_krw"], alpha=0.5, color="green")
plt.xlabel("USD/KRW")
plt.ylabel("Gasoline Price KRW")
plt.title("USD/KRW and Domestic Gasoline Price")
plt.grid(True)
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\scatter_usdkrw_gasoline.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(event_window["average_pressure_score"], event_window["gasoline_price_krw"], alpha=0.5, color="red")
plt.xlabel("International News Pressure")
plt.ylabel("Gasoline Price KRW")
plt.title("International News Pressure and Gasoline Price")
plt.grid(True)
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\scatter_intl_pressure_gasoline.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(event_window["kor_average_pressure_score"], event_window["diesel_price_krw"], alpha=0.5, color="purple")
plt.xlabel("Korean News Pressure")
plt.ylabel("Diesel Price KRW")
plt.title("Korean News Pressure and Diesel Price")
plt.grid(True)
plt.tight_layout()
plt.savefig(SUMMARY_FIGURE_DIR + r"\scatter_kor_pressure_diesel.png", dpi=300)
plt.close()

print("5번 이벤트 분석 완료")
print("사건별 요약 CSV:", PROCESSED_DIR + r"\event_analysis_summary.csv")
print("사건 구간 데이터 CSV:", PROCESSED_DIR + r"\event_window_dataset_final.csv")
print("보고서용 요약 CSV:", PROCESSED_DIR + r"\event_analysis_report_table.csv")
print("사건 구간 상관관계 CSV:", PROCESSED_DIR + r"\event_period_correlation.csv")
print("사건별 그래프 폴더:", EVENT_FIGURE_DIR)
print("요약 그래프 폴더:", SUMMARY_FIGURE_DIR)
