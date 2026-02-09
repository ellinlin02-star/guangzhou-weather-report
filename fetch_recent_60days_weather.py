#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""获取广州最近 60 天的天气数据（Open-Meteo Archive API）"""

import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd

API_URL = "https://archive-api.open-meteo.com/v1/archive"
BASE_PARAMS = {
    "latitude": 23.1291,
    "longitude": 113.2644,
    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
    "timezone": "Asia/Shanghai",
}


def fetch_last_n_days(n=60):
    end = datetime.now().date()
    start = end - timedelta(days=n - 1)
    params = {**BASE_PARAMS, "start_date": start.isoformat(), "end_date": end.isoformat()}
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{API_URL}?{query}") as resp:
        return json.loads(resp.read().decode())


def to_dataframe(raw):
    daily = raw.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "temp_max": daily.get("temperature_2m_max", []),
        "temp_min": daily.get("temperature_2m_min", []),
        "temp_mean": daily.get("temperature_2m_mean", []),
    })
    df["temp_range"] = (df["temp_max"] - df["temp_min"]).round(1)
    df["temp_max"] = df["temp_max"].round(1)
    df["temp_min"] = df["temp_min"].round(1)
    df["temp_mean"] = df["temp_mean"].round(1)
    return df


def main():
    print("正在获取广州最近 60 天天气数据...")
    raw = fetch_last_n_days(60)
    df = to_dataframe(raw)
    if df.empty:
        print("未获取到数据（可能 API 暂无近期档案）。")
        return
    print(f"共获取 {len(df)} 天数据：{df['date'].min().date()} 至 {df['date'].max().date()}\n")
    print(df.to_string(index=False))
    out_csv = "guangzhou_weather_last_60days.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {out_csv}")
    out_json = "guangzhou_weather_last_60days.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            [{"date": row["date"].strftime("%Y-%m-%d"), "temp_max": row["temp_max"], "temp_min": row["temp_min"], "temp_mean": row["temp_mean"]}
            for _, row in df.iterrows()
        ],
        f,
        ensure_ascii=False,
        indent=2,
    )
    print(f"已保存: {out_json}")


if __name__ == "__main__":
    main()
