#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
广州日气温数据探索与核心指标统计（pandas）
数据源: Open-Meteo Archive API (2016-01-01 至 2025-12-31)
"""

import json
import urllib.request
import urllib.parse
import pandas as pd
from pathlib import Path

API_URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "latitude": 23.1291,
    "longitude": 113.2644,
    "start_date": "2016-01-01",
    "end_date": "2025-12-31",
    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
    "timezone": "Asia/Shanghai",
}


def fetch_data():
    """从 API 获取原始 JSON 数据"""
    query = urllib.parse.urlencode(PARAMS)
    with urllib.request.urlopen(f"{API_URL}?{query}") as resp:
        return json.loads(resp.read().decode())


def to_dataframe(raw):
    """将 API 返回的 daily 结构转为 DataFrame"""
    daily = raw.get("daily", {})
    df = pd.DataFrame({
        "date": daily.get("time", []),
        "temp_max": daily.get("temperature_2m_max", []),
        "temp_min": daily.get("temperature_2m_min", []),
        "temp_mean": daily.get("temperature_2m_mean", []),
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


def explore_and_stats(df):
    """用 pandas 做数据探索 + 质量检查 + 核心指标统计"""
    out = {}

    # ---------- 1. 概览 ----------
    out["1_概览"] = {
        "行数": int(df.shape[0]),
        "列数": int(df.shape[1]),
        "列名": list(df.columns),
        "时间范围": {
            "起始": df["date"].min().strftime("%Y-%m-%d"),
            "结束": df["date"].max().strftime("%Y-%m-%d"),
        },
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "位置": {
            "纬度": None,
            "经度": None,
            "时区": "Asia/Shanghai",
        },
    }

    # ---------- 2. 质量 ----------
    missing = df.isna().sum().to_dict()
    out["2_数据质量"] = {
        "缺失值": {k: int(v) for k, v in missing.items()},
        "重复行数": int(df.duplicated().sum()),
        "结论": "无缺失、无重复" if (df.isna().sum().sum() == 0 and df.duplicated().sum() == 0) else "存在缺失或重复",
    }

    # ---------- 3. 描述统计（全量） ----------
    desc = df[["temp_max", "temp_min", "temp_mean"]].describe().round(2)
    out["3_描述统计_全量"] = {
        col: desc[col].to_dict() for col in desc.columns
    }
    # 转为与标准库输出一致的 key（count/mean/std/min/25%/50%/75%/max）
    for col in out["3_描述统计_全量"]:
        d = out["3_描述统计_全量"][col]
        out["3_描述统计_全量"][col] = {k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in d.items()}

    # 日温差
    df = df.copy()
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    out["3_日温差统计"] = {
        "mean": round(float(df["temp_range"].mean()), 2),
        "std": round(float(df["temp_range"].std()), 2),
        "min": round(float(df["temp_range"].min()), 2),
        "max": round(float(df["temp_range"].max()), 2),
    }

    # ---------- 4. 按年统计 ----------
    df["year"] = df["date"].dt.year
    by_year = df.groupby("year").agg(
        temp_max_mean=("temp_max", "mean"),
        temp_min_mean=("temp_min", "mean"),
        temp_mean_mean=("temp_mean", "mean"),
        temp_max_max=("temp_max", "max"),
        temp_min_min=("temp_min", "min"),
        days=("date", "count"),
    ).round(2)
    out["4_按年统计"] = {
        str(k): v.to_dict() for k, v in by_year.iterrows()
    }
    for y in out["4_按年统计"]:
        out["4_按年统计"][y] = {k: (round(v, 2) if isinstance(v, (int, float)) else int(v)) for k, v in out["4_按年统计"][y].items()}

    # ---------- 5. 按月统计（跨年） ----------
    df["month"] = df["date"].dt.month
    by_month = df.groupby("month").agg(
        temp_mean_avg=("temp_mean", "mean"),
        temp_max_avg=("temp_max", "mean"),
        temp_min_avg=("temp_min", "mean"),
    ).round(2)
    out["5_按月统计_跨年平均"] = {
        str(k): v.to_dict() for k, v in by_month.iterrows()
    }
    for m in out["5_按月统计_跨年平均"]:
        out["5_按月统计_跨年平均"][m] = {k: round(v, 2) for k, v in out["5_按月统计_跨年平均"][m].items()}

    # ---------- 6. 极值记录 ----------
    idx_max = df["temp_max"].idxmax()
    idx_min = df["temp_min"].idxmin()
    out["6_极值记录"] = {
        "最高温日": {
            "date": df.loc[idx_max, "date"].strftime("%Y-%m-%d"),
            "temp_max": round(float(df.loc[idx_max, "temp_max"]), 2),
            "temp_min": round(float(df.loc[idx_max, "temp_min"]), 2),
        },
        "最低温日": {
            "date": df.loc[idx_min, "date"].strftime("%Y-%m-%d"),
            "temp_max": round(float(df.loc[idx_min, "temp_max"]), 2),
            "temp_min": round(float(df.loc[idx_min, "temp_min"]), 2),
        },
    }

    return out, df


def main():
    print("正在获取数据...")
    raw = fetch_data()
    df = to_dataframe(raw)
    print(f"已加载 {len(df)} 行 (pandas DataFrame)")

    # pandas 概览
    print("\n--- df.info() ---")
    df.info()
    print("\n--- df.head() ---")
    print(df.head())
    print("\n--- df.describe() ---")
    print(df[["temp_max", "temp_min", "temp_mean"]].describe().round(2))

    print("\n进行数据探索与统计...")
    stats, df_enhanced = explore_and_stats(df)

    # 补充位置信息
    stats["1_概览"]["位置"] = {
        "纬度": raw.get("latitude"),
        "经度": raw.get("longitude"),
        "时区": raw.get("timezone"),
    }

    # 保存 JSON 结果
    out_path = Path(__file__).parent / "guangzhou_temp_explore_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")

    # 控制台摘要
    print("\n" + "=" * 55)
    print("广州日气温数据探索 — 核心指标摘要 (pandas)")
    print("=" * 55)
    print("\n【1. 概览】", json.dumps(stats["1_概览"], ensure_ascii=False, indent=2))
    print("\n【2. 数据质量】", json.dumps(stats["2_数据质量"], ensure_ascii=False, indent=2))
    print("\n【3. 描述统计（全量）】")
    for k, v in stats["3_描述统计_全量"].items():
        print(f"  {k}: {v}")
    print("\n【日温差】", stats["3_日温差统计"])
    print("\n【4. 按年统计】")
    print(pd.DataFrame(stats["4_按年统计"]).T.to_string())
    print("\n【6. 极值记录】", json.dumps(stats["6_极值记录"], ensure_ascii=False, indent=2))
    print("\n完成。")

    return stats, df_enhanced


if __name__ == "__main__":
    main()
