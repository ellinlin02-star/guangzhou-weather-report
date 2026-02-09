#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气候数据统计
对 Open-Meteo 历史气候数据做完整统计描述
"""

import json
import statistics
import urllib.request
import urllib.parse
from datetime import datetime
from collections import defaultdict

def fetch_weather_data(latitude, longitude, start_date, end_date):
    """获取天气数据"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
        "timezone": "Asia/Shanghai"
    }
    query_string = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{query_string}") as response:
        return json.loads(response.read().decode())

def desc_stats(values):
    """描述统计：样本量、均值、标准差、最小值、25%分位、中位数、75%分位、最大值"""
    v = [x for x in values if x is not None]
    if not v:
        return None
    n = len(v)
    sorted_v = sorted(v)
    q1 = sorted_v[n // 4] if n >= 4 else sorted_v[0]
    q3 = sorted_v[(3 * n) // 4] if n >= 4 else sorted_v[-1]
    return {
        "样本量": n,
        "均值": round(statistics.mean(v), 2),
        "标准差": round(statistics.stdev(v), 2) if n > 1 else 0,
        "最小值": round(min(v), 2),
        "25%分位": round(q1, 2),
        "中位数": round(statistics.median(v), 2),
        "75%分位": round(q3, 2),
        "最大值": round(max(v), 2),
    }

def main():
    data = fetch_weather_data(23.1291, 113.2644, "2016-01-01", "2025-12-31")
    daily = data.get("daily", {})
    times = daily.get("time", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])
    tmean = daily.get("temperature_2m_mean", [])

    # 日温差
    daily_range = []
    for i in range(len(times)):
        if tmax[i] is not None and tmin[i] is not None:
            daily_range.append(tmax[i] - tmin[i])

    # 整体描述统计
    stats_tmax = desc_stats(tmax)
    stats_tmin = desc_stats(tmin)
    stats_tmean = desc_stats(tmean)
    stats_range = desc_stats(daily_range)

    # 按年统计（年平均温度）
    by_year = defaultdict(list)
    for i, d in enumerate(times):
        if tmean[i] is not None:
            by_year[d[:4]].append(tmean[i])
    yearly = {y: round(statistics.mean(v), 2) for y in sorted(by_year) for v in [by_year[y]]}

    # 按月统计（全时期各月）
    by_month = defaultdict(list)
    for i, d in enumerate(times):
        if tmean[i] is not None:
            m = datetime.strptime(d, "%Y-%m-%d").month
            by_month[m].append(tmean[i])
    month_names = "1月,2月,3月,4月,5月,6月,7月,8月,9月,10月,11月,12月".split(",")
    monthly = {month_names[m - 1]: round(statistics.mean(by_month[m]), 2) for m in sorted(by_month)}

    # 极端日
    tmean_clean = [t for t in tmean if t is not None]
    tmax_clean = [t for t in tmax if t is not None]
    tmin_clean = [t for t in tmin if t is not None]
    idx_max = tmean.index(max(tmean_clean)) if tmean_clean else -1
    idx_min = tmean.index(min(tmean_clean)) if tmean_clean else -1
    idx_tmax_day = tmax.index(max(tmax_clean)) if tmax_clean else -1
    idx_tmin_day = tmin.index(min(tmin_clean)) if tmin_clean else -1

    # 输出
    print("=" * 60)
    print("气候数据统计")
    print("=" * 60)
    print("\n【数据信息】")
    print(f"  位置: 纬度 {data.get('latitude')}°, 经度 {data.get('longitude')}° (广州)")
    print(f"  时区: {data.get('timezone')}  海拔: {data.get('elevation')} m")
    print(f"  时间范围: {times[0]} ~ {times[-1]}")
    print(f"  总天数: {len(times)}")

    print("\n【日最高温度 °C】")
    if stats_tmax:
        for k, v in stats_tmax.items():
            print(f"  {k}: {v}")

    print("\n【日最低温度 °C】")
    if stats_tmin:
        for k, v in stats_tmin.items():
            print(f"  {k}: {v}")

    print("\n【日平均温度 °C】")
    if stats_tmean:
        for k, v in stats_tmean.items():
            print(f"  {k}: {v}")

    print("\n【日温差 (最高-最低) °C】")
    if stats_range:
        for k, v in stats_range.items():
            print(f"  {k}: {v}")

    print("\n【极端日】")
    if idx_tmax_day >= 0:
        print(f"  日最高温极值: {tmax[idx_tmax_day]}°C  日期: {times[idx_tmax_day]}")
    if idx_tmin_day >= 0:
        print(f"  日最低温极值: {tmin[idx_tmin_day]}°C  日期: {times[idx_tmin_day]}")
    if idx_max >= 0:
        print(f"  日平均温最高: {tmean[idx_max]}°C  日期: {times[idx_max]}")
    if idx_min >= 0:
        print(f"  日平均温最低: {tmean[idx_min]}°C  日期: {times[idx_min]}")

    print("\n【各月平均温度 °C】")
    for mo, t in monthly.items():
        print(f"  {mo}: {t}")

    print("\n【各年平均温度 °C】")
    for yr, t in yearly.items():
        print(f"  {yr}: {t}")

    # 保存 JSON
    out = {
        "数据信息": {
            "纬度": data.get("latitude"),
            "经度": data.get("longitude"),
            "时区": data.get("timezone"),
            "海拔": data.get("elevation"),
            "时间范围": [times[0], times[-1]],
            "总天数": len(times),
        },
        "日最高温度": stats_tmax,
        "日最低温度": stats_tmin,
        "日平均温度": stats_tmean,
        "日温差": stats_range,
        "各月平均温度": monthly,
        "各年平均温度": yearly,
        "极端日": {
            "日最高温极值_°C_日期": [tmax[idx_tmax_day] if idx_tmax_day >= 0 else None, times[idx_tmax_day] if idx_tmax_day >= 0 else None],
            "日最低温极值_°C_日期": [tmin[idx_tmin_day] if idx_tmin_day >= 0 else None, times[idx_tmin_day] if idx_tmin_day >= 0 else None],
        },
    }
    with open("climate_stats_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n统计结果已保存: climate_stats_result.json")

if __name__ == "__main__":
    main()
