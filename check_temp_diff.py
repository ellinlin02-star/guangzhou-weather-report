#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查日温差数据"""

import json
import urllib.request
import urllib.parse

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
    full_url = f"{url}?{query_string}"
    
    with urllib.request.urlopen(full_url) as response:
        data = json.loads(response.read().decode())
    return data

# 获取数据
data = fetch_weather_data(23.1291, 113.2644, "2016-01-01", "2025-12-31")
daily = data.get("daily", {})
times = daily.get("time", [])
temp_max = daily.get("temperature_2m_max", [])
temp_min = daily.get("temperature_2m_min", [])

# 计算所有日温差
daily_diffs = []
for i in range(len(times)):
    if temp_max[i] is not None and temp_min[i] is not None:
        diff = temp_max[i] - temp_min[i]
        daily_diffs.append({
            "日期": times[i],
            "最高": temp_max[i],
            "最低": temp_min[i],
            "温差": round(diff, 2)
        })

# 按温差排序
daily_diffs.sort(key=lambda x: x["温差"], reverse=True)

print("日温差最大的前20天：")
print("-" * 60)
for i, day in enumerate(daily_diffs[:20], 1):
    print(f"{i:2d}. {day['日期']}: 最高{day['最高']:5.1f}°C, 最低{day['最低']:5.1f}°C, 温差{day['温差']:5.1f}°C")

print(f"\n总共有 {len(daily_diffs)} 天的完整数据")
print(f"最大日温差: {daily_diffs[0]['温差']}°C ({daily_diffs[0]['日期']})")
print(f"最小日温差: {daily_diffs[-1]['温差']}°C ({daily_diffs[-1]['日期']})")

# 统计超过不同阈值的天数
thresholds = [10, 15, 20, 25, 30]
print("\n日温差统计：")
for threshold in thresholds:
    count = sum(1 for day in daily_diffs if day["温差"] >= threshold)
    print(f"  温差 ≥ {threshold}°C 的天数: {count} 天")
