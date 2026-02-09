#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冬季温度分析和暖冬预测
"""

import json
import statistics
import urllib.request
import urllib.parse
from collections import defaultdict
from datetime import datetime

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

def get_winter_data(data, year):
    """获取指定年份的冬季数据（12月、1月、2月）"""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temp_mean = daily.get("temperature_2m_mean", [])
    
    winter_temps = []
    winter_dates = []
    
    for i, date_str in enumerate(times):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        month = date_obj.month
        
        # 冬季定义为12月、1月、2月
        # 对于跨年冬季，12月属于前一年，1-2月属于当年
        if year == "2015-2016":
            # 2015年12月 + 2016年1-2月
            if (date_obj.year == 2015 and month == 12) or (date_obj.year == 2016 and month in [1, 2]):
                if temp_mean[i] is not None:
                    winter_temps.append(temp_mean[i])
                    winter_dates.append(date_str)
        elif year == "2024-2025":
            # 2024年12月 + 2025年1-2月
            if (date_obj.year == 2024 and month == 12) or (date_obj.year == 2025 and month in [1, 2]):
                if temp_mean[i] is not None:
                    winter_temps.append(temp_mean[i])
                    winter_dates.append(date_str)
        elif year == "2025-2026":
            # 2025年12月 + 2026年1-2月（如果数据存在）
            if (date_obj.year == 2025 and month == 12) or (date_obj.year == 2026 and month in [1, 2]):
                if temp_mean[i] is not None:
                    winter_temps.append(temp_mean[i])
                    winter_dates.append(date_str)
        else:
            # 其他年份格式如"2016-2017"：前一年12月 + 当年1-2月
            if "-" in year:
                # 跨年格式，取后一年作为基准年
                year_int = int(year.split("-")[1])
            else:
                year_int = int(year)
            if (date_obj.year == year_int - 1 and month == 12) or (date_obj.year == year_int and month in [1, 2]):
                if temp_mean[i] is not None:
                    winter_temps.append(temp_mean[i])
                    winter_dates.append(date_str)
    
    return winter_temps, winter_dates

def analyze_winter_trends(data):
    """分析冬季温度趋势"""
    # 定义冬季年份（跨年）
    winter_years = [
        "2015-2016", "2016-2017", "2017-2018", "2018-2019", 
        "2019-2020", "2020-2021", "2021-2022", "2022-2023",
        "2023-2024", "2024-2025"
    ]
    
    winter_stats = {}
    
    for winter_year in winter_years:
        temps, dates = get_winter_data(data, winter_year)
        if temps:
            winter_stats[winter_year] = {
                "平均温度": round(statistics.mean(temps), 2),
                "最高温度": round(max(temps), 2),
                "最低温度": round(min(temps), 2),
                "天数": len(temps)
            }
    
    return winter_stats

def forecast_winter_2025_2026(data, winter_stats):
    """预测2025-2026年冬季"""
    # 获取2024-2025年冬季的部分数据（2024年12月 + 2025年1-2月）
    temps_2024_2025, dates_2024_2025 = get_winter_data(data, "2024-2025")
    
    # 获取2025年12月的数据（如果存在）
    temps_2025_2026, dates_2025_2026 = get_winter_data(data, "2025-2026")
    
    # 计算历史冬季平均温度
    historical_avg_temps = [stats["平均温度"] for stats in winter_stats.values()]
    overall_avg = statistics.mean(historical_avg_temps)
    
    # 计算最近几年的趋势
    recent_winters = list(winter_stats.values())[-5:]  # 最近5个冬季
    recent_avg = statistics.mean([w["平均温度"] for w in recent_winters])
    
    # 分析趋势
    all_temps = [stats["平均温度"] for stats in winter_stats.values()]
    if len(all_temps) >= 3:
        # 计算线性趋势（简单方法：比较前一半和后一半的平均值）
        mid_point = len(all_temps) // 2
        early_avg = statistics.mean(all_temps[:mid_point])
        late_avg = statistics.mean(all_temps[mid_point:])
        trend = late_avg - early_avg
    else:
        trend = 0
    
    forecast = {
        "历史冬季平均温度": round(overall_avg, 2),
        "最近5年冬季平均温度": round(recent_avg, 2),
        "温度趋势": round(trend, 2),
        "2024-2025年冬季平均温度": round(statistics.mean(temps_2024_2025), 2) if temps_2024_2025 else None,
        "2025年12月数据": len(temps_2025_2026) > 0,
        "预测": {}
    }
    
    # 基于趋势预测
    if temps_2024_2025:
        # 如果2024-2025年冬季数据完整，可以作为参考
        base_temp = statistics.mean(temps_2024_2025)
    else:
        # 否则使用最近几年的平均值
        base_temp = recent_avg
    
    # 预测2025-2026年冬季平均温度（考虑趋势）
    predicted_temp = base_temp + trend * 0.5  # 趋势影响因子
    
    forecast["预测"]["2025-2026年冬季预测平均温度"] = round(predicted_temp, 2)
    forecast["预测"]["与历史平均比较"] = round(predicted_temp - overall_avg, 2)
    
    # 判断是否为暖冬（高于历史平均0.5°C以上）
    if predicted_temp > overall_avg + 0.5:
        forecast["预测"]["暖冬概率"] = "高"
        forecast["预测"]["结论"] = "预测为暖冬"
    elif predicted_temp > overall_avg:
        forecast["预测"]["暖冬概率"] = "中等"
        forecast["预测"]["结论"] = "可能为暖冬"
    else:
        forecast["预测"]["暖冬概率"] = "低"
        forecast["预测"]["结论"] = "可能为正常或偏冷冬季"
    
    return forecast

def print_analysis(winter_stats, forecast):
    """打印分析结果"""
    print("=" * 70)
    print("冬季温度分析和2025-2026年暖冬预测")
    print("=" * 70)
    
    print("\n【历史冬季温度统计】")
    print("-" * 70)
    print(f"{'冬季':<12} {'平均温度(°C)':<15} {'最高温度(°C)':<15} {'最低温度(°C)':<15} {'天数':<10}")
    print("-" * 70)
    
    for winter_year, stats in winter_stats.items():
        print(f"{winter_year:<12} {stats['平均温度']:<15} {stats['最高温度']:<15} {stats['最低温度']:<15} {stats['天数']:<10}")
    
    print("\n【2025-2026年冬季预测】")
    print("-" * 70)
    print(f"历史冬季平均温度: {forecast['历史冬季平均温度']}°C")
    print(f"最近5年冬季平均温度: {forecast['最近5年冬季平均温度']}°C")
    print(f"温度趋势: {forecast['温度趋势']:+.2f}°C（正数表示升温趋势）")
    
    if forecast['2024-2025年冬季平均温度']:
        print(f"2024-2025年冬季平均温度: {forecast['2024-2025年冬季平均温度']}°C")
    
    print(f"\n预测结果:")
    print(f"  2025-2026年冬季预测平均温度: {forecast['预测']['2025-2026年冬季预测平均温度']}°C")
    print(f"  与历史平均比较: {forecast['预测']['与历史平均比较']:+.2f}°C")
    print(f"  暖冬概率: {forecast['预测']['暖冬概率']}")
    print(f"  结论: {forecast['预测']['结论']}")
    
    print("\n【分析说明】")
    print("-" * 70)
    print("1. 冬季定义为12月、1月、2月（跨年）")
    print("2. 暖冬标准：冬季平均温度高于历史平均0.5°C以上")
    print("3. 预测基于历史趋势和最近几年的数据")
    print("4. 实际天气受多种因素影响，预测仅供参考")

def main():
    """主函数"""
    latitude = 23.1291
    longitude = 113.2644
    start_date = "2015-12-01"  # 从2015年12月开始，以便获取完整冬季数据
    end_date = "2025-12-31"
    
    print("正在获取天气数据...")
    try:
        data = fetch_weather_data(latitude, longitude, start_date, end_date)
        print("数据获取成功！\n")
        
        print("正在分析冬季数据...")
        winter_stats = analyze_winter_trends(data)
        
        print("正在生成预测...")
        forecast = forecast_winter_2025_2026(data, winter_stats)
        
        print_analysis(winter_stats, forecast)
        
        # 保存结果
        result = {
            "历史冬季统计": winter_stats,
            "预测": forecast
        }
        
        with open("winter_forecast_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n分析结果已保存到 winter_forecast_result.json")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
