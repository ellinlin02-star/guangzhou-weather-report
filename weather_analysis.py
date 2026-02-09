#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气数据分析脚本
分析Open-Meteo API返回的历史天气数据
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
    full_url = f"{url}?{query_string}"
    
    with urllib.request.urlopen(full_url) as response:
        data = json.loads(response.read().decode())
    return data

def analyze_weather_data(data):
    """分析天气数据"""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    temp_mean = daily.get("temperature_2m_mean", [])
    
    # 过滤掉None值
    temp_max_clean = [t for t in temp_max if t is not None]
    temp_min_clean = [t for t in temp_min if t is not None]
    temp_mean_clean = [t for t in temp_mean if t is not None]
    
    # 基本统计信息
    analysis = {
        "位置信息": {
            "纬度": data.get("latitude"),
            "经度": data.get("longitude"),
            "时区": data.get("timezone"),
            "海拔": f"{data.get('elevation')}米"
        },
        "数据概览": {
            "总天数": len(times),
            "起始日期": times[0] if times else "N/A",
            "结束日期": times[-1] if times else "N/A"
        },
        "最高温度统计": {
            "最大值": max(temp_max_clean) if temp_max_clean else None,
            "最小值": min(temp_max_clean) if temp_max_clean else None,
            "平均值": round(statistics.mean(temp_max_clean), 2) if temp_max_clean else None,
            "中位数": round(statistics.median(temp_max_clean), 2) if temp_max_clean else None
        },
        "最低温度统计": {
            "最大值": max(temp_min_clean) if temp_min_clean else None,
            "最小值": min(temp_min_clean) if temp_min_clean else None,
            "平均值": round(statistics.mean(temp_min_clean), 2) if temp_min_clean else None,
            "中位数": round(statistics.median(temp_min_clean), 2) if temp_min_clean else None
        },
        "平均温度统计": {
            "最大值": max(temp_mean_clean) if temp_mean_clean else None,
            "最小值": min(temp_mean_clean) if temp_mean_clean else None,
            "平均值": round(statistics.mean(temp_mean_clean), 2) if temp_mean_clean else None,
            "中位数": round(statistics.median(temp_mean_clean), 2) if temp_mean_clean else None
        }
    }
    
    # 按年份统计
    yearly_stats = defaultdict(lambda: {"max": [], "min": [], "mean": []})
    for i, date_str in enumerate(times):
        if temp_max[i] is not None:
            year = date_str[:4]
            yearly_stats[year]["max"].append(temp_max[i])
        if temp_min[i] is not None:
            year = date_str[:4]
            yearly_stats[year]["min"].append(temp_min[i])
        if temp_mean[i] is not None:
            year = date_str[:4]
            yearly_stats[year]["mean"].append(temp_mean[i])
    
    yearly_analysis = {}
    for year in sorted(yearly_stats.keys()):
        stats = yearly_stats[year]
        yearly_analysis[year] = {
            "最高温平均": round(statistics.mean(stats["max"]), 2) if stats["max"] else None,
            "最低温平均": round(statistics.mean(stats["min"]), 2) if stats["min"] else None,
            "平均温度": round(statistics.mean(stats["mean"]), 2) if stats["mean"] else None
        }
    
    analysis["年度统计"] = yearly_analysis
    
    # 找出极端温度日期
    if temp_max_clean:
        max_temp = max(temp_max_clean)
        max_temp_idx = temp_max.index(max_temp)
        analysis["极端高温"] = {
            "温度": max_temp,
            "日期": times[max_temp_idx] if max_temp_idx < len(times) else "N/A"
        }
    
    if temp_min_clean:
        min_temp = min(temp_min_clean)
        min_temp_idx = temp_min.index(min_temp)
        analysis["极端低温"] = {
            "温度": min_temp,
            "日期": times[min_temp_idx] if min_temp_idx < len(times) else "N/A"
        }
    
    # 计算日温差（最高温度 - 最低温度）
    daily_temp_diff = []
    for i in range(len(times)):
        if temp_max[i] is not None and temp_min[i] is not None:
            diff = temp_max[i] - temp_min[i]
            daily_temp_diff.append({
                "日期": times[i],
                "最高温度": temp_max[i],
                "最低温度": temp_min[i],
                "日温差": round(diff, 2)
            })
    
    if daily_temp_diff:
        # 找出温差最大的那一天
        max_diff_day = max(daily_temp_diff, key=lambda x: x["日温差"])
        analysis["最大日温差"] = {
            "日期": max_diff_day["日期"],
            "最高温度": max_diff_day["最高温度"],
            "最低温度": max_diff_day["最低温度"],
            "日温差": max_diff_day["日温差"]
        }
        
        # 找出温差超过30°C的所有日期
        large_diff_days = [day for day in daily_temp_diff if day["日温差"] >= 30.0]
        if large_diff_days:
            # 按温差从大到小排序
            large_diff_days.sort(key=lambda x: x["日温差"], reverse=True)
            analysis["日温差超过30°C的日期"] = large_diff_days[:10]  # 只保存前10个
    
    return analysis

def print_analysis(analysis):
    """打印分析结果"""
    print("=" * 60)
    print("天气数据分析报告")
    print("=" * 60)
    
    print("\n【位置信息】")
    for key, value in analysis["位置信息"].items():
        print(f"  {key}: {value}")
    
    print("\n【数据概览】")
    for key, value in analysis["数据概览"].items():
        print(f"  {key}: {value}")
    
    print("\n【最高温度统计 (°C)】")
    for key, value in analysis["最高温度统计"].items():
        print(f"  {key}: {value}°C" if value is not None else f"  {key}: N/A")
    
    print("\n【最低温度统计 (°C)】")
    for key, value in analysis["最低温度统计"].items():
        print(f"  {key}: {value}°C" if value is not None else f"  {key}: N/A")
    
    print("\n【平均温度统计 (°C)】")
    for key, value in analysis["平均温度统计"].items():
        print(f"  {key}: {value}°C" if value is not None else f"  {key}: N/A")
    
    if "极端高温" in analysis:
        print("\n【极端高温】")
        print(f"  温度: {analysis['极端高温']['温度']}°C")
        print(f"  日期: {analysis['极端高温']['日期']}")
    
    if "极端低温" in analysis:
        print("\n【极端低温】")
        print(f"  温度: {analysis['极端低温']['温度']}°C")
        print(f"  日期: {analysis['极端低温']['日期']}")
    
    if "最大日温差" in analysis:
        print("\n【最大日温差】")
        diff_info = analysis["最大日温差"]
        print(f"  日期: {diff_info['日期']}")
        print(f"  最高温度: {diff_info['最高温度']}°C")
        print(f"  最低温度: {diff_info['最低温度']}°C")
        print(f"  日温差: {diff_info['日温差']}°C")
    
    if "日温差超过30°C的日期" in analysis:
        print("\n【日温差超过30°C的日期（前10个）】")
        for i, day in enumerate(analysis["日温差超过30°C的日期"], 1):
            print(f"  {i}. {day['日期']}: 最高{day['最高温度']}°C, 最低{day['最低温度']}°C, 温差{day['日温差']}°C")
    
    print("\n【年度统计 (°C)】")
    for year, stats in analysis["年度统计"].items():
        print(f"\n  {year}年:")
        print(f"    最高温平均: {stats['最高温平均']}°C" if stats['最高温平均'] else "    最高温平均: N/A")
        print(f"    最低温平均: {stats['最低温平均']}°C" if stats['最低温平均'] else "    最低温平均: N/A")
        print(f"    平均温度: {stats['平均温度']}°C" if stats['平均温度'] else "    平均温度: N/A")

def main():
    """主函数"""
    latitude = 23.1291
    longitude = 113.2644
    start_date = "2016-01-01"
    end_date = "2025-12-31"
    
    print("正在获取天气数据...")
    try:
        data = fetch_weather_data(latitude, longitude, start_date, end_date)
        print("数据获取成功！\n")
        
        print("正在分析数据...")
        analysis = analyze_weather_data(data)
        
        print_analysis(analysis)
        
        # 保存分析结果到JSON文件
        with open("weather_analysis_result.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print("\n分析结果已保存到 weather_analysis_result.json")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
