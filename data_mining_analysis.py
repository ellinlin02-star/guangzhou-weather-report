#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气数据价值挖掘分析
从多个维度分析天气数据的潜在价值
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

def analyze_seasonal_patterns(data):
    """分析季节性模式"""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temp_mean = daily.get("temperature_2m_mean", [])
    
    monthly_stats = defaultdict(lambda: {"temps": [], "count": 0})
    
    for i, date_str in enumerate(times):
        if temp_mean[i] is not None:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            month = date_obj.month
            monthly_stats[month]["temps"].append(temp_mean[i])
            monthly_stats[month]["count"] += 1
    
    seasonal_analysis = {}
    season_names = {12: "12月(冬)", 1: "1月(冬)", 2: "2月(冬)", 
                    3: "3月(春)", 4: "4月(春)", 5: "5月(春)",
                    6: "6月(夏)", 7: "7月(夏)", 8: "8月(夏)",
                    9: "9月(秋)", 10: "10月(秋)", 11: "11月(秋)"}
    
    for month in sorted(monthly_stats.keys()):
        temps = monthly_stats[month]["temps"]
        if temps:
            seasonal_analysis[season_names[month]] = {
                "平均温度": round(statistics.mean(temps), 2),
                "最高温度": round(max(temps), 2),
                "最低温度": round(min(temps), 2),
                "标准差": round(statistics.stdev(temps), 2) if len(temps) > 1 else 0,
                "数据天数": len(temps)
            }
    
    return seasonal_analysis

def analyze_temperature_trends(data):
    """分析温度变化趋势"""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temp_mean = daily.get("temperature_2m_mean", [])
    
    # 按年份分析
    yearly_trends = defaultdict(lambda: {"temps": [], "dates": []})
    
    for i, date_str in enumerate(times):
        if temp_mean[i] is not None:
            year = date_str[:4]
            yearly_trends[year]["temps"].append(temp_mean[i])
            yearly_trends[year]["dates"].append(date_str)
    
    trend_analysis = {}
    years = sorted(yearly_trends.keys())
    
    for year in years:
        temps = yearly_trends[year]["temps"]
        if temps:
            trend_analysis[year] = {
                "年平均温度": round(statistics.mean(temps), 2),
                "最高温度": round(max(temps), 2),
                "最低温度": round(min(temps), 2),
                "温度标准差": round(statistics.stdev(temps), 2) if len(temps) > 1 else 0,
                "数据天数": len(temps)
            }
    
    # 计算整体趋势
    if len(years) >= 2:
        early_years = years[:len(years)//2]
        late_years = years[len(years)//2:]
        
        early_avg = statistics.mean([trend_analysis[y]["年平均温度"] for y in early_years])
        late_avg = statistics.mean([trend_analysis[y]["年平均温度"] for y in late_years])
        
        trend_analysis["趋势分析"] = {
            "前半期平均": round(early_avg, 2),
            "后半期平均": round(late_avg, 2),
            "温度变化": round(late_avg - early_avg, 2),
            "变化率": round((late_avg - early_avg) / early_avg * 100, 2)
        }
    
    return trend_analysis

def analyze_extreme_events(data):
    """分析极端天气事件"""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    temp_mean = daily.get("temperature_2m_mean", [])
    
    # 计算阈值（使用平均值±2倍标准差）
    temp_mean_clean = [t for t in temp_mean if t is not None]
    if len(temp_mean_clean) > 1:
        mean_avg = statistics.mean(temp_mean_clean)
        mean_std = statistics.stdev(temp_mean_clean)
        hot_threshold = mean_avg + 2 * mean_std
        cold_threshold = mean_avg - 2 * mean_std
    else:
        hot_threshold = 30
        cold_threshold = 15
    
    extreme_events = {
        "极端高温日": [],
        "极端低温日": [],
        "大温差日": []
    }
    
    for i, date_str in enumerate(times):
        if temp_mean[i] is not None:
            # 极端高温
            if temp_mean[i] >= hot_threshold:
                extreme_events["极端高温日"].append({
                    "日期": date_str,
                    "平均温度": round(temp_mean[i], 2),
                    "最高温度": round(temp_max[i], 2) if temp_max[i] is not None else None
                })
            
            # 极端低温
            if temp_mean[i] <= cold_threshold:
                extreme_events["极端低温日"].append({
                    "日期": date_str,
                    "平均温度": round(temp_mean[i], 2),
                    "最低温度": round(temp_min[i], 2) if temp_min[i] is not None else None
                })
        
        # 大温差日（日温差>15°C）
        if temp_max[i] is not None and temp_min[i] is not None:
            diff = temp_max[i] - temp_min[i]
            if diff >= 15:
                extreme_events["大温差日"].append({
                    "日期": date_str,
                    "最高温度": round(temp_max[i], 2),
                    "最低温度": round(temp_min[i], 2),
                    "日温差": round(diff, 2)
                })
    
    # 只保留前10个
    extreme_events["极端高温日"] = sorted(extreme_events["极端高温日"], 
                                         key=lambda x: x["平均温度"], reverse=True)[:10]
    extreme_events["极端低温日"] = sorted(extreme_events["极端低温日"], 
                                         key=lambda x: x["平均温度"])[:10]
    extreme_events["大温差日"] = sorted(extreme_events["大温差日"], 
                                       key=lambda x: x["日温差"], reverse=True)[:10]
    
    return extreme_events

def analyze_quarterly_comparison(data):
    """分析季度对比"""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temp_mean = daily.get("temperature_2m_mean", [])
    
    quarterly_stats = defaultdict(lambda: {"temps": []})
    
    for i, date_str in enumerate(times):
        if temp_mean[i] is not None:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            month = date_obj.month
            year = date_obj.year
            
            # 确定季度
            if month in [12, 1, 2]:
                quarter = f"{year-1 if month==12 else year}-Q1(冬)"
            elif month in [3, 4, 5]:
                quarter = f"{year}-Q2(春)"
            elif month in [6, 7, 8]:
                quarter = f"{year}-Q3(夏)"
            else:
                quarter = f"{year}-Q4(秋)"
            
            quarterly_stats[quarter]["temps"].append(temp_mean[i])
    
    quarterly_analysis = {}
    for quarter in sorted(quarterly_stats.keys()):
        temps = quarterly_stats[quarter]["temps"]
        if temps:
            quarterly_analysis[quarter] = {
                "平均温度": round(statistics.mean(temps), 2),
                "最高温度": round(max(temps), 2),
                "最低温度": round(min(temps), 2),
                "天数": len(temps)
            }
    
    return quarterly_analysis

def analyze_heating_cooling_days(data):
    """分析供暖和制冷需求日数"""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temp_mean = daily.get("temperature_2m_mean", [])
    
    # 定义阈值（广州地区）
    heating_threshold = 10  # 低于10°C可能需要供暖
    cooling_threshold = 26  # 高于26°C可能需要制冷
    
    heating_days = defaultdict(int)
    cooling_days = defaultdict(int)
    
    for i, date_str in enumerate(times):
        if temp_mean[i] is not None:
            year = date_str[:4]
            
            if temp_mean[i] < heating_threshold:
                heating_days[year] += 1
            elif temp_mean[i] > cooling_threshold:
                cooling_days[year] += 1
    
    energy_analysis = {}
    for year in sorted(set(list(heating_days.keys()) + list(cooling_days.keys()))):
        energy_analysis[year] = {
            "供暖需求日数": heating_days.get(year, 0),
            "制冷需求日数": cooling_days.get(year, 0),
            "舒适日数": 365 - heating_days.get(year, 0) - cooling_days.get(year, 0)
        }
    
    return energy_analysis

def generate_insights(data, seasonal, trends, extremes, quarterly, energy):
    """生成数据价值洞察"""
    insights = {
        "数据概览": {
            "数据时间跨度": f"{data.get('daily', {}).get('time', [])[0] if data.get('daily', {}).get('time') else 'N/A'} 至 {data.get('daily', {}).get('time', [])[-1] if data.get('daily', {}).get('time') else 'N/A'}",
            "总数据天数": len(data.get('daily', {}).get('time', [])),
            "数据完整性": "完整"
        },
        "核心发现": [],
        "应用价值": [],
        "商业价值": [],
        "研究价值": []
    }
    
    # 核心发现
    if "趋势分析" in trends:
        trend_info = trends["趋势分析"]
        if trend_info["温度变化"] > 0:
            insights["核心发现"].append(
                f"温度呈上升趋势：近10年平均温度上升{trend_info['温度变化']}°C，变化率{trend_info['变化率']}%"
            )
        else:
            insights["核心发现"].append(
                f"温度呈下降趋势：近10年平均温度下降{abs(trend_info['温度变化'])}°C"
            )
    
    # 找出最热和最冷的月份
    if seasonal:
        hottest_month = max(seasonal.items(), key=lambda x: x[1]["平均温度"])
        coldest_month = min(seasonal.items(), key=lambda x: x[1]["平均温度"])
        insights["核心发现"].append(
            f"最热月份：{hottest_month[0]}，平均温度{hottest_month[1]['平均温度']}°C"
        )
        insights["核心发现"].append(
            f"最冷月份：{coldest_month[0]}，平均温度{coldest_month[1]['平均温度']}°C"
        )
    
    # 极端事件统计
    if extremes:
        insights["核心发现"].append(
            f"极端高温日：{len(extremes.get('极端高温日', []))}天（超过阈值）"
        )
        insights["核心发现"].append(
            f"极端低温日：{len(extremes.get('极端低温日', []))}天（低于阈值）"
        )
        insights["核心发现"].append(
            f"大温差日：{len(extremes.get('大温差日', []))}天（日温差≥15°C）"
        )
    
    # 应用价值
    insights["应用价值"] = [
        "1. 农业规划：了解季节性温度模式，优化作物种植时间",
        "2. 能源管理：预测供暖和制冷需求，优化能源消耗",
        "3. 旅游规划：识别最佳旅游季节和极端天气时段",
        "4. 健康预警：识别极端高温/低温时段，发布健康预警",
        "5. 建筑设计：为建筑保温、通风设计提供数据支持",
        "6. 服装零售：预测季节性需求，优化库存管理"
    ]
    
    # 商业价值
    insights["商业价值"] = [
        "1. 电力需求预测：基于温度数据预测用电高峰，优化电力调度",
        "2. 零售业：根据温度趋势预测季节性商品需求（如空调、取暖设备）",
        "3. 物流运输：识别极端天气时段，优化运输路线和时间",
        "4. 保险业：评估极端天气风险，制定保险费率",
        "5. 房地产：分析气候舒适度，影响房价和租金",
        "6. 农业保险：基于历史极端事件数据，设计保险产品"
    ]
    
    # 研究价值
    insights["研究价值"] = [
        "1. 气候变化研究：分析长期温度趋势，研究全球变暖影响",
        "2. 气象模式分析：识别季节性模式和异常天气事件",
        "3. 城市热岛效应：对比不同区域数据，研究城市热岛现象",
        "4. 极端天气频率：分析极端事件发生频率和强度变化",
        "5. 机器学习训练：为天气预测模型提供训练数据",
        "6. 环境政策制定：为气候适应政策提供数据支撑"
    ]
    
    return insights

def print_comprehensive_analysis(seasonal, trends, extremes, quarterly, energy, insights):
    """打印综合分析报告"""
    print("=" * 80)
    print("天气数据价值挖掘分析报告")
    print("=" * 80)
    
    print("\n【一、季节性模式分析】")
    print("-" * 80)
    print(f"{'月份':<15} {'平均温度(°C)':<15} {'最高温度(°C)':<15} {'最低温度(°C)':<15} {'标准差':<10}")
    print("-" * 80)
    for month, stats in seasonal.items():
        print(f"{month:<15} {stats['平均温度']:<15} {stats['最高温度']:<15} {stats['最低温度']:<15} {stats['标准差']:<10}")
    
    print("\n【二、年度温度趋势分析】")
    print("-" * 80)
    print(f"{'年份':<10} {'年平均温度(°C)':<18} {'最高温度(°C)':<15} {'最低温度(°C)':<15} {'标准差':<10}")
    print("-" * 80)
    for year, stats in trends.items():
        if year != "趋势分析":
            print(f"{year:<10} {stats['年平均温度']:<18} {stats['最高温度']:<15} {stats['最低温度']:<15} {stats['温度标准差']:<10}")
    
    if "趋势分析" in trends:
        trend = trends["趋势分析"]
        print(f"\n趋势分析：")
        print(f"  前半期平均温度: {trend['前半期平均']}°C")
        print(f"  后半期平均温度: {trend['后半期平均']}°C")
        print(f"  温度变化: {trend['温度变化']:+.2f}°C")
        print(f"  变化率: {trend['变化率']:+.2f}%")
    
    print("\n【三、极端天气事件分析】")
    print("-" * 80)
    print("极端高温日（前5个）：")
    for i, event in enumerate(extremes.get("极端高温日", [])[:5], 1):
        print(f"  {i}. {event['日期']}: 平均温度{event['平均温度']}°C, 最高温度{event['最高温度']}°C")
    
    print("\n极端低温日（前5个）：")
    for i, event in enumerate(extremes.get("极端低温日", [])[:5], 1):
        print(f"  {i}. {event['日期']}: 平均温度{event['平均温度']}°C, 最低温度{event['最低温度']}°C")
    
    print("\n大温差日（前5个）：")
    for i, event in enumerate(extremes.get("大温差日", [])[:5], 1):
        print(f"  {i}. {event['日期']}: 最高{event['最高温度']}°C, 最低{event['最低温度']}°C, 温差{event['日温差']}°C")
    
    print("\n【四、能源需求分析（供暖/制冷日数）】")
    print("-" * 80)
    print(f"{'年份':<10} {'供暖需求日数':<15} {'制冷需求日数':<15} {'舒适日数':<15}")
    print("-" * 80)
    for year, stats in list(energy.items())[-5:]:  # 只显示最近5年
        print(f"{year:<10} {stats['供暖需求日数']:<15} {stats['制冷需求日数']:<15} {stats['舒适日数']:<15}")
    
    print("\n【五、数据价值洞察】")
    print("-" * 80)
    print("核心发现：")
    for finding in insights["核心发现"]:
        print(f"  • {finding}")
    
    print("\n应用价值：")
    for value in insights["应用价值"]:
        print(f"  {value}")
    
    print("\n商业价值：")
    for value in insights["商业价值"]:
        print(f"  {value}")
    
    print("\n研究价值：")
    for value in insights["研究价值"]:
        print(f"  {value}")

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
        
        print("正在进行多维度分析...")
        seasonal = analyze_seasonal_patterns(data)
        trends = analyze_temperature_trends(data)
        extremes = analyze_extreme_events(data)
        quarterly = analyze_quarterly_comparison(data)
        energy = analyze_heating_cooling_days(data)
        insights = generate_insights(data, seasonal, trends, extremes, quarterly, energy)
        
        print_comprehensive_analysis(seasonal, trends, extremes, quarterly, energy, insights)
        
        # 保存结果
        result = {
            "季节性分析": seasonal,
            "年度趋势": trends,
            "极端事件": extremes,
            "季度对比": quarterly,
            "能源需求": energy,
            "价值洞察": insights
        }
        
        with open("data_mining_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n\n完整分析结果已保存到 data_mining_result.json")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
