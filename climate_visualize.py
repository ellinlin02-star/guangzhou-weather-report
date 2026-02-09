#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气候数据可视化
生成多种图表：月度柱状图、年度趋势、日序列、箱线图、日温差分布等
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 中文字体：优先使用系统自带，避免中文乱码
def setup_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'SimHei', 'sans-serif']
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

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

def fig_monthly_bar(data, outpath):
    """图1：各月平均温度柱状图"""
    daily = data["daily"]
    times = daily["time"]
    tmean = daily["temperature_2m_mean"]
    by_month = defaultdict(list)
    for i, d in enumerate(times):
        if tmean[i] is not None:
            m = datetime.strptime(d, "%Y-%m-%d").month
            by_month[m].append(tmean[i])
    months = list(range(1, 13))
    labels = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
    avgs = [sum(by_month[m]) / len(by_month[m]) if by_month[m] else 0 for m in months]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, avgs, color=plt.cm.RdYlBu_r([(t - 10) / 25 for t in avgs]), edgecolor="gray", linewidth=0.5)
    ax.set_ylabel("平均温度 (°C)", fontsize=12)
    ax.set_xlabel("月份", fontsize=12)
    ax.set_title("各月平均温度（2016-2025）", fontsize=14)
    ax.axhline(y=sum(avgs) / 12, color="gray", linestyle="--", alpha=0.8, label="全年平均")
    ax.legend()
    ax.set_ylim(0, 35)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {outpath}")

def fig_yearly_trend(data, outpath):
    """图2：各年平均温度趋势线"""
    daily = data["daily"]
    times = daily["time"]
    tmean = daily["temperature_2m_mean"]
    by_year = defaultdict(list)
    for i, d in enumerate(times):
        if tmean[i] is not None:
            by_year[d[:4]].append(tmean[i])
    years = sorted(by_year.keys())
    avgs = [sum(by_year[y]) / len(by_year[y]) for y in years]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, avgs, marker="o", linewidth=2, markersize=8, color="#1f4e79")
    ax.fill_between(years, avgs, alpha=0.2, color="#1f4e79")
    ax.axhline(y=sum(avgs) / len(avgs), color="gray", linestyle="--", alpha=0.8, label="十年平均")
    ax.set_ylabel("年平均温度 (°C)", fontsize=12)
    ax.set_xlabel("年份", fontsize=12)
    ax.set_title("年度平均温度变化趋势", fontsize=14)
    ax.legend()
    ax.set_ylim(20, 25)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {outpath}")

def fig_daily_series(data, outpath, year=2024):
    """图3：某年日平均温度曲线（可选滚动平均）"""
    daily = data["daily"]
    times = daily["time"]
    tmean = daily["temperature_2m_mean"]
    dates = []
    temps = []
    for i, d in enumerate(times):
        if d.startswith(str(year)) and tmean[i] is not None:
            dates.append(datetime.strptime(d, "%Y-%m-%d"))
            temps.append(tmean[i])
    if not dates:
        # 若指定年无数据则用最后一年
        year = times[-1][:4] if times else "2024"
        for i, d in enumerate(times):
            if d.startswith(str(year)) and tmean[i] is not None:
                dates.append(datetime.strptime(d, "%Y-%m-%d"))
                temps.append(tmean[i])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, temps, color="#2e86ab", alpha=0.7, linewidth=0.8, label="日平均温度")
    # 7日滚动平均
    if len(temps) >= 7:
        roll = [sum(temps[i:i+7])/7 for i in range(len(temps)-6)]
        roll_dates = dates[3:4] + dates[4:-3] if len(dates) > 7 else dates
        roll_dates = dates[3:-3]
        ax.plot(roll_dates, roll, color="#e94f37", linewidth=2, label="7日滚动平均")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.set_ylabel("温度 (°C)", fontsize=12)
    ax.set_xlabel("日期", fontsize=12)
    ax.set_title(f"{year}年日平均温度", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 38)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {outpath}")

def fig_box_by_month(data, outpath):
    """图4：各月温度分布箱线图"""
    daily = data["daily"]
    times = daily["time"]
    tmean = daily["temperature_2m_mean"]
    by_month = defaultdict(list)
    for i, d in enumerate(times):
        if tmean[i] is not None:
            m = datetime.strptime(d, "%Y-%m-%d").month
            by_month[m].append(tmean[i])
    months = list(range(1, 13))
    labels = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
    box_data = [by_month[m] for m in months]
    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a8dadc")
        patch.set_alpha(0.8)
    ax.set_ylabel("温度 (°C)", fontsize=12)
    ax.set_xlabel("月份", fontsize=12)
    ax.set_title("各月日平均温度分布（箱线图）", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {outpath}")

def fig_daily_range(data, outpath):
    """图5：日温差（最高-最低）分布直方图"""
    daily = data["daily"]
    tmax = daily["temperature_2m_max"]
    tmin = daily["temperature_2m_min"]
    diffs = []
    for i in range(len(tmax)):
        if tmax[i] is not None and tmin[i] is not None:
            diffs.append(tmax[i] - tmin[i])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diffs, bins=25, color="#457b9d", edgecolor="white", alpha=0.85)
    ax.axvline(x=sum(diffs)/len(diffs), color="#e94f37", linestyle="--", linewidth=2, label=f"平均日温差 {sum(diffs)/len(diffs):.1f}°C")
    ax.set_xlabel("日温差 (°C)", fontsize=12)
    ax.set_ylabel("天数", fontsize=12)
    ax.set_title("日温差分布（日最高温 − 日最低温）", fontsize=14)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {outpath}")

def fig_max_min_band(data, outpath, year=2024):
    """图6：某年日最高/最低温度带状图"""
    daily = data["daily"]
    times = daily["time"]
    tmax = daily["temperature_2m_max"]
    tmin = daily["temperature_2m_min"]
    dates = []
    highs = []
    lows = []
    for i, d in enumerate(times):
        if d.startswith(str(year)) and tmax[i] is not None and tmin[i] is not None:
            dates.append(datetime.strptime(d, "%Y-%m-%d"))
            highs.append(tmax[i])
            lows.append(tmin[i])
    if not dates:
        year = times[-1][:4] if times else "2024"
        for i, d in enumerate(times):
            if d.startswith(str(year)) and tmax[i] is not None and tmin[i] is not None:
                dates.append(datetime.strptime(d, "%Y-%m-%d"))
                highs.append(tmax[i])
                lows.append(tmin[i])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(dates, highs, lows, alpha=0.3, color="#1f4e79")
    ax.plot(dates, highs, color="#e94f37", linewidth=1, label="日最高温")
    ax.plot(dates, lows, color="#2e86ab", linewidth=1, label="日最低温")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.set_ylabel("温度 (°C)", fontsize=12)
    ax.set_xlabel("日期", fontsize=12)
    ax.set_title(f"{year}年日最高温与日最低温", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 42)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {outpath}")

def main():
    setup_chinese_font()
    print("正在获取气候数据...")
    data = fetch_weather_data(23.1291, 113.2644, "2016-01-01", "2025-12-31")
    print("数据获取成功，正在生成图表...\n")

    fig_monthly_bar(data, "chart_01_月度平均温度.png")
    fig_yearly_trend(data, "chart_02_年度温度趋势.png")
    fig_daily_series(data, "chart_03_日平均温度曲线.png")
    fig_box_by_month(data, "chart_04_各月温度箱线图.png")
    fig_daily_range(data, "chart_05_日温差分布.png")
    fig_max_min_band(data, "chart_06_日最高最低温.png")

    print("\n全部图表已生成，文件位于当前目录。")

if __name__ == "__main__":
    main()
