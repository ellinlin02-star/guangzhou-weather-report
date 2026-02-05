#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
广州天气数据报告：2025年分析、近10年分析、异常天气标注、2026年预测
"""

import base64
import io
import json
import urllib.request
import urllib.parse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import math

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    HAS_MATPLOTLIB = True
    # 查找系统中支持中文的字体：优先用字体文件路径（避免名称解析到回退字体）
    _CHINESE_FONT = None
    for _path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
    ]:
        if Path(_path).exists():
            _CHINESE_FONT = fm.FontProperties(fname=_path)
            break
    if _CHINESE_FONT is None:
        for _name in ("PingFang SC", "Heiti SC", "STHeiti", "Microsoft YaHei", "SimHei", "Arial Unicode MS"):
            if _CHINESE_FONT is None:
                try:
                    _fp = fm.FontProperties(family=_name)
                    _resolved = fm.findfont(_fp)
                    if _resolved and "DejaVu" not in _resolved:
                        _CHINESE_FONT = _fp
                        break
                except Exception:
                    pass
except ImportError:
    HAS_MATPLOTLIB = False
    fm = None
    _CHINESE_FONT = None

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
    query = urllib.parse.urlencode(PARAMS)
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
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    return df


def analyze_2025(df):
    """2025 年天气分析"""
    d = df[df["year"] == 2025]
    if d.empty:
        return {}
    desc = d[["temp_max", "temp_min", "temp_mean"]].describe().round(2)
    idx_max = d["temp_max"].idxmax()
    idx_min = d["temp_min"].idxmin()
    by_month = d.groupby("month").agg(
        temp_max_mean=("temp_max", "mean"),
        temp_min_mean=("temp_min", "mean"),
        temp_mean_mean=("temp_mean", "mean"),
        temp_max_max=("temp_max", "max"),
        temp_min_min=("temp_min", "min"),
    ).round(2)
    d_sorted = d.sort_values("date")
    daily = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "temp_max": round(float(row["temp_max"]), 1),
            "temp_min": round(float(row["temp_min"]), 1),
            "temp_mean": round(float(row["temp_mean"]), 1),
        }
        for _, row in d_sorted.iterrows()
    ]
    return {
        "days": len(d),
        "desc": {col: desc[col].to_dict() for col in desc.columns},
        "highest_day": {
            "date": d.loc[idx_max, "date"].strftime("%Y-%m-%d"),
            "temp_max": round(float(d.loc[idx_max, "temp_max"]), 1),
            "temp_min": round(float(d.loc[idx_max, "temp_min"]), 1),
        },
        "lowest_day": {
            "date": d.loc[idx_min, "date"].strftime("%Y-%m-%d"),
            "temp_max": round(float(d.loc[idx_min, "temp_max"]), 1),
            "temp_min": round(float(d.loc[idx_min, "temp_min"]), 1),
        },
        "by_month": {str(k): v.to_dict() for k, v in by_month.iterrows()},
        "daily": daily,
    }


def analyze_10years(df):
    """近 10 年天气分析"""
    by_year = df.groupby("year").agg(
        temp_max_mean=("temp_max", "mean"),
        temp_min_mean=("temp_min", "mean"),
        temp_mean_mean=("temp_mean", "mean"),
        temp_max_max=("temp_max", "max"),
        temp_min_min=("temp_min", "min"),
        days=("date", "count"),
    ).round(2)
    # 简单线性趋势（年均温）
    years = by_year.index.astype(int).values
    means = by_year["temp_mean_mean"].values
    if len(years) >= 2:
        slope = (means[-1] - means[0]) / (years[-1] - years[0])
        trend_per_decade = round(slope * 10, 2)
    else:
        trend_per_decade = 0
    return {
        "by_year": {str(k): v.to_dict() for k, v in by_year.iterrows()},
        "trend_per_decade_c": trend_per_decade,
        "overall": {
            "temp_max_mean": round(float(df["temp_max"].mean()), 2),
            "temp_min_mean": round(float(df["temp_min"].mean()), 2),
            "temp_mean_mean": round(float(df["temp_mean"].mean()), 2),
        },
    }


def get_abnormal_days(df, high_pct=95, low_pct=5):
    """异常天气：日最高温 ≥ 95 分位 或 日最低温 ≤ 5 分位（按全样本）"""
    thresh_high = df["temp_max"].quantile(high_pct / 100)
    thresh_low = df["temp_min"].quantile(low_pct / 100)
    high_days = df[df["temp_max"] >= thresh_high][["date", "temp_max", "temp_min", "year"]].copy()
    low_days = df[df["temp_min"] <= thresh_low][["date", "temp_max", "temp_min", "year"]].copy()
    high_by_year = high_days.groupby("year").size().to_dict()
    low_by_year = low_days.groupby("year").size().to_dict()
    years = sorted(set(high_by_year) | set(low_by_year))
    by_year = {str(y): {"high": int(high_by_year.get(y, 0)), "low": int(low_by_year.get(y, 0))} for y in years}
    return {
        "threshold_high_c": round(float(thresh_high), 1),
        "threshold_low_c": round(float(thresh_low), 1),
        "high_days": [
            {"date": r["date"].strftime("%Y-%m-%d"), "temp_max": round(r["temp_max"], 1), "temp_min": round(r["temp_min"], 1)}
            for _, r in high_days.iterrows()
        ],
        "low_days": [
            {"date": r["date"].strftime("%Y-%m-%d"), "temp_max": round(r["temp_max"], 1), "temp_min": round(r["temp_min"], 1)}
            for _, r in low_days.iterrows()
        ],
        "by_year": by_year,
    }


def predict_2026(df):
    """2026 年预测：使用 2016-2025 各月均值作为 2026 年月度预测（含轻微升温趋势）"""
    by_month = df.groupby("month").agg(
        temp_max_avg=("temp_max", "mean"),
        temp_min_avg=("temp_min", "mean"),
        temp_mean_avg=("temp_mean", "mean"),
    ).round(2)
    # 可选：加 0.1°C/年 的升温
    trend = 0.1
    pred = {}
    month_names = "1月,2月,3月,4月,5月,6月,7月,8月,9月,10月,11月,12月".split(",")
    for m in range(1, 13):
        row = by_month.loc[m]
        pred[str(m)] = {
            "month_name": month_names[m - 1],
            "temp_mean": round(float(row["temp_mean_avg"]) + trend, 1),
            "temp_max": round(float(row["temp_max_avg"]) + trend, 1),
            "temp_min": round(float(row["temp_min_avg"]) + trend, 1),
        }
    return {
        "method": "基于 2016-2025 年月均值，并叠加约 +0.1°C 年际升温趋势",
        "by_month": pred,
    }


def _predict_2026_daily(pred):
    """由 2026 年月度预测插值得到逐日预测（月内正弦波动，月均与预测一致）"""
    import calendar
    daily = []
    start = datetime(2026, 1, 1)
    for d in range(365):
        dt = start + timedelta(days=d)
        m = dt.month
        days_in_month = calendar.monthrange(2026, m)[1]
        day_i = dt.day - 1
        # 月内正弦波动，月均不变
        phase = 2 * math.pi * day_i / days_in_month
        amp = 1.5
        pm = pred["by_month"][str(m)]
        t_mean = round(pm["temp_mean"] + amp * math.sin(phase), 1)
        t_max = round(pm["temp_max"] + 1.2 * math.sin(phase), 1)
        t_min = round(pm["temp_min"] + 1.2 * math.sin(phase), 1)
        daily.append({
            "date": dt.strftime("%Y-%m-%d"),
            "temp_max": t_max,
            "temp_min": t_min,
            "temp_mean": t_mean,
        })
    return daily


def _outlook_2026(df, pred):
    """基于 10 年月度规律与 2026 预测，给出冬夏季定性展望（仅供参考）"""
    winter_months = [12, 1, 2]
    summer_months = [6, 7, 8]
    df_winter = df[df["month"].isin(winter_months)]
    df_summer = df[df["month"].isin(summer_months)]
    hist_winter_mean = float(df_winter["temp_mean"].mean())
    hist_summer_mean = float(df_summer["temp_mean"].mean())
    hist_summer_max_p95 = float(df_summer["temp_max"].quantile(0.95))
    pred_winter_mean = (pred["by_month"]["12"]["temp_mean"] + pred["by_month"]["1"]["temp_mean"] + pred["by_month"]["2"]["temp_mean"]) / 3
    pred_summer_mean = (pred["by_month"]["6"]["temp_mean"] + pred["by_month"]["7"]["temp_mean"] + pred["by_month"]["8"]["temp_mean"]) / 3
    pred_summer_max = max(pred["by_month"]["6"]["temp_max"], pred["by_month"]["7"]["temp_max"], pred["by_month"]["8"]["temp_max"])
    # 冬季
    if pred_winter_mean >= hist_winter_mean + 0.5:
        winter_text = "偏暖冬"
    elif pred_winter_mean <= hist_winter_mean - 0.5:
        winter_text = "偏严寒"
    else:
        winter_text = "接近常年，偏暖或偏冷幅度不大"
    # 夏季
    if pred_summer_max >= 34.5 or pred_summer_max >= hist_summer_max_p95 + 0.5:
        summer_text = "出现极端高温的概率较常年偏高，需关注防暑与用电"
    elif pred_summer_max >= 33.5:
        summer_text = "盛夏可能出现阶段性高温，建议关注气象预警"
    else:
        summer_text = "极端高温风险接近常年水平"
    return {
        "winter": winter_text,
        "summer": summer_text,
        "note": "以上基于 2016-2025 年月度统计与简单趋势，仅供参考，不替代气象部门预报。",
    }


def build_report_data(df):
    pred = predict_2026(df)
    return {
        "meta": {
            "title": "广州天气数据报告",
            "period": "2016-01-01 至 2025-12-31",
            "location": "广州（约 23.13°N, 113.26°E）",
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "year_2025": analyze_2025(df),
        "years_10": analyze_10years(df),
        "abnormal": get_abnormal_days(df),
        "predict_2026": pred,
        "outlook_2026": _outlook_2026(df, pred),
    }


def _render_charts_base64(report):
    """用 matplotlib 生成四张图，返回 base64 编码的 PNG（不依赖 CDN/JS）"""
    if not HAS_MATPLOTLIB:
        return {}
    chart_data = _chart_data(report)
    out = {}
    # 深色主题 + 中文字体（避免月、温等显示为方框）
    plt.rcParams.update({
        "figure.facecolor": "#1a2332",
        "axes.facecolor": "#1a2332",
        "axes.edgecolor": "#8b949e",
        "axes.labelcolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "legend.facecolor": "#1a2332",
        "legend.edgecolor": "#30363d",
        "text.color": "#e6edf3",
        "font.sans-serif": ["PingFang SC", "Heiti SC", "Microsoft YaHei", "SimHei", "sans-serif"],
        "axes.unicode_minus": False,
    })
    months = chart_data["chart_2025"]["labels"]
    x = list(range(len(months)))
    _legend_kw = {"loc": "upper right", "fontsize": 9}
    if _CHINESE_FONT is not None:
        _legend_kw["prop"] = _CHINESE_FONT

    def _set_ax_font(ax, ylabel=None):
        if ylabel:
            ax.set_ylabel(ylabel)
        if _CHINESE_FONT is not None:
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_fontproperties(_CHINESE_FONT)
            if ylabel:
                ax.yaxis.get_label().set_fontproperties(_CHINESE_FONT)

    # 图1: 2025 年月度
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x, chart_data["chart_2025"]["max"], "o-", color="#f85149", label="月均最高温", linewidth=2)
    ax.plot(x, chart_data["chart_2025"]["mean"], "s-", color="#58a6ff", label="月均平均温", linewidth=2)
    ax.plot(x, chart_data["chart_2025"]["min"], "^-", color="#3fb950", label="月均最低温", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend(**_legend_kw)
    _set_ax_font(ax, "温度 (°C)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["chart_2025"] = base64.b64encode(buf.getvalue()).decode("ascii")

    # 图2: 近10年
    fig, ax = plt.subplots(figsize=(8, 3.2))
    labels_10y = chart_data["chart_10y"]["labels"]
    x10 = list(range(len(labels_10y)))
    ax.plot(x10, chart_data["chart_10y"]["max"], "o-", color="#f85149", label="年均最高温", linewidth=2)
    ax.plot(x10, chart_data["chart_10y"]["mean"], "s-", color="#58a6ff", label="年均平均温", linewidth=2)
    ax.plot(x10, chart_data["chart_10y"]["min"], "^-", color="#3fb950", label="年均最低温", linewidth=2)
    ax.set_xticks(x10)
    ax.set_xticklabels(labels_10y)
    ax.legend(**_legend_kw)
    _set_ax_font(ax, "温度 (°C)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["chart_10y"] = base64.b64encode(buf.getvalue()).decode("ascii")

    # 图3: 异常天气柱状
    fig, ax = plt.subplots(figsize=(8, 2.8))
    labels_ab = chart_data["chart_ab"]["labels"]
    xab = list(range(len(labels_ab)))
    w = 0.35
    ax.bar([i - w / 2 for i in xab], chart_data["chart_ab"]["high"], w, label="高温异常", color="#f85149", alpha=0.8)
    ax.bar([i + w / 2 for i in xab], chart_data["chart_ab"]["low"], w, label="低温异常", color="#58a6ff", alpha=0.8)
    ax.set_xticks(xab)
    ax.set_xticklabels(labels_ab)
    ax.legend(**_legend_kw)
    _set_ax_font(ax, "天数")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["chart_ab"] = base64.b64encode(buf.getvalue()).decode("ascii")

    # 图4: 2026 预测
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x, chart_data["chart_2026"]["max"], "o-", color="#f85149", label="预测最高温", linewidth=2)
    ax.plot(x, chart_data["chart_2026"]["mean"], "s-", color="#58a6ff", label="预测平均温", linewidth=2)
    ax.plot(x, chart_data["chart_2026"]["min"], "^-", color="#3fb950", label="预测最低温", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend(**_legend_kw)
    _set_ax_font(ax, "温度 (°C)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["chart_2026"] = base64.b64encode(buf.getvalue()).decode("ascii")
    return out


def _chart_data(report):
    """提取图表数据（供 matplotlib 或 Chart.js）"""
    y25 = report["year_2025"]
    y10 = report["years_10"]
    ab = report["abnormal"]
    pred = report["predict_2026"]
    months_cn = "1月,2月,3月,4月,5月,6月,7月,8月,9月,10月,11月,12月".split(",")
    # 2025 年逐日
    chart_2025_daily = {"labels": [], "max": [], "min": [], "mean": []}
    if y25.get("daily"):
        chart_2025_daily["labels"] = [x["date"] for x in y25["daily"]]
        chart_2025_daily["max"] = [float(x["temp_max"]) for x in y25["daily"]]
        chart_2025_daily["min"] = [float(x["temp_min"]) for x in y25["daily"]]
        chart_2025_daily["mean"] = [float(x["temp_mean"]) for x in y25["daily"]]
    # 2025 年月度
    chart_2025 = {"labels": months_cn}
    if y25.get("by_month"):
        chart_2025["max"] = [float(y25["by_month"][str(m)]["temp_max_mean"]) for m in range(1, 13)]
        chart_2025["min"] = [float(y25["by_month"][str(m)]["temp_min_mean"]) for m in range(1, 13)]
        chart_2025["mean"] = [float(y25["by_month"][str(m)]["temp_mean_mean"]) for m in range(1, 13)]
    # 近10年
    by_year = sorted(y10["by_year"].items())
    chart_10y = {
        "labels": [yr for yr, _ in by_year],
        "max": [float(v["temp_max_mean"]) for _, v in by_year],
        "min": [float(v["temp_min_mean"]) for _, v in by_year],
        "mean": [float(v["temp_mean_mean"]) for _, v in by_year],
    }
    # 异常按年
    ab_by = ab.get("by_year", {})
    by_year_ab = sorted(ab_by.items())
    chart_ab = {
        "labels": [yr for yr, _ in by_year_ab],
        "high": [v["high"] for _, v in by_year_ab],
        "low": [v["low"] for _, v in by_year_ab],
    }
    # 2026 预测（月度 + 逐日插值）
    chart_2026 = {
        "labels": months_cn,
        "mean": [pred["by_month"][str(m)]["temp_mean"] for m in range(1, 13)],
        "max": [pred["by_month"][str(m)]["temp_max"] for m in range(1, 13)],
        "min": [pred["by_month"][str(m)]["temp_min"] for m in range(1, 13)],
    }
    chart_2026_daily = {"labels": [], "max": [], "min": [], "mean": []}
    pred_daily = _predict_2026_daily(pred)
    if pred_daily:
        chart_2026_daily["labels"] = [x["date"] for x in pred_daily]
        chart_2026_daily["max"] = [float(x["temp_max"]) for x in pred_daily]
        chart_2026_daily["min"] = [float(x["temp_min"]) for x in pred_daily]
        chart_2026_daily["mean"] = [float(x["temp_mean"]) for x in pred_daily]
    return {"chart_2025_daily": chart_2025_daily, "chart_2025": chart_2025, "chart_10y": chart_10y, "chart_ab": chart_ab, "chart_2026": chart_2026, "chart_2026_daily": chart_2026_daily}


def _tables_html(chart_data):
    """生成与各图表对应的表格 HTML，用于页面展示与下载"""
    parts = []
    d = chart_data
    if d.get("chart_2025_daily") and d["chart_2025_daily"].get("labels"):
        c = d["chart_2025_daily"]
        n = len(c["labels"])
        rows = "".join(
            f"<tr><td>{c['labels'][i]}</td><td>{c['max'][i]}</td><td>{c['mean'][i]}</td><td>{c['min'][i]}</td></tr>"
            for i in range(n)
        )
        parts.append(f'<div class="data-table-wrap"><h3>一、2025 年逐日气温</h3><table class="data-table"><thead><tr><th>日期</th><th>日最高温（°C）</th><th>日平均温（°C）</th><th>日最低温（°C）</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if d.get("chart_2025") and d["chart_2025"].get("labels"):
        c = d["chart_2025"]
        rows = "".join(f"<tr><td>{c['labels'][i]}</td><td>{c['max'][i]}</td><td>{c['mean'][i]}</td><td>{c['min'][i]}</td></tr>" for i in range(len(c["labels"])))
        parts.append(f'<div class="data-table-wrap"><h3>二、2025 年月度气温</h3><table class="data-table"><thead><tr><th>月份</th><th>月均最高温（°C）</th><th>月均平均温（°C）</th><th>月均最低温（°C）</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if d.get("chart_10y") and d["chart_10y"].get("labels"):
        c = d["chart_10y"]
        rows = "".join(f"<tr><td>{c['labels'][i]}</td><td>{c['max'][i]}</td><td>{c['mean'][i]}</td><td>{c['min'][i]}</td></tr>" for i in range(len(c["labels"])))
        parts.append(f'<div class="data-table-wrap"><h3>三、近 10 年气温趋势（2016－2025）</h3><table class="data-table"><thead><tr><th>年份</th><th>年均最高温（°C）</th><th>年均平均温（°C）</th><th>年均最低温（°C）</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if d.get("chart_ab") and d["chart_ab"].get("labels"):
        c = d["chart_ab"]
        rows = "".join(f"<tr><td>{c['labels'][i]}</td><td>{c['high'][i]}</td><td>{c['low'][i]}</td></tr>" for i in range(len(c["labels"])))
        parts.append(f'<div class="data-table-wrap"><h3>四、异常天气（高温/低温异常天数）</h3><table class="data-table"><thead><tr><th>年份</th><th>高温异常（天）</th><th>低温异常（天）</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if d.get("chart_2026") and d["chart_2026"].get("labels"):
        c = d["chart_2026"]
        rows = "".join(f"<tr><td>{c['labels'][i]}</td><td>{c['max'][i]}</td><td>{c['mean'][i]}</td><td>{c['min'][i]}</td></tr>" for i in range(len(c["labels"])))
        parts.append(f'<div class="data-table-wrap"><h3>五、2026 年预测（月均温）</h3><table class="data-table"><thead><tr><th>月份</th><th>预测最高温（°C）</th><th>预测平均温（°C）</th><th>预测最低温（°C）</th></tr></thead><tbody>{rows}</tbody></table></div>')
    return "".join(parts)


def render_html(report):
    """生成独立 HTML 报告，使用 Chart.js 可交互图表（悬停显示数据提示）"""
    r = report
    m = r["meta"]
    y25 = r["year_2025"]
    y10 = r["years_10"]
    ab = r["abnormal"]
    pred = r["predict_2026"]
    chart_data = _chart_data(report)
    tables_html = _tables_html(chart_data)
    # 嵌入页面供 JS 使用，避免 </script> 出现在字符串中
    chart_data_json = json.dumps(chart_data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    deg = "&#176;"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{m["title"]}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.6/dist/chart.umd.min.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js" crossorigin="anonymous"></script>
  <style>
    :root {{ --bg: #0f1419; --card: #1a2332; --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --border: #30363d; --warn: #f85149; --ok: #3fb950; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; min-height: 100vh; }}
    .container {{ max-width: 960px; margin: 0 auto; padding: 2rem 1rem; }}
    header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; padding: 2rem 0 3rem; border-bottom: 1px solid var(--border); }}
    header .header-text {{ flex: 1; min-width: 0; }}
    header h1 {{ font-size: 1.75rem; font-weight: 700; margin: 0 0 0.5rem; color: var(--text); }}
    header p {{ color: var(--muted); font-size: 0.95rem; margin: 0; }}
    section {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }}
    section h2 {{ font-size: 1.2rem; font-weight: 600; margin: 0 0 1rem; color: var(--accent); }}
    .chart-wrap {{ margin: 1rem 0; position: relative; height: 280px; }}
    .chart-wrap.chart-daily {{ height: 320px; }}
    .chart-wrap canvas {{ border-radius: 8px; }}
    .chart-wrap.chart-daily {{ padding: 12px; border-radius: 10px; background: linear-gradient(145deg, rgba(30,40,55,0.5) 0%%, rgba(26,35,50,0.3) 100%%); border: 1px solid rgba(48,54,61,0.5); }}
    footer {{ text-align: center; padding: 2rem 1rem; color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--border); }}
    .highlight {{ color: var(--accent); }}
    .summary {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.5rem; }}
    .btn-download {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.5rem 1rem; font-size: 0.9rem; color: var(--accent); background: rgba(88,166,255,0.15); border: 1px solid var(--accent); border-radius: 8px; cursor: pointer; font-family: inherit; transition: background 0.2s, color 0.2s; flex-shrink: 0; }}
    .btn-download:hover {{ background: rgba(88,166,255,0.25); color: var(--text); }}
    .data-table-wrap {{ margin-bottom: 1.5rem; }}
    .data-table-wrap h3 {{ font-size: 1rem; color: var(--accent); margin: 0 0 0.5rem; }}
    .chart-subtitle {{ font-size: 1rem; font-weight: 600; color: var(--accent); margin: 1rem 0 0.25rem; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .data-table th, .data-table td {{ padding: 0.4rem 0.6rem; text-align: left; border: 1px solid var(--border); }}
    .data-table th {{ background: rgba(48,54,61,0.5); color: var(--muted); }}
    .data-table tbody tr:hover td {{ background: rgba(88,166,255,0.08); }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-text">
        <h1>{m["title"]}</h1>
        <p>{m["location"]} · {m["period"]} · 生成时间 {m["generated"]}</p>
      </div>
      <button type="button" class="btn-download" id="btn-download-report" data-filename-xlsx="广州天气报告_表格数据_{m["generated"].replace(" ", "_").replace(":", "-")[:16]}.xlsx" aria-label="下载表格数据">下载表格数据</button>
    </header>

    <section>
      <h2>一、2025 年天气情况</h2>
      <p>2025 年共 <strong>{y25.get("days", 0)}</strong> 天。年均：最高温 {y25.get("desc", {}).get("temp_max", {}).get("mean", "-")}{deg}C，最低温 {y25.get("desc", {}).get("temp_min", {}).get("mean", "-")}{deg}C，平均温 {y25.get("desc", {}).get("temp_mean", {}).get("mean", "-")}{deg}C。</p>
      <p>年度极值：最高温日 <span class="highlight">{y25.get("highest_day", {}).get("date", "-")}</span>（{y25.get("highest_day", {}).get("temp_max", "-")}{deg}C），最低温日 <span class="highlight">{y25.get("lowest_day", {}).get("date", "-")}</span>（{y25.get("lowest_day", {}).get("temp_min", "-")}{deg}C）。</p>
      <div class="chart-wrap chart-daily"><canvas id="chart-2025-daily" aria-label="2025年逐日气温"></canvas></div>
      <div class="chart-wrap"><canvas id="chart-2025" aria-label="2025年月度气温"></canvas></div>
    </section>

    <section>
      <h2>二、近 10 年天气趋势（2016－2025）</h2>
      <p>近 10 年日均最高温均值 <strong>{y10.get("overall", {}).get("temp_max_mean", "-")}{deg}C</strong>，最低温均值 <strong>{y10.get("overall", {}).get("temp_min_mean", "-")}{deg}C</strong>，平均温 <strong>{y10.get("overall", {}).get("temp_mean_mean", "-")}{deg}C</strong>。年均温趋势约 <strong>{y10.get("trend_per_decade_c", 0)}{deg}C/10 年</strong>。</p>
      <div class="chart-wrap"><canvas id="chart-10y" aria-label="近10年气温趋势"></canvas></div>
    </section>

    <section>
      <h2>三、异常天气（高温/低温异常天数）</h2>
      <p>高温异常：日最高温 ≥ {ab["threshold_high_c"]}{deg}C（95% 分位），共 <strong>{len(ab["high_days"])}</strong> 天；低温异常：日最低温 ≤ {ab["threshold_low_c"]}{deg}C（5% 分位），共 <strong>{len(ab["low_days"])}</strong> 天。</p>
      <div class="chart-wrap"><canvas id="chart-ab" aria-label="异常天气天数"></canvas></div>
    </section>

    <section>
      <h2>四、2026 年预测（月均温）</h2>
      <p class="summary"><em>{pred["method"]}</em></p>
      <h3 class="chart-subtitle">每日温度预测</h3>
      <p class="summary">下图为由月均温插值得到的 2026 年逐日温度预测，悬停可看具体日期与数值。</p>
      <div class="chart-wrap chart-daily"><canvas id="chart-2026-daily" aria-label="2026年逐日预测"></canvas></div>
      <p class="summary">2026 年月度预测（月均最高温 / 平均温 / 最低温）：</p>
      <div class="chart-wrap"><canvas id="chart-2026" aria-label="2026年月度预测"></canvas></div>
      <p class="summary">注：仅供参考，不替代气象部门预报。</p>
    </section>

    <section>
      <h2>五、2026 年冬夏季展望</h2>
      <p><strong>冬季（2025/26 冬）：</strong>{r.get("outlook_2026", {}).get("winter", "—")}。</p>
      <p><strong>夏季（2026 年夏）：</strong>{r.get("outlook_2026", {}).get("summer", "—")}。</p>
      <p class="summary">{r.get("outlook_2026", {}).get("note", "")}</p>
    </section>

    <footer>
      <p>数据来源：Open-Meteo Archive API · 广州 · Asia/Shanghai</p>
    </footer>
  </div>
  <script>
(function() {{
  var data = {chart_data_json};
  var deg = "\\u00b0C";

  var btn = document.getElementById("btn-download-report");
  if (btn) {{
    btn.addEventListener("click", function() {{
      try {{
        var d = data;
        var filename = (btn.getAttribute("data-filename-xlsx") || "广州天气报告_表格数据.xlsx").replace(/[/\\\\:*?\"<>|]/g, "_");
        if (typeof XLSX !== "undefined") {{
          var wb = XLSX.utils.book_new();
          if (d.chart_2025_daily && d.chart_2025_daily.labels && d.chart_2025_daily.labels.length) {{
            var arr = [["日期", "日最高温(°C)", "日平均温(°C)", "日最低温(°C)"]];
            for (var i = 0; i < d.chart_2025_daily.labels.length; i++)
              arr.push([d.chart_2025_daily.labels[i], d.chart_2025_daily.max[i], d.chart_2025_daily.mean[i], d.chart_2025_daily.min[i]]);
            wb.SheetNames.push("2025逐日"); wb.Sheets["2025逐日"] = XLSX.utils.aoa_to_sheet(arr);
          }}
          if (d.chart_2025 && d.chart_2025.labels && d.chart_2025.labels.length) {{
            var arr = [["月份", "月均最高温(°C)", "月均平均温(°C)", "月均最低温(°C)"]];
            for (var i = 0; i < d.chart_2025.labels.length; i++)
              arr.push([d.chart_2025.labels[i], d.chart_2025.max[i], d.chart_2025.mean[i], d.chart_2025.min[i]]);
            wb.SheetNames.push("2025月度"); wb.Sheets["2025月度"] = XLSX.utils.aoa_to_sheet(arr);
          }}
          if (d.chart_10y && d.chart_10y.labels && d.chart_10y.labels.length) {{
            var arr = [["年份", "年均最高温(°C)", "年均平均温(°C)", "年均最低温(°C)"]];
            for (var i = 0; i < d.chart_10y.labels.length; i++)
              arr.push([d.chart_10y.labels[i], d.chart_10y.max[i], d.chart_10y.mean[i], d.chart_10y.min[i]]);
            wb.SheetNames.push("近10年"); wb.Sheets["近10年"] = XLSX.utils.aoa_to_sheet(arr);
          }}
          if (d.chart_ab && d.chart_ab.labels && d.chart_ab.labels.length) {{
            var arr = [["年份", "高温异常(天)", "低温异常(天)"]];
            for (var i = 0; i < d.chart_ab.labels.length; i++)
              arr.push([d.chart_ab.labels[i], d.chart_ab.high[i], d.chart_ab.low[i]]);
            wb.SheetNames.push("异常天气"); wb.Sheets["异常天气"] = XLSX.utils.aoa_to_sheet(arr);
          }}
          if (d.chart_2026 && d.chart_2026.labels && d.chart_2026.labels.length) {{
            var arr = [["月份", "预测最高温(°C)", "预测平均温(°C)", "预测最低温(°C)"]];
            for (var i = 0; i < d.chart_2026.labels.length; i++)
              arr.push([d.chart_2026.labels[i], d.chart_2026.max[i], d.chart_2026.mean[i], d.chart_2026.min[i]]);
            wb.SheetNames.push("2026预测"); wb.Sheets["2026预测"] = XLSX.utils.aoa_to_sheet(arr);
          }}
          XLSX.writeFile(wb, filename);
        }} else {{
          var parts = [];
          function escapeHtml(s) {{ return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }}
          function tableSection(title, headers, rows) {{
            var h = "<tr>" + headers.map(function(th) {{ return "<th>" + escapeHtml(th) + "</th>"; }}).join("") + "</tr>";
            var b = rows.map(function(row) {{ return "<tr>" + row.map(function(cell) {{ return "<td>" + escapeHtml(cell) + "</td>"; }}).join("") + "</tr>"; }}).join("");
            return "<section class=\\"data-section\\"><h2>" + escapeHtml(title) + "</h2><table><thead>" + h + "</thead><tbody>" + b + "</tbody></table></section>";
          }}
          if (d.chart_2025_daily && d.chart_2025_daily.labels && d.chart_2025_daily.labels.length) {{
            var rows = d.chart_2025_daily.labels.map(function(_, i) {{ return [d.chart_2025_daily.labels[i], d.chart_2025_daily.max[i], d.chart_2025_daily.mean[i], d.chart_2025_daily.min[i]]; }});
            parts.push(tableSection("一、2025 年逐日气温", ["日期", "日最高温（°C）", "日平均温（°C）", "日最低温（°C）"], rows));
          }}
          if (d.chart_2025 && d.chart_2025.labels && d.chart_2025.labels.length) {{
            var rows = d.chart_2025.labels.map(function(_, i) {{ return [d.chart_2025.labels[i], d.chart_2025.max[i], d.chart_2025.mean[i], d.chart_2025.min[i]]; }});
            parts.push(tableSection("二、2025 年月度气温", ["月份", "月均最高温（°C）", "月均平均温（°C）", "月均最低温（°C）"], rows));
          }}
          if (d.chart_10y && d.chart_10y.labels && d.chart_10y.labels.length) {{
            var rows = d.chart_10y.labels.map(function(_, i) {{ return [d.chart_10y.labels[i], d.chart_10y.max[i], d.chart_10y.mean[i], d.chart_10y.min[i]]; }});
            parts.push(tableSection("三、近 10 年气温趋势（2016－2025）", ["年份", "年均最高温（°C）", "年均平均温（°C）", "年均最低温（°C）"], rows));
          }}
          if (d.chart_ab && d.chart_ab.labels && d.chart_ab.labels.length) {{
            var rows = d.chart_ab.labels.map(function(_, i) {{ return [d.chart_ab.labels[i], d.chart_ab.high[i], d.chart_ab.low[i]]; }});
            parts.push(tableSection("四、异常天气（高温/低温异常天数）", ["年份", "高温异常（天）", "低温异常（天）"], rows));
          }}
          if (d.chart_2026 && d.chart_2026.labels && d.chart_2026.labels.length) {{
            var rows = d.chart_2026.labels.map(function(_, i) {{ return [d.chart_2026.labels[i], d.chart_2026.max[i], d.chart_2026.mean[i], d.chart_2026.min[i]]; }});
            parts.push(tableSection("五、2026 年预测（月均温）", ["月份", "预测最高温（°C）", "预测平均温（°C）", "预测最低温（°C）"], rows));
          }}
          var tablePage = "<!DOCTYPE html><html lang=\\"zh-CN\\"><head><meta charset=\\"UTF-8\\"><title>广州天气报告 - 表格数据</title><style>" +
            "body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #0f1419; color: #e6edf3; padding: 2rem; }} " +
            ".container {{ max-width: 960px; margin: 0 auto; }} " +
            "h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }} " +
            ".meta {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 2rem; }} " +
            ".data-section {{ margin-bottom: 2rem; }} " +
            ".data-section h2 {{ font-size: 1.1rem; color: #58a6ff; margin-bottom: 0.75rem; }} " +
            "table {{ width: 100%; border-collapse: collapse; background: #1a2332; border-radius: 8px; overflow: hidden; }} " +
            "th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #30363d; }} " +
            "th {{ background: rgba(48,54,61,0.5); color: #8b949e; font-weight: 600; font-size: 0.85rem; }} " +
            "td {{ font-size: 0.9rem; }} " +
            "tr:hover td {{ background: rgba(88,166,255,0.08); }}</style></head><body><div class=\\"container\\">" +
            "<h1>广州天气报告 · 表格数据</h1><p class=\\"meta\\">数据与报告图表一一对应，可复制到 Excel 使用。</p>" +
            parts.join("") +
            "</div></body></html>";
          var blob = new Blob(["\ufeff" + tablePage], {{ type: "text/html;charset=utf-8" }});
          var url = URL.createObjectURL(blob);
          var a = document.createElement("a");
          a.href = url;
          a.download = filename.replace(/\\.xlsx$/i, ".html");
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }}
      }} catch (e) {{ if (typeof console !== "undefined") console.error(e); alert("下载失败，请稍后重试或复制页面下方表格数据。"); }}
    }});
  }}

  var opts = {{
    responsive: true,
    maintainAspectRatio: false,
    color: "#8b949e",
    font: {{ family: "'PingFang SC', 'Microsoft YaHei', sans-serif", size: 11 }},
    plugins: {{
      legend: {{ position: "top", labels: {{ usePointStyle: true, padding: 14 }} }},
      tooltip: {{
        backgroundColor: "#1a2332",
        titleColor: "#e6edf3",
        bodyColor: "#e6edf3",
        borderColor: "#30363d",
        borderWidth: 1,
        padding: 10,
        callbacks: {{
          label: function(ctx) {{
            var name = ctx.dataset.label;
            var v = ctx.parsed.y;
            if (ctx.chart.config.type === "bar") return name + "：" + v + " 天";
            return name + "：" + Number(v).toFixed(1) + deg;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ grid: {{ color: "#30363d" }}, ticks: {{ maxRotation: 45 }} }},
      y: {{ grid: {{ color: "#30363d" }} }}
    }}
  }};

  function lineDatasets(labels, max, mean, min, pointRadius) {{
    pointRadius = pointRadius ?? 4;
    return [
      {{ label: labels[0], data: max, borderColor: "#f85149", backgroundColor: "rgba(248,81,73,0.1)", tension: 0.2, pointRadius: pointRadius }},
      {{ label: labels[1], data: mean, borderColor: "#58a6ff", backgroundColor: "rgba(88,166,255,0.1)", tension: 0.2, pointRadius: pointRadius }},
      {{ label: labels[2], data: min, borderColor: "#3fb950", backgroundColor: "rgba(63,185,80,0.1)", tension: 0.2, pointRadius: pointRadius }}
    ];
  }}

  var c25daily = data.chart_2025_daily;
  if (c25daily && c25daily.labels && c25daily.labels.length) {{
    var dailyCanvas = document.getElementById("chart-2025-daily");
    var dailyCtx = dailyCanvas.getContext("2d");
    var dailyH = 320;
    var gradMax = dailyCtx.createLinearGradient(0, 0, 0, dailyH);
    gradMax.addColorStop(0, "rgba(248,81,73,0.22)"); gradMax.addColorStop(1, "rgba(248,81,73,0)");
    var gradMean = dailyCtx.createLinearGradient(0, 0, 0, dailyH);
    gradMean.addColorStop(0, "rgba(88,166,255,0.22)"); gradMean.addColorStop(1, "rgba(88,166,255,0)");
    var gradMin = dailyCtx.createLinearGradient(0, 0, 0, dailyH);
    gradMin.addColorStop(0, "rgba(63,185,80,0.22)"); gradMin.addColorStop(1, "rgba(63,185,80,0)");
    var dailyDatasets = [
      {{ label: "日最高温", data: c25daily.max, borderColor: "#f85149", backgroundColor: gradMax, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBorderWidth: 2, pointHoverBackgroundColor: "#f85149" }},
      {{ label: "日平均温", data: c25daily.mean, borderColor: "#58a6ff", backgroundColor: gradMean, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBorderWidth: 2, pointHoverBackgroundColor: "#58a6ff" }},
      {{ label: "日最低温", data: c25daily.min, borderColor: "#3fb950", backgroundColor: gradMin, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBorderWidth: 2, pointHoverBackgroundColor: "#3fb950" }}
    ];
    var dailyOpts = Object.assign({{}}, opts, {{
      interaction: {{ intersect: false, mode: "index" }},
      plugins: Object.assign({{}}, opts.plugins, {{
        tooltip: Object.assign({{}}, opts.plugins.tooltip, {{
          callbacks: Object.assign({{}}, opts.plugins.tooltip.callbacks, {{
            title: function(items) {{ return items.length ? "日期：" + (items[0].label || "") : ""; }},
            label: function(ctx) {{ return ctx.dataset.label + "：" + Number(ctx.parsed.y).toFixed(1) + deg; }}
          }})
        }})
      }}),
      scales: {{
        x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 14, maxRotation: 45, color: "#8b949e", font: {{ size: 11 }} }} }},
        y: {{ grid: {{ color: "rgba(48,54,61,0.5)" }}, ticks: {{ color: "#8b949e", font: {{ size: 11 }}, stepSize: 5 }} }}
      }}
    }});
    new Chart(dailyCanvas, {{
      type: "line",
      data: {{ labels: c25daily.labels, datasets: dailyDatasets }},
      options: dailyOpts
    }});
  }}

  var c25 = data.chart_2025;
  if (c25 && c25.labels && c25.labels.length) {{
    new Chart(document.getElementById("chart-2025"), {{
      type: "line",
      data: {{ labels: c25.labels, datasets: lineDatasets(["月均最高温", "月均平均温", "月均最低温"], c25.max, c25.mean, c25.min) }},
      options: opts
    }});
  }}

  var c10 = data.chart_10y;
  if (c10 && c10.labels && c10.labels.length) {{
    new Chart(document.getElementById("chart-10y"), {{
      type: "line",
      data: {{ labels: c10.labels, datasets: lineDatasets(["年均最高温", "年均平均温", "年均最低温"], c10.max, c10.mean, c10.min) }},
      options: opts
    }});
  }}

  var cab = data.chart_ab;
  if (cab && cab.labels && cab.labels.length) {{
    new Chart(document.getElementById("chart-ab"), {{
      type: "bar",
      data: {{
        labels: cab.labels,
        datasets: [
          {{ label: "高温异常", data: cab.high, backgroundColor: "rgba(248,81,73,0.8)", borderColor: "#f85149", borderWidth: 1 }},
          {{ label: "低温异常", data: cab.low, backgroundColor: "rgba(88,166,255,0.8)", borderColor: "#58a6ff", borderWidth: 1 }}
        ]
      }},
      options: Object.assign({{}}, opts, {{
        scales: {{ x: {{ grid: {{ display: false }}, ticks: {{ maxRotation: 45 }} }}, y: {{ grid: {{ color: "#30363d" }}, ticks: {{ stepSize: 1 }} }} }}
      }})
    }});
  }}

  var c26daily = data.chart_2026_daily;
  if (c26daily && c26daily.labels && c26daily.labels.length) {{
    var canvas2026d = document.getElementById("chart-2026-daily");
    if (canvas2026d) {{
      var ctx2026d = canvas2026d.getContext("2d");
      var h = 320;
      var gMax = ctx2026d.createLinearGradient(0, 0, 0, h);
      gMax.addColorStop(0, "rgba(248,81,73,0.22)"); gMax.addColorStop(1, "rgba(248,81,73,0)");
      var gMean = ctx2026d.createLinearGradient(0, 0, 0, h);
      gMean.addColorStop(0, "rgba(88,166,255,0.22)"); gMean.addColorStop(1, "rgba(88,166,255,0)");
      var gMin = ctx2026d.createLinearGradient(0, 0, 0, h);
      gMin.addColorStop(0, "rgba(63,185,80,0.22)"); gMin.addColorStop(1, "rgba(63,185,80,0)");
      var ds2026d = [
        {{ label: "预测最高温", data: c26daily.max, borderColor: "#f85149", backgroundColor: gMax, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBorderWidth: 2, pointHoverBackgroundColor: "#f85149" }},
        {{ label: "预测平均温", data: c26daily.mean, borderColor: "#58a6ff", backgroundColor: gMean, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBorderWidth: 2, pointHoverBackgroundColor: "#58a6ff" }},
        {{ label: "预测最低温", data: c26daily.min, borderColor: "#3fb950", backgroundColor: gMin, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBorderWidth: 2, pointHoverBackgroundColor: "#3fb950" }}
      ];
      var opts2026d = Object.assign({{}}, opts, {{
        interaction: {{ intersect: false, mode: "index" }},
        plugins: Object.assign({{}}, opts.plugins, {{
          tooltip: Object.assign({{}}, opts.plugins.tooltip, {{
            callbacks: Object.assign({{}}, opts.plugins.tooltip.callbacks, {{
              title: function(items) {{ return items.length ? "日期：" + (items[0].label || "") : ""; }},
              label: function(ctx) {{ return ctx.dataset.label + "：" + Number(ctx.parsed.y).toFixed(1) + deg; }}
            }})
          }})
        }}),
        scales: {{ x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 14, maxRotation: 45 }} }}, y: {{ grid: {{ color: "#30363d" }} }} }}
      }});
      new Chart(canvas2026d, {{ type: "line", data: {{ labels: c26daily.labels, datasets: ds2026d }}, options: opts2026d }});
    }}
  }}

  var c26 = data.chart_2026;
  if (c26 && c26.labels && c26.labels.length) {{
    new Chart(document.getElementById("chart-2026"), {{
      type: "line",
      data: {{ labels: c26.labels, datasets: lineDatasets(["预测最高温", "预测平均温", "预测最低温"], c26.max, c26.mean, c26.min) }},
      options: opts
    }});
  }}

}})();
  </script>
</body>
</html>"""
    return html


def main():
    print("正在获取数据...")
    raw = fetch_data()
    df = to_dataframe(raw)
    print(f"已加载 {len(df)} 行")

    print("生成报告数据...")
    report = build_report_data(df)

    out_dir = Path(__file__).parent
    json_path = out_dir / "guangzhou_weather_report_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"已保存: {json_path}")

    html_path = out_dir / "guangzhou_weather_report.html"
    html = render_html(report)
    # 兼容性：用 HTML 实体代替度数符号，避免编码问题；添加 BOM 便于部分浏览器识别 UTF-8
    # 已统一用 &#176;，此处仅防遗漏
    if "°" in html:
        html = html.replace("°", "&#176;")
    with open(html_path, "w", encoding="utf-8-sig") as f:
        f.write(html)
    print(f"已保存: {html_path}")

    # 同时写入 docs/index.html，便于用 GitHub Pages / Netlify 等部署
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    docs_index = docs_dir / "index.html"
    with open(docs_index, "w", encoding="utf-8-sig") as f:
        f.write(html)
    print(f"已保存: {docs_index}（用于线上部署）")

    print("\n完成。请打开 guangzhou_weather_report.html 查看报告。")


if __name__ == "__main__":
    main()
