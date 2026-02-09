#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
广州天气数据报告：2025年分析、近10年分析、异常天气标注、2026年预测
"""

import base64
import io
import json
import os
import urllib.request
import urllib.parse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import math

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_stats = None

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
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
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


def fetch_forecast_16d():
    """获取未来 16 天预报（含今天），用于今日温度与未来 15 天预告"""
    params = {
        "latitude": PARAMS["latitude"],
        "longitude": PARAMS["longitude"],
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
        "timezone": PARAMS["timezone"],
        "forecast_days": 16,
    }
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{FORECAST_URL}?{query}") as resp:
        return json.loads(resp.read().decode())


def fetch_last_30d():
    """获取最近 30 天历史数据（用于趋势图）"""
    end = datetime.now().date()
    start = end - timedelta(days=29)
    params = {
        "latitude": PARAMS["latitude"],
        "longitude": PARAMS["longitude"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
        "timezone": PARAMS["timezone"],
    }
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{API_URL}?{query}") as resp:
        return json.loads(resp.read().decode())


def _temp_advice(temp_min: float, temp_max: float) -> str:
    """根据今日温度范围给出简要建议"""
    low, high = float(temp_min), float(temp_max)
    if high >= 35:
        return "高温天气，注意防暑降温、减少户外活动，多补水。"
    if high >= 30:
        return "天气较热，注意防晒与补水，午后尽量减少暴晒。"
    if low >= 25 and high >= 28:
        return "体感偏热，适宜短袖，注意通风。"
    if low >= 18 and high <= 28:
        return "气温适宜，早晚可加薄外套，注意增减衣物。"
    if low >= 10 and high <= 22:
        return "早晚偏凉，建议穿长袖或薄外套。"
    if low >= 0 and high <= 15:
        return "天气较冷，注意保暖，外出加衣。"
    if low < 0:
        return "严寒天气，注意防寒保暖，尽量减少户外停留。"
    return "注意根据体感增减衣物。"


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


def _trend_significance(years, means):
    """
    对年均温做线性回归 year ~ mean，检验斜率是否显著不为 0。
    返回 slope_per_year, stderr, p_value, ci_low, ci_high, 以及 0.1 是否落在 95% CI 内。
    若无 scipy 或 n<3，返回 None 表示未做检验。
    """
    if not HAS_SCIPY or len(years) < 3:
        return None
    years = [float(y) for y in years]
    means = [float(m) for m in means]
    res = scipy_stats.linregress(years, means)
    n = len(years)
    df = n - 2
    t_crit = scipy_stats.t.ppf(0.975, df)
    se = res.stderr  # 斜率标准误
    ci_low = res.slope - t_crit * se
    ci_high = res.slope + t_crit * se
    return {
        "slope_per_year": round(float(res.slope), 4),
        "stderr": float(se),
        "p_value": float(res.pvalue),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "significant": bool(res.pvalue < 0.05),
        "trend_01_in_ci": bool(ci_low <= 0.1 <= ci_high),
        "r_squared": round(float(res.rvalue) ** 2, 4),
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
    years = by_year.index.astype(int).values
    means = by_year["temp_mean_mean"].values
    # 趋势：回归斜率（°C/年）与显著性
    trend_sig = _trend_significance(years.tolist(), means.tolist())
    if len(years) >= 2:
        slope = (means[-1] - means[0]) / (years[-1] - years[0])
        trend_per_decade = round(slope * 10, 2)
    else:
        trend_per_decade = 0
    out = {
        "by_year": {str(k): v.to_dict() for k, v in by_year.iterrows()},
        "trend_per_decade_c": trend_per_decade,
        "overall": {
            "temp_max_mean": round(float(df["temp_max"].mean()), 2),
            "temp_min_mean": round(float(df["temp_min"].mean()), 2),
            "temp_mean_mean": round(float(df["temp_mean"].mean()), 2),
        },
    }
    if trend_sig is not None:
        out["trend_test"] = trend_sig
    return out


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


def predict_2026(df, years_10=None):
    """
    2026 年预测：基于 2016-2025 各月均值，叠加由线性回归估计的年际趋势（统计技能：数据驱动 + 95% CI）。
    若 years_10 含 trend_test，则采用回归斜率 β（°C/年）作为 2025→2026 的偏移；否则采用参考值 0.1°C/年。
    """
    by_month = df.groupby("month").agg(
        temp_max_avg=("temp_max", "mean"),
        temp_min_avg=("temp_min", "mean"),
        temp_mean_avg=("temp_mean", "mean"),
    ).round(2)
    month_names = "1月,2月,3月,4月,5月,6月,7月,8月,9月,10月,11月,12月".split(",")

    trend = 0.1  # 默认参考值（无回归时使用）
    method = "基于 2016-2025 年月均值，并叠加约 +0.1°C 年际升温趋势（参考值；未做回归估计）。"
    trend_from_regression = False

    if years_10 and years_10.get("trend_test"):
        t = years_10["trend_test"]
        # 2025→2026 仅 1 年，偏移 = 斜率 × 1
        trend = t["slope_per_year"]
        ci_low, ci_high = t["ci_low"], t["ci_high"]
        p = t["p_value"]
        trend_from_regression = True
        if p < 0.05:
            method = (
                f"基于 2016-2025 年月均值，叠加线性回归估计的年际趋势：β = {trend:.3f}°C/年，"
                f"95% CI [{ci_low:.3f}, {ci_high:.3f}]，p = {p:.3f}（显著）。"
            )
        else:
            method = (
                f"基于 2016-2025 年月均值，叠加线性回归估计的年际趋势：β = {trend:.3f}°C/年，"
                f"95% CI [{ci_low:.3f}, {ci_high:.3f}]，p = {p:.3f}（不显著，预测中仍采用该估计值供参考）。"
            )

    pred = {}
    for m in range(1, 13):
        row = by_month.loc[m]
        pred[str(m)] = {
            "month_name": month_names[m - 1],
            "temp_mean": round(float(row["temp_mean_avg"]) + trend, 1),
            "temp_max": round(float(row["temp_max_avg"]) + trend, 1),
            "temp_min": round(float(row["temp_min_avg"]) + trend, 1),
        }
    return {
        "method": method,
        "by_month": pred,
        "trend_offset_per_year": round(trend, 4),
        "trend_from_regression": trend_from_regression,
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


def build_report_data(df, today_weather=None, last_30d=None, forecast_15d=None, yesterday_weather=None):
    years_10 = analyze_10years(df)
    pred = predict_2026(df, years_10)
    out = {
        "meta": {
            "title": "广州天气数据报告",
            "period": "2016-01-01 至 2025-12-31",
            "location": "广州（约 23.13°N, 113.26°E）",
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "year_2025": analyze_2025(df),
        "years_10": years_10,
        "abnormal": get_abnormal_days(df),
        "predict_2026": pred,
        "outlook_2026": _outlook_2026(df, pred),
    }
    if today_weather is not None:
        out["today_weather"] = today_weather
    if yesterday_weather is not None:
        out["yesterday_weather"] = yesterday_weather
    if forecast_15d is not None:
        out["forecast_15d"] = forecast_15d
    return out


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
    # 未来 15 天预告（含今天共 16 天）
    chart_forecast = {"labels": [], "max": [], "min": [], "mean": []}
    if report.get("forecast_15d"):
        for x in report["forecast_15d"]:
            chart_forecast["labels"].append(x["date"])
            chart_forecast["max"].append(float(x["temp_max"]))
            chart_forecast["min"].append(float(x["temp_min"]))
            chart_forecast["mean"].append(float(x["temp_mean"]))
    return {
        "chart_2025_daily": chart_2025_daily,
        "chart_2025": chart_2025,
        "chart_10y": chart_10y,
        "chart_ab": chart_ab,
        "chart_2026": chart_2026,
        "chart_2026_daily": chart_2026_daily,
        "chart_forecast": chart_forecast,
    }


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


def _section_today_and_trends(r, deg):
    """报告最前：今日/昨日温度与建议、未来 15 天预告（打开页面时按当前日期加载）"""
    parts = []
    today = r.get("today_weather")
    yesterday = r.get("yesterday_weather")
    if today:
        yesterday_line = ""
        if yesterday:
            yesterday_line = f"\n      <p class=\"summary\"><strong>昨日（{yesterday['date']}）温度范围：</strong> {yesterday['temp_min']}{deg}C ~ {yesterday['temp_max']}{deg}C</p>"
        parts.append(f"""
    <section class="highlight-section">
      <h2>今日天气与近期预告</h2>
      <p id="today-summary"><strong>今日（{today["date"]}）温度范围：</strong> {today["temp_min"]}{deg}C ~ {today["temp_max"]}{deg}C</p>{yesterday_line}
      <p class="summary" id="today-advice"><strong>温度建议：</strong>{today["advice"]}</p>
      <h3 class="chart-subtitle">今日逐时温度预告</h3>
      <p class="summary" id="hourly-today-note">打开页面时自动加载当前日期的逐时预报，悬停可看具体时刻与温度。</p>
      <div id="hourly-today-wrap" class="chart-wrap chart-daily" style="min-height: 200px;">
        <p id="hourly-loading" class="summary">正在加载今日逐时温度…</p>
        <canvas id="chart-hourly-today" aria-label="今日逐时气温" style="display: none;"></canvas>
      </div>
    </section>""")
    parts.append("""
    <section>
      <h3 class="chart-subtitle">未来 15 天天气预告</h3>
      <p id="forecast-loading" class="summary">正在加载当前日期的未来 15 天预报…</p>
      <div class="chart-wrap chart-daily" id="wrap-forecast" style="display: none;"><canvas id="chart-forecast" aria-label="未来15天预报"></canvas></div>
      <p class="summary">打开页面时自动按当前日期更新，数据来源：Open-Meteo Forecast API。</p>
    </section>""")
    return "\n".join(parts) if parts else ""

def _trend_sig_text(y10):
    """近 10 年趋势显著性说明（用于第二节）"""
    t = y10.get("trend_test")
    if not t:
        return "（趋势显著性检验需 scipy；安装后重新生成报告即可显示：pip install scipy。）"
    p = t["p_value"]
    slope = t["slope_per_year"]
    ci_low, ci_high = t["ci_low"], t["ci_high"]
    if p < 0.05:
        return f"经线性回归检验，2016－2025 年年均温随时间呈显著上升趋势（斜率 {slope:.3f}°C/年，95% CI [{ci_low:.3f}, {ci_high:.3f}]，p = {p:.3f}）。"
    return f"经线性回归检验，2016－2025 年年均温随时间的变化在统计上不显著（斜率 {slope:.3f}°C/年，95% CI [{ci_low:.3f}, {ci_high:.3f}]，p = {p:.3f}）。"


def _trend_01_note(y10, pred):
    """说明 2026 预测采用的趋势是否来自回归估计及统计学含义（用于第四节）"""
    if pred.get("trend_from_regression"):
        t = y10.get("trend_test")
        if t and t["p_value"] < 0.05:
            return "预测采用由 2016－2025 年年均温线性回归得到的斜率估计（见第二节），95% CI 已在上文给出，趋势在统计上显著。"
        return "预测采用回归估计的斜率供参考；该年际趋势在 2016－2025 年数据中不显著（p ≥ 0.05），结果仅供定性参考。"
    t = y10.get("trend_test")
    if not t:
        return "预测采用参考值 +0.1°C/年；安装 scipy 后重新生成报告可改为基于回归估计的斜率。"
    return "预测采用参考值；本节方法说明中已给出回归估计的斜率与 95% CI，可与数据对照。"


def _report_context_for_llm(report):
    """生成供大模型使用的报告摘要文本（自然语言查数上下文）"""
    r = report
    m = r.get("meta", {})
    deg = "°C"
    parts = [
        f"报告：{m.get('title', '')}。{m.get('location', '')}，数据时段 {m.get('period', '')}，生成时间 {m.get('generated', '')}。"
    ]
    if r.get("today_weather"):
        t = r["today_weather"]
        parts.append(f"今日（{t.get('date')}）温度范围：{t.get('temp_min')}～{t.get('temp_max')}{deg}；穿衣建议：{t.get('advice', '')}")
    if r.get("yesterday_weather"):
        y = r["yesterday_weather"]
        parts.append(f"昨日（{y.get('date')}）温度范围：{y.get('temp_min')}～{y.get('temp_max')}{deg}")
    y25 = r.get("year_2025", {})
    if y25:
        desc = y25.get("desc", {})
        parts.append(
            f"2025年：共{y25.get('days', 0)}天；年均最高温{desc.get('temp_max', {}).get('mean', '-')}{deg}，"
            f"最低温{desc.get('temp_min', {}).get('mean', '-')}{deg}，平均温{desc.get('temp_mean', {}).get('mean', '-')}{deg}。"
        )
        h, l = y25.get("highest_day", {}), y25.get("lowest_day", {})
        parts.append(f"年度最高温日：{h.get('date', '-')}（{h.get('temp_max', '-')}{deg}）；最低温日：{l.get('date', '-')}（{l.get('temp_min', '-')}{deg}）。")
    y10 = r.get("years_10", {})
    if y10:
        o = y10.get("overall", {})
        parts.append(
            f"近10年（2016-2025）：日均最高温均值{o.get('temp_max_mean', '-')}{deg}，"
            f"最低温均值{o.get('temp_min_mean', '-')}{deg}，平均温{o.get('temp_mean_mean', '-')}{deg}；"
            f"年均温趋势约{y10.get('trend_per_decade_c', '-')}{deg}/10年。"
        )
    ab = r.get("abnormal", {})
    if ab:
        parts.append(
            f"异常天气定义：高温异常为日最高温≥{ab.get('threshold_high_c', '-')}{deg}（95%分位），共{len(ab.get('high_days', []))}天；"
            f"低温异常为日最低温≤{ab.get('threshold_low_c', '-')}{deg}（5%分位），共{len(ab.get('low_days', []))}天。"
        )
    pred = r.get("predict_2026", {})
    if pred:
        parts.append(f"2026年预测方法：{pred.get('method', '')}。")
    out = r.get("outlook_2026", {})
    if out:
        parts.append(f"2026年冬季展望：{out.get('winter', '')}；夏季展望：{out.get('summer', '')}。")
    return "\n".join(parts)


def render_html(report, chat_api_url=""):
    """生成独立 HTML 报告，使用 Chart.js 可交互图表（悬停显示数据提示）。chat_api_url 为大模型问答接口，留空则仅用 FAQ。"""
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
    report_context_text = _report_context_for_llm(report)
    report_context_json = json.dumps(report_context_text, ensure_ascii=False).replace("</", "\\u003c/")
    chat_api_url_js = json.dumps(chat_api_url)
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
    section.highlight-section {{ border-color: var(--accent); background: linear-gradient(135deg, rgba(26,35,50,0.98) 0%%, rgba(30,45,65,0.95) 100%%); }}
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
    .report-module {{ margin-bottom: 3rem; }}
    .report-module-title {{ font-size: 1.35rem; font-weight: 700; color: var(--text); margin: 0 0 1.25rem; padding-bottom: 0.75rem; border-bottom: 2px solid var(--accent); }}
    .charts-row {{ display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }}
    .charts-row .chart-cell {{ flex: 1 1 45%; min-width: 280px; }}
    /* 智能客服聊天 */
    #chat-widget {{ position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 9999; font-family: inherit; }}
    #chat-toggle {{ width: 56px; height: 56px; border-radius: 50%; border: 2px solid var(--accent); background: var(--card); color: var(--accent); cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.3); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; transition: transform 0.2s, background 0.2s; }}
    #chat-toggle:hover {{ background: rgba(88,166,255,0.2); transform: scale(1.05); }}
    #chat-panel {{ display: none; position: absolute; bottom: 70px; right: 0; width: 360px; max-width: calc(100vw - 2rem); max-height: 480px; background: var(--card); border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); flex-direction: column; overflow: hidden; }}
    #chat-panel.open {{ display: flex; }}
    #chat-header {{ padding: 0.75rem 1rem; background: rgba(88,166,255,0.15); border-bottom: 1px solid var(--border); font-weight: 600; color: var(--accent); display: flex; align-items: center; justify-content: space-between; }}
    #chat-close {{ background: none; border: none; color: var(--muted); cursor: pointer; font-size: 1.2rem; padding: 0 0.25rem; line-height: 1; }}
    #chat-close:hover {{ color: var(--text); }}
    #chat-messages {{ flex: 1; overflow-y: auto; padding: 1rem; min-height: 200px; max-height: 320px; }}
    .chat-msg {{ margin-bottom: 0.75rem; max-width: 90%; }}
    .chat-msg.bot {{ margin-right: auto; }}
    .chat-msg.user {{ margin-left: auto; }}
    .chat-msg .bubble {{ padding: 0.6rem 0.9rem; border-radius: 12px; font-size: 0.9rem; line-height: 1.5; }}
    .chat-msg.bot .bubble {{ background: rgba(48,54,61,0.6); border: 1px solid var(--border); color: var(--text); }}
    .chat-msg.user .bubble {{ background: rgba(88,166,255,0.2); border: 1px solid var(--accent); color: var(--text); }}
    #chat-quick {{ padding: 0 1rem 0.5rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }}
    #chat-quick button {{ padding: 0.35rem 0.6rem; font-size: 0.8rem; color: var(--accent); background: rgba(88,166,255,0.12); border: 1px solid var(--border); border-radius: 8px; cursor: pointer; font-family: inherit; }}
    #chat-quick button:hover {{ background: rgba(88,166,255,0.2); }}
    #chat-form {{ display: flex; padding: 0.75rem 1rem; border-top: 1px solid var(--border); gap: 0.5rem; }}
    #chat-input {{ flex: 1; padding: 0.5rem 0.75rem; border: 1px solid var(--border); border-radius: 8px; background: var(--bg); color: var(--text); font-size: 0.9rem; font-family: inherit; }}
    #chat-input:focus {{ outline: none; border-color: var(--accent); }}
    #chat-send {{ padding: 0.5rem 1rem; background: var(--accent); color: #0f1419; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; font-family: inherit; font-size: 0.9rem; }}
    #chat-send:hover {{ filter: brightness(1.1); }}
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

    <div class="report-module">
      <h2 class="report-module-title">一、近期天气与预告</h2>
      {_section_today_and_trends(r, deg)}
    </div>

    <div class="report-module">
      <h2 class="report-module-title">二、历史与年度分析</h2>
    <section>
      <h2>1. 2025 年天气情况</h2>
      <p>2025 年共 <strong>{y25.get("days", 0)}</strong> 天。年均：最高温 {y25.get("desc", {}).get("temp_max", {}).get("mean", "-")}{deg}C，最低温 {y25.get("desc", {}).get("temp_min", {}).get("mean", "-")}{deg}C，平均温 {y25.get("desc", {}).get("temp_mean", {}).get("mean", "-")}{deg}C。</p>
      <p>年度极值：最高温日 <span class="highlight">{y25.get("highest_day", {}).get("date", "-")}</span>（{y25.get("highest_day", {}).get("temp_max", "-")}{deg}C），最低温日 <span class="highlight">{y25.get("lowest_day", {}).get("date", "-")}</span>（{y25.get("lowest_day", {}).get("temp_min", "-")}{deg}C）。</p>
      <div class="chart-wrap chart-daily"><canvas id="chart-2025-daily" aria-label="2025年逐日气温"></canvas></div>
      <div class="chart-wrap"><canvas id="chart-2025" aria-label="2025年月度气温"></canvas></div>
    </section>

    <section>
      <h2>2. 近 10 年天气趋势（2016－2025）</h2>
      <p>近 10 年日均最高温均值 <strong>{y10.get("overall", {}).get("temp_max_mean", "-")}{deg}C</strong>，最低温均值 <strong>{y10.get("overall", {}).get("temp_min_mean", "-")}{deg}C</strong>，平均温 <strong>{y10.get("overall", {}).get("temp_mean_mean", "-")}{deg}C</strong>。年均温趋势约 <strong>{y10.get("trend_per_decade_c", 0)}{deg}C/10 年</strong>。</p>
      <p class="summary">{_trend_sig_text(y10)}</p>
      <div class="chart-wrap"><canvas id="chart-10y" aria-label="近10年气温趋势"></canvas></div>
    </section>

    <section>
      <h2>3. 异常天气（高温/低温异常天数）</h2>
      <p>高温异常：日最高温 ≥ {ab["threshold_high_c"]}{deg}C（95% 分位），共 <strong>{len(ab["high_days"])}</strong> 天；低温异常：日最低温 ≤ {ab["threshold_low_c"]}{deg}C（5% 分位），共 <strong>{len(ab["low_days"])}</strong> 天。</p>
      <div class="chart-wrap"><canvas id="chart-ab" aria-label="异常天气天数"></canvas></div>
    </section>

    <section>
      <h2>4. 2026 年预测（月均温）</h2>
      <p class="summary"><em>{pred["method"]}</em></p>
      <p class="summary">{_trend_01_note(y10, pred)}</p>
      <h3 class="chart-subtitle">每日温度预测</h3>
      <p class="summary">下图为由月均温插值得到的 2026 年逐日温度预测，悬停可看具体日期与数值。</p>
      <div class="chart-wrap chart-daily"><canvas id="chart-2026-daily" aria-label="2026年逐日预测"></canvas></div>
      <p class="summary">2026 年月度预测（月均最高温 / 平均温 / 最低温）：</p>
      <div class="chart-wrap"><canvas id="chart-2026" aria-label="2026年月度预测"></canvas></div>
      <p class="summary">注：仅供参考，不替代气象部门预报。</p>
    </section>

    <section>
      <h2>5. 2026 年冬夏季展望</h2>
      <p><strong>冬季（2025/26 冬）：</strong>{r.get("outlook_2026", {}).get("winter", "—")}。</p>
      <p><strong>夏季（2026 年夏）：</strong>{r.get("outlook_2026", {}).get("summer", "—")}。</p>
      <p class="summary">{r.get("outlook_2026", {}).get("note", "")}</p>
    </section>
    </div>

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
          if (d.chart_forecast && d.chart_forecast.labels && d.chart_forecast.labels.length) {{
            var arr = [["日期", "预报最高温(°C)", "预报平均温(°C)", "预报最低温(°C)"]];
            for (var i = 0; i < d.chart_forecast.labels.length; i++)
              arr.push([d.chart_forecast.labels[i], d.chart_forecast.max[i], d.chart_forecast.mean[i], d.chart_forecast.min[i]]);
            wb.SheetNames.push("未来15天"); wb.Sheets["未来15天"] = XLSX.utils.aoa_to_sheet(arr);
          }}
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
          if (d.chart_forecast && d.chart_forecast.labels && d.chart_forecast.labels.length) {{
            var rows = d.chart_forecast.labels.map(function(_, i) {{ return [d.chart_forecast.labels[i], d.chart_forecast.max[i], d.chart_forecast.mean[i], d.chart_forecast.min[i]]; }});
            parts.push(tableSection("一、未来 15 天预报", ["日期", "预报最高温（°C）", "预报平均温（°C）", "预报最低温（°C）"], rows));
          }}
          if (d.chart_2025_daily && d.chart_2025_daily.labels && d.chart_2025_daily.labels.length) {{
            var rows = d.chart_2025_daily.labels.map(function(_, i) {{ return [d.chart_2025_daily.labels[i], d.chart_2025_daily.max[i], d.chart_2025_daily.mean[i], d.chart_2025_daily.min[i]]; }});
            parts.push(tableSection("二、2025 年逐日气温", ["日期", "日最高温（°C）", "日平均温（°C）", "日最低温（°C）"], rows));
          }}
          if (d.chart_2025 && d.chart_2025.labels && d.chart_2025.labels.length) {{
            var rows = d.chart_2025.labels.map(function(_, i) {{ return [d.chart_2025.labels[i], d.chart_2025.max[i], d.chart_2025.mean[i], d.chart_2025.min[i]]; }});
            parts.push(tableSection("三、2025 年月度气温", ["月份", "月均最高温（°C）", "月均平均温（°C）", "月均最低温（°C）"], rows));
          }}
          if (d.chart_10y && d.chart_10y.labels && d.chart_10y.labels.length) {{
            var rows = d.chart_10y.labels.map(function(_, i) {{ return [d.chart_10y.labels[i], d.chart_10y.max[i], d.chart_10y.mean[i], d.chart_10y.min[i]]; }});
            parts.push(tableSection("四、近 10 年气温趋势（2016－2025）", ["年份", "年均最高温（°C）", "年均平均温（°C）", "年均最低温（°C）"], rows));
          }}
          if (d.chart_ab && d.chart_ab.labels && d.chart_ab.labels.length) {{
            var rows = d.chart_ab.labels.map(function(_, i) {{ return [d.chart_ab.labels[i], d.chart_ab.high[i], d.chart_ab.low[i]]; }});
            parts.push(tableSection("五、异常天气（高温/低温异常天数）", ["年份", "高温异常（天）", "低温异常（天）"], rows));
          }}
          if (d.chart_2026 && d.chart_2026.labels && d.chart_2026.labels.length) {{
            var rows = d.chart_2026.labels.map(function(_, i) {{ return [d.chart_2026.labels[i], d.chart_2026.max[i], d.chart_2026.mean[i], d.chart_2026.min[i]]; }});
            parts.push(tableSection("六、2026 年预测（月均温）", ["月份", "预测最高温（°C）", "预测平均温（°C）", "预测最低温（°C）"], rows));
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

  // Visualization-expert: Clarity, Honesty, Simplicity, Accessibility (colorblind-friendly palette)
  var palette = {{ high: "#e65c2b", mean: "#0173b2", low: "#029e73" }};
  var paletteRgba = {{ high: "rgba(230,92,43,0.15)", mean: "rgba(1,115,178,0.15)", low: "rgba(2,158,115,0.15)" }};
  var gridColor = "rgba(48,54,61,0.4)";
  var tickColor = "#8b949e";

  var opts = {{
    responsive: true,
    maintainAspectRatio: false,
    color: tickColor,
    font: {{ family: "'PingFang SC', 'Microsoft YaHei', sans-serif", size: 12 }},
    plugins: {{
      legend: {{ position: "top", labels: {{ usePointStyle: true, padding: 12, boxWidth: 12 }} }},
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
      x: {{ grid: {{ display: false }}, ticks: {{ maxRotation: 45, color: tickColor, font: {{ size: 11 }} }} }},
      y: {{
        grid: {{ color: gridColor, drawBorder: false }},
        ticks: {{ color: tickColor, font: {{ size: 11 }}, stepSize: 5 }},
        title: {{ display: true, text: "温度 (" + deg + ")", color: tickColor, font: {{ size: 11 }} }}
      }}
    }}
  }};

  function lineDatasets(labels, max, mean, min, pointRadius) {{
    pointRadius = pointRadius ?? 5;
    return [
      {{ label: labels[0], data: max, borderColor: palette.high, backgroundColor: paletteRgba.high, borderWidth: 2, tension: 0.25, pointRadius: pointRadius, pointHoverRadius: 7 }},
      {{ label: labels[1], data: mean, borderColor: palette.mean, backgroundColor: paletteRgba.mean, borderWidth: 2, tension: 0.25, pointRadius: pointRadius, pointHoverRadius: 7 }},
      {{ label: labels[2], data: min, borderColor: palette.low, backgroundColor: paletteRgba.low, borderWidth: 2, tension: 0.25, pointRadius: pointRadius, pointHoverRadius: 7 }}
    ];
  }}

  function renderLineChart(canvasId, chartData, optsOverride) {{
    if (!chartData || !chartData.labels || chartData.labels.length === 0) return;
    var el = document.getElementById(canvasId);
    if (!el) return;
    var o = Object.assign({{}}, opts, optsOverride || {{}});
    new Chart(el, {{ type: "line", data: {{ labels: chartData.labels, datasets: lineDatasets(["最高温", "平均温", "最低温"], chartData.max, chartData.mean, chartData.min, 3) }}, options: o }});
  }}

  // 未来 15 天：打开页面时按当前日期请求并绘制
  (function loadForecast15() {{
    var forecastUrl = "https://api.open-meteo.com/v1/forecast?latitude=23.1291&longitude=113.2644&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&timezone=Asia/Shanghai&forecast_days=16";
    fetch(forecastUrl).then(function(r) {{ return r.json(); }})
      .then(function(fc) {{
        var cFc = {{ labels: [], max: [], min: [], mean: [] }};
        if (fc.daily && fc.daily.time) {{
          for (var j = 0; j < fc.daily.time.length; j++) {{
            cFc.labels.push(fc.daily.time[j]);
            cFc.max.push(fc.daily.temperature_2m_max[j]);
            cFc.min.push(fc.daily.temperature_2m_min[j]);
            cFc.mean.push(fc.daily.temperature_2m_mean[j]);
          }};
        }};
        data.chart_forecast = cFc;
        var loadingFc = document.getElementById("forecast-loading");
        var wrapFc = document.getElementById("wrap-forecast");
        if (cFc.labels.length) {{
          if (loadingFc) loadingFc.style.display = "none";
          if (wrapFc) wrapFc.style.display = "block";
          var canvasFc = document.getElementById("chart-forecast");
          if (canvasFc) {{
            var existingFc = typeof Chart !== "undefined" && Chart.getChart(canvasFc);
            if (existingFc) existingFc.destroy();
            renderLineChart("chart-forecast", cFc, {{ interaction: {{ intersect: false, mode: "index" }}, scales: {{ x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 16, maxRotation: 45, color: tickColor }} }}, y: {{ grid: {{ color: gridColor }}, ticks: {{ color: tickColor, stepSize: 5 }}, title: {{ display: true, text: "温度 (" + deg + ")", color: tickColor, font: {{ size: 11 }} }} }} }} }});
          }};
        }} else {{ if (loadingFc) loadingFc.textContent = "暂无未来 15 天预报"; }}
      }})
      .catch(function() {{ var loadingFc = document.getElementById("forecast-loading"); if (loadingFc) loadingFc.textContent = "加载失败，请刷新或检查网络"; }});
  }})();

  var c25daily = data.chart_2025_daily;
  if (c25daily && c25daily.labels && c25daily.labels.length) {{
    var dailyCanvas = document.getElementById("chart-2025-daily");
    var dailyCtx = dailyCanvas.getContext("2d");
    var dailyH = 320;
    var gH = dailyCtx.createLinearGradient(0, 0, 0, dailyH);
    gH.addColorStop(0, "rgba(230,92,43,0.2)"); gH.addColorStop(1, "rgba(230,92,43,0)");
    var gM = dailyCtx.createLinearGradient(0, 0, 0, dailyH);
    gM.addColorStop(0, "rgba(1,115,178,0.2)"); gM.addColorStop(1, "rgba(1,115,178,0)");
    var gL = dailyCtx.createLinearGradient(0, 0, 0, dailyH);
    gL.addColorStop(0, "rgba(2,158,115,0.2)"); gL.addColorStop(1, "rgba(2,158,115,0)");
    var dailyDatasets = [
      {{ label: "日最高温", data: c25daily.max, borderColor: palette.high, backgroundColor: gH, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBackgroundColor: palette.high }},
      {{ label: "日平均温", data: c25daily.mean, borderColor: palette.mean, backgroundColor: gM, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBackgroundColor: palette.mean }},
      {{ label: "日最低温", data: c25daily.min, borderColor: palette.low, backgroundColor: gL, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBackgroundColor: palette.low }}
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
        x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 14, maxRotation: 45, color: tickColor, font: {{ size: 11 }} }} }},
        y: {{ grid: {{ color: gridColor }}, ticks: {{ color: tickColor, font: {{ size: 11 }}, stepSize: 5 }}, title: {{ display: true, text: "温度 (" + deg + ")", color: tickColor, font: {{ size: 11 }} }} }}
      }}
    }});
    new Chart(dailyCanvas, {{ type: "line", data: {{ labels: c25daily.labels, datasets: dailyDatasets }}, options: dailyOpts }});
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
          {{ label: "高温异常", data: cab.high, backgroundColor: "rgba(230,92,43,0.85)", borderColor: palette.high, borderWidth: 1 }},
          {{ label: "低温异常", data: cab.low, backgroundColor: "rgba(1,115,178,0.85)", borderColor: palette.mean, borderWidth: 1 }}
        ]
      }},
      options: Object.assign({{}}, opts, {{
        scales: {{
          x: {{ grid: {{ display: false }}, ticks: {{ maxRotation: 45, color: tickColor }} }},
          y: {{ grid: {{ color: gridColor }}, ticks: {{ stepSize: 1, color: tickColor }}, title: {{ display: true, text: "天数", color: tickColor, font: {{ size: 11 }} }} }}
        }}
      }})
    }});
  }}

  var c26daily = data.chart_2026_daily;
  if (c26daily && c26daily.labels && c26daily.labels.length) {{
    var canvas2026d = document.getElementById("chart-2026-daily");
    if (canvas2026d) {{
      var ctx2026d = canvas2026d.getContext("2d");
      var h = 320;
      var gH26 = ctx2026d.createLinearGradient(0, 0, 0, h);
      gH26.addColorStop(0, "rgba(230,92,43,0.2)"); gH26.addColorStop(1, "rgba(230,92,43,0)");
      var gM26 = ctx2026d.createLinearGradient(0, 0, 0, h);
      gM26.addColorStop(0, "rgba(1,115,178,0.2)"); gM26.addColorStop(1, "rgba(1,115,178,0)");
      var gL26 = ctx2026d.createLinearGradient(0, 0, 0, h);
      gL26.addColorStop(0, "rgba(2,158,115,0.2)"); gL26.addColorStop(1, "rgba(2,158,115,0)");
      var ds2026d = [
        {{ label: "预测最高温", data: c26daily.max, borderColor: palette.high, backgroundColor: gH26, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBackgroundColor: palette.high }},
        {{ label: "预测平均温", data: c26daily.mean, borderColor: palette.mean, backgroundColor: gM26, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBackgroundColor: palette.mean }},
        {{ label: "预测最低温", data: c26daily.min, borderColor: palette.low, backgroundColor: gL26, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 6, hitRadius: 12, pointHoverBackgroundColor: palette.low }}
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
        scales: {{
          x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 14, maxRotation: 45, color: tickColor }} }},
          y: {{ grid: {{ color: gridColor }}, ticks: {{ color: tickColor, stepSize: 5 }}, title: {{ display: true, text: "温度 (" + deg + ")", color: tickColor, font: {{ size: 11 }} }} }}
        }}
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

  // 今日逐时温度：每次打开页面自动请求当前日期的逐时预报并更新「今日」区块
  (function loadHourlyToday() {{
    var wrap = document.getElementById("hourly-today-wrap");
    var loadingEl = document.getElementById("hourly-loading");
    var canvasEl = document.getElementById("chart-hourly-today");
    if (!wrap || !canvasEl) return;
    var apiUrl = "https://api.open-meteo.com/v1/forecast?latitude=23.1291&longitude=113.2644&hourly=temperature_2m&timezone=Asia/Shanghai&forecast_days=2";
    fetch(apiUrl)
      .then(function(r) {{ return r.json(); }})
      .then(function(res) {{
        var h = res.hourly;
        if (!h || !h.time || !h.temperature_2m) {{ if (loadingEl) loadingEl.textContent = "暂无逐时数据"; return; }}
        var times = h.time;
        var temps = h.temperature_2m;
        var todayStr = times[0].slice(0, 10);
        var todayIndices = [];
        for (var i = 0; i < times.length; i++) {{
          if (times[i].slice(0, 10) === todayStr) todayIndices.push(i);
        }}
        if (todayIndices.length === 0) {{ if (loadingEl) loadingEl.textContent = "今日逐时数据暂无"; return; }}
        var labels = [];
        var data = [];
        for (var j = 0; j < todayIndices.length; j++) {{
          var idx = todayIndices[j];
          labels.push(times[idx].slice(11, 16));
          data.push(Number(temps[idx].toFixed(1)));
        }}
        var tMin = Math.min.apply(null, data);
        var tMax = Math.max.apply(null, data);
        var summaryEl = document.getElementById("today-summary");
        if (summaryEl) summaryEl.innerHTML = "<strong>今日（" + todayStr + "）温度范围：</strong> " + tMin + deg + " ~ " + tMax + deg;
        var adviceEl = document.getElementById("today-advice");
        if (adviceEl) {{
          var advice = (tMax >= 35) ? "高温天气，注意防暑降温、减少户外活动，多补水。" :
            (tMax >= 30) ? "天气较热，注意防晒与补水，午后尽量减少暴晒。" :
            (tMin >= 18 && tMax <= 28) ? "气温适宜，早晚可加薄外套，注意增减衣物。" :
            (tMin >= 10 && tMax <= 22) ? "早晚偏凉，建议穿长袖或薄外套。" :
            (tMin >= 0 && tMax <= 15) ? "天气较冷，注意保暖，外出加衣。" :
            (tMin < 0) ? "严寒天气，注意防寒保暖。" : "注意根据体感增减衣物。";
          adviceEl.innerHTML = "<strong>温度建议：</strong>" + advice;
        }}
        if (loadingEl) loadingEl.style.display = "none";
        canvasEl.style.display = "block";
        new Chart(canvasEl, {{
          type: "line",
          data: {{
            labels: labels,
            datasets: [{{
              label: "气温",
              data: data,
              borderColor: palette.mean,
              backgroundColor: "rgba(1,115,178,0.2)",
              borderWidth: 2,
              fill: true,
              tension: 0.3,
              pointRadius: 2,
              pointHoverRadius: 6
            }}]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ intersect: false, mode: "index" }},
            plugins: {{
              legend: {{ display: false }},
              tooltip: {{
                backgroundColor: "#1a2332",
                titleColor: "#e6edf3",
                bodyColor: "#e6edf3",
                borderColor: "#30363d",
                callbacks: {{
                  title: function(items) {{ return items.length ? todayStr + " " + (items[0].label || "") : ""; }},
                  label: function(ctx) {{ return "温度：" + Number(ctx.parsed.y).toFixed(1) + deg; }}
                }}
              }}
            }},
            scales: {{
              x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 12, color: tickColor, font: {{ size: 10 }} }} }},
              y: {{ grid: {{ color: gridColor }}, ticks: {{ color: tickColor, stepSize: 2 }}, title: {{ display: true, text: "温度 (" + deg + ")", color: tickColor, font: {{ size: 11 }} }} }}
            }}
          }}
        }});
      }})
      .catch(function() {{ if (loadingEl) loadingEl.textContent = "加载失败，请刷新或检查网络"; }});
  }})();

}})();

  </script>
  <script type="application/json" id="report-context">{report_context_json}</script>
  <!-- 智能客服：支持大模型自然语言查数，未配置 API 时使用 FAQ -->
  <div id="chat-widget">
    <button type="button" id="chat-toggle" aria-label="打开客服">💬</button>
    <div id="chat-panel">
      <div id="chat-header">报告小助手 <button type="button" id="chat-close" aria-label="关闭">×</button></div>
      <div id="chat-messages"></div>
      <div id="chat-quick"></div>
      <form id="chat-form">
        <input type="text" id="chat-input" placeholder="输入问题，例如：数据从哪来？" autocomplete="off" />
        <button type="submit" id="chat-send">发送</button>
      </form>
    </div>
  </div>
  <script>
(function() {{
  var CHAT_API_URL = {chat_api_url_js};  // 大模型接口：留空则仅用 FAQ；部署到 Vercel 并设环境变量 CHAT_API_URL=/api/chat 时自动填入
  var panel = document.getElementById("chat-panel");
  var toggle = document.getElementById("chat-toggle");
  var closeBtn = document.getElementById("chat-close");
  var messages = document.getElementById("chat-messages");
  var quick = document.getElementById("chat-quick");
  var form = document.getElementById("chat-form");
  var input = document.getElementById("chat-input");

  var reportContext = "";
  try {{
    var ctxEl = document.getElementById("report-context");
    if (ctxEl && ctxEl.textContent) reportContext = JSON.parse(ctxEl.textContent);
  }} catch (e) {{}}

  var faq = [
    {{ keys: ["数据来源", "数据从哪", "哪来的", "open-meteo", "api"], answer: "本报告历史数据来自 Open-Meteo Archive API（广州站），今日与未来预报来自 Open-Meteo Forecast API。页面底部有数据来源说明。" }},
    {{ keys: ["今日", "今天", "温度范围", "逐时"], answer: "报告顶部「今日天气与近期预告」中有今日温度范围与穿衣建议；「今日逐时温度预告」图会在打开页面时按当前日期加载逐时预报，悬停可看具体时刻与温度。" }},
    {{ keys: ["昨日", "昨天"], answer: "在「今日天气与近期预告」区块中，今日温度下方会显示昨日（具体日期）的温度范围，便于对比。" }},
    {{ keys: ["未来15天", "15天", "预报"], answer: "「未来 15 天天气预告」图表在打开页面时按当前日期自动加载，数据来自 Open-Meteo Forecast API，悬停可看每日温度。" }},
    {{ keys: ["2025", "去年", "年度"], answer: "「二、历史与年度分析」下「2025 年天气情况」为 2025 年全年统计：年均最高温/最低温/平均温、年度极值日，以及逐日与月度气温图。" }},
    {{ keys: ["近10年", "10年", "趋势", "线性回归", "p值", "显著性"], answer: "近 10 年趋势基于 2016－2025 年数据，用线性回归得到年均温变化（°C/10 年）。若 p < 0.05 表示趋势在统计上显著；p ≥ 0.05 表示不显著，报告中会注明「仅供定性参考」。" }},
    {{ keys: ["高温异常", "低温异常", "95%", "5%", "分位"], answer: "高温异常：日最高温 ≥ 95% 分位（约 33.6°C），表示该日最高温在历史中偏高。低温异常：日最低温 ≤ 5% 分位（约 8.4°C），表示该日最低温在历史中偏低。报告中有各年异常天数的柱状图。" }},
    {{ keys: ["2026", "预测", "月均温", "展望"], answer: "2026 年预测基于 2016－2025 年月均值，并叠加线性回归得到的年际趋势（若显著则采用，否则注明不显著、供参考）。「每日温度预测」由月均温插值得到；「冬夏季展望」为定性描述，不替代气象部门预报。" }},
    {{ keys: ["下载", "表格", "xlsx", "excel"], answer: "点击页面右上角「下载表格数据」可下载 XLSX 表格，内含本报告中的图表数据（如 2025 年逐日、近 10 年、异常天气、2026 预测等），便于在 Excel 中进一步分析。" }},
    {{ keys: ["怎么用", "如何看", "怎么看", "什么意思"], answer: "报告从上到下依次为：近期天气与预告（今日/昨日/逐时/未来15天）、历史与年度分析（2025 年、近 10 年趋势、异常天气、2026 预测与冬夏季展望）。每个图表悬停可看具体数值。有疑问可以继续问我。" }}
  ];

  function normalize(s) {{ return (s || "").toLowerCase().replace(/\\s+/g, ""); }}
  function findAnswer(q) {{
    var nq = normalize(q);
    if (!nq) return null;
    var best = null, bestScore = 0;
    for (var i = 0; i < faq.length; i++) {{
      var score = 0;
      for (var j = 0; j < faq[i].keys.length; j++) {{
        if (nq.indexOf(normalize(faq[i].keys[j])) !== -1) score++;
      }}
      if (score > bestScore) {{ bestScore = score; best = faq[i]; }}
    }}
    return best ? best.answer : null;
  }}

  function addMsg(text, isUser, isLoading) {{
    var div = document.createElement("div");
    div.className = "chat-msg " + (isUser ? "user" : "bot");
    if (isLoading) div.classList.add("loading");
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    div.appendChild(bubble);
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  }}

  var welcome = "你好，我是报告小助手。你可以用自然语言提问，例如：「2025年最热是哪天」「近10年升温多少」「今天适合穿什么」。若已配置大模型接口，我会根据报告数据回答；否则使用预设问答。";
  var quickQuestions = ["2025年最热是哪天？", "近10年升温趋势怎样？", "什么是高温异常？", "今天适合穿什么？"];

  function showWelcome() {{
    addMsg(welcome, false);
    quickQuestions.forEach(function(q) {{
      var btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = q;
      btn.addEventListener("click", function() {{ input.value = q; form.dispatchEvent(new Event("submit")); }});
      quick.appendChild(btn);
    }});
  }}

  toggle.addEventListener("click", function() {{
    panel.classList.toggle("open");
    if (panel.classList.contains("open") && messages.children.length === 0) showWelcome();
    if (panel.classList.contains("open")) input.focus();
  }});
  closeBtn.addEventListener("click", function() {{ panel.classList.remove("open"); }});

  form.addEventListener("submit", function(e) {{
    e.preventDefault();
    var q = (input.value || "").trim();
    if (!q) return;
    input.value = "";
    addMsg(q, true);

    if (CHAT_API_URL) {{
      var loadingEl = addMsg("思考中…", false, true);
      fetch(CHAT_API_URL, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ question: q, reportContext: reportContext }})
      }})
        .then(function(r) {{ return r.json(); }})
        .then(function(data) {{
          var bubble = loadingEl.querySelector(".bubble");
          if (bubble) bubble.textContent = data.answer || data.error || "回答生成失败";
          loadingEl.classList.remove("loading");
        }})
        .catch(function() {{
          var bubble = loadingEl.querySelector(".bubble");
          if (bubble) bubble.textContent = findAnswer(q) || "网络或服务异常，请稍后重试或检查是否已配置大模型接口。";
          loadingEl.classList.remove("loading");
        }});
    }} else {{
      setTimeout(function() {{
        addMsg(findAnswer(q) || "抱歉，没找到相关说明。请配置 CHAT_API_URL 接入大模型后可自然语言查数，或试试：数据从哪来？什么是高温异常？", false);
      }}, 200);
    }}
  }});
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

    today_weather = None
    last_30d = None
    forecast_15d = None
    try:
        print("获取今日与未来 15 天预报...")
        fc = fetch_forecast_16d()
        daily = fc.get("daily", {})
        times = daily.get("time", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        tmean = daily.get("temperature_2m_mean", [])
        if times and len(times) >= 1:
            temp_min = round(tmin[0], 1)
            temp_max = round(tmax[0], 1)
            today_weather = {
                "date": times[0],
                "temp_min": temp_min,
                "temp_max": temp_max,
                "advice": _temp_advice(temp_min, temp_max),
            }
        if len(times) >= 16:
            forecast_15d = [
                {"date": times[i], "temp_max": round(tmax[i], 1), "temp_min": round(tmin[i], 1), "temp_mean": round(tmean[i], 1)}
                for i in range(16)
            ]
        else:
            forecast_15d = [
                {"date": times[i], "temp_max": round(tmax[i], 1), "temp_min": round(tmin[i], 1), "temp_mean": round(tmean[i], 1)}
                for i in range(len(times))
            ] if times else None
    except Exception as e:
        print(f"预报获取失败（将不显示今日与未来预告）: {e}")

    yesterday_weather = None
    try:
        print("获取最近 2 天历史（用于昨日温度）...")
        raw30 = fetch_last_30d()
        daily30 = raw30.get("daily", {})
        t30 = daily30.get("time", [])
        if t30:
            last_30d = [
                {
                    "date": t30[i],
                    "temp_max": round(daily30["temperature_2m_max"][i], 1),
                    "temp_min": round(daily30["temperature_2m_min"][i], 1),
                    "temp_mean": round(daily30["temperature_2m_mean"][i], 1),
                }
                for i in range(len(t30))
            ]
            # 昨日 = 今日 - 1 天，从 last_30d 中取昨日数据
            if today_weather:
                today_str = today_weather["date"]
                yesterday_dt = datetime.strptime(today_str, "%Y-%m-%d").date() - timedelta(days=1)
                yesterday_str = yesterday_dt.strftime("%Y-%m-%d")
            else:
                yesterday_str = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
            for row in last_30d:
                if row["date"] == yesterday_str:
                    yesterday_weather = {"date": yesterday_str, "temp_min": row["temp_min"], "temp_max": row["temp_max"]}
                    break
    except Exception as e:
        print(f"昨日数据获取失败（将不显示昨日温度）: {e}")

    print("生成报告数据...")
    report = build_report_data(df, today_weather=today_weather, yesterday_weather=yesterday_weather, forecast_15d=forecast_15d)

    out_dir = Path(__file__).parent
    json_path = out_dir / "guangzhou_weather_report_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"已保存: {json_path}")

    html_path = out_dir / "guangzhou_weather_report.html"
    chat_api_url = os.environ.get("CHAT_API_URL", "")
    html = render_html(report, chat_api_url=chat_api_url)
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
