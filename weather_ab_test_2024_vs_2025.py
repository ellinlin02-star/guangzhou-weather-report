#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2024年(组A) vs 2025年(组B) 广州天气 A/B 检验
使用 ab-testing-analyzer 技能进行独立样本 t 检验与效应量分析。
"""

import sys
from pathlib import Path

# 添加 ab-testing-analyzer 技能的 scripts 路径（仅用 statistical_tests，避免 sklearn 依赖）
skill_scripts = Path(__file__).parent / ".cursor" / "skills" / "ab-testing-analyzer" / "scripts"
sys.path.insert(0, str(skill_scripts))

import pandas as pd

# 使用广州天气报告的数据源
from guangzhou_weather_report import fetch_data, to_dataframe


def load_ab_data():
    """加载 2024、2025 年日度数据，并标记组 A / 组 B"""
    raw = fetch_data()
    df = to_dataframe(raw)
    df = df[df["year"].isin([2024, 2025])].copy()
    df["group"] = df["year"].map({2024: "A", 2025: "B"})
    return df


def run_ab_analysis():
    from statistical_tests import StatisticalTests

    print("=" * 60)
    print("广州天气 A/B 检验：2024年(组A) vs 2025年(组B)")
    print("=" * 60)

    data = load_ab_data()
    n_a = (data["group"] == "A").sum()
    n_b = (data["group"] == "B").sum()
    print(f"\n组 A (2024年): {n_a} 天")
    print(f"组 B (2025年): {n_b} 天")

    stats_tests = StatisticalTests(alpha=0.05)
    metrics = [
        ("temp_mean", "日平均气温 (°C)"),
        ("temp_max", "日最高气温 (°C)"),
        ("temp_min", "日最低气温 (°C)"),
    ]

    results = []
    for metric_col, metric_name in metrics:
        t_result = stats_tests.t_test(
            data,
            group_col="group",
            metric_col=metric_col,
            test_type="independent",
            equal_var=True,
            alternative="two-sided",
        )
        results.append((metric_name, metric_col, t_result))

    # 报告
    print("\n" + "-" * 60)
    print("一、独立样本 t 检验结果 (组 A vs 组 B)")
    print("-" * 60)

    for metric_name, metric_col, r in results:
        g1, g2 = r["group1_stats"], r["group2_stats"]
        print(f"\n【{metric_name}】")
        print(f"  组 A (2024): 均值 = {g1['mean']:.2f}, 标准差 = {g1['std']:.2f}, n = {g1['n']}")
        print(f"  组 B (2025): 均值 = {g2['mean']:.2f}, 标准差 = {g2['std']:.2f}, n = {g2['n']}")
        print(f"  均值差 (A - B) = {r['mean_difference']:.3f}")
        ci = r["confidence_interval"]
        print(f"  95% 置信区间: [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  t = {r['statistic']:.3f}, df = {r['degrees_of_freedom']}, p = {r['p_value']:.4f}")
        print(f"  Cohen's d = {r['effect_size']:.3f} ({r['effect_size_interpretation']})")
        print(f"  结论: {'有统计学差异 (p < 0.05)' if r['is_significant'] else '无统计学差异 (p ≥ 0.05)'}")

    # 综合结论
    print("\n" + "=" * 60)
    print("二、综合结论")
    print("=" * 60)
    sig_metrics = [name for name, _, r in results if r["is_significant"]]
    if sig_metrics:
        print(f"  以下指标在 2024 与 2025 年间存在统计学差异 (α=0.05)：")
        for m in sig_metrics:
            print(f"    · {m}")
    else:
        print("  在 α=0.05 水平下，2024 年与 2025 年的日平均气温、日最高气温、日最低气温均无统计学差异。")
    print("\n注：检验基于逐日观测的独立样本 t 检验，两组样本量充足。")
    return results, data


if __name__ == "__main__":
    run_ab_analysis()
