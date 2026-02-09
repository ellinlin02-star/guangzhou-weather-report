#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把图表 PNG 转为 base64 并生成单文件 HTML，可直接用浏览器打开。"""

import base64
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CHARTS = [
    ("chart_01_月度平均温度.png", "1. 各月平均温度", "2016–2025 年各月份平均温度柱状图，虚线为全年平均。"),
    ("chart_02_年度温度趋势.png", "2. 年度温度趋势", "各年年平均温度变化及十年平均线。"),
    ("chart_03_日平均温度曲线.png", "3. 日平均温度曲线", "单年每日平均温度及 7 日滚动平均。"),
    ("chart_04_各月温度箱线图.png", "4. 各月温度分布（箱线图）", "各月日平均温度的分布与离散程度。"),
    ("chart_05_日温差分布.png", "5. 日温差分布", "日最高温与日最低温之差的分布直方图。"),
    ("chart_06_日最高最低温.png", "6. 日最高温与日最低温", "单年每日最高温、最低温及二者之间的区间。"),
]

# CSS 里的大括号用 {{ }} 转义，只保留 {sections} 占位
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>气候数据可视化报告</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --border: #30363d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
    }}
    .container {{ max-width: 960px; margin: 0 auto; padding: 2rem 1rem; }}
    header {{
      text-align: center;
      padding: 2rem 0 3rem;
      border-bottom: 1px solid var(--border);
    }}
    header h1 {{
      font-size: 1.75rem;
      font-weight: 700;
      margin: 0 0 0.5rem;
      color: var(--text);
    }}
    header p {{
      color: var(--muted);
      font-size: 0.95rem;
      margin: 0;
    }}
    .chart-section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 2rem;
    }}
    .chart-section h2 {{
      font-size: 1.1rem;
      font-weight: 600;
      margin: 0 0 0.5rem;
      color: var(--accent);
    }}
    .chart-section p {{
      font-size: 0.875rem;
      color: var(--muted);
      margin: 0 0 1rem;
    }}
    .chart-section img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      display: block;
      background: #0d1117;
    }}
    footer {{
      text-align: center;
      padding: 2rem 1rem;
      color: var(--muted);
      font-size: 0.85rem;
      border-top: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>气候数据可视化报告</h1>
      <p>广州 · 2016-01-01 至 2025-12-31 · Open-Meteo 日度数据</p>
    </header>

    {sections}

    <footer>
      数据来源：Open-Meteo Archive API · 图表由 climate_visualize.py 生成
    </footer>
  </div>
</body>
</html>
"""

SECTION_TEMPLATE = """
    <section class="chart-section">
      <h2>{title}</h2>
      <p>{desc}</p>
      <img src="data:image/png;base64,{b64}" alt="{alt}" loading="lazy">
    </section>"""


def main():
    sections = []
    for filename, title, desc in CHARTS:
        path = os.path.join(SCRIPT_DIR, filename)
        if not os.path.isfile(path):
            print(f"警告: 未找到 {filename}")
            continue
        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
        sections.append(
            SECTION_TEMPLATE.format(
                title=title,
                desc=desc,
                b64=b64,
                alt=title,
            )
        )
    html = HTML_TEMPLATE.format(sections="".join(sections))
    out_path = os.path.join(SCRIPT_DIR, "climate_report_standalone.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成: {out_path}")
    print("用浏览器直接打开该文件即可查看，无需联网。")


if __name__ == "__main__":
    main()
