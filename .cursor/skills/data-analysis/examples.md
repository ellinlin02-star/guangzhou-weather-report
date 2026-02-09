# 示例工作流与现有脚本对接

## 示例工作流：从 CSV 到报告

**场景**：本地有一份 `sales.csv`（日期、地区、销售额），需要一份简要分析报告。

1. **加载与概览**
   ```python
   import pandas as pd
   df = pd.read_csv("sales.csv", encoding="utf-8")
   df["日期"] = pd.to_datetime(df["日期"])
   print(df.shape, df.dtypes, df.head())
   ```

2. **质量检查**
   ```python
   print(df.isna().sum())
   print(df.duplicated().sum())
   ```

3. **描述统计与分组**
   ```python
   summary = df.groupby("地区").agg(销售额=("销售额", ["sum", "mean", "count"]))
   monthly = df.set_index("日期").resample("ME")["销售额"].sum()
   ```

4. **可视化**（可选）
   ```python
   import matplotlib.pyplot as plt
   monthly.plot(title="月度销售额"); plt.tight_layout(); plt.savefig("monthly_sales.png")
   ```

5. **输出**
   - 将 `summary`、关键数字写入 Markdown 报告，按 SKILL.md 中的「分析报告结构」组织。
   - 需要时 `df.to_csv("result.csv", index=False, encoding="utf-8-sig")`。

---

## 与现有脚本对接：weather_analysis.py

项目中的 `weather_analysis.py` 从 Open-Meteo API 拉取日度天气，用纯 Python 做统计并输出 `weather_analysis_result.json`。可与本技能这样配合：

### 方式 A：在脚本结果上做进一步分析

脚本已产出 **JSON 结果**（如 `weather_analysis_result.json`）。若要做可视化或额外分析，可用 pandas 读入后走技能流程：

```python
import pandas as pd
import json

with open("weather_analysis_result.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

# 年度统计 → DataFrame，便于画图
yearly = raw.get("年度统计", {})
df_year = pd.DataFrame(yearly).T  # 行=年份，列=最高温平均、最低温平均、平均温度
df_year.plot(kind="bar", title="各年度平均温度"); plt.savefig("yearly_temps.png")
```

其他嵌套块（如「极端高温」「日温差超过30°C的日期」）也可用 `pd.json_normalize()` 或手写列表转 DataFrame，再做描述统计或可视化。

### 方式 B：把 API 原始数据转成 DataFrame 再分析

若希望用 pandas 做时间序列或与技能流程完全一致，可在脚本中或单独脚本里把 API 返回的 `daily` 转成表再分析：

```python
# 在取得 data = fetch_weather_data(...) 之后
daily = data.get("daily", {})
df = pd.DataFrame({
    "date": pd.to_datetime(daily["time"]),
    "temp_max": daily["temperature_2m_max"],
    "temp_min": daily["temperature_2m_min"],
    "temp_mean": daily["temperature_2m_mean"]
})
# 之后按技能流程：df.describe()、按年/月 groupby、画图、输出报告
```

这样既保留现有脚本的「获取 + 基础统计」逻辑，又用同一份数据走「概览 → 质量 → 统计 → 可视化 → 报告」的通用流程。

### 方式 C：扩展现有脚本而不改主流程

- 在 `weather_analysis.py` 中 `analyze_weather_data()` 返回的 `analysis` 已可写入 JSON。
- 需要图表时：在脚本末尾或单独脚本中读取 `weather_analysis_result.json`，按方式 A 用 pandas/matplotlib 画图并保存。
- 需要 Markdown 报告时：用 SKILL.md 中的报告模板，把 `analysis` 里的关键字段填进去（数据概览、各统计块、结论），可手写或由 Agent 生成。

---

## 与 data_mining_analysis.py 的对接

`data_mining_analysis.py` 在天气数据上做了季节性、趋势等价值挖掘。对接方式类似：

- **结果 JSON**：若脚本输出结构化 JSON，同样用 `pd.read_json()` 或 `json.load` + `pd.json_normalize()` 转成 DataFrame，再做交叉分析或可视化。
- **统一报告**：把「基础统计」（如 weather_analysis）和「挖掘结果」（如 data_mining）合并进同一份 Markdown 报告的不同小节，保持 SKILL.md 中的报告结构即可。

---

## 小结

| 场景 | 做法 |
|------|------|
| 纯本地文件分析 | 按 SKILL.md 五步流程 + 本页示例工作流 |
| 已有脚本出 JSON | 读 JSON → 转 DataFrame → 技能中的统计/可视化/报告 |
| 已有脚本出 API 原始数据 | 把 daily 等转成 DataFrame → 走完整技能流程 |
| 扩展脚本功能 | 不改主逻辑，另写脚本读结果做图表或报告 |
