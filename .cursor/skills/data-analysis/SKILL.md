---
name: data-analysis
description: Load, explore, clean, and analyze tabular or time-series data; generate summaries and visualizations. Use when the user works with CSV, Excel, JSON, or similar data, asks for data exploration, cleaning, statistics, charts, or analysis reports.
---

# Data Analysis

## 何时使用

- 用户提供数据文件（CSV、Excel、JSON 等）并要求分析、统计或可视化
- 用户要求探索数据、清洗数据、做描述统计或生成分析报告
- 用户提到「数据分析」「数据探索」「可视化」「统计」等

## 通用流程

按需执行，不必全部：

```
进度：
- [ ] 1. 加载与概览
- [ ] 2. 质量与缺失检查
- [ ] 3. 描述统计与分布
- [ ] 4. 可视化（如需要）
- [ ] 5. 结论与输出（报告/文件）
```

---

## 1. 加载与概览

**CSV**
```python
import pandas as pd
df = pd.read_csv("file.csv", encoding="utf-8")  # 中文用 utf-8，乱码可试 gbk
```

**Excel**
```python
df = pd.read_excel("file.xlsx", sheet_name=0)  # 第一张表，或写表名
```

**JSON（行式或列表）**
```python
df = pd.read_json("file.json")  # 或 pd.json_normalize(data) 展平嵌套
```

**概览**
```python
df.shape
df.dtypes
df.head()
df.info()
df.describe(include="all")  # 含非数值列
```

---

## 2. 质量与缺失

- 缺失：`df.isna().sum()`、`df.dropna()` / `df.fillna(...)`
- 重复：`df.duplicated().sum()`、`df.drop_duplicates()`
- 异常/类型：检查数值列范围、日期格式、对象列唯一值

---

## 3. 描述统计与分组

- 数值：`df.describe()`、分位数、方差
- 分类：`df["col"].value_counts()`、占比
- 分组：`df.groupby("col").agg(...)`、`pd.crosstab()`
- 时间序列：先 `pd.to_datetime(df["date"])`，再按日/月/年聚合

---

## 4. 可视化

优先用 matplotlib 或 seaborn，保持简洁。

- 分布：`df["col"].hist()` 或 `sns.histplot()`
- 关系：`sns.scatterplot()`、`sns.lineplot()`
- 分类对比：`sns.barplot()`、`sns.boxplot()`
- 多变量：`sns.pairplot()` 或子图

**约定**：中文标签用 `plt.rcParams["font.sans-serif"] = ["SimHei"]` 或系统已有中文字体；图标题、轴标签要清晰。

---

## 5. 输出与报告

**保存结果**
- 表格：`df.to_csv("result.csv", index=False, encoding="utf-8-sig")` 或 `to_excel()`
- 图表：`plt.savefig("fig.png", dpi=150, bbox_inches="tight")`

**分析报告结构（Markdown）**

```markdown
# [数据集/主题] 分析报告

## 1. 数据概览
- 行/列数、主要列类型、时间范围（如有）

## 2. 数据质量
- 缺失、重复、异常值简要说明

## 3. 主要发现
- 描述统计要点
- 关键趋势或分组差异

## 4. 可视化说明
- 每张图的结论（可附图或文件名）

## 5. 结论与建议
- 3–5 条可操作结论
```

---

## 工具选择

| 场景           | 推荐           |
|----------------|----------------|
| 表格分析       | pandas         |
| 统计/假设检验  | scipy.stats    |
| 绘图           | matplotlib, seaborn |
| 大数据/内存紧  | 分块 `read_csv(chunksize=...)` 或 dask |

---

## 示例工作流与现有脚本对接

**示例工作流**：从 CSV 加载 → 质量检查 → 分组/时间聚合 → 可选可视化 → 按「分析报告结构」写 Markdown 并保存结果文件。完整步骤见 [examples.md](examples.md)。

**与现有脚本（如 weather_analysis.py）对接**：

- 脚本已产出 **JSON**（如 `weather_analysis_result.json`）：用 `json.load` 读入，将需要的块（如「年度统计」）转为 `pd.DataFrame`，再按本技能做描述统计、画图、写报告。
- 脚本产出 **API 原始数据**：把 `daily` 等列表转成 DataFrame（`date`、`temp_max`、`temp_min` 等列），然后走「概览 → 质量 → 统计 → 可视化 → 报告」流程。
- 只扩展不改主流程：在脚本外另写脚本读取结果 JSON，用 pandas/matplotlib 做图表或生成 Markdown 报告。

更多对接方式和代码片段见 [examples.md](examples.md)。

---

## 注意事项

- 大文件先看前几行和 `df.info()`，再决定是否全量读入或分块。
- 时间列统一用 `pd.to_datetime()`，时区需要时再处理。
- 列名含空格或特殊字符时，可先 `df.columns = df.columns.str.strip()` 或重命名。
- 与用户确认：分析目标、关键指标、报告受众，以便控制深度和表述。
