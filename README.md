# 黄凯研｜数据与 AI 应用作品集

首都经济贸易大学数据科学与大数据技术本科生，求职方向为数据分析、机器学习、AI 数据与评测、数据工程实习。

这个仓库用于集中展示 AI 数据评测、算法建模和数据工程方向的代表项目。涉及平台内部任务的内容只保留脱敏的案例说明，不上传原始题面、隐藏测试或平台内部材料。

## 代表项目

| 项目 | 核心工作 | 关键结果 |
| --- | --- | --- |
| [Crash-Safe SQLite Job Queue](projects/crash-safe-sqlite-job-queue/) | 设计可由测试自动验证的并发任务队列工程任务 | 覆盖并发领取、租约 fencing、崩溃恢复、幂等副作用和数据库迁移等 8 类风险 |
| [Near-Duplicate Corpus Audit](projects/near-duplicate-audit/) | 设计近重复语料检测 CLI 任务与多级验证方案 | 用 oracle、空操作和两类错误解验证评分区分度；开发机 5,000 文档约 0.39 秒 |
| [IMDB 电影评论情感分析](projects/imdb-sentiment-analysis/) | 比较词袋法与 TF-IDF；Multinomial Naive Bayes 分类；ROC-AUC 与 5 折验证 | 测试集准确率 83.71%，AUC 0.9180；5 折准确率 81.36% ± 0.69% |
| [个贷违约数据清洗与特征工程](projects/credit-risk-preprocessing/) | 缺失值、类别编码、偏度变换、截尾与共线性处理 | 10,000 × 29 原始表；固定 80% 样本；交付 51 列数据与统计摘要 |
| [FoodMart 销售数据仓库 ETL](projects/foodmart-etl/) | SQL Server / SSIS；6 个维度表 + 1 个销售事实表；派生利润 | 合并三年销售记录，校验事实表行数与 gain = sales - cost |

## 技术栈

Python · Pandas · NumPy · scikit-learn · SQL · R · SQL Server · SSIS · Git · Linux · pytest

## 项目约定

- 不提交受许可限制或无再分发权限的原始数据。
- Talents AI 相关工作仅以脱敏案例展示，不公开平台题面、隐藏测试和评分资产。
- 代码中的本地路径、数据库连接、机器名均使用占位符。
- 指标来自对应实验记录，同时报告交叉验证或验证步骤，避免只展示单次结果。
- 每个项目目录包含独立 README；数据和输出目录默认不进入版本库。

## 联系方式

- GitHub: [Khalil-Duang](https://github.com/Khalil-Duang)
- Email: 18810085776@163.com

