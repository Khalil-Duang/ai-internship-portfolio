# IMDB 电影评论情感分析

## 项目目标

比较词袋法（Bag of Words）与 TF-IDF 两种文本表示对朴素贝叶斯分类器的影响，并使用准确率、Macro F1、混淆矩阵、ROC-AUC 和 5 折交叉验证评估模型。

## 数据

使用 IMDB Large Movie Review Dataset 的 `aclImdb/train` 和 `aclImdb/test` 目录。数据集不包含在仓库中，请从原数据源获取。

## 方法

1. 读取 `pos/neg` 两个类别目录，清理 `<br />` 等排版噪声。
2. 以 `CountVectorizer` 和 `TfidfVectorizer` 构造最多 20,000 维特征。
3. 使用 `MultinomialNB(alpha=1.0)` 训练。
4. 在独立测试集上计算 Accuracy、Macro Precision/Recall/F1、Confusion Matrix、ROC-AUC。
5. 在训练集上补充 5 折交叉验证，比较单次留出结果与稳定性。

## 结果

| 方案 | Accuracy | Macro F1 | AUC | 5 折 Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 词袋法 + NB | 0.8244 | 0.8238 | 0.8976 | 0.7910 ± 0.0096 |
| TF-IDF + NB | 0.8371 | 0.8368 | 0.9180 | 0.8136 ± 0.0069 |

TF-IDF 相比词袋法将测试集准确率提高 1.27 个百分点，AUC 提高 0.0204。5 折结果低于单次测试结果，说明结果会受数据划分与方差影响。

## 运行

```bash
python -m pip install -r requirements.txt
python train_and_evaluate.py --data data/aclImdb --out results
```

## 我的复盘

- 结果不能只报一个漂亮的测试集数字；同时报告交叉验证均值与标准差，才能说明稳定性。
- 文本表示会显著影响模型表现，模型名称本身并不能代表最终效果。
- 项目应把配置、指标和图表一起保存，便于复现与解释。

