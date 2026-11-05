# -*- coding: utf-8 -*-
"""
电影评论情感分析 —— 词袋法 / TF-IDF  +  朴素贝叶斯
====================================================================
作业要求
--------
1. 先用「词袋法(Bag of Words)」对电影评论做特征提取, 再建立朴素贝叶斯模型对
   评论分类, 最后用混淆矩阵、precision、recall、ROC 曲线、AUC 值评估分类效果。
2. 先用「TF-IDF」对电影评论做特征提取, 再建立朴素贝叶斯模型对评论分类,
   同样用混淆矩阵、precision、recall、ROC 曲线、AUC 值评估分类效果。
3. 提交内容: 代码 + PDF 报告。

数据集
------
IMDB Large Movie Review Dataset (aclImdb):
    data/aclImdb/train/{pos,neg}  25000 条训练评论
    data/aclImdb/test/{pos,neg}   25000 条测试评论
标签: 0 = neg(负面), 1 = pos(正面), 正负样本各占 50%。

运行方式
--------
    python movie_sentiment_naive_bayes.py
    python movie_sentiment_naive_bayes.py --data data/aclImdb --out results

输出
----
    <out>/summary.json          两个实验的全部评估指标
    <out>/report.txt            可读的文本版结果
    <out>/figures/*.png         混淆矩阵、ROC 曲线等图片
    <out>/cross_validation.json 5 折交叉验证结果(模型稳定性)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # 无窗口环境也要能出图
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ----------------------------------------------------------------------------
# 0. 绘图风格(保证中文标签正常显示)
# ----------------------------------------------------------------------------
def setup_plot_style() -> None:
    """优先使用系统中文字体, 避免图中中文变成方框。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 170
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25


# ----------------------------------------------------------------------------
# 1. 数据读取与预处理
# ----------------------------------------------------------------------------
def load_reviews(root: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """读取 aclImdb 目录下的 pos / neg 评论文本。

    load_files 会自动把子文件夹名(pos, neg)当作类别名, 并按字母序编号:
        neg -> 0,  pos -> 1
    返回 (文本列表, 标签数组, 类别名列表)。
    """
    bunch = load_files(root, encoding="utf-8", shuffle=False)
    class_names = list(bunch.target_names)
    # IMDB 评论里用 <br /> 表示换行, 属于排版噪声, 统一替换成空格
    texts = [re.sub(r"<br\s*/?>", " ", doc) for doc in bunch.data]
    labels = np.asarray(bunch.target, dtype=int)
    return texts, labels, class_names


# ----------------------------------------------------------------------------
# 2. 评估: 混淆矩阵 / precision / recall / ROC / AUC
# ----------------------------------------------------------------------------
def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
) -> Dict:
    """计算作业要求的全部指标, 返回可直接写进 json 的字典。"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # ROC 曲线: 用正面类(标签 1)的预测概率作为打分
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    auc = roc_auc_score(y_true, y_score)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in (0, 1)
        },
        "macro_avg": {
            "precision": float(prec_macro),
            "recall": float(rec_macro),
            "f1": float(f1_macro),
        },
        "weighted_avg": {
            "precision": float(prec_w),
            "recall": float(rec_w),
            "f1": float(f1_w),
        },
        "auc": float(auc),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=class_names, digits=4
        ),
    }


# ----------------------------------------------------------------------------
# 3. 画图
# ----------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, class_names: List[str], title: str, path: str) -> None:
    """左侧原始计数, 右侧按行归一化(百分比), 更直观地看误判情况。"""
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    for ax, mat, fmt, subtitle in (
        (axes[0], cm, ".0f", "混淆矩阵(样本数)"),
        (axes[1], cm_norm, ".3f", "混淆矩阵(按真实类别归一化)"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() if fmt == ".0f" else 1.0)
        ax.set_xticks([0, 1], class_names)
        ax.set_yticks([0, 1], class_names)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(subtitle)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                value = mat[i, j]
                ax.text(
                    j, i, format(value, fmt),
                    ha="center", va="center", fontsize=12,
                    color="white" if value > mat.max() * 0.55 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(
    results: Dict[str, Dict], path: str, title: str = "ROC 曲线与 AUC 值"
) -> None:
    """把若干模型的 ROC 曲线画在同一张图上便于比较。"""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for (name, res), color in zip(results.items(), colors):
        fpr = np.asarray(res["roc"]["fpr"])
        tpr = np.asarray(res["roc"]["tpr"])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {res['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="随机猜测 (AUC = 0.5000)")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("假正例率 FPR")
    ax.set_ylabel("真正例率 TPR")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metric_comparison(results: Dict[str, Dict], path: str) -> None:
    """把两种特征提取方式的 准确率 / precision / recall / F1 / AUC 放在一起对比。"""
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels = ["准确率", "precision", "recall", "F1", "AUC"]
    names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for k, name in enumerate(names):
        res = results[name]
        values = [
            res["accuracy"],
            res["macro_avg"]["precision"],
            res["macro_avg"]["recall"],
            res["macro_avg"]["f1"],
            res["auc"],
        ]
        bars = ax.bar(x + (k - 0.5) * width, values, width, label=name)
        for rect, v in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2, v + 0.008, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8.5,
            )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("指标取值")
    ax.set_title("词袋法 vs TF-IDF: 朴素贝叶斯分类效果对比")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_top_words(
    model: MultinomialNB, vectorizer, class_names: List[str], path: str, top_n: int = 15
) -> None:
    """朴素贝叶斯学到的"最能代表正/负面"的词: log P(词|类别) 之差排序。"""
    vocab = np.array(vectorizer.get_feature_names_out())
    diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    pos_idx = np.argsort(diff)[-top_n:][::-1]
    neg_idx = np.argsort(diff)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, idx, color, label in (
        (axes[0], pos_idx, "#2ca02c", f"更倾向 {class_names[1]} (正面)"),
        (axes[1], neg_idx, "#d62728", f"更倾向 {class_names[0]} (负面)"),
    ):
        words = vocab[idx][::-1]
        scores = diff[idx][::-1]
        ax.barh(words, scores, color=color, alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel("log P(词|正) - log P(词|负)")
        ax.tick_params(labelsize=8.5)
    fig.suptitle("朴素贝叶斯模型权重最高的特征词(TF-IDF 特征空间)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. 两个实验
# ----------------------------------------------------------------------------
def run_experiment(
    name: str,
    vectorizer,
    text_train: List[str],
    y_train: np.ndarray,
    text_test: List[str],
    y_test: np.ndarray,
    class_names: List[str],
    out_dir: str,
    alpha: float = 1.0,
) -> Dict:
    """特征提取 -> 训练朴素贝叶斯 -> 评估 -> 出图。"""
    print(f"\n===== 实验: {name} =====")
    t0 = time.time()

    # (1) 特征提取: 只在训练集上 fit, 再 transform 测试集, 避免信息泄漏
    x_train = vectorizer.fit_transform(text_train)
    x_test = vectorizer.transform(text_test)
    print(
        f"特征矩阵: train {x_train.shape}, test {x_test.shape}, "
        f"非零元素占比 {x_train.nnz / (x_train.shape[0] * x_train.shape[1]):.4%}"
    )

    # (2) 朴素贝叶斯分类器 (多项式模型 + 拉普拉斯平滑)
    model = MultinomialNB(alpha=alpha)
    model.fit(x_train, y_train)

    # (3) 预测: 类别 + 正面类概率(用于画 ROC 曲线)
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    # (4) 评估
    res = evaluate(y_test, y_pred, y_score, class_names)
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    res["n_features"] = int(x_train.shape[1])
    res["alpha"] = alpha
    res["vectorizer"] = type(vectorizer).__name__

    fig_dir = os.path.join(out_dir, "figures")
    tag = "bow" if "Count" in type(vectorizer).__name__ else "tfidf"
    plot_confusion(
        np.asarray(res["confusion_matrix"]),
        class_names,
        f"{name}: 混淆矩阵",
        os.path.join(fig_dir, f"{tag}_confusion_matrix.png"),
    )
    plot_roc({name: res}, os.path.join(fig_dir, f"{tag}_roc_curve.png"), f"{name}: ROC 曲线")
    if isinstance(model, MultinomialNB) and hasattr(model, "feature_log_prob_"):
        plot_top_words(
            model, vectorizer, class_names,
            os.path.join(fig_dir, f"{tag}_top_words.png"),
        )

    print(res["classification_report"])
    print(f"AUC = {res['auc']:.4f}, 用时 {res['elapsed_seconds']}s")
    return res


def cross_validate(name: str, vectorizer, texts, labels, cv: int = 5) -> Dict:
    """5 折交叉验证, 检查模型在训练集内部的稳定性(选用)。"""
    print(f"\n----- {name}: {cv} 折交叉验证 -----")
    pipe = Pipeline([("vec", vectorizer), ("nb", MultinomialNB(alpha=1.0))])
    acc = cross_val_score(pipe, texts, labels, cv=cv, scoring="accuracy", n_jobs=-1)
    auc = cross_val_score(pipe, texts, labels, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"准确率 {acc.mean():.4f} ± {acc.std():.4f} | AUC {auc.mean():.4f} ± {auc.std():.4f}")
    return {
        "cv_folds": cv,
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "accuracy_per_fold": acc.tolist(),
        "auc_mean": float(auc.mean()),
        "auc_std": float(auc.std()),
        "auc_per_fold": auc.tolist(),
    }


# ----------------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="IMDB 电影评论情感分析: 词袋法 / TF-IDF + 朴素贝叶斯")
    parser.add_argument("--data", default="data/aclImdb", help="aclImdb 数据集根目录")
    parser.add_argument("--out", default="results", help="结果输出目录")
    parser.add_argument("--max-features", type=int, default=20000, help="词表大小上限")
    parser.add_argument("--alpha", type=float, default=1.0, help="朴素贝叶斯拉普拉斯平滑系数")
    parser.add_argument("--no-cv", action="store_true", help="跳过 5 折交叉验证")
    args = parser.parse_args()

    setup_plot_style()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    # ---- 读取数据 ----
    print("读取训练集 ...")
    text_train, y_train, class_names = load_reviews(os.path.join(args.data, "train"))
    print("读取测试集 ...")
    text_test, y_test, _ = load_reviews(os.path.join(args.data, "test"))
    print(f"训练样本 {len(text_train)} 条, 测试样本 {len(text_test)} 条, 类别 {class_names}")
    print(f"训练集类别分布 {np.bincount(y_train)}, 测试集类别分布 {np.bincount(y_test)}")

    # ---- 实验一: 词袋法 (CountVectorizer) ----
    bow_vectorizer = CountVectorizer(
        stop_words="english",   # 去掉英文停用词
        max_features=args.max_features,
        min_df=2,               # 至少在 2 篇评论中出现过才保留
        ngram_range=(1, 1),     # 一元词袋
    )
    bow_res = run_experiment(
        "词袋法 + 朴素贝叶斯", bow_vectorizer,
        text_train, y_train, text_test, y_test, class_names, args.out, alpha=args.alpha,
    )

    # ---- 实验二: TF-IDF ----
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        min_df=2,
        sublinear_tf=True,      # 用 1+log(tf) 压缩高频词
        ngram_range=(1, 1),
    )
    tfidf_res = run_experiment(
        "TF-IDF + 朴素贝叶斯", tfidf_vectorizer,
        text_train, y_train, text_test, y_test, class_names, args.out, alpha=args.alpha,
    )

    # ---- 汇总图 ----
    all_res = {"词袋法 + 朴素贝叶斯": bow_res, "TF-IDF + 朴素贝叶斯": tfidf_res}
    plot_roc(all_res, os.path.join(args.out, "figures", "roc_comparison.png"))
    plot_metric_comparison(all_res, os.path.join(args.out, "figures", "metrics_comparison.png"))

    # ---- 交叉验证 ----
    cv_result = {}
    if not args.no_cv:
        cfg = {"stop_words": "english", "max_features": args.max_features, "min_df": 2}
        cv_result["词袋法 + 朴素贝叶斯"] = cross_validate(
            "词袋法", CountVectorizer(**cfg), text_train, y_train
        )
        cv_result["TF-IDF + 朴素贝叶斯"] = cross_validate(
            "TF-IDF", TfidfVectorizer(sublinear_tf=True, **cfg), text_train, y_train
        )
        with open(os.path.join(args.out, "cross_validation.json"), "w", encoding="utf-8") as f:
            json.dump(cv_result, f, ensure_ascii=False, indent=2)

    # ---- 保存结果 ----
    summary = {
        "dataset": os.path.abspath(args.data),
        "n_train": len(text_train),
        "n_test": len(text_test),
        "class_names": class_names,
        "experiments": all_res,
        "cross_validation": cv_result,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["IMDB 电影评论情感分析 —— 实验结果", "=" * 60]
    for name, res in all_res.items():
        lines += [
            f"\n【{name}】",
            f"特征维数: {res['n_features']}  平滑系数 alpha={res['alpha']}",
            f"准确率 accuracy : {res['accuracy']:.4f}",
            f"宏平均 precision: {res['macro_avg']['precision']:.4f}",
            f"宏平均 recall   : {res['macro_avg']['recall']:.4f}",
            f"宏平均 F1       : {res['macro_avg']['f1']:.4f}",
            f"AUC             : {res['auc']:.4f}",
            f"混淆矩阵: {res['confusion_matrix']}",
            "",
            res["classification_report"],
        ]
        if name in cv_result:
            cv = cv_result[name]
            lines.append(
                f"5 折交叉验证: 准确率 {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}, "
                f"AUC {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}"
            )
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n全部结果已保存到:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
