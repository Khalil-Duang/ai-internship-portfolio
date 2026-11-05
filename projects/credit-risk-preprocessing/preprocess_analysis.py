# -*- coding: utf-8 -*-
"""
第一次作业: 描述与预处理
数据集: CCF BDCI 个贷违约数据集 (CCF2021-2026秋用xls.xls)

流程:
    原始数据 -> 随机抽取 80% 得到 data0
    data0 -> 缺失值处理      -> data1
    data1 -> 定性变量处理    -> data2
    data2 -> 定量变量处理    -> data3

输出:
    data0.csv ~ data3.csv   各阶段数据
    figures/*.png           报告用图
    stats.json              报告需要的全部数字

运行: python preprocess_analysis.py --xls 原始数据.xls --out 输出目录
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats as sps

plt.rcParams["axes.unicode_minus"] = False
for _f in ("Microsoft YaHei", "SimHei"):
    if any(x.name == _f for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

SEED = 2026

# 原始数据中与本问题无关的字段: 标识类、日期类, 以及数据说明中标注删除的字段
DROP_COLS = [
    "loan_id",             # 贷款编号, 唯一标识
    "censor_status",       # 数据说明: 不需要管理, 可删除
    "app_type",            # 数据说明: 删除
    "f1",                  # 数据说明: 删除(且缺失 8.58%)
    "issue_date",          # 放款日期
    "earlies_credit_mon",  # 最早信用月份(日期)
    "user_id",             # 用户编号
    "post_code",           # 邮编
    "region",              # 地区编码
]

# 变量分组
CAT_COLS = [
    "year_of_loan", "employer_type", "industry", "house_exist", "use",
    "initial_list_status", "title", "del_in_18month", "known_dero",
    "pub_dero_bankrup", "early_return", "class", "work_year",
]
NUM_COLS = [
    "total_loan", "interest", "monthly_payment", "debt_loan_ratio",
    "scoring_low", "scoring_high", "known_outstanding_loan", "recircle_b",
    "recircle_u", "early_return_amount", "early_return_amount_3mon",
    "f0", "f2", "f3", "f4",
]
TARGET = "y"


def load_raw(xls_path: str) -> pd.DataFrame:
    """读取原始 Excel 表, 去掉空列与无关字段。"""
    df = pd.read_excel(xls_path, sheet_name="CCF个贷违约train")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    print("原始数据:", df.shape)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print("删除标识/日期/说明中要求删除的字段后:", df.shape)
    return df


# ---------------------------------------------------------------- 工具函数
def freq_table(s: pd.Series) -> pd.DataFrame:
    """频数分布表: 取值、频数、百分比(缺失单独一行)。"""
    vc = s.value_counts(dropna=False)
    out = pd.DataFrame({
        "取值": [("缺失" if pd.isna(k) else k) for k in vc.index],
        "频数": vc.values,
        "百分比": [round(100 * v / len(s), 2) for v in vc.values],
    })
    return out.reset_index(drop=True)


def col_summary(s: pd.Series) -> dict:
    """定性变量的分布概况: 缺失率、类别数、最大/最小类占比。"""
    n = len(s)
    miss = int(s.isna().sum())
    vc = s.value_counts(dropna=True)
    shares = vc / n
    return {
        "n_unique": int(vc.shape[0]),
        "missing": miss,
        "missing_pct": round(100 * miss / n, 2),
        "max_share": round(100 * float(shares.max()), 2),
        "min_share": round(100 * float(shares.min()), 2),
        "top": [[str(k), int(v)] for k, v in vc.head(5).items()],
    }


def num_summary(s: pd.Series) -> dict:
    """定量变量的描述统计量。"""
    v = pd.to_numeric(s, errors="coerce")
    return {
        "n": int(v.notna().sum()),
        "missing": int(v.isna().sum()),
        "missing_pct": round(100 * float(v.isna().mean()), 2),
        "mean": round(float(v.mean()), 3),
        "std": round(float(v.std()), 3),
        "min": round(float(v.min()), 3),
        "p25": round(float(v.quantile(0.25)), 3),
        "median": round(float(v.median()), 3),
        "p75": round(float(v.quantile(0.75)), 3),
        "max": round(float(v.max()), 3),
        "skew": round(float(sps.skew(v.dropna())), 3),
        "kurtosis": round(float(sps.kurtosis(v.dropna())), 3),
    }


def merge_low_freq(s: pd.Series, threshold: float = 0.03, other: str = "其他") -> pd.Series:
    """占比低于阈值的类别合并为"其他"。"""
    share = s.value_counts(normalize=True)
    keep = share[share >= threshold].index
    return s.where(s.isin(keep), other)


def map_values(s: pd.Series, mapping: dict, rest_label: str) -> pd.Series:
    """按取值映射合并: 命中 mapping 的用映射值, 其余非空值归入 rest_label。"""
    out = s.map(lambda v: mapping.get(v) if (not pd.isna(v) and v in mapping) else
                (None if pd.isna(v) else rest_label))
    return out


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", default="data/raw.xls")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    S: dict = {"dataset": os.path.basename(args.xls)}

    # ======================= 1 研究问题与数据 =======================
    df = load_raw(args.xls)
    S["raw_shape"] = list(df.shape)
    S["n_raw"] = int(df.shape[0])
    S["target"] = TARGET
    S["cat_cols"] = CAT_COLS
    S["num_cols"] = NUM_COLS
    S["dropped_cols"] = DROP_COLS
    S["y_rate_raw"] = round(100 * float(df[TARGET].mean()), 2)

    # 随机抽取 80% 作为 data0
    data0 = df.sample(frac=0.8, random_state=SEED).reset_index(drop=True)
    S["seed"] = SEED
    S["n_data0"] = int(data0.shape[0])
    S["y_rate_data0"] = round(100 * float(data0[TARGET].mean()), 2)
    S["y_counts_data0"] = [int(v) for v in data0[TARGET].value_counts().sort_index().values]
    print("\ndata0: {}, 违约率 {}%".format(data0.shape, S["y_rate_data0"]))

    # 变量清单
    var_list = []
    for c in CAT_COLS:
        s = col_summary(data0[c])
        var_list.append({
            "name": c, "type": "定性", "n_unique": s["n_unique"],
            "missing": s["missing"], "missing_pct": s["missing_pct"],
            "max_share": s["max_share"], "min_share": s["min_share"],
        })
    for c in NUM_COLS:
        s = num_summary(data0[c])
        var_list.append({
            "name": c, "type": "定量", "n_unique": None,
            "missing": s["missing"], "missing_pct": s["missing_pct"],
            "mean": s["mean"], "median": s["median"], "skew": s["skew"],
        })
    S["var_list"] = var_list

    # ======================= 2 数据描述 =======================
    # 2.1 因变量分布
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    cnt = data0[TARGET].value_counts().sort_index()
    bars = ax.bar(["未违约 (y=0)", "违约 (y=1)"], cnt.values,
                  color=["#4C78A8", "#E45756"], width=0.55)
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 80,
                "{}\n({:.1f}%)".format(v, 100 * v / len(data0)),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, cnt.values.max() * 1.22)
    ax.set_ylabel("样本数")
    ax.set_title("data0 因变量 y 的分布")
    fig.savefig(os.path.join(fig_dir, "fig1_y_dist.png"))
    plt.close(fig)

    # 2.2 定性变量分布
    cat_summary = {c: col_summary(data0[c]) for c in CAT_COLS}
    S["cat_summary"] = cat_summary
    S["cat_freq_tables"] = {c: freq_table(data0[c]).to_dict("records") for c in CAT_COLS}

    show = ["employer_type", "industry", "class", "work_year", "use", "house_exist"]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5))
    for ax, c in zip(axes.ravel(), show):
        vc = data0[c].value_counts().sort_values(ascending=True)
        labels = [("缺失" if pd.isna(k) else str(k)) for k in vc.index]
        labels = [(l if len(l) <= 22 else l[:20] + "...") for l in labels]
        ax.barh(labels, vc.values, color="#4C78A8", alpha=0.9)
        for i, v in enumerate(vc.values):
            ax.text(v + max(vc.values) * 0.01, i,
                    "{} ({:.1f}%)".format(v, 100 * v / len(data0)), va="center", fontsize=8)
        ax.set_xlim(0, max(vc.values) * 1.30)
        ax.set_title("{} (类别数 {})".format(c, data0[c].nunique(dropna=True)), fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("data0 主要定性变量的频数分布", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_cat_dist.png"))
    plt.close(fig)

    # 高基数变量 title 的分布(前 15 类 + 其他)
    vc = data0["title"].value_counts()
    top15, rest = vc.head(15), vc.iloc[15:].sum()
    x = ["{:g}".format(k) for k in top15.index] + ["其他(893 类)"]
    y = list(top15.values) + [int(rest)]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    b = ax.bar(x, y, color="#72B7B2")
    ax.bar_label(b, labels=["{:.1f}%".format(100 * v / len(data0)) for v in y], fontsize=8)
    ax.set_title("高基数变量 title 的取值分布(前 15 个取值)")
    ax.set_ylabel("样本数")
    ax.set_xlabel("title 取值")
    fig.savefig(os.path.join(fig_dir, "fig3_title_dist.png"))
    plt.close(fig)
    S["title_top15"] = [[str(k), int(v)] for k, v in top15.items()]
    S["title_n_unique"] = int(vc.shape[0])
    S["title_low_freq_share"] = round(100 * float(vc.iloc[15:].sum() / len(data0)), 2)

    # 2.3 定量变量分布
    num_summary_all = {c: num_summary(data0[c]) for c in NUM_COLS}
    S["num_summary"] = num_summary_all
    nrow = int(np.ceil(len(NUM_COLS) / 3))
    fig, axes = plt.subplots(nrow, 3, figsize=(12, 3.1 * nrow))
    for ax, c in zip(axes.ravel(), NUM_COLS):
        v = pd.to_numeric(data0[c], errors="coerce").dropna()
        ax.hist(v, bins=40, color="#4C78A8", alpha=0.85)
        ax.set_title("{}\n偏度 {:.2f}".format(c, sps.skew(v)), fontsize=9.5)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(NUM_COLS):]:
        ax.axis("off")
    fig.suptitle("data0 各定量变量的直方图", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_num_hist.png"))
    plt.close(fig)

    # 箱线图(标准化后同图比较, 便于看异常值)
    z = (data0[NUM_COLS] - data0[NUM_COLS].mean()) / data0[NUM_COLS].std()
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.boxplot([z[c].dropna() for c in NUM_COLS], showfliers=True,
               flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.set_xticks(range(1, len(NUM_COLS) + 1), NUM_COLS)
    ax.axhline(3, color="r", ls="--", lw=0.8)
    ax.axhline(-3, color="r", ls="--", lw=0.8)
    ax.set_ylabel("标准化取值")
    ax.set_title("data0 定量变量箱线图(标准化后, 红色虚线为 ±3 倍标准差)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)
    fig.savefig(os.path.join(fig_dir, "fig5_num_box.png"))
    plt.close(fig)

    # 缺失值总览
    miss = data0.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=True)
    S["missing_raw"] = {k: int(v) for k, v in miss.items()}
    S["missing_raw_pct"] = {k: round(100 * float(v) / len(data0), 2) for k, v in miss.items()}
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    b = ax.barh(list(miss.index), miss.values, color="#E45756", alpha=0.9)
    ax.bar_label(b, labels=["{} ({:.2f}%)".format(v, 100 * v / len(data0)) for v in miss.values],
                 fontsize=8, padding=2)
    ax.set_xlim(0, miss.max() * 1.4)
    ax.set_xlabel("缺失样本数")
    ax.set_title("data0 存在缺失的变量")
    fig.savefig(os.path.join(fig_dir, "fig6_missing.png"))
    plt.close(fig)

    # ======================= 3.1 缺失值处理 =======================
    data1 = data0.copy()
    fill_log = []
    for c in CAT_COLS:
        if data1[c].isna().any():
            mode = data1[c].mode(dropna=True)[0]
            n = int(data1[c].isna().sum())
            data1[c] = data1[c].fillna(mode)
            fill_log.append({"var": c, "type": "定性", "n": n,
                             "pct": round(100 * n / len(data0), 2),
                             "method": "众数填补", "value": str(mode)})
    for c in NUM_COLS:
        if data1[c].isna().any():
            med = pd.to_numeric(data1[c], errors="coerce").median()
            n = int(data1[c].isna().sum())
            data1[c] = pd.to_numeric(data1[c], errors="coerce").fillna(med)
            fill_log.append({"var": c, "type": "定量", "n": n,
                             "pct": round(100 * n / len(data0), 2),
                             "method": "中位数填补", "value": "{:.3f}".format(med)})
    S["fill_log"] = fill_log
    S["missing_after_fill"] = int(data1.isna().sum().sum())
    S["data1_shape"] = list(data1.shape)
    # 填补效果证据: 抽样展示"填补前缺失 -> 填补后取值"
    fill_examples = []
    for item in fill_log:
        c = item["var"]
        miss_idx = data0.index[data0[c].isna()].to_numpy()
        for i in miss_idx[:2]:
            fill_examples.append({
                "var": c, "row": int(i), "before": "缺失(空)", "after": str(data1.at[i, c]),
            })
    S["fill_examples"] = fill_examples[:12]
    print("\ndata1 缺失值合计: {}".format(S["missing_after_fill"]))

    # ======================= 3.2 定性变量处理 =======================
    # 3.2.1 0-1 型变量中少数类占比 < 3% 的删除检查
    check01 = []
    for c in CAT_COLS:
        vals = set(pd.Series(data1[c].dropna().unique()))
        if vals <= {0, 1, 0.0, 1.0}:
            share = data1[c].value_counts(normalize=True)
            check01.append({
                "var": c,
                "minority": round(100 * float(share.min()), 2),
                "majority": round(100 * float(share.max()), 2),
                "action": "删除" if share.min() < 0.03 else "保留",
            })
    S["check_01"] = check01
    S["drop_01"] = [r["var"] for r in check01 if r["action"] == "删除"]
    data2 = data1.drop(columns=S["drop_01"], errors="ignore")

    # 3.2.2 类别合并
    cat_after = data2.copy()
    cat_after["house_exist"] = map_values(
        data2["house_exist"], {0: "0套", 1: "1套", 2: "2套"}, "3套及以上")
    cat_after["del_in_18month"] = map_values(
        data2["del_in_18month"], {0: "0次", 1: "1-2次", 2: "1-2次"}, "3次及以上")
    cat_after["known_dero"] = map_values(
        data2["known_dero"], {0: "0次", 1: "1-2次", 2: "1-2次"}, "3次及以上")
    cat_after["early_return"] = map_values(
        data2["early_return"], {0: "0次", 1: "1次", 2: "2次", 3: "3次"}, "4次及以上")
    cat_after["pub_dero_bankrup"] = map_values(
        data2["pub_dero_bankrup"], {0: "0次", 1: "1次"}, "2次及以上")
    wy_map = {
        "< 1 year": "1年以内", "1 year": "1-3年", "2 years": "1-3年", "3 years": "1-3年",
        "4 years": "4-9年", "5 years": "4-9年", "6 years": "4-9年",
        "7 years": "4-9年", "8 years": "4-9年", "9 years": "4-9年",
        "10+ years": "10年及以上",
    }
    cat_after["work_year"] = data2["work_year"].map(wy_map)
    ind_map = {
        "金融业": "金融与房地产业", "房地产业": "金融与房地产业",
        "公共服务、社会组织": "公共服务与社会组织", "国际组织": "公共服务与社会组织",
        "电力、热力生产供应业": "工业与建筑", "采矿业": "工业与建筑",
        "制造业": "工业与建筑", "建筑业": "工业与建筑",
        "信息传输、软件和信息技术服务业": "信息技术与文化", "文化和体育业": "信息技术与文化",
        "住宿和餐饮业": "批发零售与生活服务", "批发和零售业": "批发零售与生活服务",
        "交通运输、仓储和邮政业": "交通物流与农林牧渔", "农、林、牧、渔业": "交通物流与农林牧渔",
    }
    cat_after["industry"] = data2["industry"].map(ind_map)
    S["industry_map"] = ind_map
    S["work_year_map"] = wy_map
    cat_after["title"] = merge_low_freq(data2["title"], 0.03)
    cat_after["use"] = merge_low_freq(data2["use"], 0.03)
    S["merge_rule"] = ("类别数大于 10 的变量: 取值型有序变量按区间合并; "
                       "名义变量保留占比不小于 3% 的类别, 其余归入“其他”")

    # 3.2.3 数字编码
    cat_final = pd.DataFrame(index=data2.index)
    encode_log = []
    keep_cat = [c for c in CAT_COLS if c not in S["drop_01"]]
    for c in keep_cat:
        s = cat_after[c]
        vals = set(pd.Series(s.dropna().unique()))
        new_name = c + "n"
        if vals <= {0, 1, 0.0, 1.0}:
            cat_final[new_name] = s.astype(int)
            order = [0, 1]
            kind = "0-1 型变量直接编码为 0-1"
        elif c in ("house_exist", "year_of_loan", "class", "work_year",
                   "del_in_18month", "known_dero", "early_return", "pub_dero_bankrup"):
            preset = {
                "house_exist": ["0套", "1套", "2套", "3套及以上"],
                "year_of_loan": [3, 5],
                "class": ["A", "B", "C", "D", "E", "F", "G"],
                "work_year": ["1年以内", "1-3年", "4-9年", "10年及以上"],
                "del_in_18month": ["0次", "1-2次", "3次及以上"],
                "known_dero": ["0次", "1-2次", "3次及以上"],
                "early_return": ["0次", "1次", "2次", "3次", "4次及以上"],
                "pub_dero_bankrup": ["0次", "1次", "2次及以上"],
            }[c]
            order = [v for v in preset if v in vals]
            cat_final[new_name] = s.map({v: i + 1 for i, v in enumerate(order)})
            kind = "有序变量按等级从 1 开始编码"
        else:
            order = list(s.value_counts().index)
            cat_final[new_name] = s.map({v: i + 1 for i, v in enumerate(order)})
            kind = "名义变量按频数降序从 1 开始编码"
        encode_log.append({
            "old": c, "new": new_name, "kind": kind,
            "n_before": int(data2[c].nunique(dropna=True)),
            "n_after": int(s.nunique(dropna=True)),
            "order": [str(v) for v in order][:14],
        })
    S["encode_log"] = encode_log
    S["cat_freq_after"] = {c: freq_table(cat_after[c]).to_dict("records") for c in keep_cat}
    S["merge_compare"] = [
        {"var": c, "before": int(data2[c].nunique(dropna=True)),
         "after": int(cat_after[c].nunique(dropna=True))} for c in keep_cat
    ]

    # 组装 data2: 新变量在前, 旧变量移到后面
    old_cat = [c for c in data2.columns if c in CAT_COLS]
    num_part = [c for c in data2.columns if c in NUM_COLS]
    data2_out = pd.concat([data2[[TARGET]], cat_final, data2[num_part], data2[old_cat]], axis=1)
    S["data2_shape"] = list(data2_out.shape)

    # 合并前后类别数对比图
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    names = [r["var"] for r in S["merge_compare"]]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["before"] for r in S["merge_compare"]], 0.4,
           label="合并前", color="#F58518")
    ax.bar(x + 0.2, [r["after"] for r in S["merge_compare"]], 0.4,
           label="合并后", color="#4C78A8")
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("类别数")
    ax.set_yscale("log")
    ax.set_title("定性变量合并前后的类别数(纵轴为对数刻度)")
    ax.legend()
    fig.savefig(os.path.join(fig_dir, "fig7_merge.png"))
    plt.close(fig)

    # 合并后定性变量分布图
    show2 = ["house_exist", "work_year", "industry", "use", "early_return", "title"]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5))
    for ax, c in zip(axes.ravel(), show2):
        vc = cat_after[c].value_counts()
        labels = [str(k) if len(str(k)) <= 18 else str(k)[:16] + "..." for k in vc.index]
        ax.bar(range(len(vc)), vc.values, color="#72B7B2")
        ax.set_xticks(range(len(vc)), labels, rotation=30, ha="right", fontsize=8)
        for i, v in enumerate(vc.values):
            ax.text(i, v + max(vc.values) * 0.015, "{:.1f}%".format(100 * v / len(cat_after)),
                    ha="center", fontsize=8)
        ax.set_ylim(0, max(vc.values) * 1.18)
        ax.set_title("{} 合并后 (类别数 {})".format(c, len(vc)), fontsize=10)
    fig.suptitle("定性变量合并后的频数分布", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig8_cat_after.png"))
    plt.close(fig)

    # ======================= 3.3 定量变量处理 =======================
    num_df = data2[num_part].astype(float)
    corr = num_df.corr()
    S["corr"] = {c: {d: round(float(corr.loc[c, d]), 3) for d in corr.columns} for c in corr.index}
    cols = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            if abs(r) >= 0.7:
                pairs.append({"a": cols[i], "b": cols[j], "r": round(r, 3)})
    pairs.sort(key=lambda d: -abs(d["r"]))
    S["high_corr"] = [p for p in pairs if abs(p["r"]) >= 0.85]
    S["watch_corr"] = [p for p in pairs if abs(p["r"]) < 0.85]
    corr_y = {c: round(float(np.corrcoef(num_df[c], data2[TARGET])[0, 1]), 4) for c in num_df.columns}
    S["corr_y"] = corr_y

    drop_collinear = []
    for p in S["high_corr"]:
        a, b = p["a"], p["b"]
        if a in drop_collinear or b in drop_collinear:
            continue
        keep, drop = (a, b) if abs(corr_y[a]) >= abs(corr_y[b]) else (b, a)
        if abs(abs(corr_y[a]) - abs(corr_y[b])) < 0.02:
            keep, drop = ((a, b) if num_summary_all[a]["skew"] <= num_summary_all[b]["skew"]
                          else (b, a))
        drop_collinear.append(drop)
    S["drop_collinear"] = drop_collinear
    kept_num = [c for c in num_part if c not in drop_collinear]
    print("\n共线性剔除:", drop_collinear)

    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if abs(v) >= 0.5:
                ax.text(j, i, "{:.2f}".format(v), ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.75 else "black")
    ax.grid(False)
    ax.set_title("定量自变量相关系数矩阵")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(fig_dir, "fig9_corr.png"))
    plt.close(fig)

    # 3.3.2 偏度与数据转换
    def winsor(v: pd.Series) -> pd.Series:
        """极端值替换: 用 1% 与 99% 分位数替换超过该范围的取值。"""
        return v.clip(v.quantile(0.01), v.quantile(0.99))

    transform_log = []
    num_final = pd.DataFrame(index=num_df.index)
    for c in kept_num:
        v = num_df[c].copy()
        skew0 = float(sps.skew(v))
        nonneg = bool((v >= 0).all())
        candidates = {"不转换": v}
        if nonneg:
            candidates["平方根变换"] = np.sqrt(v)
            candidates["对数变换 ln(1+x)"] = np.log1p(v)
        candidates["极端值截断(1%/99%)"] = winsor(v)
        if nonneg:
            candidates["平方根 + 极端值截断"] = winsor(np.sqrt(v))
            candidates["对数变换 + 极端值截断"] = winsor(np.log1p(v))
        skews = {k: float(sps.skew(x)) for k, x in candidates.items()}
        method = min(skews, key=lambda k: abs(skews[k])) if abs(skew0) > 1 else "不转换"
        v2 = candidates[method]
        new_name = c if method == "不转换" else c + "n"
        num_final[new_name] = v2
        transform_log.append({
            "var": c, "new": new_name, "method": method,
            "skew_before": round(skew0, 3), "skew_after": round(float(sps.skew(v2)), 3),
            "candidates": {k: round(x, 3) for k, x in skews.items()},
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
            "mean": round(float(v.mean()), 3), "median": round(float(v.median()), 3),
        })
    S["transform_log"] = transform_log
    S["kept_num"] = kept_num

    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    names = [r["var"] for r in transform_log]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["skew_before"] for r in transform_log], 0.4, label="处理前", color="#E45756")
    ax.bar(x + 0.2, [r["skew_after"] for r in transform_log], 0.4, label="处理后", color="#4C78A8")
    ax.axhline(1, color="gray", ls="--", lw=0.8)
    ax.axhline(-1, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("偏度")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("定量变量处理前后的偏度(灰线为 ±1)")
    ax.legend()
    fig.savefig(os.path.join(fig_dir, "fig10_skew.png"))
    plt.close(fig)

    plot_cols = [r["new"] for r in transform_log]
    nrow = int(np.ceil(len(plot_cols) / 3))
    fig, axes = plt.subplots(nrow, 3, figsize=(12, 3.1 * nrow))
    for ax, c in zip(axes.ravel(), plot_cols):
        v = num_final[c]
        ax.hist(v, bins=40, color="#54A24B", alpha=0.85)
        ax.set_title("{}\n偏度 {:.2f}".format(c, sps.skew(v)), fontsize=9.5)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(plot_cols):]:
        ax.axis("off")
    fig.suptitle("转换后各定量变量的直方图", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig11_num_after.png"))
    plt.close(fig)

    # data3: 新变量在前, 旧变量一律移到后面
    data3 = pd.concat(
        [data2_out[[TARGET]], cat_final, num_final, data2_out[old_cat], data2[num_part]], axis=1)
    data3 = data3.loc[:, ~data3.columns.duplicated()]
    S["data3_shape"] = list(data3.shape)
    S["data3_columns"] = list(data3.columns)
    sample_cols2 = [TARGET, "industryn", "titlen", "work_yearn", "early_returnn",
                    "classn", "total_loan", "interest", "early_return_amount"]
    sample_cols2 = [c for c in sample_cols2 if c in data2_out.columns]
    S["data2_sample"] = data2_out[sample_cols2].head(5).to_dict("records")
    S["data2_sample_cols"] = sample_cols2
    sample_cols3 = [TARGET, "industryn", "titlen", "work_yearn", "classn",
                    "total_loan", "recircle_bn", "early_return_amountn"]
    sample_cols3 = [c for c in sample_cols3 if c in data3.columns]
    S["data3_sample"] = data3[sample_cols3].head(5).round(3).to_dict("records")
    S["data3_sample_cols"] = sample_cols3

    # ======================= 4 输出 =======================
    for name, d in (("data0", data0), ("data1", data1), ("data2", data2_out), ("data3", data3)):
        p = os.path.join(out_dir, name + ".csv")
        d.to_csv(p, index=False, encoding="utf-8-sig")
        print("已输出 {} {}".format(p, d.shape))

    S["pipeline"] = [
        {"stage": "原始数据", "rows": S["n_raw"], "vars": int(df.shape[1] - 1)},
        {"stage": "data0 (随机 80%)", "rows": S["n_data0"], "vars": int(data0.shape[1] - 1)},
        {"stage": "data1 (缺失值处理后)", "rows": int(data1.shape[0]), "vars": int(data1.shape[1] - 1)},
        {"stage": "data2 (定性变量处理后)", "rows": int(data2_out.shape[0]),
         "vars": int(data2_out.shape[1] - 1)},
        {"stage": "data3 (定量变量处理后)", "rows": int(data3.shape[0]),
         "vars": int(data3.shape[1] - 1)},
    ]
    S["y_rate_data3"] = round(100 * float(data3[TARGET].mean()), 2)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=2)
    print("\n统计结果已保存到 stats.json")


if __name__ == "__main__":
    main()
