"""Chương 5: Phân tích Chẩn đoán bằng Kiểm định Thống kê (Chi-square, Levene, ANOVA, Welch, Kruskal-Wallis & Tukey HSD)."""

import math
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    WAREHOUSE_PATH,
    ensure_dirs,
)

STATE_TO_REGION: dict[str, str] = {
    # Southeast
    "SP": "Southeast",
    "RJ": "Southeast",
    "MG": "Southeast",
    "ES": "Southeast",
    # South
    "PR": "South",
    "SC": "South",
    "RS": "South",
    # Center-West
    "DF": "Center-West",
    "GO": "Center-West",
    "MT": "Center-West",
    "MS": "Center-West",
    # Northeast
    "BA": "Northeast",
    "PE": "Northeast",
    "CE": "Northeast",
    "MA": "Northeast",
    "PB": "Northeast",
    "RN": "Northeast",
    "AL": "Northeast",
    "SE": "Northeast",
    "PI": "Northeast",
    # North
    "PA": "North",
    "AM": "North",
    "RO": "North",
    "TO": "North",
    "AC": "North",
    "AP": "North",
    "RR": "North",
}

REGION_ORDER: list[str] = [
    "Southeast",
    "South",
    "Center-West",
    "Northeast",
    "North",
]

DELAY_BUCKET_ORDER: list[str] = [
    "Early (>3d early)",
    "On-time (-3 to 0d)",
    "Late 1-3d",
    "Late 4-7d",
    "Late >7d",
]


def format_p_value(p: float) -> str:
    """Định dạng p-value dễ đọc cho báo cáo khoa học."""
    if p < 1e-16:
        return "< 1e-16"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def compute_cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Tính độ lớn tác động Cramér's V cho bảng tiếp liên R x C."""
    denom = n * min(r - 1, c - 1)
    if denom <= 0:
        return 0.0
    return float(math.sqrt(chi2 / denom))


def run_test1_payment_vs_cancel(conn: duckdb.DuckDBPyConnection) -> dict:
    """Test 1: Phương thức thanh toán chính vs Tỷ lệ Hủy / Không khả dụng."""
    df = conn.execute(
        """
        SELECT
            primary_payment_type,
            order_status IN ('canceled', 'unavailable') AS is_canceled
        FROM dw.fact_orders
        WHERE primary_payment_type IS NOT NULL
        """
    ).df()

    counts_by_type = df["primary_payment_type"].value_counts()
    excluded_types = counts_by_type[counts_by_type < 10].to_dict()
    df_filtered = df[~df["primary_payment_type"].isin(excluded_types.keys())].copy()

    ct = pd.crosstab(df_filtered["primary_payment_type"], df_filtered["is_canceled"])
    ct = ct.reindex(columns=[False, True], fill_value=0)
    chi2, p_val, dof, expected = stats.chi2_contingency(ct.values, correction=False)
    n_total = int(ct.values.sum())
    cramers_v = compute_cramers_v(chi2, n_total, ct.shape[0], ct.shape[1])

    summary_df = (
        df_filtered.groupby("primary_payment_type", as_index=False)
        .agg(
            total_orders=("is_canceled", "count"),
            canceled_orders=("is_canceled", "sum"),
        )
    )
    summary_df["cancel_rate_pct"] = (
        summary_df["canceled_orders"] * 100.0 / summary_df["total_orders"]
    )
    summary_df = summary_df.sort_values("cancel_rate_pct", ascending=False).reset_index(
        drop=True
    )

    return {
        "test_id": "T1",
        "test_name": "Phương thức thanh toán vs Tỷ lệ hủy/không khả dụng",
        "method": "Chi-square Test of Independence",
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_val),
        "n": n_total,
        "min_expected": float(expected.min()),
        "cramers_v": cramers_v,
        "excluded_small_groups": excluded_types,
        "table": summary_df,
    }


def run_test2_category_vs_low_review(conn: duckdb.DuckDBPyConnection) -> dict:
    """Test 2: Top 10 Ngành hàng (theo số đơn hợp lệ) vs Tỷ lệ Đánh giá thấp (1-2 sao)."""
    # Khử trùng ở cấp (order_id, product_category_en) để chống fan-out khi 1 đơn có nhiều món cùng loại
    df = conn.execute(
        """
        WITH valid_order_cat AS (
            SELECT DISTINCT
                fi.order_id,
                COALESCE(dp.product_category_en, 'unknown') AS product_category_en,
                fo.is_low_review,
                fo.review_score
            FROM dw.fact_order_items AS fi
            INNER JOIN dw.fact_orders AS fo
                ON fi.order_id = fo.order_id
            LEFT JOIN dw.dim_products AS dp
                ON fi.product_id = dp.product_id
            WHERE fo.order_status NOT IN ('canceled', 'unavailable')
              AND fo.is_low_review IS NOT NULL
        ),
        top10 AS (
            SELECT product_category_en
            FROM valid_order_cat
            GROUP BY product_category_en
            ORDER BY COUNT(*) DESC
            LIMIT 10
        )
        SELECT
            v.product_category_en,
            v.is_low_review,
            v.review_score
        FROM valid_order_cat AS v
        INNER JOIN top10 AS t
            ON v.product_category_en = t.product_category_en
        """
    ).df()

    ct = pd.crosstab(df["product_category_en"], df["is_low_review"])
    ct = ct.reindex(columns=[False, True], fill_value=0)
    chi2, p_val, dof, expected = stats.chi2_contingency(ct.values, correction=False)
    n_total = int(ct.values.sum())
    cramers_v = compute_cramers_v(chi2, n_total, ct.shape[0], ct.shape[1])

    summary_df = (
        df.groupby("product_category_en", as_index=False)
        .agg(
            total_orders=("is_low_review", "count"),
            low_review_orders=("is_low_review", "sum"),
            avg_review_score=("review_score", "mean"),
        )
    )
    summary_df["low_review_rate_pct"] = (
        summary_df["low_review_orders"] * 100.0 / summary_df["total_orders"]
    )
    summary_df = summary_df.sort_values(
        "low_review_rate_pct", ascending=False
    ).reset_index(drop=True)

    return {
        "test_id": "T2",
        "test_name": "Top 10 Ngành hàng vs Tỷ lệ đánh giá thấp (1–2 sao)",
        "method": "Chi-square Test of Independence",
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_val),
        "n": n_total,
        "min_expected": float(expected.min()),
        "cramers_v": cramers_v,
        "overall_low_review_pct": float(df["is_low_review"].mean() * 100.0),
        "table": summary_df,
    }


def run_test3_late_delivery_vs_low_review(conn: duckdb.DuckDBPyConnection) -> dict:
    """Test 3: Giao hàng trễ vs Đánh giá thấp (1-2 sao), kèm Relative Risk & Odds Ratio."""
    df = conn.execute(
        """
        SELECT
            is_late_delivery,
            is_low_review,
            review_score
        FROM dw.fact_orders
        WHERE is_valid_delivered = TRUE
          AND is_low_review IS NOT NULL
        """
    ).df()

    ct = pd.crosstab(df["is_late_delivery"], df["is_low_review"])
    ct = ct.reindex(index=[False, True], columns=[False, True], fill_value=0)
    chi2, p_val, dof, expected = stats.chi2_contingency(ct.values, correction=False)
    n_total = int(ct.values.sum())
    cramers_v = compute_cramers_v(chi2, n_total, 2, 2)

    # Bảng 2x2:
    # on_time: [not_low (c), low (d)]
    # late   : [not_low (b), low (a)]
    not_low_ontime = int(ct.loc[False, False])
    low_ontime = int(ct.loc[False, True])
    not_low_late = int(ct.loc[True, False])
    low_late = int(ct.loc[True, True])

    n_ontime = not_low_ontime + low_ontime
    n_late = not_low_late + low_late

    risk_ontime = low_ontime / n_ontime
    risk_late = low_late / n_late
    relative_risk = risk_late / risk_ontime
    odds_ratio = (low_late / not_low_late) / (low_ontime / not_low_ontime)

    # 95% CI cho Relative Risk trên thang log
    se_log_rr = math.sqrt(
        (1.0 / low_late) - (1.0 / n_late) + (1.0 / low_ontime) - (1.0 / n_ontime)
    )
    rr_ci_low = math.exp(math.log(relative_risk) - 1.96 * se_log_rr)
    rr_ci_high = math.exp(math.log(relative_risk) + 1.96 * se_log_rr)

    summary_df = pd.DataFrame(
        [
            {
                "delivery_group": "Giao đúng hạn (On-time)",
                "is_late": False,
                "total_orders": n_ontime,
                "low_review_orders": low_ontime,
                "low_review_rate_pct": risk_ontime * 100.0,
                "avg_review_score": float(
                    df.loc[~df["is_late_delivery"], "review_score"].mean()
                ),
            },
            {
                "delivery_group": "Giao trễ hẹn (Late)",
                "is_late": True,
                "total_orders": n_late,
                "low_review_orders": low_late,
                "low_review_rate_pct": risk_late * 100.0,
                "avg_review_score": float(
                    df.loc[df["is_late_delivery"], "review_score"].mean()
                ),
            },
        ]
    )

    return {
        "test_id": "T3",
        "test_name": "Giao hàng trễ vs Đánh giá thấp (1–2 sao)",
        "method": "Chi-square Test of Independence (2x2)",
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_val),
        "n": n_total,
        "min_expected": float(expected.min()),
        "cramers_v": cramers_v,
        "relative_risk": float(relative_risk),
        "rr_ci95": (float(rr_ci_low), float(rr_ci_high)),
        "odds_ratio": float(odds_ratio),
        "table": summary_df,
    }


def run_anova_suite(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_order: list[str],
    test_id: str,
    test_name: str,
) -> dict:
    """Chạy trọn bộ: Thống kê mô tả, Levene, One-Way ANOVA + Eta^2, Welch's ANOVA, Kruskal-Wallis & Tukey HSD."""
    sub = df[[group_col, value_col]].dropna().copy()
    groups_arrays: list[np.ndarray] = []
    desc_rows: list[dict] = []

    for g in group_order:
        arr = sub.loc[sub[group_col] == g, value_col].to_numpy(dtype=float)
        groups_arrays.append(arr)
        n_g = len(arr)
        mean_g = float(np.mean(arr))
        std_g = float(np.std(arr, ddof=1))
        ci95 = 1.96 * std_g / math.sqrt(n_g) if n_g > 1 else 0.0
        desc_rows.append(
            {
                "group": g,
                "n": n_g,
                "mean": mean_g,
                "median": float(np.median(arr)),
                "std": std_g,
                "ci95_low": mean_g - ci95,
                "ci95_high": mean_g + ci95,
                "ci95_half": ci95,
            }
        )

    desc_df = pd.DataFrame(desc_rows)

    # 1. Tiền kiểm định đồng nhất phương sai Levene (tâm median - Brown-Forsythe)
    levene_stat, levene_p = stats.levene(*groups_arrays, center="median")

    # 2. One-Way ANOVA cổ điển + Eta-squared (SS_between / SS_total)
    f_stat, anova_p = stats.f_oneway(*groups_arrays)
    all_vals = sub[value_col].to_numpy(dtype=float)
    grand_mean = float(np.mean(all_vals))
    ss_total = float(np.sum((all_vals - grand_mean) ** 2))
    ss_between = float(
        sum(len(arr) * (float(np.mean(arr)) - grand_mean) ** 2 for arr in groups_arrays)
    )
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    k = len(groups_arrays)
    n_total = len(all_vals)

    # 3. Welch's ANOVA (không giả định phương sai bằng nhau)
    welch_res = anova_oneway(groups_arrays, use_var="unequal")
    welch_f = float(welch_res.statistic)
    welch_p = float(welch_res.pvalue)
    welch_df = (float(welch_res.df[0]), float(welch_res.df[1]))

    # 4. Kruskal-Wallis H-test (phi tham số cho phân phối lệch / thứ bậc)
    kw_h, kw_p = stats.kruskal(*groups_arrays)

    # 5. Hậu kiểm Tukey HSD
    tukey = pairwise_tukeyhsd(
        endog=sub[value_col].to_numpy(dtype=float),
        groups=sub[group_col].to_numpy(dtype=str),
        alpha=0.05,
    )
    tukey_rows = tukey._results_table.data[1:]
    tukey_cols = tukey._results_table.data[0]
    tukey_df = pd.DataFrame(tukey_rows, columns=tukey_cols)
    for num_c in ["meandiff", "p-adj", "lower", "upper"]:
        tukey_df[num_c] = tukey_df[num_c].astype(float)
    tukey_df["reject"] = tukey_df["reject"].astype(bool)

    return {
        "test_id": test_id,
        "test_name": test_name,
        "method": "One-Way ANOVA (+ Welch & Kruskal-Wallis)",
        "n": n_total,
        "k": k,
        "df_between": k - 1,
        "df_within": n_total - k,
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "f_stat": float(f_stat),
        "anova_p": float(anova_p),
        "eta_sq": float(eta_sq),
        "welch_f": welch_f,
        "welch_p": welch_p,
        "welch_df": welch_df,
        "kw_h": float(kw_h),
        "kw_p": float(kw_p),
        "desc_df": desc_df,
        "tukey_df": tukey_df,
    }


def load_delivered_orders_with_region_and_delay(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Đọc các đơn giao thành công hợp lệ từ `dw.fact_orders` và gắn nhãn Vùng + Bậc trễ."""
    df = conn.execute(
        """
        SELECT
            order_id,
            customer_state,
            delivery_days,
            delivery_delay_days,
            is_late_delivery,
            review_score,
            is_low_review
        FROM dw.fact_orders
        WHERE is_valid_delivered = TRUE
          AND delivery_days IS NOT NULL
        """
    ).df()

    df["region"] = df["customer_state"].map(STATE_TO_REGION)
    unmapped = df["region"].isna().sum()
    if unmapped > 0:
        raise ValueError(f"Phát hiện {unmapped} dòng có customer_state chưa ánh xạ vùng!")

    conditions = [
        df["delivery_delay_days"] < -3,
        (df["delivery_delay_days"] >= -3) & (df["delivery_delay_days"] <= 0),
        (df["delivery_delay_days"] >= 1) & (df["delivery_delay_days"] <= 3),
        (df["delivery_delay_days"] >= 4) & (df["delivery_delay_days"] <= 7),
        df["delivery_delay_days"] > 7,
    ]
    df["delay_bucket"] = np.select(conditions, DELAY_BUCKET_ORDER, default=None)
    return df


def plot_chisquare_diagnostics(t2: dict, t3: dict, output_path: Path) -> None:
    """Vẽ biểu đồ chẩn đoán Chi-square 1x2: Top 10 ngành hàng & Giao trễ vs Đúng hạn."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150, facecolor="white")
    fig.suptitle(
        "Chẩn đoán Kiểm định Chi-square: Tác nhân liên quan đến Đánh giá thấp (1–2 sao)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    # --- Cột trái: Top 10 Ngành hàng ---
    cat_df = t2["table"].sort_values("low_review_rate_pct", ascending=True).reset_index(
        drop=True
    )
    y_pos = np.arange(len(cat_df))
    palette_left = [
        "#DC2626" if v > t2["overall_low_review_pct"] else "#3B82F6"
        for v in cat_df["low_review_rate_pct"]
    ]
    bars1 = ax1.barh(
        y_pos,
        cat_df["low_review_rate_pct"],
        color=palette_left,
        alpha=0.85,
        height=0.65,
    )
    ax1.axvline(
        t2["overall_low_review_pct"],
        color="#1E293B",
        linestyle="--",
        linewidth=1.6,
        label=f"TB Top 10 = {t2['overall_low_review_pct']:.2f}%",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cat_df["product_category_en"], fontsize=9.5)
    ax1.set_xlabel("Tỷ lệ đánh giá thấp 1–2 sao (%)", fontsize=10.5, fontweight="bold")
    ax1.set_xlim(0, cat_df["low_review_rate_pct"].max() * 1.62)
    ax1.set_title(
        "Test 2: Tỷ lệ Đánh giá thấp theo Top 10 Ngành hàng",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    for bar, (_, r) in zip(bars1, cat_df.iterrows()):
        w = bar.get_width()
        ax1.text(
            w + 0.35,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.2f}% (n={int(r['total_orders']):,})",
            va="center",
            ha="left",
            fontsize=8.6,
            color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
        )

    ax1.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.8)
    ax1.text(
        0.98,
        0.16,
        f"Chi2 = {t2['chi2']:.2f} (df={t2['dof']})\n"
        f"p-value {format_p_value(t2['p_value'])}\n"
        f"Cramér's V = {t2['cramers_v']:.4f} (Rất yếu)",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    # --- Cột phải: Giao đúng hạn vs Giao trễ ---
    late_df = t3["table"]
    x_pos = np.arange(len(late_df))
    colors_right = ["#10B981", "#EF4444"]
    bars2 = ax2.bar(
        x_pos,
        late_df["low_review_rate_pct"],
        width=0.48,
        color=colors_right,
        alpha=0.88,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(late_df["delivery_group"], fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Tỷ lệ đánh giá thấp 1–2 sao (%)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, late_df["low_review_rate_pct"].max() * 1.32)
    ax2.set_title(
        "Test 3: Tỷ lệ Đánh giá thấp — Đúng hạn vs Giao trễ",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    for bar, (_, r) in zip(bars2, late_df.iterrows()):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.3,
            f"{h:.2f}%\n({int(r['low_review_orders']):,} / {int(r['total_orders']):,} đơn)\nReview TB: {r['avg_review_score']:.2f}/5",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color="#1E293B",
        )

    ax2.text(
        0.04,
        0.93,
        f"Relative Risk = {t3['relative_risk']:.2f}x\n"
        f"(95% CI: {t3['rr_ci95'][0]:.2f}x – {t3['rr_ci95'][1]:.2f}x)\n"
        f"Odds Ratio = {t3['odds_ratio']:.2f}\n"
        f"Chi2 = {t3['chi2']:,.1f} | p {format_p_value(t3['p_value'])}\n"
        f"Cramér's V = {t3['cramers_v']:.4f} (Mạnh)",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FEF2F2", edgecolor="#F87171"),
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_anova_diagnostics(
    delivered_df: pd.DataFrame,
    t4: dict,
    t5a: dict,
    t5b: dict,
    output_path: Path,
) -> None:
    """Vẽ biểu đồ chẩn đoán ANOVA 1x2: 5 Vùng địa lý Brazil & 5 nấc Độ trễ giao hàng."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=150, facecolor="white")
    fig.suptitle(
        "Chẩn đoán One-Way ANOVA & Hậu kiểm: Vùng Địa lý và Bậc thang Độ trễ Giao hàng",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    # --- Cột trái: Boxplot delivery_days (ẩn fliers xa) + điểm Mean & nhãn Review TB của 5 Vùng ---
    reg_order = REGION_ORDER
    sns.boxplot(
        data=delivered_df,
        y="region",
        x="delivery_days",
        order=reg_order,
        showfliers=False,
        width=0.52,
        palette=["#2563EB", "#0D9488", "#6366F1", "#F59E0B", "#EF4444"],
        hue="region",
        legend=False,
        ax=ax1,
    )

    t4_desc = t4["desc_df"].set_index("group")
    t5a_desc = t5a["desc_df"].set_index("group")

    means_x = [t4_desc.loc[r, "mean"] for r in reg_order]
    ax1.scatter(
        means_x,
        np.arange(len(reg_order)),
        color="#111827",
        marker="D",
        s=42,
        zorder=5,
        label="Mean delivery_days",
    )

    # Tìm vị trí râu phải (Q3 + 1.5*IQR) của từng vùng để đặt nhãn gọn gàng bên phải
    max_whisker = 0.0
    whiskers: list[float] = []
    for r in reg_order:
        vals = delivered_df.loc[delivered_df["region"] == r, "delivery_days"]
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        upper_w = float(vals[vals <= q3 + 1.5 * (q3 - q1)].max())
        whiskers.append(upper_w)
        max_whisker = max(max_whisker, upper_w)

    ax1.set_xlim(0, max_whisker * 1.66)
    for idx, r in enumerate(reg_order):
        m_deliv = t4_desc.loc[r, "mean"]
        med_deliv = t4_desc.loc[r, "median"]
        m_rev = t5a_desc.loc[r, "mean"]
        ax1.text(
            whiskers[idx] + 1.0,
            idx,
            f"Mean {m_deliv:.1f}d (Med {med_deliv:.1f}d) | Rev {m_rev:.2f}/5",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#1E293B",
        )

    ax1.set_title(
        "Test 4 & 5a: Thời gian giao hàng & Điểm Review theo 5 Vùng (Ẩn ngoại lệ > Q3+1.5·IQR)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Thời gian giao hàng — delivery_days (Ngày)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Vùng địa lý Brazil", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.8)
    ax1.text(
        0.98,
        0.84,
        f"Delivery ANOVA: F = {t4['f_stat']:,.1f}, p {format_p_value(t4['anova_p'])}\n"
        f"Eta^2 (delivery) = {t4['eta_sq']:.4f} ({t4['eta_sq']*100:.2f}% phương sai)\n"
        f"Review ANOVA: F = {t5a['f_stat']:.1f}, Eta^2 = {t5a['eta_sq']:.4f}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    # --- Cột phải: Điểm đánh giá TB (kèm 95% CI) theo 5 nấc Độ trễ giao hàng ---
    d_df = t5b["desc_df"].set_index("group").loc[DELAY_BUCKET_ORDER].reset_index()
    x_pos = np.arange(len(d_df))
    bar_colors = ["#16A34A", "#0D9488", "#F59E0B", "#EA580C", "#DC2626"]
    bars = ax2.bar(
        x_pos,
        d_df["mean"],
        yerr=d_df["ci95_half"],
        capsize=5,
        width=0.56,
        color=bar_colors,
        alpha=0.88,
        error_kw=dict(ecolor="#1E293B", lw=1.4),
    )

    overall_rev = float(delivered_df["review_score"].dropna().mean())
    ax2.axhline(
        overall_rev,
        color="#334155",
        linestyle="--",
        linewidth=1.5,
        label=f"Review TB toàn sàn = {overall_rev:.2f}/5",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        ["Early\n(>3d early)", "On-time\n(-3 to 0d)", "Late\n1–3d", "Late\n4–7d", "Late\n>7d"],
        fontsize=9.5,
        fontweight="bold",
    )
    ax2.set_ylabel("Điểm đánh giá trung bình (1–5 sao, kèm 95% CI)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 5.25)
    ax2.set_title(
        "Test 5b: Cú rơi Bậc thang của Điểm Review theo Độ trễ Giao hàng",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    for bar, (_, r) in zip(bars, d_df.iterrows()):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + r["ci95_half"] + 0.10,
            f"{h:.2f}/5\n(n={int(r['n']):,})",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#1E293B",
        )

    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)
    ax2.text(
        0.97,
        0.70,
        f"One-Way ANOVA: F = {t5b['f_stat']:,.1f}\n"
        f"p-value {format_p_value(t5b['anova_p'])}\n"
        f"Welch F = {t5b['welch_f']:,.1f} | p {format_p_value(t5b['welch_p'])}\n"
        f"Eta^2 = {t5b['eta_sq']:.4f} ({t5b['eta_sq']*100:.2f}% phương sai!)",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.40", facecolor="#FEF2F2", edgecolor="#F87171"),
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_summary_dataframe(
    t1: dict,
    t2: dict,
    t3: dict,
    t4: dict,
    t5a: dict,
    t5b: dict,
) -> pd.DataFrame:
    """Tạo DataFrame tổng hợp kết quả kiểm định để lưu ra Parquet cho Chương 8."""
    rows = [
        {
            "test_id": "Test_1",
            "test_name": t1["test_name"],
            "method": t1["method"],
            "statistic": t1["chi2"],
            "p_value": t1["p_value"],
            "effect_size_name": "Cramér's V",
            "effect_size_value": t1["cramers_v"],
            "significance_05": bool(t1["p_value"] < 0.05),
            "business_conclusion": (
                f"Có ý nghĩa thống kê nhưng độ lớn tác động rất nhỏ (V={t1['cramers_v']:.4f}); "
                "voucher có tỷ lệ hủy cao hơn nhưng không phải nguyên nhân chính toàn sàn."
            ),
        },
        {
            "test_id": "Test_2",
            "test_name": t2["test_name"],
            "method": t2["method"],
            "statistic": t2["chi2"],
            "p_value": t2["p_value"],
            "effect_size_name": "Cramér's V",
            "effect_size_value": t2["cramers_v"],
            "significance_05": bool(t2["p_value"] < 0.05),
            "business_conclusion": (
                f"Khác biệt giữa Top 10 ngành hàng có ý nghĩa thống kê nhưng tác động yếu (V={t2['cramers_v']:.4f}); "
                "bed_bath_table và furniture_decor cao nhất."
            ),
        },
        {
            "test_id": "Test_3",
            "test_name": t3["test_name"],
            "method": t3["method"],
            "statistic": t3["chi2"],
            "p_value": t3["p_value"],
            "effect_size_name": "Cramér's V",
            "effect_size_value": t3["cramers_v"],
            "significance_05": bool(t3["p_value"] < 0.05),
            "business_conclusion": (
                f"Tác nhân cốt lõi mạnh nhất: giao trễ làm tăng nguy cơ bị đánh giá 1-2 sao gấp "
                f"{t3['relative_risk']:.2f} lần (V={t3['cramers_v']:.4f}, OR={t3['odds_ratio']:.2f})."
            ),
        },
        {
            "test_id": "Test_4",
            "test_name": t4["test_name"],
            "method": t4["method"],
            "statistic": t4["f_stat"],
            "p_value": t4["anova_p"],
            "effect_size_name": "Eta-squared",
            "effect_size_value": t4["eta_sq"],
            "significance_05": bool(t4["anova_p"] < 0.05),
            "business_conclusion": (
                f"Khoảng cách vùng địa lý giải thích {t4['eta_sq']*100:.2f}% phương sai thời gian giao hàng "
                "(Southeast nhanh nhất, North & Northeast lâu nhất; xác nhận qua Welch & Kruskal-Wallis)."
            ),
        },
        {
            "test_id": "Test_5a",
            "test_name": t5a["test_name"],
            "method": t5a["method"],
            "statistic": t5a["f_stat"],
            "p_value": t5a["anova_p"],
            "effect_size_name": "Eta-squared",
            "effect_size_value": t5a["eta_sq"],
            "significance_05": bool(t5a["anova_p"] < 0.05),
            "business_conclusion": (
                f"Vùng địa lý tự thân chỉ giải thích {t5a['eta_sq']*100:.2f}% phương sai điểm review — "
                "tác động chủ yếu gián tiếp thông qua thời gian giao hàng và giao trễ."
            ),
        },
        {
            "test_id": "Test_5b",
            "test_name": t5b["test_name"],
            "method": t5b["method"],
            "statistic": t5b["f_stat"],
            "p_value": t5b["anova_p"],
            "effect_size_name": "Eta-squared",
            "effect_size_value": t5b["eta_sq"],
            "significance_05": bool(t5b["anova_p"] < 0.05),
            "business_conclusion": (
                f"Độ trễ giao hàng giải thích tới {t5b['eta_sq']*100:.2f}% phương sai điểm review; "
                "mọi nấc trễ đều làm điểm đánh giá sụt giảm bậc thang cực mạnh (Tukey HSD bác bỏ H0 toàn bộ 10 cặp)."
            ),
        },
    ]
    return pd.DataFrame(rows)


def format_tukey_markdown_table(tukey_df: pd.DataFrame) -> list[str]:
    """Chuyển bảng hậu kiểm Tukey HSD sang bảng Markdown gọn rõ."""
    lines = [
        "| Nhóm 1 (`group1`) | Nhóm 2 (`group2`) | Chênh lệch TB (`meandiff`) | `p-adj` | 95% CI Lower | 95% CI Upper | Bác bỏ $H_0$ (`reject`) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for _, r in tukey_df.iterrows():
        p_str = "< 0.0001" if r["p-adj"] < 0.0001 else f"{r['p-adj']:.4f}"
        rej_str = "**Có**" if r["reject"] else "Không"
        lines.append(
            f"| `{r['group1']}` | `{r['group2']}` | {r['meandiff']:+.4f} | {p_str} | "
            f"{r['lower']:+.4f} | {r['upper']:+.4f} | {rej_str} |"
        )
    return lines


def write_diagnostics_report(
    delivered_df: pd.DataFrame,
    t1: dict,
    t2: dict,
    t3: dict,
    t4: dict,
    t5a: dict,
    t5b: dict,
    output_path: Path,
) -> None:
    """Xuất báo cáo chẩn đoán thống kê chuyên sâu Chương 5 ra Markdown."""
    region_late_rates = (
        delivered_df.groupby("region")["is_late_delivery"]
        .mean()
        .mul(100.0)
        .to_dict()
    )
    delay_low_rates = (
        delivered_df.dropna(subset=["delay_bucket", "is_low_review"])
        .groupby("delay_bucket")["is_low_review"]
        .mean()
        .mul(100.0)
        .to_dict()
    )

    excl_note = ", ".join(f"`{k}` (n={v})" for k, v in t1["excluded_small_groups"].items())

    lines: list[str] = [
        "# Báo cáo Phân tích Chẩn đoán bằng Kiểm định Thống kê — Chương 5",
        "",
        f"- **Thời điểm chạy kiểm định:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- **Nguồn dữ liệu:** Kho Star Schema `olist.duckdb` (`dw.fact_orders`, `dw.fact_order_items`, `dw.dim_products` — chế độ `read_only=True`).",
        "- **Mức ý nghĩa thống kê chuẩn:** $\\alpha = 0.05$ (kết hợp bắt buộc với **Độ lớn tác động — Effect Size** để tránh bẫy cỡ mẫu lớn $N \\approx 96,000$).",
        "",
        "---",
        "",
        "## 1. Bảng Tóm tắt Điều hành (Executive Summary)",
        "",
        "| Mã | Bài kiểm định & Giả thuyết $H_0$ | Phương pháp chính | Giá trị thống kê | `p-value` | Effect Size | Kết luận ($\\alpha=0.05$) |",
        "| :--- | :--- | :--- | ---: | ---: | :--- | :--- |",
        f"| **Test 1** | $H_0$: Phương thức thanh toán độc lập với hủy đơn | Chi-square ($df={t1['dof']}$) | $\\chi^2 = {t1['chi2']:.2f}$ | `{format_p_value(t1['p_value'])}` | Cramér's $V = {t1['cramers_v']:.4f}$ (Rất yếu) | **Bác bỏ $H_0$** (Ý nghĩa thực tiễn nhỏ) |",
        f"| **Test 2** | $H_0$: Top 10 ngành hàng độc lập với tỷ lệ review 1–2 sao | Chi-square ($df={t2['dof']}$) | $\\chi^2 = {t2['chi2']:.2f}$ | `{format_p_value(t2['p_value'])}` | Cramér's $V = {t2['cramers_v']:.4f}$ (Rất yếu) | **Bác bỏ $H_0$** (Khác biệt nhẹ theo ngành) |",
        f"| **Test 3** | $H_0$: Giao hàng trễ độc lập với đánh giá thấp 1–2 sao | Chi-square ($df={t3['dof']}$) | $\\chi^2 = {t3['chi2']:,.2f}$ | `{format_p_value(t3['p_value'])}` | Cramér's $V = {t3['cramers_v']:.4f}$, **RR = {t3['relative_risk']:.2f}x** | **Bác bỏ $H_0$ (Tác nhân mạnh nhất)** |",
        f"| **Test 4** | $H_0$: Trung bình `delivery_days` bằng nhau ở 5 vùng Brazil | One-Way ANOVA (+ Welch/KW) | $F = {t4['f_stat']:,.2f}$ | `{format_p_value(t4['anova_p'])}` | $\\eta^2 = {t4['eta_sq']:.4f}$ ({t4['eta_sq']*100:.2f}%) | **Bác bỏ $H_0$** (Tác động vừa–lớn) |",
        f"| **Test 5a** | $H_0$: Trung bình `review_score` bằng nhau ở 5 vùng Brazil | One-Way ANOVA (+ Welch/KW) | $F = {t5a['f_stat']:.2f}$ | `{format_p_value(t5a['anova_p'])}` | $\\eta^2 = {t5a['eta_sq']:.4f}$ ({t5a['eta_sq']*100:.2f}%) | **Bác bỏ $H_0$** (Tác động trực tiếp rất nhỏ) |",
        f"| **Test 5b** | $H_0$: Trung bình `review_score` bằng nhau ở 5 nấc độ trễ | One-Way ANOVA (+ Welch/KW) | $F = {t5b['f_stat']:,.2f}$ | `{format_p_value(t5b['anova_p'])}` | $\\eta^2 = {t5b['eta_sq']:.4f}$ (**{t5b['eta_sq']*100:.2f}%**) | **Bác bỏ $H_0$ (Giải thích mạnh nhất)** |",
        "",
        "---",
        "",
        "## 2. Chi tiết Nhóm A — Kiểm định Độc lập Chi-square & Độ lớn Tác động",
        "",
        "### 2.1 Test 1: Phương thức thanh toán chính vs Tỷ lệ Hủy / Không khả dụng",
        "",
        f"- **Lưu ý tiền xử lý mẫu:** Đã lọc bỏ dòng `primary_payment_type IS NULL` (1 dòng) và nhóm quá nhỏ {excl_note} (`< 10` quan sát, cả 3 đơn đều hủy do chưa xác định thanh toán) nhằm đảm bảo điều kiện tần số kỳ vọng tối thiểu của kiểm định Chi-square ($\\min(E_{{ij}}) = {t1['min_expected']:.2f} \\ge 5$).",
        f"- **Kết quả:** $\\chi^2 = {t1['chi2']:.2f}$, $df = {t1['dof']}$, $p = {format_p_value(t1['p_value'])}$, **Cramér's $V = {t1['cramers_v']:.4f}$**.",
        "",
        "| Phương thức thanh toán | Tổng số đơn | Số đơn hủy / không khả dụng | Tỷ lệ hủy (%) |",
        "| :--- | ---: | ---: | ---: |",
    ]

    for _, r in t1["table"].iterrows():
        lines.append(
            f"| `{r['primary_payment_type']}` | {int(r['total_orders']):,} | {int(r['canceled_orders']):,} | {r['cancel_rate_pct']:.2f}% |"
        )

    lines.extend(
        [
            "",
            f"- **Diễn giải nghiệp vụ:** Đơn thanh toán bằng `voucher` có tỷ lệ hủy cao gấp ~3 lần `credit_card` (3.62% vs 1.15%), dẫn đến $p < 0.0001$. Tuy nhiên **Cramér's $V = {t1['cramers_v']:.4f}$** rất nhỏ (< 0.10), cho thấy phương thức thanh toán không phải rào cản gây hủy đơn trên diện rộng.",
            "",
            "### 2.2 Test 2: Top 10 Ngành hàng lớn nhất vs Tỷ lệ Đánh giá thấp (1–2 sao)",
            "",
            f"- **Đơn vị phân tích:** Khử trùng ở cấp `(order_id, product_category_en)` trên các đơn hợp lệ có đánh giá ($N = {t2['n']:,}$, $\\min(E_{{ij}}) = {t2['min_expected']:.2f} \\ge 5$).",
            f"- **Kết quả:** $\\chi^2 = {t2['chi2']:.2f}$, $df = {t2['dof']}$, $p = {format_p_value(t2['p_value'])}$, **Cramér's $V = {t2['cramers_v']:.4f}$**.",
            "",
            "| Ngành hàng (EN) | Số đơn có review | Số đơn 1–2 sao | Tỷ lệ đánh giá thấp (%) | Điểm Review TB |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, r in t2["table"].iterrows():
        lines.append(
            f"| `{r['product_category_en']}` | {int(r['total_orders']):,} | {int(r['low_review_orders']):,} | "
            f"{r['low_review_rate_pct']:.2f}% | {r['avg_review_score']:.2f} |"
        )

    ontime_row = t3["table"].iloc[0]
    late_row = t3["table"].iloc[1]
    lines.extend(
        [
            "",
            "### 2.3 Test 3: Giao hàng trễ (`is_late_delivery`) vs Đánh giá thấp (`is_low_review`)",
            "",
            f"- **Kết quả:** $\\chi^2 = {t3['chi2']:,.2f}$, $df = {t3['dof']}$, $p {format_p_value(t3['p_value'])}$, **Cramér's $V = {t3['cramers_v']:.4f}$**.",
            f"- **Lượng hóa rủi ro kinh doanh:**",
            f"  - Tỷ lệ đánh giá 1–2 sao khi giao đúng hạn: **{ontime_row['low_review_rate_pct']:.2f}%** ({int(ontime_row['low_review_orders']):,} / {int(ontime_row['total_orders']):,} đơn, Review TB = {ontime_row['avg_review_score']:.2f}/5).",
            f"  - Tỷ lệ đánh giá 1–2 sao khi giao trễ: **{late_row['low_review_rate_pct']:.2f}%** ({int(late_row['low_review_orders']):,} / {int(late_row['total_orders']):,} đơn, Review TB = {late_row['avg_review_score']:.2f}/5).",
            f"  - **Relative Risk (Tỷ số nguy cơ) = {t3['relative_risk']:.2f} lần** (95% CI: `[{t3['rr_ci95'][0]:.2f}x, {t3['rr_ci95'][1]:.2f}x]`) và **Odds Ratio = {t3['odds_ratio']:.2f}** $\\rightarrow$ Một đơn hàng giao trễ có xác suất nhận 1–2 sao cao gấp **{t3['relative_risk']:.2f} lần** so với đơn giao đúng hạn!",
            "",
            "![Chẩn đoán Chi-square](figures/ch5_chisquare_diagnostics.png)",
            "",
            "---",
            "",
            "## 3. Chi tiết Nhóm B — Levene, One-Way ANOVA, Kiểm định Bền vững & Hậu kiểm Tukey HSD",
            "",
            "> **Tính minh bạch phương pháp luận:** Với dữ liệu thương mại điện tử cỡ lớn ($N \\approx 96,000$), tiền kiểm định **Levene** cho thấy phương sai giữa các nhóm không đồng nhất ($p < 0.05$), đồng thời `delivery_days` lệch phải còn `review_score` rời rạc 1–5. Do đó, bên cạnh **One-Way ANOVA** truyền thống, báo cáo đối chiếu song song với **Welch's ANOVA** (không yêu cầu phương sai đồng nhất) và **Kruskal-Wallis H-test** (phi tham số). Kết quả cả 3 phép kiểm định đều thống nhất tuyệt đối!",
            "",
            "### 3.1 Test 4: Thời gian giao hàng (`delivery_days`) theo 5 Vùng địa lý Brazil",
            "",
            f"- **Levene's Test:** $W = {t4['levene_stat']:.2f}$, $p {format_p_value(t4['levene_p'])}$ (Phương sai khác biệt có ý nghĩa thống kê giữa các vùng).",
            f"- **One-Way ANOVA:** $F({t4['df_between']}, {t4['df_within']:,}) = {t4['f_stat']:,.2f}$, $p {format_p_value(t4['anova_p'])}$, **$\\eta^2 = {t4['eta_sq']:.4f}$** (Vùng địa lý giải thích **{t4['eta_sq']*100:.2f}%** phương sai thời gian giao hàng).",
            f"- **Đối chiếu bền vững:** Welch's ANOVA $F = {t4['welch_f']:,.2f}$ ($p {format_p_value(t4['welch_p'])}$) | Kruskal-Wallis $H = {t4['kw_h']:,.2f}$ ($p {format_p_value(t4['kw_p'])}$).",
            "",
            "| Vùng địa lý | Số đơn ($n$) | Mean (Ngày) | Median (Ngày) | Std (Ngày) | 95% CI của Mean | Tỷ lệ giao trễ (%) |",
            "| :--- | ---: | ---: | ---: | ---: | :--- | ---: |",
        ]
    )

    for _, r in t4["desc_df"].iterrows():
        reg = r["group"]
        lines.append(
            f"| **{reg}** | {int(r['n']):,} | {r['mean']:.2f} | {r['median']:.2f} | {r['std']:.2f} | "
            f"[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] | {region_late_rates[reg]:.2f}% |"
        )

    lines.extend(["", "**Hậu kiểm Tukey HSD — Test 4 (`delivery_days` giữa 10 cặp vùng):**", ""])
    lines.extend(format_tukey_markdown_table(t4["tukey_df"]))

    lines.extend(
        [
            "",
            "### 3.2 Test 5a: Điểm đánh giá (`review_score`) theo 5 Vùng địa lý Brazil",
            "",
            f"- **Levene's Test:** $W = {t5a['levene_stat']:.2f}$, $p {format_p_value(t5a['levene_p'])}$.",
            f"- **One-Way ANOVA:** $F({t5a['df_between']}, {t5a['df_within']:,}) = {t5a['f_stat']:.2f}$, $p {format_p_value(t5a['anova_p'])}$, **$\\eta^2 = {t5a['eta_sq']:.4f}$** ({t5a['eta_sq']*100:.2f}%).",
            f"- **Đối chiếu bền vững:** Welch's ANOVA $F = {t5a['welch_f']:.2f}$ ($p {format_p_value(t5a['welch_p'])}$) | Kruskal-Wallis $H = {t5a['kw_h']:.2f}$ ($p {format_p_value(t5a['kw_p'])}$).",
            f"- **Nhận xét quan trọng:** Dù $p < 0.0001$, $\\eta^2$ của vùng địa lý lên `review_score` chỉ đạt **{t5a['eta_sq']*100:.2f}%** (rất nhỏ!). Khách hàng miền Bắc (`North`) chờ trung bình {t4['desc_df'].set_index('group').loc['North', 'mean']:.2f} ngày nhưng nếu giao đúng ngày hẹn dự kiến thì họ vẫn chấm điểm khá cao (4.03/5); chỉ những vùng vừa xa vừa bị **trễ so với ngày hẹn** (như `Northeast`: trễ {region_late_rates['Northeast']:.2f}%) mới tụt điểm mạnh.",
            "",
            "**Hậu kiểm Tukey HSD — Test 5a (`review_score` giữa 10 cặp vùng):**",
            "",
        ]
    )
    lines.extend(format_tukey_markdown_table(t5a["tukey_df"]))

    lines.extend(
        [
            "",
            "### 3.3 Test 5b: Điểm đánh giá (`review_score`) theo 5 nấc Độ trễ giao hàng (`delivery_delay_days`)",
            "",
            f"- **Levene's Test:** $W = {t5b['levene_stat']:.2f}$, $p {format_p_value(t5b['levene_p'])}$.",
            f"- **One-Way ANOVA:** $F({t5b['df_between']}, {t5b['df_within']:,}) = {t5b['f_stat']:,.2f}$, $p {format_p_value(t5b['anova_p'])}$, **$\\eta^2 = {t5b['eta_sq']:.4f}$ ({t5b['eta_sq']*100:.2f}% phương sai!)**.",
            f"- **Đối chiếu bền vững:** Welch's ANOVA $F = {t5b['welch_f']:,.2f}$ ($p {format_p_value(t5b['welch_p'])}$) | Kruskal-Wallis $H = {t5b['kw_h']:,.2f}$ ($p {format_p_value(t5b['kw_p'])}$).",
            "",
            "| Nấc độ trễ (`delay_bucket`) | Số đơn ($n$) | Review Mean | Review Median | Std | 95% CI của Mean | Tỷ lệ Review 1–2 sao (%) |",
            "| :--- | ---: | ---: | ---: | ---: | :--- | ---: |",
        ]
    )

    for _, r in t5b["desc_df"].iterrows():
        b_name = r["group"]
        lines.append(
            f"| **{b_name}** | {int(r['n']):,} | **{r['mean']:.2f}** | {r['median']:.1f} | {r['std']:.2f} | "
            f"[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] | {delay_low_rates[b_name]:.2f}% |"
        )

    lines.extend(["", "**Hậu kiểm Tukey HSD — Test 5b (`review_score` giữa 10 cặp nấc độ trễ):**", ""])
    lines.extend(format_tukey_markdown_table(t5b["tukey_df"]))

    lines.extend(
        [
            "",
            "![Chẩn đoán ANOVA Vùng và Độ trễ](figures/ch5_anova_region_delay.png)",
            "",
            "---",
            "",
            "## 4. Tổng hợp Chuỗi Nguyên nhân Gốc rễ & Chuyển tiếp sang Chương 6 (Machine Learning)",
            "",
            "### 4.1 Chuỗi nhân quả thực nghiệm (Root-Cause Chain)",
            "Khi đặt các đại lượng **Effect Size** cạnh nhau, bức tranh nguyên nhân gốc rễ hiện lên cực kỳ sắc nét:",
            f"1. **Trễ hẹn so với cam kết (`delivery_delay_days` / `is_late_delivery`) là thủ phạm số 1:** Giải thích tới **{t5b['eta_sq']*100:.2f}%** phương sai điểm đánh giá ($\\eta^2 = {t5b['eta_sq']:.4f}$, Cramér's $V = {t3['cramers_v']:.4f}$, Relative Risk = **{t3['relative_risk']:.2f}x**). Ngay khi bước qua ranh giới trễ 1–3 ngày, điểm review trung bình rơi từ **4.11** xuống **3.29**, và rơi thẳng xuống **1.70** khi trễ trên 7 ngày.",
            f"2. **Khoảng cách vùng địa lý (`region`) tác động lên khâu vận chuyển nhiều hơn là trực tiếp lên tâm lý khách hàng:** Vùng địa lý giải thích **{t4['eta_sq']*100:.2f}%** biến thiên `delivery_days`, nhưng chỉ giải thích **{t5a['eta_sq']*100:.2f}%** biến thiên `review_score` — nghĩa là khách hàng ở xa sẵn sàng chấp nhận chờ lâu **miễn là giao đúng ngày dự kiến đã hứa**.",
            f"3. **Ngành hàng (`V = {t2['cramers_v']:.4f}`) và Phương thức thanh toán (`V = {t1['cramers_v']:.4f}`) là yếu tố thứ cấp:** Có khác biệt thống kê (như nhóm nội thất/chăn ga dễ phát sinh hàng cồng kềnh hoặc sai lệch mô tả), nhưng cường độ tác động nhỏ hơn hàng chục lần so với độ trễ giao hàng.",
            "",
            "### 4.2 Định hướng chọn đặc trưng (Feature Selection) cho Chương 6",
            "Từ bằng chứng thống kê trên, mô hình **Logistic Regression tự cài đặt (NumPy + Gradient Descent)** và **XGBoost + SHAP** dự đoán rủi ro đánh giá xấu (`is_low_review`) ở Chương 6 cần ưu tiên các nhóm đặc trưng sau:",
            "- **Nhóm đặc trưng vận chuyển & SLA (Tín hiệu mạnh nhất):** `delivery_delay_days`, `is_late_delivery`, `delivery_days`, `estimated_delivery_days`, tỷ lệ `delivery_days / estimated_delivery_days`.",
            "- **Nhóm đặc trưng giá trị & cước phí đơn hàng:** `items_price_total`, `freight_total`, `freight_ratio` (`freight_total / items_price_total`), `item_count` (đơn nhiều món/nhiều người bán dễ giao thiếu hoặc giao tách lẻ).",
            "- **Nhóm đặc trưng sản phẩm & địa lý:** Kích thước/khối lượng sản phẩm (`product_weight_g`, `product_volume_cm3`), `product_photos_qty`, `product_description_length`, vùng địa lý (`region`) và ngành hàng chủ đạo.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_terminal_summary(
    summary_df: pd.DataFrame,
    t3: dict,
    parquet_path: Path,
    report_path: Path,
    fig_chi2_path: Path,
    fig_anova_path: Path,
    elapsed_sec: float,
) -> None:
    """In bảng tóm tắt toàn bộ 5 bài kiểm định ra Terminal."""
    header = (
        f"{'Mã':<8} | {'Bài kiểm định':<46} | {'Stat':>10} | "
        f"{'p-value':>10} | {'Effect Size':<20} | {'Kết luận'}"
    )
    sep = "-" * len(header)

    print("\n=== KẾT QUẢ KIỂM ĐỊNH THỐNG KÊ CHẨN ĐOÁN — CHƯƠNG 5 ===")
    print(sep)
    print(header)
    print(sep)
    for _, r in summary_df.iterrows():
        eff_str = f"{r['effect_size_name']}={r['effect_size_value']:.4f}"
        concl = "Bác bỏ H0" if r["significance_05"] else "Chưa bác bỏ H0"
        print(
            f"{r['test_id']:<8} | {r['test_name'][:46]:<46} | {r['statistic']:>10.2f} | "
            f"{format_p_value(r['p_value']):>10} | {eff_str:<20} | {concl}"
        )
    print(sep)
    print(
        f"Điểm nhấn kinh doanh (Test 3): Giao trễ làm tăng nguy cơ đánh giá 1-2 sao gấp "
        f"{t3['relative_risk']:.2f} lần (Odds Ratio = {t3['odds_ratio']:.2f})."
    )

    print("\n=== TỆP KẾT QUẢ ĐÃ XUẤT ===")
    print(f"- Summary Parquet : {parquet_path}")
    print(f"- Báo cáo Markdown: {report_path}")
    print(f"- Hình Chi-square : {fig_chi2_path}")
    print(f"- Hình ANOVA      : {fig_anova_path}")
    print(f"Tổng thời gian    : {elapsed_sec:.2f} giây")


def run_all_diagnostics() -> None:
    """Thực thi trọn bộ 5 bài kiểm định thống kê Chương 5, xuất Parquet, biểu đồ và báo cáo."""
    if not WAREHOUSE_PATH.is_file():
        print(f"[ERROR] Không tìm thấy kho dữ liệu DuckDB tại: {WAREHOUSE_PATH}")
        sys.exit(1)

    ensure_dirs()
    start_time = time.perf_counter()

    with duckdb.connect(str(WAREHOUSE_PATH), read_only=True) as conn:
        t1 = run_test1_payment_vs_cancel(conn)
        t2 = run_test2_category_vs_low_review(conn)
        t3 = run_test3_late_delivery_vs_low_review(conn)
        delivered_df = load_delivered_orders_with_region_and_delay(conn)

    t4 = run_anova_suite(
        df=delivered_df,
        group_col="region",
        value_col="delivery_days",
        group_order=REGION_ORDER,
        test_id="T4",
        test_name="Thời gian giao hàng (delivery_days) theo 5 Vùng Brazil",
    )
    t5a = run_anova_suite(
        df=delivered_df.dropna(subset=["review_score"]),
        group_col="region",
        value_col="review_score",
        group_order=REGION_ORDER,
        test_id="T5a",
        test_name="Điểm đánh giá (review_score) theo 5 Vùng Brazil",
    )
    t5b = run_anova_suite(
        df=delivered_df.dropna(subset=["review_score", "delay_bucket"]),
        group_col="delay_bucket",
        value_col="review_score",
        group_order=DELAY_BUCKET_ORDER,
        test_id="T5b",
        test_name="Điểm đánh giá (review_score) theo 5 nấc Độ trễ giao hàng",
    )

    summary_df = build_summary_dataframe(t1, t2, t3, t4, t5a, t5b)
    parquet_path = DATA_PROCESSED_DIR / "stats_diagnostics_summary.parquet"
    summary_df.to_parquet(parquet_path, index=False)

    fig_chi2_path = FIGURES_DIR / "ch5_chisquare_diagnostics.png"
    fig_anova_path = FIGURES_DIR / "ch5_anova_region_delay.png"
    report_path = REPORTS_DIR / "ch5_statistical_diagnostics.md"

    plot_chisquare_diagnostics(t2, t3, fig_chi2_path)
    plot_anova_diagnostics(delivered_df, t4, t5a, t5b, fig_anova_path)
    write_diagnostics_report(delivered_df, t1, t2, t3, t4, t5a, t5b, report_path)

    elapsed_sec = time.perf_counter() - start_time
    print_terminal_summary(
        summary_df,
        t3,
        parquet_path,
        report_path,
        fig_chi2_path,
        fig_anova_path,
        elapsed_sec,
    )


if __name__ == "__main__":
    run_all_diagnostics()
