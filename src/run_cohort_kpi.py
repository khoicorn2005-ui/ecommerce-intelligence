"""Thực thi SQL tính KPI & Cohort Retention, xuất Parquet, vẽ biểu đồ và tạo báo cáo insight."""

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    SQL_DIR,
    WAREHOUSE_PATH,
    ensure_dirs,
)

MART_TABLES: tuple[str, ...] = (
    "mart_monthly_kpi",
    "mart_cohort_retention",
    "mart_category_performance",
    "mart_state_performance",
)


def export_marts_to_parquet(
    conn: duckdb.DuckDBPyConnection,
    processed_dir: Path,
) -> list[tuple[str, int, int, float]]:
    """Xuất 4 bảng `dw.mart_*` sang Parquet và trả về thống kê dung lượng."""
    stats: list[tuple[str, int, int, float]] = []
    for table in MART_TABLES:
        out_path = processed_dir / f"{table}.parquet"
        conn.execute(f"COPY dw.{table} TO '{out_path.as_posix()}' (FORMAT PARQUET);")
        rows = conn.execute(f"SELECT COUNT(*) FROM dw.{table}").fetchone()[0]
        cols = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'dw' AND table_name = ?
            """,
            [table],
        ).fetchone()[0]
        size_mb = out_path.stat().st_size / (1024 * 1024)
        stats.append((f"dw.{table}", rows, cols, size_mb))
    return stats


def fetch_overall_kpis(conn: duckdb.DuckDBPyConnection) -> dict:
    """Tính các chỉ số KPI tổng hợp toàn kỳ trên các đơn hàng hợp lệ."""
    row = conn.execute(
        """
        WITH valid_orders AS (
            SELECT *
            FROM dw.fact_orders
            WHERE order_status NOT IN ('canceled', 'unavailable')
        ),
        cust_orders AS (
            SELECT customer_unique_id, COUNT(*) AS n_orders
            FROM valid_orders
            GROUP BY customer_unique_id
        )
        SELECT
            COUNT(*) AS total_orders,
            COUNT(DISTINCT customer_unique_id) AS unique_customers,
            (SELECT COUNT(*) FROM cust_orders WHERE n_orders >= 2) AS repeat_customers,
            SUM(payment_total) AS total_gmv,
            SUM(items_price_total) AS net_product_revenue,
            SUM(freight_total) AS total_freight,
            SUM(payment_total) / NULLIF(COUNT(*), 0) AS aov,
            AVG(CASE WHEN is_valid_delivered THEN delivery_days END) AS avg_delivery_days,
            AVG(
                CASE
                    WHEN is_valid_delivered
                    THEN CASE WHEN is_late_delivery THEN 100.0 ELSE 0.0 END
                END
            ) AS late_delivery_rate,
            AVG(review_score) AS avg_review_score,
            AVG(
                CASE
                    WHEN is_low_review IS NOT NULL
                    THEN CASE WHEN is_low_review THEN 100.0 ELSE 0.0 END
                END
            ) AS low_review_rate
        FROM valid_orders
        """
    ).fetchone()

    total_orders, unique_customers, repeat_customers = row[0], row[1], row[2]
    return {
        "total_orders": int(total_orders),
        "unique_customers": int(unique_customers),
        "repeat_customers": int(repeat_customers),
        "repeat_rate_pct": (repeat_customers / unique_customers * 100.0) if unique_customers else 0.0,
        "total_gmv": float(row[3]),
        "net_product_revenue": float(row[4]),
        "total_freight": float(row[5]),
        "aov": float(row[6]),
        "avg_delivery_days": float(row[7]),
        "late_delivery_rate": float(row[8]),
        "avg_review_score": float(row[9]),
        "low_review_rate": float(row[10]),
    }


def plot_monthly_kpi_trends(monthly_df: pd.DataFrame, output_path: Path) -> None:
    """Vẽ biểu đồ xu hướng KPI theo tháng (2017-01 đến 2018-08) gồm 2 đồ thị xếp dọc."""
    df = monthly_df[
        (monthly_df["purchase_year_month"] >= "2017-01")
        & (monthly_df["purchase_year_month"] <= "2018-08")
    ].copy()
    df = df.sort_values("month_start_date").reset_index(drop=True)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=150, facecolor="white")
    fig.suptitle(
        "Xu hướng Tăng trưởng & Chất lượng Vận hành theo Tháng (2017-01 đến 2018-08)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(df))
    labels = df["purchase_year_month"].tolist()
    gmv_millions = df["gmv_payment"] / 1e6

    # --- Subplot 1: GMV (cột) + Số đơn hàng (đường) ---
    bars = ax1.bar(
        x,
        gmv_millions,
        width=0.62,
        color="#2563EB",
        alpha=0.82,
        label="GMV theo tháng (Triệu BRL)",
    )
    ax1.set_ylabel("GMV (Triệu BRL)", fontsize=10.5, fontweight="bold", color="#1E3A8A")
    ax1.set_ylim(0, gmv_millions.max() * 1.25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax1.set_title(
        "Quy mô GMV & Số lượng Đơn hàng hợp lệ (Đã loại các tháng biên cắt dữ liệu: 2016 & 2018-09/10)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    ax1_r = ax1.twinx()
    ax1_r.grid(False)
    line_orders = ax1_r.plot(
        x,
        df["total_orders"],
        color="#EA580C",
        marker="o",
        linewidth=2.3,
        markersize=5.5,
        label="Số đơn hàng hợp lệ",
    )
    ax1_r.set_ylabel("Số đơn hàng", fontsize=10.5, fontweight="bold", color="#9A3412")
    ax1_r.set_ylim(0, df["total_orders"].max() * 1.25)

    peak_idx = int(df["gmv_payment"].idxmax())
    peak_row = df.iloc[peak_idx]
    ax1.annotate(
        f"Đỉnh Black Friday ({peak_row['purchase_year_month']})\n"
        f"GMV: {peak_row['gmv_payment']/1e6:.2f}M BRL | {int(peak_row['total_orders']):,} đơn",
        xy=(peak_idx, gmv_millions.iloc[peak_idx]),
        xytext=(max(peak_idx - 4.3, 0.5), gmv_millions.max() * 1.07),
        fontsize=9,
        fontweight="bold",
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF3C7", edgecolor="#F59E0B"),
        arrowprops=dict(arrowstyle="->", color="#B45309", lw=1.4),
    )

    handles1 = [bars] + line_orders
    labels1 = [h.get_label() for h in handles1]
    ax1.legend(
        handles1,
        labels1,
        loc="upper left",
        frameon=True,
        facecolor="white",
        fontsize=9.5,
    )

    # --- Subplot 2: Tỷ lệ giao trễ (%) + Điểm đánh giá TB ---
    line_late = ax2.plot(
        x,
        df["late_delivery_rate"],
        color="#DC2626",
        marker="s",
        linewidth=2.3,
        markersize=5.5,
        label="Tỷ lệ giao hàng trễ (%)",
    )
    ax2.set_ylabel("Tỷ lệ giao trễ (%)", fontsize=10.5, fontweight="bold", color="#991B1B")
    ax2.set_ylim(0, df["late_delivery_rate"].max() * 1.28)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax2.set_title(
        "Tương quan Nghịch biến giữa Tỷ lệ Giao hàng Trễ và Điểm Đánh giá Khách hàng",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    ax2_r = ax2.twinx()
    ax2_r.grid(False)
    line_rev = ax2_r.plot(
        x,
        df["avg_review_score"],
        color="#0D9488",
        marker="D",
        linewidth=2.3,
        markersize=5.2,
        linestyle="-.",
        label="Điểm đánh giá TB (1–5 sao)",
    )
    ax2_r.set_ylabel("Điểm đánh giá TB (1–5)", fontsize=10.5, fontweight="bold", color="#115E59")
    ax2_r.set_ylim(3.55, 4.45)

    worst_idx = int(df["late_delivery_rate"].idxmax())
    worst_row = df.iloc[worst_idx]
    ax2.annotate(
        f"Đỉnh giao trễ ({worst_row['purchase_year_month']})\n"
        f"Trễ: {worst_row['late_delivery_rate']:.2f}% | Review: {worst_row['avg_review_score']:.2f}/5",
        xy=(worst_idx, worst_row["late_delivery_rate"]),
        xytext=(max(worst_idx - 5.2, 0.5), worst_row["late_delivery_rate"] + 2.8),
        fontsize=8.8,
        fontweight="bold",
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEE2E2", edgecolor="#EF4444"),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.4),
    )

    best_idx = int(df["late_delivery_rate"].idxmin())
    best_row = df.iloc[best_idx]
    ax2.annotate(
        f"Phục hồi ({best_row['purchase_year_month']})\n"
        f"Trễ: {best_row['late_delivery_rate']:.2f}% | Review: {best_row['avg_review_score']:.2f}/5",
        xy=(best_idx, best_row["late_delivery_rate"]),
        xytext=(best_idx - 0.6, 12.8),
        ha="center",
        fontsize=8.8,
        fontweight="bold",
        color="#065F46",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#D1FAE5", edgecolor="#10B981"),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.4),
    )

    handles2 = line_late + line_rev
    labels2 = [h.get_label() for h in handles2]
    ax2.legend(
        handles2,
        labels2,
        loc="upper left",
        frameon=True,
        facecolor="white",
        fontsize=9.5,
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_cohort_matrix_for_heatmap(
    cohort_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chuẩn bị ma trận Cohort Retention (2017-01..2018-08, Month 1..12) và mask right-censoring."""
    cohorts = sorted(
        cohort_df[
            (cohort_df["cohort_year_month"] >= "2017-01")
            & (cohort_df["cohort_year_month"] <= "2018-08")
        ]["cohort_year_month"].unique()
    )
    max_obs_period = pd.Period("2018-08", freq="M")
    month_cols = list(range(1, 13))

    sizes = (
        cohort_df[cohort_df["cohort_index"] == 0]
        .set_index("cohort_year_month")["cohort_size"]
        .to_dict()
    )

    lookup = (
        cohort_df[
            (cohort_df["cohort_year_month"].isin(cohorts))
            & (cohort_df["cohort_index"].isin(month_cols))
        ]
        .set_index(["cohort_year_month", "cohort_index"])["retention_rate_pct"]
        .to_dict()
    )

    matrix = pd.DataFrame(index=cohorts, columns=month_cols, dtype=float)
    for c_ym in cohorts:
        c_period = pd.Period(c_ym, freq="M")
        for m_idx in month_cols:
            if c_period + m_idx <= max_obs_period:
                matrix.loc[c_ym, m_idx] = float(lookup.get((c_ym, m_idx), 0.0))
            else:
                matrix.loc[c_ym, m_idx] = np.nan

    y_labels = [f"{c_ym} (n={int(sizes[c_ym]):,})" for c_ym in cohorts]
    matrix_labeled = matrix.copy()
    matrix_labeled.index = y_labels
    matrix_labeled.columns = [f"M+{m}" for m in month_cols]

    return matrix, matrix_labeled


def plot_cohort_heatmap(matrix_labeled: pd.DataFrame, output_path: Path) -> None:
    """Vẽ Heatmap Ma trận Cohort Retention tách riêng Month 1..12 khỏi Month 0 = 100%."""
    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(14, 9.5), dpi=150, facecolor="white")

    annot_labels = matrix_labeled.map(
        lambda v: f"{v:.2f}%" if pd.notna(v) else ""
    )
    mask = matrix_labeled.isna()
    ax.set_facecolor("#F1F5F9")

    sns.heatmap(
        matrix_labeled,
        mask=mask,
        annot=annot_labels,
        fmt="",
        cmap="YlGnBu",
        linewidths=0.6,
        linecolor="white",
        vmin=0.0,
        vmax=float(np.nanmax(matrix_labeled.values)),
        cbar_kws={"label": "Tỷ lệ giữ chân khách hàng (%)", "shrink": 0.85},
        annot_kws={"size": 8.5},
        ax=ax,
    )

    ax.set_title(
        "Ma trận Cohort Retention Rate (Month 1 – Month 12, tách khỏi Month 0 = 100%)",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Số tháng sau đơn hàng đầu tiên (Cohort Index)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Tháng Cohort & Quy mô ban đầu (Month 0 = 100%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.tick_params(axis="x", labelsize=9.5, rotation=0)
    ax.tick_params(axis="y", labelsize=9, rotation=0)

    fig.text(
        0.5,
        0.005,
        "Ghi chú: Ô màu xám nhạt là các tháng vượt quá mốc dữ liệu đầy đủ 2018-08 (Right-Censoring).",
        ha="center",
        fontsize=9,
        color="#475569",
    )

    fig.tight_layout(rect=[0, 0.025, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_insights_report(
    overall: dict,
    monthly_df: pd.DataFrame,
    cohort_matrix_raw: pd.DataFrame,
    category_df: pd.DataFrame,
    state_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Xuất báo cáo phân tích KPI & Cohort Insights Chương 4 ra file Markdown."""
    valid_months = monthly_df[
        (monthly_df["purchase_year_month"] >= "2017-01")
        & (monthly_df["purchase_year_month"] <= "2018-08")
    ].copy()
    peak_gmv_row = valid_months.loc[valid_months["gmv_payment"].idxmax()]
    worst_late_row = valid_months.loc[valid_months["late_delivery_rate"].idxmax()]
    best_late_row = valid_months.loc[valid_months["late_delivery_rate"].idxmin()]

    m1_avg = float(cohort_matrix_raw[1].dropna().mean())
    m2_avg = float(cohort_matrix_raw[2].dropna().mean())
    m3_avg = float(cohort_matrix_raw[3].dropna().mean())
    m6_avg = float(cohort_matrix_raw[6].dropna().mean())
    m1_max = float(cohort_matrix_raw[1].dropna().max())
    m1_max_cohort = str(cohort_matrix_raw[1].dropna().idxmax())

    top10_cats = category_df.head(10)
    top5_states_gmv = state_df.head(5)
    slow_states = state_df.sort_values("avg_delivery_days", ascending=False).head(5)
    low_rev_states = state_df.sort_values("avg_review_score", ascending=True).head(5)

    lines: list[str] = [
        "# Báo cáo Phân tích KPI Kinh doanh & Cohort Retention — Chương 4",
        "",
        f"- **Thời điểm tạo báo cáo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- **Phạm vi dữ liệu:** Các đơn hàng hợp lệ (`order_status NOT IN ('canceled', 'unavailable')`) từ kho `olist.duckdb` (`dw.*`).",
        "",
        "---",
        "",
        "## 1. Tổng quan KPI Toàn kỳ",
        "",
        "| Chỉ số KPI | Giá trị thực tế | Ghi chú nghiệp vụ |",
        "| :--- | ---: | :--- |",
        f"| Tổng đơn hàng hợp lệ | **{overall['total_orders']:,}** | Không tính đơn `canceled` và `unavailable` |",
        f"| Tổng khách hàng duy nhất (`customer_unique_id`) | **{overall['unique_customers']:,}** | Định danh con người thật |",
        f"| Khách mua lặp lại (≥ 2 đơn) | **{overall['repeat_customers']:,}** ({overall['repeat_rate_pct']:.2f}%) | Phản ánh mức độ trung thành rất thấp |",
        f"| Tổng GMV (`payment_total`) | **{overall['total_gmv']:,.2f} BRL** | Tổng giá trị thanh toán gồm sản phẩm & vận chuyển |",
        f"| Doanh thu sản phẩm ròng (`items_price_total`) | **{overall['net_product_revenue']:,.2f} BRL** | Tổng tiền hàng trước phí ship |",
        f"| Giá trị đơn hàng trung bình (AOV) | **{overall['aov']:,.2f} BRL** | `total_gmv / total_orders` |",
        f"| Thời gian giao hàng trung bình | **{overall['avg_delivery_days']:.2f} ngày** | Tính trên đơn `is_valid_delivered = TRUE` |",
        f"| Tỷ lệ giao hàng trễ hẹn | **{overall['late_delivery_rate']:.2f}%** | So với `order_estimated_delivery_date` |",
        f"| Điểm đánh giá trung bình | **{overall['avg_review_score']:.2f} / 5.00** | Trên các đơn có đánh giá |",
        f"| Tỷ lệ đánh giá thấp (1–2 sao) | **{overall['low_review_rate']:.2f}%** | Tỷ trọng nhóm rủi ro trải nghiệm xấu |",
        "",
        "---",
        "",
        "## 2. Xu hướng Tăng trưởng & Chất lượng Vận hành theo Tháng",
        "",
        "> **Lưu ý phạm vi biểu đồ:** Các tháng biên `2016-09`, `2016-10`, `2016-12` và `2018-09` chỉ có 1–293 đơn do cắt đầu/cuối tập dữ liệu thu thập (*edge-period censoring*). Biểu đồ tập trung vào **20 tháng vận hành đầy đủ từ `2017-01` đến `2018-08`** để tránh làm méo đường xu hướng và tỷ lệ tăng trưởng MoM.",
        "",
        "![Xu hướng KPI theo tháng](figures/ch4_monthly_kpi_trends.png)",
        "",
        f"- **Đà tăng trưởng & Đỉnh Black Friday:** GMV tăng mạnh xuyên suốt năm 2017 và đạt đỉnh lịch sử vào **{peak_gmv_row['purchase_year_month']}** với **{peak_gmv_row['gmv_payment']:,.2f} BRL** ({int(peak_gmv_row['total_orders']):,} đơn hàng), phản ánh hiệu ứng khuyến mãi cuối năm.",
        f"- **Điểm nghẽn Logistics đầu năm 2018:** Tỷ lệ giao hàng trễ vọt lên đỉnh **{worst_late_row['late_delivery_rate']:.2f}%** vào **{worst_late_row['purchase_year_month']}** (từ mức 15.99% của `2018-02`), kéo điểm đánh giá trung bình rơi xuống đáy **{worst_late_row['avg_review_score']:.2f}★**.",
        f"- **Phục hồi vận hành:** Từ `2018-04` đến `2018-08`, khi tỷ lệ giao trễ được kéo giảm về đáy **{best_late_row['late_delivery_rate']:.2f}%** ở **{best_late_row['purchase_year_month']}**, điểm đánh giá trung bình nhanh chóng hồi phục lên **{best_late_row['avg_review_score']:.2f}★** — cho thấy sự nhạy cảm rất rõ giữa tiến độ giao hàng và sự hài lòng.",
        "",
        "---",
        "",
        "## 3. Phân tích Ma trận Cohort Giữ chân Khách hàng",
        "",
        "![Ma trận Cohort Retention](figures/ch4_cohort_retention_heatmap.png)",
        "",
        "| Mốc tuổi Cohort | Tỷ lệ giữ chân trung bình (2017-01 – 2018-08) | Ghi chú |",
        "| :--- | ---: | :--- |",
        f"| **Month 1 (M+1)** | **{m1_avg:.2f}%** | Cao nhất đạt {m1_max:.2f}% ở cohort `{m1_max_cohort}` |",
        f"| **Month 2 (M+2)** | **{m2_avg:.2f}%** | Giảm sâu sau tháng đầu |",
        f"| **Month 3 (M+3)** | **{m3_avg:.2f}%** | Duy trì quanh mức ~0.25% |",
        f"| **Month 6 (M+6)** | **{m6_avg:.2f}%** | Gần như không có nhịp quay lại tự nhiên |",
        "",
        "**Phân tích bối cảnh kinh doanh tại sao Retention của Olist rất thấp (~0.2%–0.5%):**",
        "1. **Mô hình sàn trung gian (Aggregator Marketplace):** Khách hàng tại Brazil mua qua nhiều kênh đối tác gắn nhãn Olist; trải nghiệm thương hiệu gắn với cửa hàng/sản phẩm hơn là nhận diện sàn Olist, nên thiếu động lực quay lại trực tiếp.",
        "2. **Cơ cấu ngành hàng mua sắm thưa (Low-frequency Categories):** Các ngành chủ lực như đồng hồ/quà tặng, đồ nội thất/gia dụng (`bed_bath_table`), điện tử/phụ kiện có chu kỳ thay mới dài, khác với hàng tiêu dùng nhanh (FMCG).",
        "3. **Không có cơ chế giữ chân sau bán (CRM/Loyalty):** Thiếu chương trình tích điểm, voucher tái mua trong 30 ngày đầu (*M+1*) và bị ảnh hưởng bởi các đợt giao trễ (8.11% toàn kỳ).",
        "4. **Gợi ý hành động:** Tập trung chiến dịch tái kích hoạt (*Win-back / Voucher*) vào cửa sổ vàng **15–45 ngày sau đơn đầu (M+1)** cho nhóm ngành hàng có tần suất lặp tự nhiên cao (như `health_beauty`, `pet_shop`) và ưu tiên cải thiện đúng hạn cho đơn đầu tiên.",
        "",
        "---",
        "",
        "## 4. Hiệu suất theo Ngành hàng & theo Bang",
        "",
        "### 4.1 Top 10 Ngành hàng có Doanh thu Sản phẩm cao nhất",
        "",
        "| Hạng | Ngành hàng (EN) | Số đơn | Số sản phẩm | Doanh thu SP (BRL) | Giá TB (BRL) | Tỷ lệ giao trễ (%) | Điểm Review TB | Tỷ lệ Review thấp (%) |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for idx, (_, r) in enumerate(top10_cats.iterrows(), start=1):
        lines.append(
            f"| {idx} | `{r['product_category_en']}` | {int(r['total_orders']):,} | {int(r['total_items_sold']):,} | "
            f"{r['total_product_revenue']:,.2f} | {r['avg_item_price']:,.2f} | "
            f"{r['late_delivery_rate']:.2f}% | {r['avg_review_score']:.2f} | {r['low_review_rate']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "### 4.2 Top 5 Bang có GMV lớn nhất",
            "",
            "| Bang | Số đơn | Khách duy nhất | Tổng GMV (BRL) | AOV (BRL) | Giao TB (Ngày) | Giao Median (Ngày) | Tỷ lệ trễ (%) | Review TB |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, r in top5_states_gmv.iterrows():
        lines.append(
            f"| **{r['customer_state']}** | {int(r['total_orders']):,} | {int(r['unique_customers']):,} | "
            f"{r['total_gmv']:,.2f} | {r['aov']:,.2f} | {r['avg_delivery_days']:.2f} | "
            f"{r['median_delivery_days']:.2f} | {r['late_delivery_rate']:.2f}% | {r['avg_review_score']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### 4.3 Top 5 Bang giao hàng lâu nhất & Top 5 Bang có điểm Review thấp nhất",
            "",
            "| Nhóm | Bang | Số đơn | Giao TB (Ngày) | Giao Median (Ngày) | Tỷ lệ trễ (%) | Điểm Review TB | Tỷ lệ Review thấp (%) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, r in slow_states.iterrows():
        lines.append(
            f"| Giao lâu nhất | **{r['customer_state']}** | {int(r['total_orders']):,} | {r['avg_delivery_days']:.2f} | "
            f"{r['median_delivery_days']:.2f} | {r['late_delivery_rate']:.2f}% | {r['avg_review_score']:.2f} | {r['low_review_rate']:.2f}% |"
        )
    for _, r in low_rev_states.iterrows():
        lines.append(
            f"| Review thấp nhất | **{r['customer_state']}** | {int(r['total_orders']):,} | {r['avg_delivery_days']:.2f} | "
            f"{r['median_delivery_days']:.2f} | {r['late_delivery_rate']:.2f}% | {r['avg_review_score']:.2f} | {r['low_review_rate']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Giả thuyết chuyển tiếp sang Chương 5 (Chẩn đoán Thống kê)",
            "",
            "Từ các mẫu hình mô tả ở Chương 4 (như bang `RJ` giao TB 15.31 ngày / trễ 13.47% / review 3.90★ so với `SP` giao 8.76 ngày / trễ 5.89% / review 4.21★, và đỉnh giao trễ tháng 2018-03 làm review tụt xuống 3.76★), ta đề xuất **3 giả thuyết kiểm định thống kê** cho Chương 5:",
            "",
            "1. **Giả thuyết 1 (Khác biệt địa lý — Levene + One-Way ANOVA + Tukey HSD):** Thời gian giao hàng (`delivery_days`) và điểm đánh giá (`review_score`) có sự khác biệt có ý nghĩa thống kê giữa các bang/vùng địa lý hay không (đặc biệt giữa nhóm bang lõi Đông Nam như `SP` với `RJ` và các bang xa miền Bắc/Đông Bắc như `RR`, `AP`, `AL`, `MA`)?",
            "2. **Giả thuyết 2 (Tác nhân gây đánh giá thấp & hủy đơn — Chi-square Test of Independence):** Việc giao hàng trễ (`is_late_delivery`), ngành hàng sản phẩm (`product_category_en`) hoặc phương thức thanh toán (`primary_payment_type`) có mối liên hệ độc lập/phụ thuộc có ý nghĩa thống kê với tỷ lệ đánh giá xấu (`is_low_review`) và trạng thái hủy đơn (`canceled`) hay không?",
            "3. **Giả thuyết 3 (Cường độ tác động của trễ hẹn lên điểm đánh giá — Mann-Whitney U / Independent Test):** Chênh lệch thời gian giao hàng và số ngày trễ hẹn (`delivery_delay_days`) giữa nhóm đơn hàng bị 1–2 sao và nhóm 4–5 sao có khác biệt đáng kể về mặt thống kê hay không, nhằm định lượng ngưỡng SLA giao hàng mục tiêu?",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_terminal_summary(
    overall: dict,
    mart_stats: list[tuple[str, int, int, float]],
    report_path: Path,
    fig_trends_path: Path,
    fig_cohort_path: Path,
    elapsed_sec: float,
) -> None:
    """In tóm tắt KPI toàn kỳ, 4 bảng Data Mart và đường dẫn file đã xuất."""
    print("\n=== TÓM TẮT KPI TOÀN KỲ (ĐƠN HÀNG HỢP LỆ) ===")
    print(f"- Tổng đơn hàng hợp lệ  : {overall['total_orders']:,}")
    print(f"- Khách hàng duy nhất   : {overall['unique_customers']:,}")
    print(f"- Khách mua lặp lại     : {overall['repeat_customers']:,} ({overall['repeat_rate_pct']:.2f}%)")
    print(f"- Tổng GMV (Payment)    : {overall['total_gmv']:,.2f} BRL")
    print(f"- Doanh thu SP ròng     : {overall['net_product_revenue']:,.2f} BRL")
    print(f"- AOV trung bình        : {overall['aov']:,.2f} BRL")
    print(f"- Giao hàng TB / Trễ    : {overall['avg_delivery_days']:.2f} ngày | {overall['late_delivery_rate']:.2f}%")
    print(f"- Điểm Review TB / Thấp : {overall['avg_review_score']:.2f}★ | {overall['low_review_rate']:.2f}%")

    header = (
        f"{'Bảng Data Mart':<30} | {'Số dòng':>10} | {'Số cột':>7} | {'Parquet (MB)':>12}"
    )
    sep = "-" * len(header)
    print("\n=== 4 BẢNG DATA MART & FILE PARQUET ===")
    print(sep)
    print(header)
    print(sep)
    for tbl, rows, cols, size_mb in mart_stats:
        print(f"{tbl:<30} | {rows:>10,} | {cols:>7} | {size_mb:>12.2f}")
    print(sep)

    print("\n=== TỆP BÁO CÁO & BIỂU ĐỒ ĐÃ XUẤT ===")
    print(f"- Thư mục Parquet  : {DATA_PROCESSED_DIR}")
    print(f"- Báo cáo Insight  : {report_path}")
    print(f"- Biểu đồ KPI      : {fig_trends_path}")
    print(f"- Heatmap Cohort   : {fig_cohort_path}")
    print(f"Tổng thời gian chạy: {elapsed_sec:.2f} giây")


def run_cohort_kpi() -> None:
    """Chạy toàn bộ pipeline Chương 4 Chặng 2: SQL Marts, Parquet, 2 biểu đồ và báo cáo."""
    sql_file = SQL_DIR / "02_kpi_and_cohort.sql"
    if not WAREHOUSE_PATH.is_file():
        print(f"[ERROR] Không tìm thấy file DuckDB tại: {WAREHOUSE_PATH}")
        sys.exit(1)
    if not sql_file.is_file():
        print(f"[ERROR] Không tìm thấy file SQL tại: {sql_file}")
        sys.exit(1)

    ensure_dirs()
    start_time = time.perf_counter()
    sql_script = sql_file.read_text(encoding="utf-8")

    with duckdb.connect(str(WAREHOUSE_PATH)) as conn:
        conn.execute(sql_script)
        mart_stats = export_marts_to_parquet(conn, DATA_PROCESSED_DIR)
        overall = fetch_overall_kpis(conn)
        monthly_df = conn.execute("SELECT * FROM dw.mart_monthly_kpi ORDER BY month_start_date").df()
        cohort_df = conn.execute(
            "SELECT * FROM dw.mart_cohort_retention ORDER BY cohort_month, cohort_index"
        ).df()
        category_df = conn.execute(
            "SELECT * FROM dw.mart_category_performance ORDER BY total_product_revenue DESC"
        ).df()
        state_df = conn.execute(
            "SELECT * FROM dw.mart_state_performance ORDER BY total_gmv DESC"
        ).df()

    fig_trends_path = FIGURES_DIR / "ch4_monthly_kpi_trends.png"
    fig_cohort_path = FIGURES_DIR / "ch4_cohort_retention_heatmap.png"
    report_path = REPORTS_DIR / "ch4_kpi_cohort_insights.md"

    plot_monthly_kpi_trends(monthly_df, fig_trends_path)
    cohort_matrix_raw, cohort_matrix_labeled = build_cohort_matrix_for_heatmap(cohort_df)
    plot_cohort_heatmap(cohort_matrix_labeled, fig_cohort_path)
    write_insights_report(
        overall,
        monthly_df,
        cohort_matrix_raw,
        category_df,
        state_df,
        report_path,
    )

    elapsed_sec = time.perf_counter() - start_time
    print_terminal_summary(
        overall,
        mart_stats,
        report_path,
        fig_trends_path,
        fig_cohort_path,
        elapsed_sec,
    )


if __name__ == "__main__":
    run_cohort_kpi()
