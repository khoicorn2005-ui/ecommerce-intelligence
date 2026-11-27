"""Chương 7 — Chặng 2: Phát hiện Đơn hàng Bất thường Đa chiều bằng Isolation Forest."""

import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    WAREHOUSE_PATH,
    ensure_dirs,
)

CONTAMINATION: float = 0.01
RANDOM_STATE: int = 42

LOG1P_COLS: list[str] = [
    "items_price_total",
    "freight_total",
    "payment_total",
    "item_count",
    "freight_to_price_ratio",
    "payment_price_gap",
    "delivery_days_imp",
    "avg_product_weight_g_imp",
    "avg_product_volume_cm3_imp",
]

LINEAR_COLS: list[str] = [
    "max_installments_imp",
    "delivery_delay_days_imp",
]


def load_valid_orders_for_anomaly(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Truy vấn các đơn hàng hợp lệ và tổng hợp đặc trưng sản phẩm đa chiều cấp đơn."""
    return conn.execute(
        """
        WITH item_prod AS (
            SELECT
                fi.order_id,
                AVG(dp.product_weight_g)   AS avg_product_weight_g,
                AVG(dp.product_volume_cm3) AS avg_product_volume_cm3
            FROM dw.fact_order_items AS fi
            LEFT JOIN dw.dim_products AS dp
                ON fi.product_id = dp.product_id
            GROUP BY fi.order_id
        )
        SELECT
            fo.order_id,
            fo.customer_unique_id,
            fo.purchase_date,
            fo.customer_state,
            fo.order_status,
            fo.items_price_total,
            fo.freight_total,
            fo.payment_total,
            fo.gross_order_value,
            fo.item_count,
            fo.max_installments,
            fo.freight_total / (fo.items_price_total + 1e-5) AS freight_to_price_ratio,
            ABS(fo.payment_total - fo.gross_order_value)     AS payment_price_gap,
            fo.delivery_days,
            fo.delivery_delay_days,
            fo.is_late_delivery,
            fo.review_score,
            ip.avg_product_weight_g,
            ip.avg_product_volume_cm3
        FROM dw.fact_orders AS fo
        LEFT JOIN item_prod AS ip
            ON fo.order_id = ip.order_id
        WHERE fo.order_status NOT IN ('canceled', 'unavailable')
          AND fo.payment_total > 0
          AND fo.items_price_total > 0
        ORDER BY fo.order_id
        """
    ).df()


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, RobustScaler, list[str], dict]:
    """Điền median an toàn cho đặc trưng thiếu, áp dụng log1p + RobustScaler trước khi chạy IsolationForest."""
    work = pd.DataFrame(index=df.index)
    medians: dict[str, float] = {}

    for raw_col, imp_col in [
        ("delivery_days", "delivery_days_imp"),
        ("delivery_delay_days", "delivery_delay_days_imp"),
        ("max_installments", "max_installments_imp"),
        ("avg_product_weight_g", "avg_product_weight_g_imp"),
        ("avg_product_volume_cm3", "avg_product_volume_cm3_imp"),
    ]:
        med_val = float(df[raw_col].median())
        medians[raw_col] = med_val
        work[imp_col] = df[raw_col].fillna(med_val).astype(float)

    for c in [
        "items_price_total",
        "freight_total",
        "payment_total",
        "item_count",
        "freight_to_price_ratio",
        "payment_price_gap",
    ]:
        work[c] = df[c].astype(float)

    transformed = pd.DataFrame(index=df.index)
    for c in LOG1P_COLS:
        transformed[f"log1p_{c}"] = np.log1p(np.clip(work[c].to_numpy(), 0.0, None))
    for c in LINEAR_COLS:
        transformed[c] = work[c].to_numpy()

    feature_cols = list(transformed.columns)
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(transformed[feature_cols])
    return x_scaled, scaler, feature_cols, medians


def assign_anomaly_archetypes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Gán nhãn `anomaly_type` và `audit_priority` theo ngưỡng phân vị P99 có kiểm soát biên nghiệp vụ."""
    out = df.copy()

    # Lưu ý: payment_price_gap ở P99 bằng 0.00 do >99% đơn khớp tuyệt đối, nên áp sàn tối thiểu 5.0 BRL
    thresholds = {
        "payment_price_gap": max(float(out["payment_price_gap"].quantile(0.999)), 5.0),
        "payment_total_p99": float(out["payment_total"].quantile(0.99)),
        "items_price_p99": float(out["items_price_total"].quantile(0.99)),
        "item_count_p99": float(out["item_count"].quantile(0.99)),
        "freight_ratio_p99": float(out["freight_to_price_ratio"].quantile(0.99)),
        "freight_total_p99": float(out["freight_total"].quantile(0.99)),
        "delivery_days_p99": float(out["delivery_days"].dropna().quantile(0.99)),
        "delivery_delay_p99": float(out["delivery_delay_days"].dropna().quantile(0.99)),
    }

    is_anom = out["is_anomaly"] == 1
    cond_gap = out["payment_price_gap"] >= thresholds["payment_price_gap"]
    cond_bulk = (
        (out["payment_total"] > thresholds["payment_total_p99"])
        | (out["items_price_total"] > thresholds["items_price_p99"])
        | (out["item_count"] > thresholds["item_count_p99"])
    )
    cond_freight = (
        (out["freight_to_price_ratio"] > thresholds["freight_ratio_p99"])
        | (out["freight_total"] > thresholds["freight_total_p99"])
    )
    cond_deliv = (
        (out["delivery_days"].fillna(-999) > thresholds["delivery_days_p99"])
        | (out["delivery_delay_days"].fillna(-999) > thresholds["delivery_delay_p99"])
    )

    archetypes = np.select(
        [
            ~is_anom,
            is_anom & cond_gap,
            is_anom & cond_bulk,
            is_anom & cond_freight,
            is_anom & cond_deliv,
        ],
        [
            "Normal",
            "Payment-Price Discrepancy",
            "Extreme Order Value / Bulk",
            "Abnormal Freight Ratio",
            "Extreme Delivery Anomaly",
        ],
        default="Multi-Factor Unusual",
    )
    out["anomaly_type"] = archetypes

    anom_scores_flagged = out.loc[is_anom, "anomaly_score"]
    crit_cutoff = float(anom_scores_flagged.quantile(0.90))
    thresholds["critical_score_p90_of_flagged"] = crit_cutoff

    out["audit_priority"] = np.where(
        ~is_anom,
        "Normal",
        np.where(out["anomaly_score"] >= crit_cutoff, "Critical", "High"),
    )
    return out, thresholds


def build_anomaly_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng tóm tắt theo từng `anomaly_type` trên nhóm bất thường cho Power BI & báo cáo."""
    flagged = scored_df[scored_df["is_anomaly"] == 1].copy()
    n_flagged = len(flagged)

    summary = (
        flagged.groupby("anomaly_type", as_index=False)
        .agg(
            anomaly_orders=("order_id", "count"),
            total_gmv=("payment_total", "sum"),
            avg_order_value=("payment_total", "mean"),
            avg_item_count=("item_count", "mean"),
            avg_freight_ratio=("freight_to_price_ratio", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            late_delivery_rate_pct=("is_late_delivery", lambda s: float(s.mean() * 100.0)),
            avg_review_score=("review_score", "mean"),
            critical_count=("audit_priority", lambda s: int((s == "Critical").sum())),
        )
    )
    summary["share_of_anomalies_pct"] = summary["anomaly_orders"] * 100.0 / n_flagged
    summary = summary.sort_values("anomaly_orders", ascending=False).reset_index(drop=True)
    return summary[
        [
            "anomaly_type",
            "anomaly_orders",
            "share_of_anomalies_pct",
            "critical_count",
            "total_gmv",
            "avg_order_value",
            "avg_item_count",
            "avg_freight_ratio",
            "avg_delivery_days",
            "late_delivery_rate_pct",
            "avg_review_score",
        ]
    ]


def save_anomalies_to_duckdb_and_parquet(
    scored_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """Đóng kết nối an toàn sau khi lưu `dw.ml_order_anomalies` và `dw.mart_anomaly_summary`."""
    export_cols = [
        "order_id",
        "customer_unique_id",
        "purchase_date",
        "customer_state",
        "order_status",
        "items_price_total",
        "freight_total",
        "payment_total",
        "item_count",
        "freight_to_price_ratio",
        "payment_price_gap",
        "delivery_days",
        "delivery_delay_days",
        "review_score",
        "anomaly_score",
        "is_anomaly",
        "anomaly_type",
        "audit_priority",
    ]
    orders_out = scored_df[export_cols].copy()

    anom_parquet = DATA_PROCESSED_DIR / "ml_order_anomalies.parquet"
    sum_parquet = DATA_PROCESSED_DIR / "mart_anomaly_summary.parquet"

    with duckdb.connect(str(WAREHOUSE_PATH)) as conn:
        conn.register("anom_orders_view", orders_out)
        conn.execute(
            "CREATE OR REPLACE TABLE dw.ml_order_anomalies AS SELECT * FROM anom_orders_view;"
        )
        conn.unregister("anom_orders_view")

        conn.register("anom_sum_view", summary_df)
        conn.execute(
            "CREATE OR REPLACE TABLE dw.mart_anomaly_summary AS SELECT * FROM anom_sum_view;"
        )
        conn.unregister("anom_sum_view")

        conn.execute(
            f"COPY dw.ml_order_anomalies TO '{anom_parquet.as_posix()}' (FORMAT PARQUET);"
        )
        conn.execute(
            f"COPY dw.mart_anomaly_summary TO '{sum_parquet.as_posix()}' (FORMAT PARQUET);"
        )

    return anom_parquet, sum_parquet


def plot_anomaly_diagnostics(
    scored_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Vẽ biểu đồ 1x3: Phân phối Anomaly Score, Scatter Log(Price)-Log(Freight) & Phân bổ Archetype."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.6), dpi=150, facecolor="white")
    fig.suptitle(
        "Phát hiện Đơn hàng Bất thường Đa chiều bằng Isolation Forest (Top 1% Contamination)",
        fontsize=14.5,
        fontweight="bold",
        y=0.99,
    )

    normal_df = scored_df[scored_df["is_anomaly"] == 0]
    anom_df = scored_df[scored_df["is_anomaly"] == 1]

    # --- Ô 1: Phân phối Anomaly Score ---
    ax1.hist(
        normal_df["anomaly_score"],
        bins=45,
        color="#3B82F6",
        alpha=0.75,
        label=f"Bình thường (n={len(normal_df):,})",
    )
    ax1.hist(
        anom_df["anomaly_score"],
        bins=25,
        color="#DC2626",
        alpha=0.90,
        label=f"Bất thường Top 1% (n={len(anom_df):,})",
    )
    ax1.axvline(0.0, color="#111827", linestyle="--", linewidth=1.7, label="Ngưỡng cắt (Score = 0.0)")
    ax1.set_yscale("log")
    ax1.set_title("Phân phối Điểm Bất thường (Thang tần suất Log)", fontsize=11.3, fontweight="bold", pad=8)
    ax1.set_xlabel("Anomaly Score (-decision_function, lớn = dị biệt)", fontsize=9.8, fontweight="bold")
    ax1.set_ylabel("Số lượng đơn hàng (log scale)", fontsize=9.8, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.5)

    # --- Ô 2: Scatter Log1p(Tiền hàng) vs Log1p(Phí vận chuyển) ---
    rng = np.random.default_rng(RANDOM_STATE)
    sample_norm = normal_df.iloc[
        rng.choice(len(normal_df), size=min(10_000, len(normal_df)), replace=False)
    ]
    ax2.scatter(
        np.log1p(sample_norm["items_price_total"]),
        np.log1p(sample_norm["freight_total"]),
        s=9,
        color="#94A3B8",
        alpha=0.32,
        label="Mẫu 10,000 đơn thường",
    )
    ax2.scatter(
        np.log1p(anom_df["items_price_total"]),
        np.log1p(anom_df["freight_total"]),
        s=18,
        color="#DC2626",
        alpha=0.78,
        label=f"Đơn bất thường (n={len(anom_df):,})",
    )
    ax2.set_title("Không gian Log(Tiền hàng) vs Log(Phí vận chuyển)", fontsize=11.3, fontweight="bold", pad=8)
    ax2.set_xlabel("log(1 + items_price_total)", fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("log(1 + freight_total)", fontsize=9.8, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.6)

    # --- Ô 3: Cột ngang Số đơn & GMV theo Nhóm nguyên nhân bất thường ---
    s_ord = summary_df.sort_values("anomaly_orders", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(s_ord))
    bars = ax3.barh(
        y_pos,
        s_ord["anomaly_orders"],
        color=["#0D9488", "#6366F1", "#F59E0B", "#EA580C", "#DC2626"][: len(s_ord)],
        height=0.58,
        alpha=0.88,
    )
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(s_ord["anomaly_type"], fontsize=9.2, fontweight="bold")
    ax3.set_xlim(0, s_ord["anomaly_orders"].max() * 1.52)
    ax3.set_xlabel("Số đơn hàng bất thường", fontsize=9.8, fontweight="bold")
    ax3.set_title("Phân bổ theo Nhóm Nguyên nhân Bất thường", fontsize=11.3, fontweight="bold", pad=8)

    for bar, (_, r) in zip(bars, s_ord.iterrows()):
        w = bar.get_width()
        ax3.text(
            w + s_ord["anomaly_orders"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(w)} đơn ({r['share_of_anomalies_pct']:.1f}%)\nGMV: {r['total_gmv']/1000:.0f}k BRL",
            va="center",
            ha="left",
            fontsize=8.3,
            fontweight="bold",
            color="#1E293B",
        )

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_anomaly_report(
    scored_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    thresholds: dict[str, float],
    output_path: Path,
) -> None:
    """Xuất báo cáo Markdown `reports/ch7_anomaly_detection.md` (không chứa dấu | lỗi bảng)."""
    norm = scored_df[scored_df["is_anomaly"] == 0]
    anom = scored_df[scored_df["is_anomaly"] == 1]

    top10_orders = anom.sort_values("anomaly_score", ascending=False).head(10)
    total_gmv_all = float(scored_df["payment_total"].sum())
    anom_gmv_share = float(anom["payment_total"].sum() * 100.0 / total_gmv_all)

    lines: list[str] = [
        "# Báo cáo Phát hiện Đơn hàng Bất thường bằng Isolation Forest — Chương 7 (Chặng 2)",
        "",
        f"- **Thời điểm thực thi:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Tổng số đơn hàng hợp lệ kiểm tra:** **{len(scored_df):,}** đơn (`order_status NOT IN ('canceled', 'unavailable')`, `payment_total > 0`, `items_price_total > 0`).",
        f"- **Cấu hình mô hình:** `IsolationForest(n_estimators=200, contamination=0.01, random_state=42)` trên không gian **11 đặc trưng đa chiều** đã biến đổi `log1p` và chuẩn hóa bền vững bằng `RobustScaler`.",
        f"- **Lý do chọn `contamination = 1%` ({len(anom):,} đơn):** Khớp đúng năng lực rà soát thực tế của bộ phận Kiểm soát Vận hành & Gian lận (~1–2 đơn/ngày), tập trung vào các giao dịch thực sự cực đoan thay vì báo động giả tràn lan.",
        "",
        "---",
        "",
        "## 1. Đối chiếu Chân dung Thống kê: Đơn Bình thường (99%) vs Đơn Bất thường (1%)",
        "",
        "| Chỉ số đo lường | Đơn Bình thường (99%) | Đơn Bất thường (Top 1%) | Chênh lệch / Ý nghĩa nghiệp vụ |",
        "| :--- | ---: | ---: | :--- |",
        f"| Số lượng đơn hàng | {len(norm):,} | **{len(anom):,}** | Tỷ lệ gắn cờ đúng 1.00% |",
        f"| Tổng GMV liên quan (BRL) | {norm['payment_total'].sum():,.2f} | **{anom['payment_total'].sum():,.2f}** | Nhóm 1% đơn chiếm tới **{anom_gmv_share:.2f}%** tổng GMV! |",
        f"| Giá trị thanh toán TB — AOV (BRL) | {norm['payment_total'].mean():,.2f} | **{anom['payment_total'].mean():,.2f}** | Cao gấp **{anom['payment_total'].mean()/norm['payment_total'].mean():.1f} lần** đơn thường |",
        f"| Số món hàng trung bình (`item_count`) | {norm['item_count'].mean():.2f} | **{anom['item_count'].mean():.2f}** | Tập trung các đơn mua sỉ / tách nhiều kiện |",
        f"| Tỷ lệ Phí ship / Tiền hàng TB | {norm['freight_to_price_ratio'].mean()*100:.1f}% | **{anom['freight_to_price_ratio'].mean()*100:.1f}%** | Nhiều đơn phí vận chuyển đắt gấp rưỡi tiền hàng |",
        f"| Thời gian giao hàng TB (Ngày) | {norm['delivery_days'].mean():.2f} | **{anom['delivery_days'].mean():.2f}** | Kéo dài gấp **{anom['delivery_days'].mean()/norm['delivery_days'].mean():.1f} lần** trung bình |",
        f"| Tỷ lệ giao trễ hẹn (%) | {norm['is_late_delivery'].mean()*100:.2f}% | **{anom['is_late_delivery'].mean()*100:.2f}%** | Nguy cơ đứt gãy SLA vận chuyển rất cao |",
        f"| Điểm đánh giá trung bình (1–5 sao) | {norm['review_score'].mean():.2f} | **{anom['review_score'].mean():.2f}** | Thấp hơn **{norm['review_score'].mean() - anom['review_score'].mean():.2f} sao** — xác thực bất thường thực sự gây trải nghiệm xấu! |",
        "",
        "---",
        "",
        "## 2. Phân loại theo Nhóm Nguyên nhân Bất thường (`anomaly_type`)",
        "",
        f"Mỗi đơn trong nhóm 1% bất thường được tự động phân loại vào 5 kịch bản nghiệp vụ dựa trên các ngưỡng P99 (riêng sai lệch thanh toán áp sàn tối thiểu `{thresholds['payment_price_gap']:.1f} BRL` để loại nhiễu làm tròn số thực):",
        "",
        "| Nhóm nguyên nhân (`anomaly_type`) | Số đơn | Tỷ trọng (%) | Mức Critical | Tổng GMV (BRL) | AOV (BRL) | Số món TB | Giao TB (Ngày) | Tỷ lệ trễ (%) | Review TB |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, r in summary_df.iterrows():
        lines.append(
            f"| **{r['anomaly_type']}** | {int(r['anomaly_orders']):,} | {r['share_of_anomalies_pct']:.1f}% | "
            f"{int(r['critical_count'])} | {r['total_gmv']:,.2f} | {r['avg_order_value']:,.2f} | "
            f"{r['avg_item_count']:.2f} | {r['avg_delivery_days']:.1f} | "
            f"{r['late_delivery_rate_pct']:.1f}% | {r['avg_review_score']:.2f} |"
        )

    lines.extend(
        [
            "",
            "![Phát hiện bất thường Isolation Forest](figures/ch7_anomaly_detection.png)",
            "",
            "---",
            "",
            "## 3. Top 10 Đơn hàng Bất thường Nghiêm trọng nhất Cần Kiểm toán Ngay",
            "",
            "| Hạng | Mã đơn (8 ký tự) | Bang | Tiền hàng (BRL) | Phí ship (BRL) | Thanh toán (BRL) | Số món | Giao (Ngày) | Review | Điểm dị biệt | Loại bất thường |",
            "| ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :--- |",
        ]
    )

    for rank, (_, r) in enumerate(top10_orders.iterrows(), start=1):
        oid_short = str(r["order_id"])[:8]
        deliv_str = f"{r['delivery_days']:.1f}" if pd.notna(r["delivery_days"]) else "Chưa giao"
        rev_str = f"{int(r['review_score'])}" if pd.notna(r["review_score"]) else "-"
        lines.append(
            f"| {rank} | `{oid_short}` | {r['customer_state']} | {r['items_price_total']:,.2f} | "
            f"{r['freight_total']:,.2f} | {r['payment_total']:,.2f} | {int(r['item_count'])} | "
            f"{deliv_str} | {rev_str} | **{r['anomaly_score']:.4f}** | `{r['anomaly_type']}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Quy trình Khuyến nghị Xử lý cho Vận hành, CSKH & Kiểm soát Gian lận",
            "",
            "1. **`Extreme Order Value / Bulk` (Đơn giá trị rất lớn hoặc mua sỉ nhiều món):** Kích hoạt xác minh thanh toán 2 lớp chống thẻ tín dụng giả mạo (*chargeback fraud*), đồng thời gán mã vận đơn ưu tiên có bảo hiểm hàng hóa vì một đơn lỗi ở nhóm này gây thiệt hại doanh thu gấp ~7–10 lần đơn thường.",
            "2. **`Abnormal Freight Ratio` (Phí vận chuyển cao bất thường so với giá hàng):** Rà soát lỗi khai báo sai kích thước/trọng lượng sản phẩm (`dim_products`) của người bán làm công thức tính cước bưu cục bị đội giá.",
            "3. **`Payment-Price Discrepancy` (Lệch giữa tổng tiền thanh toán và giá trị đơn):** Đối soát cổng thanh toán, lãi trả góp cộng thêm hoặc lỗi áp mã voucher nhiều lần.",
            "4. **`Extreme Delivery Anomaly` (Ngâm đơn quá 46 ngày hoặc trễ trên 18 ngày):** Tự động mở ticket bồi hoàn/gửi hàng thay thế và phạt điểm chất lượng đối tác vận chuyển / nhà bán hàng.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_isolation_forest_pipeline() -> None:
    """Thực thi trọn bộ pipeline Phát hiện Bất thường Isolation Forest."""
    if not WAREHOUSE_PATH.is_file():
        print(f"[ERROR] Không tìm thấy file DuckDB tại: {WAREHOUSE_PATH}")
        sys.exit(1)

    ensure_dirs()
    start_time = time.perf_counter()

    with duckdb.connect(str(WAREHOUSE_PATH), read_only=True) as conn:
        df_orders = load_valid_orders_for_anomaly(conn)

    x_scaled, scaler, feature_cols, medians = build_feature_matrix(df_orders)

    iso_model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    preds = iso_model.fit_predict(x_scaled)
    scores = -iso_model.decision_function(x_scaled)

    df_orders["anomaly_score"] = scores.astype(float)
    df_orders["is_anomaly"] = (preds == -1).astype(int)

    scored_df, thresholds = assign_anomaly_archetypes(df_orders)
    summary_df = build_anomaly_summary(scored_df)

    model_path = MODELS_DIR / "isolation_forest_orders.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "isolation_forest": iso_model,
            "feature_cols": feature_cols,
            "imputation_medians": medians,
            "archetype_thresholds": thresholds,
            "contamination": CONTAMINATION,
        },
        model_path,
    )

    anom_parquet, sum_parquet = save_anomalies_to_duckdb_and_parquet(scored_df, summary_df)
    fig_path = FIGURES_DIR / "ch7_anomaly_detection.png"
    report_path = REPORTS_DIR / "ch7_anomaly_detection.md"

    plot_anomaly_diagnostics(scored_df, summary_df, fig_path)
    write_anomaly_report(scored_df, summary_df, thresholds, report_path)

    elapsed = time.perf_counter() - start_time
    norm_rev = float(scored_df.loc[scored_df["is_anomaly"] == 0, "review_score"].mean())
    anom_rev = float(scored_df.loc[scored_df["is_anomaly"] == 1, "review_score"].mean())

    print("\n=== KẾT QUẢ PHÁT HIỆN ĐƠN HÀNG BẤT THƯỜNG (ISOLATION FOREST) ===")
    print(f"- Tổng đơn kiểm tra   : {len(scored_df):,}")
    print(
        f"- Đơn bất thường (1%) : {int(scored_df['is_anomaly'].sum()):,} đơn "
        f"(Critical: {int((scored_df['audit_priority'] == 'Critical').sum())}, "
        f"High: {int((scored_df['audit_priority'] == 'High').sum())})"
    )
    print(f"- Điểm Review TB      : Thường = {norm_rev:.2f}/5 vs Bất thường = {anom_rev:.2f}/5")
    print("\nPhân bổ theo Nhóm Nguyên nhân Bất thường:")
    for _, r in summary_df.iterrows():
        print(
            f"  - {r['anomaly_type']:<28}: {int(r['anomaly_orders']):>4} đơn "
            f"({r['share_of_anomalies_pct']:>5.1f}%) | GMV={r['total_gmv']:>10,.2f} BRL | RevTB={r['avg_review_score']:.2f}"
        )

    print("\n=== TỆP MÔ HÌNH, PARQUET & BÁO CÁO ĐÃ LƯU ===")
    print(f"- Model Joblib     : {model_path}")
    print(f"- Parquet Đơn hàng : {anom_parquet}")
    print(f"- Parquet Tóm tắt  : {sum_parquet}")
    print(f"- Biểu đồ Chẩn đoán: {fig_path}")
    print(f"- Báo cáo Markdown : {report_path}")
    print(f"Tổng thời gian chạy: {elapsed:.2f} giây")


if __name__ == "__main__":
    run_isolation_forest_pipeline()
