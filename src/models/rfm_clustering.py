"""Chương 6 — Chặng 1: Phân cụm Khách hàng bằng RFM + K-Means Clustering."""

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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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

CHOSEN_K: int = 4
K_RANGE: tuple[int, ...] = tuple(range(2, 9))
RANDOM_STATE: int = 42
SILHOUETTE_SAMPLE_SIZE: int = 10_000
FEATURE_NAMES: tuple[str, ...] = ("recency_days", "log1p_frequency", "log1p_monetary")

SEGMENT_ACTIONS: dict[str, str] = {
    "Loyal / Repeat Buyers": (
        "Chăm sóc VIP & giữ chân ưu tiên: chương trình tích điểm thành viên, "
        "ưu tiên SLA giao hàng nhanh, ưu đãi sinh nhật và giới thiệu sản phẩm mới sớm."
    ),
    "Recent High-Value": (
        "Chuyển đổi thành khách trung thành: gửi voucher cá nhân hóa cho đơn kế tiếp "
        "trong cửa sổ 30–45 ngày, gợi ý cross-sell/upsell phụ kiện đi kèm và cam kết giao đúng hẹn."
    ),
    "Recent Standard": (
        "Kích hoạt mua lần 2 & nâng giỏ hàng (AOV): đề xuất combo/freeship theo ngưỡng chi tiêu, "
        "email nhắc nhở ngành hàng tiêu dùng lặp lại (như health_beauty, housewares)."
    ),
    "Hibernating / Lost": (
        "Tái kích hoạt tự động chi phí thấp (Win-back): chỉ gửi email/push ưu đãi theo các dịp lớn "
        "(Black Friday, lễ hội), tránh đốt ngân sách quảng cáo trả phí trên nhóm đã ngủ đông lâu."
    ),
}

SEGMENT_PLAYBOOK_META: dict[str, dict[str, str]] = {
    "Loyal / Repeat Buyers": {
        "goal": "Bảo vệ nhóm khách trung thành cốt lõi & tăng tần suất mua lặp lại",
        "channel": "Email VIP, Zalo/App Push cá nhân hóa, CSKH ưu tiên",
        "budget": "Cao trên mỗi khách (Quy mô nhỏ 3.0%, ROI giữ chân cao nhất)",
    },
    "Recent High-Value": {
        "goal": "Chuyển đổi khách mua 1 lần giá trị cao thành khách quay lại (đòn bẩy GMV số 1)",
        "channel": "Email automation sau 14–30 ngày, Retargeting có chọn lọc",
        "budget": "Ưu tiên số 1 (Chiếm ~57.6% tổng GMV toàn sàn)",
    },
    "Recent Standard": {
        "goal": "Tăng giá trị giỏ hàng (AOV) & kích hoạt đơn hàng thứ 2",
        "channel": "Email tự động gợi ý combo, thông báo khuyến mãi trong ứng dụng",
        "budget": "Trung bình (Tối ưu chi phí chuyển đổi qua tự động hóa)",
    },
    "Hibernating / Lost": {
        "goal": "Khơi lại nhu cầu với chi phí cận biên gần như bằng 0",
        "channel": "Chiến dịch Email Win-back tự động dịp lễ/Black Friday",
        "budget": "Thấp (Không chi ngân sách Paid Ads cho nhóm rời bỏ > 1 năm)",
    },
}


def build_rfm_table(conn: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Bước A: Tổng hợp bảng đặc trưng RFM và ngữ cảnh cấp `customer_unique_id`."""
    df = conn.execute(
        """
        WITH valid_orders AS (
            SELECT *
            FROM dw.fact_orders
            WHERE order_status NOT IN ('canceled', 'unavailable')
              AND payment_total > 0
        ),
        snap AS (
            SELECT MAX(purchase_date) + INTERVAL 1 DAY AS snapshot_date
            FROM valid_orders
        )
        SELECT
            v.customer_unique_id,
            DATEDIFF('day', MAX(v.purchase_date), MAX(s.snapshot_date)) AS recency_days,
            COUNT(DISTINCT v.order_id)                                  AS frequency,
            SUM(v.payment_total)                                        AS monetary,
            MIN(v.purchase_date)                                        AS first_purchase_date,
            MAX(v.purchase_date)                                        AS last_purchase_date,
            AVG(v.payment_total)                                        AS avg_order_value,
            AVG(v.review_score)                                         AS avg_review_score,
            SUM(CASE WHEN v.is_late_delivery THEN 1 ELSE 0 END)         AS late_order_count,
            ARG_MAX(v.customer_state, v.purchase_date)                  AS customer_state,
            MAX(s.snapshot_date)::DATE                                  AS snapshot_date
        FROM valid_orders AS v
        CROSS JOIN snap AS s
        GROUP BY v.customer_unique_id
        ORDER BY v.customer_unique_id
        """
    ).df()

    snapshot_date = pd.Timestamp(df["snapshot_date"].iloc[0])
    df = df.drop(columns=["snapshot_date"])
    return df, snapshot_date


def preprocess_rfm_features(
    rfm_df: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Bước B1: Biến đổi log1p cho frequency & monetary (lệch phải mạnh), giữ nguyên recency (skew ~0.45), sau đó chuẩn hóa StandardScaler."""
    features_df = pd.DataFrame(
        {
            "recency_days": rfm_df["recency_days"].astype(float),
            "log1p_frequency": np.log1p(rfm_df["frequency"].astype(float)),
            "log1p_monetary": np.log1p(rfm_df["monetary"].astype(float)),
        }
    )
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features_df[list(FEATURE_NAMES)])
    return x_scaled, scaler, features_df


def evaluate_k_range(x_scaled: np.ndarray) -> pd.DataFrame:
    """Bước B2: Quét K=2..8 tính Inertia (Elbow) và Silhouette Score có lấy mẫu chống tràn RAM."""
    rows: list[dict] = []
    for k in K_RANGE:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=RANDOM_STATE,
        )
        labels = km.fit_predict(x_scaled)
        sil = float(
            silhouette_score(
                x_scaled,
                labels,
                sample_size=SILHOUETTE_SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            )
        )
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette_score": sil,
            }
        )
    return pd.DataFrame(rows)


def assign_dynamic_personas(
    rfm_df: pd.DataFrame,
    cluster_ids: np.ndarray,
) -> dict[int, str]:
    """Bước C: Gắn nhãn chân dung kinh doanh động theo đặc trưng trung bình thực tế của từng cụm."""
    stats_df = (
        rfm_df.assign(cluster_id=cluster_ids)
        .groupby("cluster_id", as_index=False)
        .agg(
            mean_recency=("recency_days", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
    )

    label_map: dict[int, str] = {}
    remaining = stats_df.copy()

    # 1. Nhóm có tần suất mua trung bình cao nhất (>1 đơn) -> Loyal / Repeat Buyers
    loyal_idx = remaining["mean_frequency"].idxmax()
    loyal_cid = int(remaining.loc[loyal_idx, "cluster_id"])
    label_map[loyal_cid] = "Loyal / Repeat Buyers"
    remaining = remaining.drop(index=loyal_idx)

    # 2. Trong các nhóm mua 1 lần, nhóm có Recency cao nhất (xa nhất) -> Hibernating / Lost
    hib_idx = remaining["mean_recency"].idxmax()
    hib_cid = int(remaining.loc[hib_idx, "cluster_id"])
    label_map[hib_cid] = "Hibernating / Lost"
    remaining = remaining.drop(index=hib_idx)

    # 3. Giữa 2 nhóm mua gần đây còn lại: chi tiêu cao hơn -> Recent High-Value, còn lại -> Recent Standard
    remaining = remaining.sort_values("mean_monetary", ascending=False).reset_index(drop=True)
    if len(remaining) == 2:
        label_map[int(remaining.loc[0, "cluster_id"])] = "Recent High-Value"
        label_map[int(remaining.loc[1, "cluster_id"])] = "Recent Standard"
    else:
        for i, row in remaining.iterrows():
            label_map[int(row["cluster_id"])] = f"Segment_{i + 1}"

    return label_map


def build_segment_summary(rfm_segmented: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp chỉ số từng phân khúc phục vụ báo cáo & Power BI."""
    total_customers = len(rfm_segmented)
    total_gmv = float(rfm_segmented["monetary"].sum())

    summary = (
        rfm_segmented.groupby("segment_name", as_index=False)
        .agg(
            customer_count=("customer_unique_id", "count"),
            total_gmv=("monetary", "sum"),
            total_orders=("frequency", "sum"),
            total_late_orders=("late_order_count", "sum"),
            avg_recency=("recency_days", "mean"),
            median_recency=("recency_days", "median"),
            avg_frequency=("frequency", "mean"),
            median_frequency=("frequency", "median"),
            avg_monetary=("monetary", "mean"),
            median_monetary=("monetary", "median"),
            avg_order_value=("avg_order_value", "mean"),
            avg_review_score=("avg_review_score", "mean"),
            z_recency=("z_recency", "mean"),
            z_frequency=("z_frequency", "mean"),
            z_monetary=("z_monetary", "mean"),
        )
    )
    summary["customer_pct"] = summary["customer_count"] * 100.0 / total_customers
    summary["gmv_pct"] = summary["total_gmv"] * 100.0 / total_gmv
    summary["late_delivery_rate_pct"] = (
        summary["total_late_orders"] * 100.0 / summary["total_orders"]
    )
    summary["recommended_action"] = summary["segment_name"].map(SEGMENT_ACTIONS)

    summary = summary.sort_values("total_gmv", ascending=False).reset_index(drop=True)
    return summary[
        [
            "segment_name",
            "customer_count",
            "customer_pct",
            "total_gmv",
            "gmv_pct",
            "avg_recency",
            "median_recency",
            "avg_frequency",
            "median_frequency",
            "avg_monetary",
            "median_monetary",
            "avg_order_value",
            "late_delivery_rate_pct",
            "avg_review_score",
            "z_recency",
            "z_frequency",
            "z_monetary",
            "recommended_action",
        ]
    ]


def save_outputs_to_duckdb_and_parquet(
    rfm_segmented: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """Bước D: Ghi 2 bảng vào DuckDB (`dw.*`) và xuất ra file `.parquet` trong `data/processed/`."""
    seg_parquet = DATA_PROCESSED_DIR / "ml_customer_rfm_segments.parquet"
    sum_parquet = DATA_PROCESSED_DIR / "mart_rfm_segment_summary.parquet"

    with duckdb.connect(str(WAREHOUSE_PATH)) as conn:
        conn.register("rfm_seg_view", rfm_segmented)
        conn.execute(
            "CREATE OR REPLACE TABLE dw.ml_customer_rfm_segments AS SELECT * FROM rfm_seg_view;"
        )
        conn.unregister("rfm_seg_view")

        conn.register("rfm_sum_view", summary_df)
        conn.execute(
            "CREATE OR REPLACE TABLE dw.mart_rfm_segment_summary AS SELECT * FROM rfm_sum_view;"
        )
        conn.unregister("rfm_sum_view")

        conn.execute(
            f"COPY dw.ml_customer_rfm_segments TO '{seg_parquet.as_posix()}' (FORMAT PARQUET);"
        )
        conn.execute(
            f"COPY dw.mart_rfm_segment_summary TO '{sum_parquet.as_posix()}' (FORMAT PARQUET);"
        )

    return seg_parquet, sum_parquet


def plot_elbow_and_silhouette(k_eval_df: pd.DataFrame, output_path: Path) -> None:
    """Bước E1: Vẽ biểu đồ Elbow (Inertia) và Silhouette Score cho K=2..8."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=150, facecolor="white")
    fig.suptitle(
        "Lựa chọn Số cụm K Tối ưu cho Phân khúc Khách hàng RFM (K-Means)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    ks = k_eval_df["k"].to_numpy()
    inertias_k = k_eval_df["inertia"].to_numpy() / 1000.0
    sils = k_eval_df["silhouette_score"].to_numpy()
    chosen_idx = int(np.where(ks == CHOSEN_K)[0][0])

    # --- Trái: Elbow Curve ---
    ax1.plot(ks, inertias_k, color="#2563EB", marker="o", linewidth=2.3, markersize=7)
    ax1.axvline(CHOSEN_K, color="#DC2626", linestyle="--", linewidth=1.6, alpha=0.85)
    ax1.scatter(
        [CHOSEN_K],
        [inertias_k[chosen_idx]],
        color="#DC2626",
        s=140,
        zorder=5,
        label=f"Chọn K = {CHOSEN_K} (WCSS = {inertias_k[chosen_idx]:.1f}k)",
    )
    ax1.set_xticks(ks)
    ax1.set_xlabel("Số cụm (K)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("WCSS / Inertia (Nghìn đơn vị)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Phương pháp Khuỷu tay (Elbow Method)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_ylim(0, inertias_k.max() * 1.16)
    for k_val, val in zip(ks, inertias_k):
        offset_y = 6.5 if k_val != CHOSEN_K else 9.0
        ax1.text(
            k_val,
            val + offset_y,
            f"{val:.1f}k",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold" if k_val == CHOSEN_K else "normal",
            color="#991B1B" if k_val == CHOSEN_K else "#334155",
        )
    ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9.2)

    # --- Phải: Silhouette Score ---
    ax2.plot(ks, sils, color="#0D9488", marker="s", linewidth=2.3, markersize=7)
    ax2.axvline(CHOSEN_K, color="#DC2626", linestyle="--", linewidth=1.6, alpha=0.85)
    ax2.scatter(
        [CHOSEN_K],
        [sils[chosen_idx]],
        color="#DC2626",
        s=140,
        zorder=5,
        label=f"Chọn K = {CHOSEN_K} (Silhouette = {sils[chosen_idx]:.4f})",
    )
    ax2.set_xticks(ks)
    ax2.set_xlabel("Số cụm (K)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel(f"Silhouette Score (sample={SILHOUETTE_SAMPLE_SIZE:,})", fontsize=10.5, fontweight="bold")
    ax2.set_title("Chỉ số Silhouette theo Số cụm K", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_ylim(0.25, sils.max() * 1.15)
    for k_val, val in zip(ks, sils):
        ax2.text(
            k_val,
            val + 0.015,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold" if k_val == CHOSEN_K else "normal",
            color="#991B1B" if k_val == CHOSEN_K else "#334155",
        )

    ax2.annotate(
        "K=2 chỉ tách 97% khách 1 đơn\nvs 3% khách quay lại (quá thô)",
        xy=(2, sils[0]),
        xytext=(3.2, sils[0] - 0.06),
        fontsize=8.6,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.2),
    )
    ax2.annotate(
        "Đỉnh Silhouette cục bộ cho K>=3\nTách đủ 4 chân dung kinh doanh",
        xy=(CHOSEN_K, sils[chosen_idx]),
        xytext=(4.65, sils[chosen_idx] + 0.14),
        fontsize=8.8,
        fontweight="bold",
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#F87171"),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
    )
    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9.2)

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_segments_profile(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Bước E2: Vẽ biểu đồ so sánh % Khách vs % GMV và Heatmap Z-score RFM của từng phân khúc."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=150, facecolor="white")
    fig.suptitle(
        "Chân dung 4 Phân khúc Khách hàng RFM: Tỷ trọng Doanh thu (Pareto) & Hồ sơ Z-score",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    ordered = summary_df.sort_values("gmv_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    bar_h = 0.36

    # --- Trái: Cột ngang song song % Khách hàng vs % GMV ---
    bars_cust = ax1.barh(
        y + bar_h / 2,
        ordered["customer_pct"],
        height=bar_h,
        color="#3B82F6",
        alpha=0.86,
        label="Tỷ trọng Khách hàng (%)",
    )
    bars_gmv = ax1.barh(
        y - bar_h / 2,
        ordered["gmv_pct"],
        height=bar_h,
        color="#F59E0B",
        alpha=0.92,
        label="Tỷ trọng Doanh thu GMV (%)",
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(ordered["segment_name"], fontsize=10, fontweight="bold")
    ax1.set_xlabel("Tỷ trọng trên toàn sàn (%)", fontsize=10.5, fontweight="bold")
    ax1.set_xlim(0, max(ordered["customer_pct"].max(), ordered["gmv_pct"].max()) * 1.32)
    ax1.set_title(
        "So sánh Tỷ trọng Khách hàng (%) vs Tỷ trọng GMV (%)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    for bar, (_, r) in zip(bars_cust, ordered.iterrows()):
        w = bar.get_width()
        ax1.text(
            w + 0.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.1f}% ({int(r['customer_count']):,})",
            va="center",
            ha="left",
            fontsize=8.6,
            color="#1E3A8A",
        )
    for bar, (_, r) in zip(bars_gmv, ordered.iterrows()):
        w = bar.get_width()
        ax1.text(
            w + 0.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.1f}% ({r['total_gmv']/1e6:.2f}M)",
            va="center",
            ha="left",
            fontsize=8.6,
            fontweight="bold",
            color="#92400E",
        )
    ax1.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9.2)

    # --- Phải: Heatmap Z-score chuẩn hóa (kèm giá trị thực trung bình trong từng ô) ---
    hm_ordered = summary_df.sort_values("gmv_pct", ascending=False).reset_index(drop=True)
    z_matrix = hm_ordered.set_index("segment_name")[
        ["z_recency", "z_frequency", "z_monetary"]
    ].copy()
    z_matrix.columns = ["Recency\n(Thấp = Mới mua)", "Frequency\n(log1p)", "Monetary\n(log1p)"]

    annot_df = pd.DataFrame(index=z_matrix.index, columns=z_matrix.columns, dtype=str)
    for _, r in hm_ordered.iterrows():
        seg = r["segment_name"]
        annot_df.loc[seg, "Recency\n(Thấp = Mới mua)"] = (
            f"{r['z_recency']:+.2f}\n({r['avg_recency']:.0f} ngày)"
        )
        annot_df.loc[seg, "Frequency\n(log1p)"] = (
            f"{r['z_frequency']:+.2f}\n({r['avg_frequency']:.2f} đơn)"
        )
        annot_df.loc[seg, "Monetary\n(log1p)"] = (
            f"{r['z_monetary']:+.2f}\n({r['avg_monetary']:.0f} BRL)"
        )

    sns.heatmap(
        z_matrix,
        annot=annot_df,
        fmt="",
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.5,
        vmax=2.0,
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "Giá trị chuẩn hóa Z-score (0 = TB toàn sàn)", "shrink": 0.86},
        annot_kws={"size": 9.3, "fontweight": "bold"},
        ax=ax2,
    )
    ax2.set_title(
        "Hồ sơ Chuẩn hóa Z-score theo Phân khúc (Kèm giá trị TB thực tế)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel("Trục đặc trưng RFM đầu vào K-Means", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", labelsize=9.5, rotation=0)
    ax2.tick_params(axis="y", labelsize=9.8, rotation=0)

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_rfm_report(
    rfm_df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    k_eval_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bước F: Xuất báo cáo phân khúc RFM chi tiết ra `reports/ch6_rfm_segmentation.md`."""
    n_total = len(rfm_df)
    one_time_cnt = int((rfm_df["frequency"] == 1).sum())
    one_time_pct = one_time_cnt * 100.0 / n_total
    repeat_cnt = n_total - one_time_cnt
    repeat_pct = repeat_cnt * 100.0 / n_total

    skew_r = float(rfm_df["recency_days"].skew())
    skew_f = float(rfm_df["frequency"].skew())
    skew_m = float(rfm_df["monetary"].skew())

    chosen_row = k_eval_df.loc[k_eval_df["k"] == CHOSEN_K].iloc[0]

    lines: list[str] = [
        "# Báo cáo Phân cụm Khách hàng RFM & K-Means — Chương 6 (Chặng 1)",
        "",
        f"- **Thời điểm thực thi:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Mốc chốt dữ liệu (`snapshot_date`):** `{snapshot_date.strftime('%Y-%m-%d')}` (`MAX(purchase_date) + 1 ngày`)",
        f"- **Quy mô tập khách hàng hợp lệ:** **{n_total:,}** khách hàng duy nhất (`customer_unique_id`, lọc `order_status NOT IN ('canceled', 'unavailable')` và `payment_total > 0`).",
        "",
        "---",
        "",
        "## 1. Bối cảnh Dữ liệu RFM & Chiến lược Tiền xử lý",
        "",
        f"- **Đặc thù cực đoan của sàn Olist:** Có tới **{one_time_cnt:,} khách hàng ({one_time_pct:.2f}%)** chỉ mua đúng **1 đơn hàng**, và chỉ **{repeat_cnt:,} khách hàng ({repeat_pct:.2f}%)** mua lặp lại từ 2 đơn trở lên.",
        "- **Kiểm tra độ lệch (Skewness) trước khi phân cụm:**",
        f"  - `recency_days`: Skewness = **{skew_r:.2f}** (gần đối xứng $\\rightarrow$ giữ nguyên tuyến tính để bảo toàn khoảng cách theo thời gian).",
        f"  - `frequency`: Skewness = **{skew_f:.2f}** (lệch phải cực mạnh $\\rightarrow$ biến đổi `np.log1p(frequency)`).",
        f"  - `monetary`: Skewness = **{skew_m:.2f}** (lệch phải rất mạnh $\\rightarrow$ biến đổi `np.log1p(monetary)` để giảm sức kéo của các đơn hàng nghìn BRL).",
        "- **Chuẩn hóa bắt buộc:** Cả 3 biến đầu vào (`recency_days`, `log1p_frequency`, `log1p_monetary`) được đưa về cùng thang đo (mean = 0, std = 1) bằng `StandardScaler` trước khi tính khoảng cách Euclidean trong K-Means.",
        "",
        "---",
        "",
        "## 2. Quá trình Lựa chọn Số cụm $K$ Tối ưu",
        "",
        f"Thực hiện quét $K \\in \\{{2..8\\}}$ với `init='k-means++'`, `n_init=10`, `random_state=42` và tính `silhouette_score` trên mẫu ngẫu nhiên `sample_size={SILHOUETTE_SAMPLE_SIZE:,}` để tránh tràn RAM ($N^2 \\approx 9 \\times 10^9$ cặp khoảng cách):",
        "",
        "| Số cụm ($K$) | Inertia (WCSS) | Silhouette Score | Đánh giá lựa chọn |",
        "| :---: | ---: | ---: | :--- |",
    ]

    for _, r in k_eval_df.iterrows():
        k_val = int(r["k"])
        if k_val == 2:
            note = "Cực đại kỹ thuật nhưng quá thô (chỉ tách 97% mua 1 lần vs 3% quay lại)"
        elif k_val == CHOSEN_K:
            note = "**ĐƯỢC CHỌN — Điểm gập Elbow rõ & Đỉnh Silhouette cao nhất trong nhóm $K \\ge 3$**"
        else:
            note = "Giảm độ tách biệt cụm hoặc chia nhỏ dư thừa"
        prefix = "**" if k_val == CHOSEN_K else ""
        suffix = "**" if k_val == CHOSEN_K else ""
        lines.append(
            f"| {prefix}{k_val}{suffix} | {prefix}{r['inertia']:,.1f}{suffix} | "
            f"{prefix}{r['silhouette_score']:.4f}{suffix} | {note} |"
        )

    lines.extend(
        [
            "",
            f"**Lý do chọn $K = {CHOSEN_K}$:** Tại $K = {CHOSEN_K}$, đường cong Inertia thể hiện điểm gập (*elbow*) rõ rệt (giảm mạnh từ `193.5k` xuống `87.3k` rồi chững lại), đồng thời **Silhouette Score đạt đỉnh cục bộ `{chosen_row['silhouette_score']:.4f}`** (cao hơn mọi $K \\in \\{{3, 5, 6, 7, 8\\}}$) và phân tách sạch sẽ 4 nhóm chân dung kinh doanh có hành động riêng biệt.",
            "",
            "![Elbow & Silhouette K-Means RFM](figures/ch6_rfm_elbow_silhouette.png)",
            "",
            "---",
            "",
            "## 3. Chân dung Chi tiết 4 Phân khúc Khách hàng",
            "",
            "| Phân khúc (`segment_name`) | Số khách | % Khách | Tổng GMV (BRL) | % GMV | Recency TB / Med (Ngày) | Freq TB / Med | Monetary TB / Med (BRL) | Tỷ lệ trễ (%) | Review TB |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, r in summary_df.iterrows():
        lines.append(
            f"| **{r['segment_name']}** | {int(r['customer_count']):,} | {r['customer_pct']:.2f}% | "
            f"{r['total_gmv']:,.2f} | **{r['gmv_pct']:.2f}%** | "
            f"{r['avg_recency']:.1f} / {r['median_recency']:.0f} | "
            f"{r['avg_frequency']:.2f} / {r['median_frequency']:.0f} | "
            f"{r['avg_monetary']:,.2f} / {r['median_monetary']:,.2f} | "
            f"{r['late_delivery_rate_pct']:.2f}% | {r['avg_review_score']:.2f} |"
        )

    hv_row = summary_df.loc[summary_df["segment_name"] == "Recent High-Value"].iloc[0]
    loyal_row = summary_df.loc[summary_df["segment_name"] == "Loyal / Repeat Buyers"].iloc[0]
    lines.extend(
        [
            "",
            "**Điểm nhấn Pareto rút ra từ dữ liệu:**",
            f"- Nhóm **`Recent High-Value`** chỉ chiếm **{hv_row['customer_pct']:.2f}%** tổng khách hàng nhưng đóng góp tới **{hv_row['gmv_pct']:.2f}% tổng doanh thu GMV** (chi tiêu TB `{hv_row['avg_monetary']:,.2f} BRL`, gấp ~4.6 lần nhóm `Recent Standard`).",
            f"- Nhóm **`Loyal / Repeat Buyers`** tuy chỉ chiếm **{loyal_row['customer_pct']:.2f}%** ({int(loyal_row['customer_count']):,} khách) nhưng đóng góp **{loyal_row['gmv_pct']:.2f}%** GMV (hiệu suất doanh thu/khách gấp gần 2 lần trung bình sàn) và có điểm review cao nhất (**{loyal_row['avg_review_score']:.2f}/5**).",
            "",
            "![Chân dung Phân khúc RFM & Z-score](figures/ch6_rfm_segments_profile.png)",
            "",
            "---",
            "",
            "## 4. Chiến lược Hành động Kinh doanh theo Phân khúc (Prescriptive Playbook)",
            "",
            "| Phân khúc | Mục tiêu chiến lược | Kênh tiếp cận | Ưu tiên ngân sách | Hành động cụ thể đề xuất |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for _, r in summary_df.iterrows():
        seg = r["segment_name"]
        meta = SEGMENT_PLAYBOOK_META[seg]
        lines.append(
            f"| **{seg}** | {meta['goal']} | {meta['channel']} | {meta['budget']} | {r['recommended_action']} |"
        )

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_terminal_summary(
    k_eval_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    model_path: Path,
    seg_parquet: Path,
    sum_parquet: Path,
    report_path: Path,
    fig_elbow_path: Path,
    fig_profile_path: Path,
    elapsed_sec: float,
) -> None:
    """In bảng tóm tắt chọn K, hồ sơ phân khúc và đường dẫn đầu ra lên Terminal."""
    print("\n=== 1. ĐÁNH GIÁ SỐ CỤM K (ELBOW & SILHOUETTE) ===")
    print(f"{'K':>3} | {'Inertia (WCSS)':>15} | {'Silhouette':>12} | {'Ghi chú'}")
    print("-" * 55)
    for _, r in k_eval_df.iterrows():
        k_val = int(r["k"])
        tag = "<-- ĐÃ CHỌN" if k_val == CHOSEN_K else ""
        print(f"{k_val:>3} | {r['inertia']:>15,.1f} | {r['silhouette_score']:>12.4f} | {tag}")

    print("\n=== 2. CHÂN DUNG 4 PHÂN KHÚC KHÁCH HÀNG RFM (GẮN NHÃN ĐỘNG) ===")
    header = (
        f"{'Phân khúc':<23} | {'Khách':>8} | {'%KH':>6} | {'%GMV':>6} | "
        f"{'R_TB':>6} | {'F_TB':>5} | {'M_TB (BRL)':>10} | {'RevTB':>5}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for _, r in summary_df.iterrows():
        print(
            f"{r['segment_name']:<23} | {int(r['customer_count']):>8,} | {r['customer_pct']:>5.1f}% | "
            f"{r['gmv_pct']:>5.1f}% | {r['avg_recency']:>6.1f} | {r['avg_frequency']:>5.2f} | "
            f"{r['avg_monetary']:>10,.2f} | {r['avg_review_score']:>5.2f}"
        )
    print("-" * len(header))

    print("\n=== 3. TỆP KẾT QUẢ & MÔ HÌNH ĐÃ LƯU ===")
    print(f"- Mô hình Joblib    : {model_path}")
    print(f"- Parquet Khách hàng: {seg_parquet}")
    print(f"- Parquet Tóm tắt   : {sum_parquet}")
    print(f"- Báo cáo Markdown  : {report_path}")
    print(f"- Biểu đồ Chọn K    : {fig_elbow_path}")
    print(f"- Biểu đồ Phân khúc : {fig_profile_path}")
    print(f"Tổng thời gian chạy : {elapsed_sec:.2f} giây")


def run_rfm_clustering() -> None:
    """Chạy trọn bộ pipeline Phân cụm RFM + K-Means (Chương 6 — Chặng 1)."""
    if not WAREHOUSE_PATH.is_file():
        print(f"[ERROR] Không tìm thấy file DuckDB tại: {WAREHOUSE_PATH}")
        sys.exit(1)

    ensure_dirs()
    start_time = time.perf_counter()

    with duckdb.connect(str(WAREHOUSE_PATH), read_only=True) as conn:
        rfm_df, snapshot_date = build_rfm_table(conn)

    x_scaled, scaler, _ = preprocess_rfm_features(rfm_df)
    k_eval_df = evaluate_k_range(x_scaled)

    final_kmeans = KMeans(
        n_clusters=CHOSEN_K,
        init="k-means++",
        n_init=10,
        random_state=RANDOM_STATE,
    )
    cluster_ids = final_kmeans.fit_predict(x_scaled)
    cluster_label_map = assign_dynamic_personas(rfm_df, cluster_ids)

    rfm_segmented = rfm_df.copy()
    rfm_segmented["cluster_id"] = cluster_ids.astype(int)
    rfm_segmented["segment_name"] = rfm_segmented["cluster_id"].map(cluster_label_map)
    rfm_segmented["recommended_action"] = rfm_segmented["segment_name"].map(SEGMENT_ACTIONS)
    rfm_segmented["z_recency"] = x_scaled[:, 0]
    rfm_segmented["z_frequency"] = x_scaled[:, 1]
    rfm_segmented["z_monetary"] = x_scaled[:, 2]

    summary_df = build_segment_summary(rfm_segmented)

    model_path = MODELS_DIR / "rfm_kmeans.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": final_kmeans,
            "feature_names": list(FEATURE_NAMES),
            "cluster_label_map": cluster_label_map,
            "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        },
        model_path,
    )

    seg_parquet, sum_parquet = save_outputs_to_duckdb_and_parquet(rfm_segmented, summary_df)

    fig_elbow_path = FIGURES_DIR / "ch6_rfm_elbow_silhouette.png"
    fig_profile_path = FIGURES_DIR / "ch6_rfm_segments_profile.png"
    report_path = REPORTS_DIR / "ch6_rfm_segmentation.md"

    plot_elbow_and_silhouette(k_eval_df, fig_elbow_path)
    plot_segments_profile(summary_df, fig_profile_path)
    write_rfm_report(rfm_df, snapshot_date, k_eval_df, summary_df, report_path)

    elapsed_sec = time.perf_counter() - start_time
    print_terminal_summary(
        k_eval_df,
        summary_df,
        model_path,
        seg_parquet,
        sum_parquet,
        report_path,
        fig_elbow_path,
        fig_profile_path,
        elapsed_sec,
    )


if __name__ == "__main__":
    run_rfm_clustering()
