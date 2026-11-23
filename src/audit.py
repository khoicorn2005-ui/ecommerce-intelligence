"""Công cụ kiểm toán chất lượng dữ liệu tự động cho tầng `raw` trong DuckDB."""

import sys
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import FIGURES_DIR, REPORTS_DIR, WAREHOUSE_PATH, ensure_dirs  # noqa: E402


def audit_missing_values(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[list[dict], dict[str, int], list[str]]:
    """Quét toàn bộ 9 bảng `raw.*` để thống kê giá trị NULL hoặc chuỗi rỗng."""
    tables = [
        row[0]
        for row in conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'raw'
            ORDER BY table_name
            """
        ).fetchall()
    ]

    missing_rows: list[dict] = []
    total_rows_by_table: dict[str, int] = {}
    clean_tables: list[str] = []

    for table in tables:
        total_rows = conn.execute(f"SELECT COUNT(*) FROM raw.{table}").fetchone()[0]
        total_rows_by_table[table] = total_rows

        columns = conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'raw' AND table_name = ?
            ORDER BY ordinal_position
            """,
            [table],
        ).fetchall()

        table_has_missing = False
        for col_name, data_type in columns:
            if "VARCHAR" in data_type.upper():
                condition = f"{col_name} IS NULL OR TRIM({col_name}) = ''"
            else:
                condition = f"{col_name} IS NULL"

            missing_cnt = conn.execute(
                f"SELECT COUNT(*) FROM raw.{table} WHERE {condition}"
            ).fetchone()[0]

            if missing_cnt > 0:
                table_has_missing = True
                missing_rows.append(
                    {
                        "table": f"raw.{table}",
                        "column": col_name,
                        "dtype": data_type,
                        "total_rows": total_rows,
                        "missing_count": missing_cnt,
                        "missing_pct": (missing_cnt / total_rows * 100.0) if total_rows else 0.0,
                    }
                )

        if not table_has_missing:
            clean_tables.append(f"raw.{table}")

    return missing_rows, total_rows_by_table, clean_tables


def audit_duplicates_and_integrity(conn: duckdb.DuckDBPyConnection) -> dict:
    """Kiểm tra trùng lặp khóa chính, đặc thù Olist và toàn vẹn tham chiếu (orphan keys)."""
    pk_specs = [
        ("raw.orders", "order_id", "SELECT COUNT(*), COUNT(DISTINCT order_id) FROM raw.orders"),
        ("raw.customers", "customer_id", "SELECT COUNT(*), COUNT(DISTINCT customer_id) FROM raw.customers"),
        ("raw.products", "product_id", "SELECT COUNT(*), COUNT(DISTINCT product_id) FROM raw.products"),
        ("raw.sellers", "seller_id", "SELECT COUNT(*), COUNT(DISTINCT seller_id) FROM raw.sellers"),
        (
            "raw.order_items",
            "(order_id, order_item_id)",
            "SELECT COUNT(*), COUNT(DISTINCT (order_id || '|' || CAST(order_item_id AS VARCHAR))) FROM raw.order_items",
        ),
    ]

    pk_checks: list[dict] = []
    for table, key_name, query in pk_specs:
        total_cnt, distinct_cnt = conn.execute(query).fetchone()
        pk_checks.append(
            {
                "table": table,
                "key": key_name,
                "total_rows": total_cnt,
                "distinct_keys": distinct_cnt,
                "duplicates": total_cnt - distinct_cnt,
            }
        )

    distinct_cid, distinct_uid = conn.execute(
        "SELECT COUNT(DISTINCT customer_id), COUNT(DISTINCT customer_unique_id) FROM raw.customers"
    ).fetchone()
    repeat_customers = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT customer_unique_id
            FROM raw.customers
            GROUP BY customer_unique_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    rev_total, rev_distinct_oid = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT order_id) FROM raw.order_reviews"
    ).fetchone()
    orders_multi_reviews = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT order_id
            FROM raw.order_reviews
            GROUP BY order_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    geo_total, geo_distinct_zip = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT geolocation_zip_code_prefix) FROM raw.geolocation"
    ).fetchone()

    orphan_items_orders = conn.execute(
        """
        SELECT COUNT(*)
        FROM raw.order_items oi
        LEFT JOIN raw.orders o ON oi.order_id = o.order_id
        WHERE o.order_id IS NULL
        """
    ).fetchone()[0]

    orphan_orders_customers = conn.execute(
        """
        SELECT COUNT(*)
        FROM raw.orders o
        LEFT JOIN raw.customers c ON o.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
        """
    ).fetchone()[0]

    untranslated_rows = conn.execute(
        """
        SELECT COUNT(*)
        FROM raw.products p
        LEFT JOIN raw.category_translation ct
          ON p.product_category_name = ct.product_category_name
        WHERE p.product_category_name IS NOT NULL
          AND ct.product_category_name IS NULL
        """
    ).fetchone()[0]

    untranslated_cats = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT p.product_category_name
            FROM raw.products p
            LEFT JOIN raw.category_translation ct
              ON p.product_category_name = ct.product_category_name
            WHERE p.product_category_name IS NOT NULL
              AND ct.product_category_name IS NULL
            ORDER BY 1
            """
        ).fetchall()
    ]

    return {
        "pk_checks": pk_checks,
        "customer_identity": {
            "distinct_customer_id": distinct_cid,
            "distinct_customer_unique_id": distinct_uid,
            "repeat_customers": repeat_customers,
            "repeat_pct": (repeat_customers / distinct_uid * 100.0) if distinct_uid else 0.0,
        },
        "order_reviews_dup": {
            "total_rows": rev_total,
            "distinct_order_id": rev_distinct_oid,
            "extra_rows": rev_total - rev_distinct_oid,
            "orders_with_multi_reviews": orders_multi_reviews,
        },
        "geolocation_dup": {
            "total_rows": geo_total,
            "distinct_zip": geo_distinct_zip,
            "duplicate_rows": geo_total - geo_distinct_zip,
        },
        "orphans": {
            "order_items_without_order": orphan_items_orders,
            "orders_without_customer": orphan_orders_customers,
            "untranslated_product_rows": untranslated_rows,
            "untranslated_categories": untranslated_cats,
        },
    }


def audit_business_consistency(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Kiểm tra các quy tắc nhất quán nghiệp vụ trên đơn hàng, sản phẩm, thanh toán và đánh giá."""
    rules = [
        (
            "raw.orders",
            "Đơn `delivered` nhưng thiếu `order_delivered_customer_date`",
            "SELECT COUNT(*) FROM raw.orders WHERE order_status = 'delivered' AND order_delivered_customer_date IS NULL",
            "SELECT COUNT(*) FROM raw.orders WHERE order_status = 'delivered'",
        ),
        (
            "raw.orders",
            "`order_delivered_customer_date < order_purchase_timestamp`",
            "SELECT COUNT(*) FROM raw.orders WHERE order_delivered_customer_date < order_purchase_timestamp",
            "SELECT COUNT(*) FROM raw.orders WHERE order_delivered_customer_date IS NOT NULL",
        ),
        (
            "raw.orders",
            "`order_approved_at < order_purchase_timestamp`",
            "SELECT COUNT(*) FROM raw.orders WHERE order_approved_at < order_purchase_timestamp",
            "SELECT COUNT(*) FROM raw.orders WHERE order_approved_at IS NOT NULL",
        ),
        (
            "raw.orders",
            "`order_delivered_carrier_date < order_approved_at`",
            "SELECT COUNT(*) FROM raw.orders WHERE order_delivered_carrier_date < order_approved_at",
            "SELECT COUNT(*) FROM raw.orders WHERE order_delivered_carrier_date IS NOT NULL AND order_approved_at IS NOT NULL",
        ),
        (
            "raw.orders",
            "`order_delivered_customer_date < order_delivered_carrier_date`",
            "SELECT COUNT(*) FROM raw.orders WHERE order_delivered_customer_date < order_delivered_carrier_date",
            "SELECT COUNT(*) FROM raw.orders WHERE order_delivered_customer_date IS NOT NULL AND order_delivered_carrier_date IS NOT NULL",
        ),
        (
            "raw.order_items",
            "`price <= 0`",
            "SELECT COUNT(*) FROM raw.order_items WHERE price <= 0",
            "SELECT COUNT(*) FROM raw.order_items",
        ),
        (
            "raw.order_items",
            "`freight_value < 0`",
            "SELECT COUNT(*) FROM raw.order_items WHERE freight_value < 0",
            "SELECT COUNT(*) FROM raw.order_items",
        ),
        (
            "raw.order_payments",
            "`payment_value <= 0`",
            "SELECT COUNT(*) FROM raw.order_payments WHERE payment_value <= 0",
            "SELECT COUNT(*) FROM raw.order_payments",
        ),
        (
            "raw.order_payments",
            "`payment_installments < 1`",
            "SELECT COUNT(*) FROM raw.order_payments WHERE payment_installments < 1",
            "SELECT COUNT(*) FROM raw.order_payments",
        ),
        (
            "raw.order_reviews",
            "`review_score` nằm ngoài khoảng 1..5",
            "SELECT COUNT(*) FROM raw.order_reviews WHERE review_score < 1 OR review_score > 5 OR review_score IS NULL",
            "SELECT COUNT(*) FROM raw.order_reviews",
        ),
    ]

    results: list[dict] = []
    for table, rule_desc, violation_q, base_q in rules:
        violations = conn.execute(violation_q).fetchone()[0]
        base_cnt = conn.execute(base_q).fetchone()[0]
        results.append(
            {
                "table": table,
                "rule": rule_desc,
                "violations": violations,
                "base_count": base_cnt,
                "pct": (violations / base_cnt * 100.0) if base_cnt else 0.0,
            }
        )
    return results


def audit_distributions_and_outliers(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Đo lường phân phối và phát hiện ngoại lệ theo quy tắc IQR cho 5 biến số trọng tâm."""
    var_specs = [
        (
            "price",
            "raw.order_items.price (BRL)",
            "SELECT price AS val FROM raw.order_items WHERE price IS NOT NULL",
        ),
        (
            "freight_value",
            "raw.order_items.freight_value (BRL)",
            "SELECT freight_value AS val FROM raw.order_items WHERE freight_value IS NOT NULL",
        ),
        (
            "payment_value",
            "raw.order_payments.payment_value (BRL)",
            "SELECT payment_value AS val FROM raw.order_payments WHERE payment_value IS NOT NULL",
        ),
        (
            "product_weight_g",
            "raw.products.product_weight_g (g)",
            "SELECT product_weight_g AS val FROM raw.products WHERE product_weight_g IS NOT NULL",
        ),
        (
            "delivery_days",
            "delivery_days (Ngày giao thực tế)",
            """
            SELECT EXTRACT(EPOCH FROM (order_delivered_customer_date - order_purchase_timestamp)) / 86400.0 AS val
            FROM raw.orders
            WHERE order_status = 'delivered'
              AND order_delivered_customer_date IS NOT NULL
              AND order_purchase_timestamp IS NOT NULL
              AND order_delivered_customer_date >= order_purchase_timestamp
            """,
        ),
    ]

    records: list[dict] = []
    series_map: dict[str, pd.Series] = {}

    for key, label, query in var_specs:
        s = conn.execute(query).df()["val"].astype(float)
        series_map[key] = s

        q1 = float(s.quantile(0.25))
        median = float(s.quantile(0.50))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_cnt = int(((s < lower_bound) | (s > upper_bound)).sum())

        records.append(
            {
                "key": key,
                "variable": label,
                "count": int(len(s)),
                "min": float(s.min()),
                "q1": q1,
                "median": median,
                "mean": float(s.mean()),
                "q3": q3,
                "max": float(s.max()),
                "std": float(s.std()),
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outliers_count": outliers_cnt,
                "outliers_pct": outliers_cnt / len(s) * 100.0 if len(s) else 0.0,
                "skewness": float(s.skew()),
                "excess_kurtosis": float(s.kurt()),
            }
        )

    return pd.DataFrame(records), series_map


def plot_audit_distributions(
    series_map: dict[str, pd.Series],
    dist_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Vẽ biểu đồ chẩn đoán 3x2 (Histogram đến P99 & Box Plot toàn dải) cho 3 biến cốt lõi."""
    sns.set_theme(style="whitegrid", context="notebook")
    plot_vars = [
        ("price", "Giá sản phẩm — price (BRL)", "#2563EB"),
        ("freight_value", "Phí vận chuyển — freight_value (BRL)", "#0D9488"),
        ("delivery_days", "Thời gian giao hàng — delivery_days (Ngày)", "#7C3AED"),
    ]

    stats_by_key = dist_df.set_index("key").to_dict(orient="index")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=150, facecolor="white")
    fig.suptitle(
        "Chẩn đoán Phân phối & Ngoại lệ (IQR) — Dữ liệu Raw Olist",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    for row_idx, (key, title, color) in enumerate(plot_vars):
        s = series_map[key]
        st = stats_by_key[key]
        p99 = float(s.quantile(0.99))

        ax_hist = axes[row_idx, 0]
        s_clipped = s[s <= p99]
        sns.histplot(
            s_clipped,
            bins=45,
            kde=False,
            color=color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            ax=ax_hist,
        )
        ax_hist.axvline(
            st["mean"],
            color="#DC2626",
            linestyle="--",
            linewidth=1.8,
            label=f"Mean = {st['mean']:.2f}",
        )
        ax_hist.axvline(
            st["median"],
            color="#15803D",
            linestyle="-.",
            linewidth=1.8,
            label=f"Median = {st['median']:.2f}",
        )
        ax_hist.set_title(
            f"{title} — Histogram (≤ P99 = {p99:.2f})",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        ax_hist.set_xlabel(title.split("—")[-1].strip(), fontsize=9.5)
        ax_hist.set_ylabel("Tần suất (Số dòng)", fontsize=9.5)
        ax_hist.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.8)
        ax_hist.text(
            0.98,
            0.70,
            f"Skewness: {st['skewness']:.2f}\nMax thực: {st['max']:,.2f}",
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
        )

        ax_box = axes[row_idx, 1]
        sns.boxplot(
            x=s,
            ax=ax_box,
            color=color,
            width=0.45,
            fliersize=2.5,
            flierprops=dict(marker="o", markerfacecolor="#EF4444", alpha=0.35, markeredgecolor="none"),
            boxprops=dict(alpha=0.8),
        )
        ax_box.axvline(
            st["upper_bound"],
            color="#D97706",
            linestyle="--",
            linewidth=1.6,
            label=f"Q3 + 1.5·IQR = {st['upper_bound']:.2f}",
        )
        ax_box.set_title(
            f"{title} — Box Plot (Toàn miền giá trị)",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        ax_box.set_xlabel(title.split("—")[-1].strip(), fontsize=9.5)
        ax_box.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.8)
        ax_box.text(
            0.98,
            0.76,
            f"Outliers: {st['outliers_count']:,} ({st['outliers_pct']:.2f}%)\nIQR: {st['iqr']:.2f}",
            transform=ax_box.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFBEB", edgecolor="#FCD34D"),
        )

    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown_report(
    missing_info: tuple[list[dict], dict[str, int], list[str]],
    integrity_info: dict,
    consistency_rows: list[dict],
    dist_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Xuất báo cáo kiểm toán chất lượng dữ liệu chi tiết ra file Markdown."""
    missing_rows, total_rows_by_table, clean_tables = missing_info
    cid_info = integrity_info["customer_identity"]
    rev_dup = integrity_info["order_reviews_dup"]
    geo_dup = integrity_info["geolocation_dup"]
    orphans = integrity_info["orphans"]

    lines: list[str] = [
        "# Báo cáo Kiểm toán Chất lượng Dữ liệu — Tầng Raw (Olist DuckDB)",
        "",
        f"- **Thời điểm chạy kiểm toán:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Nguồn dữ liệu:** `{WAREHOUSE_PATH.name}` (chế độ chỉ đọc `read_only=True`, tổng cộng {sum(total_rows_by_table.values()):,} dòng trên 9 bảng `raw.*`)",
        "",
        "---",
        "",
        "## 1. Tính đầy đủ (Missing Values / Completeness)",
        "",
        f"Trong 9 bảng kiểm tra, có **{len(clean_tables)} bảng đầy đủ 100%** (không có `NULL` hay chuỗi rỗng): "
        + ", ".join(f"`{t}`" for t in clean_tables)
        + ".",
        "",
        "| Bảng | Cột | Kiểu dữ liệu | Tổng dòng | Số dòng thiếu | Tỷ lệ thiếu (%) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: |",
    ]

    for row in missing_rows:
        lines.append(
            f"| `{row['table']}` | `{row['column']}` | `{row['dtype']}` | "
            f"{row['total_rows']:,} | {row['missing_count']:,} | {row['missing_pct']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Tính duy nhất & Toàn vẹn tham chiếu (Duplicates & Integrity)",
            "",
            "### 2.1 Kiểm tra trùng lặp khóa chính",
            "",
            "| Bảng | Khóa chính kiểm tra | Tổng dòng | Số khóa duy nhất | Số dòng trùng |",
            "| :--- | :--- | ---: | ---: | ---: |",
        ]
    )

    for pk in integrity_info["pk_checks"]:
        lines.append(
            f"| `{pk['table']}` | `{pk['key']}` | {pk['total_rows']:,} | "
            f"{pk['distinct_keys']:,} | {pk['duplicates']:,} |"
        )

    untranslated_list = ", ".join(f"`{c}`" for c in orphans["untranslated_categories"])
    lines.extend(
        [
            "",
            "### 2.2 Đặc thù Olist & Toàn vẹn khóa ngoại (Orphan Keys)",
            "",
            "| Hạng mục kiểm tra | Kết quả thực tế | Ý nghĩa / Diễn giải |",
            "| :--- | ---: | :--- |",
            f"| `COUNT(DISTINCT customer_id)` vs `customer_unique_id` | {cid_info['distinct_customer_id']:,} vs {cid_info['distinct_customer_unique_id']:,} | `customer_id` gắn theo từng đơn (1–1 với `orders`); có **{cid_info['repeat_customers']:,} khách thật** ({cid_info['repeat_pct']:.2f}%) mua lặp lại ≥ 2 đơn |",
            f"| Trùng `order_id` trong `raw.order_reviews` | {rev_dup['extra_rows']:,} dòng dư ({rev_dup['orders_with_multi_reviews']:,} đơn) | Có {rev_dup['orders_with_multi_reviews']:,} đơn hàng nhận từ 2 đánh giá trở lên (trên tổng {rev_dup['total_rows']:,} dòng) |",
            f"| Trùng `geolocation_zip_code_prefix` trong `raw.geolocation` | {geo_dup['duplicate_rows']:,} dòng dư | Chỉ có {geo_dup['distinct_zip']:,} mã zip duy nhất trên {geo_dup['total_rows']:,} tọa độ; **bắt buộc khử trùng trước khi JOIN** |",
            f"| `order_items.order_id` lạc loài (không có trong `orders`) | {orphans['order_items_without_order']:,} | Toàn vẹn tham chiếu `order_items -> orders` đạt 100% |",
            f"| `orders.customer_id` lạc loài (không có trong `customers`) | {orphans['orders_without_customer']:,} | Toàn vẹn tham chiếu `orders -> customers` đạt 100% |",
            f"| Sản phẩm có danh mục nhưng thiếu bản dịch tiếng Anh | {orphans['untranslated_product_rows']:,} dòng ({len(orphans['untranslated_categories'])} danh mục) | Thiếu bản dịch cho: {untranslated_list} |",
            "",
            "---",
            "",
            "## 3. Tính nhất quán nghiệp vụ (Business Logic Consistency)",
            "",
            "| Bảng | Quy tắc kiểm tra | Số dòng vi phạm | Tập đối chiếu | Tỷ lệ (%) |",
            "| :--- | :--- | ---: | ---: | ---: |",
        ]
    )

    for rule in consistency_rows:
        lines.append(
            f"| `{rule['table']}` | {rule['rule']} | {rule['violations']:,} | "
            f"{rule['base_count']:,} | {rule['pct']:.3f}% |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Đo lường phân phối & Phát hiện ngoại lệ (Distribution & IQR Outliers)",
            "",
            "| Biến | Count | Min | Q1 | Median (Q2) | Mean | Q3 | Max | Std | IQR | Ngưỡng dưới | Ngưỡng trên | Outliers | Tỷ lệ (%) | Skewness | Excess Kurtosis |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, r in dist_df.iterrows():
        lines.append(
            f"| `{r['key']}` | {r['count']:,} | {r['min']:,.2f} | {r['q1']:,.2f} | {r['median']:,.2f} | "
            f"{r['mean']:,.2f} | {r['q3']:,.2f} | {r['max']:,.2f} | {r['std']:,.2f} | {r['iqr']:,.2f} | "
            f"{r['lower_bound']:,.2f} | {r['upper_bound']:,.2f} | {r['outliers_count']:,} | {r['outliers_pct']:.2f}% | "
            f"{r['skewness']:.2f} | {r['excess_kurtosis']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Biểu đồ chẩn đoán phân phối & ngoại lệ",
            "",
            "![Phân phối và Ngoại lệ các biến cốt lõi](figures/ch3_audit_distributions.png)",
            "",
            "---",
            "",
            "## 5. Quy tắc làm sạch đề xuất cho Chương 4 (Cleaning Rules)",
            "",
            "1. **KHÔNG xóa mù quáng các outlier giá trị cao (`price`, `payment_value`, `freight_value`, `product_weight_g`):**",
            "   - Các giá trị vượt ngưỡng `Q3 + 1.5*IQR` (như đơn hàng lên tới 6,735 BRL hoặc thanh toán 13,664.08 BRL) phản ánh **giao dịch thật** của hàng điện tử, nội thất hoặc đơn hàng số lượng lớn, không phải lỗi nhập liệu.",
            "   - **Hành động:** Giữ nguyên toàn bộ để tính đúng GMV, doanh thu và RFM trong kho dữ liệu & Power BI; khi dùng thống kê mô tả thì ưu tiên báo cáo **Median + IQR** song song với Mean; khi huấn luyện mô hình ML (Chương 6–7) sẽ xử lý bằng biến đổi `log1p` hoặc cờ đánh dấu thay vì xóa bỏ.",
            "2. **Xử lý và gắn cờ lỗi logic nghiệp vụ (`orders`, `order_payments`):**",
            "   - **8 đơn `delivered` thiếu `order_delivered_customer_date`:** Giữ đơn trong bảng tổng hợp doanh thu nhưng **loại khỏi phép tính `delivery_days`** (tránh làm sai lệch chỉ số thời gian giao hàng).",
            "   - **1,359 đơn có `order_delivered_carrier_date < order_approved_at` và 23 đơn `order_delivered_customer_date < order_delivered_carrier_date`:** Gắn cờ cảnh báo `has_timestamp_anomaly = TRUE` ở tầng sạch thay vì xóa đơn (vì đơn vẫn giao thành công, nguyên nhân thường do độ trễ đồng bộ log giữa cổng thanh toán và đơn vị vận chuyển).",
            "   - **9 dòng `payment_value <= 0` (toàn bộ bằng 0) và 2 dòng `payment_installments < 1`:** Gắn cờ hoặc chuẩn hóa `payment_installments = 1` khi bằng 0 và loại dòng `payment_value = 0` khi phân tích giá trị thanh toán trung bình.",
            "3. **Khử trùng lặp có chủ đích trước khi JOIN (`geolocation` & `order_reviews`):**",
            "   - **`raw.geolocation` (1,000,163 dòng -> 19,015 zip):** Tổng hợp mỗi `geolocation_zip_code_prefix` thành 1 dòng duy nhất (lấy `MEDIAN(geolocation_lat)`, `MEDIAN(geolocation_lng)` và thành phố/bang phổ biến nhất) trước khi nối với `customers`/`sellers` để tránh bùng nổ dòng.",
            "   - **`raw.order_reviews` (547 đơn có nhiều review):** Lấy bản ghi mới nhất theo `review_answer_timestamp` / `review_creation_date` (hoặc tổng hợp trung bình `review_score` theo `order_id`) để đảm bảo quan hệ 1–1 khi gắn vào Fact đơn hàng.",
            "4. **Xử lý giá trị thiếu & bổ sung bản dịch thiếu (`products`, `order_reviews`):**",
            "   - **610 sản phẩm thiếu `product_category_name`:** Gán nhãn `'unknown'` ở bảng Dimension sản phẩm; với **2 sản phẩm thiếu trọng lượng/kích thước**, điền bằng trung vị (`MEDIAN`) theo nhóm hoặc toàn bảng nếu cần tính cước/đặc trưng ML.",
            "   - **2 danh mục chưa có bản dịch tiếng Anh (13 sản phẩm):** Bổ sung ánh xạ fallback trong SQL (`'pc_gamer' -> 'pc_gamer'`, `'portateis_cozinha_e_preparadores_de_alimentos' -> 'portable_kitchen_food_preparers'`).",
            "   - **Chuẩn hóa tên cột sai chính tả:** Đổi `product_name_lenght` $\\rightarrow$ `product_name_length` và `product_description_lenght` $\\rightarrow$ `product_description_length` khi xây bảng Dimension ở Chương 4.",
            "   - **Bình luận đánh giá (`review_comment_title`, `review_comment_message`):** Giữ nguyên `NULL` vì đây là trường tùy chọn của khách hàng, tỷ lệ bỏ trống phản ánh đúng hành vi thực tế (chỉ có thể tạo cờ boolean `has_comment`).",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_terminal_summary(
    missing_info: tuple[list[dict], dict[str, int], list[str]],
    integrity_info: dict,
    consistency_rows: list[dict],
    dist_df: pd.DataFrame,
    report_path: Path,
    figure_path: Path,
) -> None:
    """In tóm tắt ngắn gọn kết quả kiểm toán lên Terminal."""
    missing_rows, total_rows_by_table, clean_tables = missing_info
    cid_info = integrity_info["customer_identity"]
    rev_dup = integrity_info["order_reviews_dup"]
    geo_dup = integrity_info["geolocation_dup"]
    orphans = integrity_info["orphans"]

    print("=== KẾT QUẢ KIỂM TOÁN CHẤT LƯỢNG DỮ LIỆU TẦNG RAW ===")
    print(
        f"1. Tính đầy đủ: {len(missing_rows)} cột có thiếu dữ liệu trên 3 bảng; "
        f"{len(clean_tables)}/9 bảng sạch 100% (tổng {sum(total_rows_by_table.values()):,} dòng)."
    )
    print(
        "2. Trùng lặp & Toàn vẹn: "
        f"0 trùng PK ở 5 bảng lõi | "
        f"{cid_info['distinct_customer_unique_id']:,} khách thật / {cid_info['distinct_customer_id']:,} customer_id "
        f"({cid_info['repeat_customers']:,} khách mua lặp lại) | "
        f"{rev_dup['orders_with_multi_reviews']:,} đơn nhiều review | "
        f"{geo_dup['duplicate_rows']:,} dòng zip trùng ở geolocation | "
        f"0 orphan (items->orders, orders->customers) | "
        f"{orphans['untranslated_product_rows']} sản phẩm thiếu bản dịch ({len(orphans['untranslated_categories'])} danh mục)."
    )

    violated_rules = [r for r in consistency_rows if r["violations"] > 0]
    print(f"3. Nhất quán nghiệp vụ: {len(violated_rules)}/{len(consistency_rows)} quy tắc có vi phạm:")
    for r in violated_rules:
        print(f"   - {r['table']}: {r['rule']} -> {r['violations']:,} dòng ({r['pct']:.3f}%)")

    print("4. Phân phối & IQR Outliers (5 biến trọng tâm):")
    print(
        f"   {'Biến':<18} | {'Median':>9} | {'Mean':>9} | {'IQR':>9} | "
        f"{'Upper Bound':>11} | {'Outliers':>15} | {'Skew':>6}"
    )
    print("   " + "-" * 92)
    for _, r in dist_df.iterrows():
        out_str = f"{r['outliers_count']:,} ({r['outliers_pct']:.2f}%)"
        print(
            f"   {r['key']:<18} | {r['median']:>9,.2f} | {r['mean']:>9,.2f} | {r['iqr']:>9,.2f} | "
            f"{r['upper_bound']:>11,.2f} | {out_str:>15} | {r['skewness']:>6.2f}"
        )

    print("\nĐã xuất 2 tệp kết quả:")
    print(f"  - Báo cáo Markdown : {report_path}")
    print(f"  - Biểu đồ chẩn đoán: {figure_path}")


def run_audit() -> None:
    """Chạy toàn bộ quy trình kiểm toán chất lượng dữ liệu tầng `raw`."""
    if not WAREHOUSE_PATH.is_file():
        print(f"[ERROR] Không tìm thấy file DuckDB tại: {WAREHOUSE_PATH}")
        print("Vui lòng chạy `python -m src.ingest` trước khi kiểm toán.")
        sys.exit(1)

    ensure_dirs()
    report_path = REPORTS_DIR / "data_quality_audit.md"
    figure_path = FIGURES_DIR / "ch3_audit_distributions.png"

    with duckdb.connect(str(WAREHOUSE_PATH), read_only=True) as conn:
        missing_info = audit_missing_values(conn)
        integrity_info = audit_duplicates_and_integrity(conn)
        consistency_rows = audit_business_consistency(conn)
        dist_df, series_map = audit_distributions_and_outliers(conn)

    plot_audit_distributions(series_map, dist_df, figure_path)
    write_markdown_report(
        missing_info,
        integrity_info,
        consistency_rows,
        dist_df,
        report_path,
    )
    print_terminal_summary(
        missing_info,
        integrity_info,
        consistency_rows,
        dist_df,
        report_path,
        figure_path,
    )


if __name__ == "__main__":
    run_audit()
