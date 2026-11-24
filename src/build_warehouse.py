"""Thực thi SQL xây dựng Star Schema (`dw.*`), đối soát dữ liệu và xuất Parquet."""

import sys
import time
from pathlib import Path

import duckdb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import DATA_PROCESSED_DIR, SQL_DIR, WAREHOUSE_PATH, ensure_dirs  # noqa: E402

DW_TABLES: tuple[str, ...] = (
    "dim_geolocation",
    "dim_customers",
    "dim_sellers",
    "dim_products",
    "dim_date",
    "fact_orders",
    "fact_order_items",
)


def run_reconciliation_checks(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Chạy bộ kiểm thử đối soát số dòng, khóa chính, tài chính và kiểu dữ liệu zip."""
    results: list[tuple[str, str]] = []

    # 1. Đối soát số dòng chống fan-out
    dw_orders_cnt = conn.execute("SELECT COUNT(*) FROM dw.fact_orders").fetchone()[0]
    raw_orders_cnt = conn.execute("SELECT COUNT(*) FROM raw.orders").fetchone()[0]
    if dw_orders_cnt != raw_orders_cnt:
        raise AssertionError(
            f"Lệch số dòng fact_orders ({dw_orders_cnt:,}) vs raw.orders ({raw_orders_cnt:,})"
        )
    results.append(
        ("Row count: dw.fact_orders == raw.orders", f"{dw_orders_cnt:,} == {raw_orders_cnt:,}")
    )

    dw_items_cnt = conn.execute("SELECT COUNT(*) FROM dw.fact_order_items").fetchone()[0]
    raw_items_cnt = conn.execute("SELECT COUNT(*) FROM raw.order_items").fetchone()[0]
    if dw_items_cnt != raw_items_cnt:
        raise AssertionError(
            f"Lệch số dòng fact_order_items ({dw_items_cnt:,}) vs raw.order_items ({raw_items_cnt:,})"
        )
    results.append(
        (
            "Row count: dw.fact_order_items == raw.order_items",
            f"{dw_items_cnt:,} == {raw_items_cnt:,}",
        )
    )

    # 2. Khóa chính duy nhất 100% cho toàn bộ 7 bảng dw.*
    pk_specs = [
        ("dw.dim_geolocation", "zip_code_prefix"),
        ("dw.dim_customers", "customer_id"),
        ("dw.dim_sellers", "seller_id"),
        ("dw.dim_products", "product_id"),
        ("dw.dim_date", "date_key"),
        ("dw.fact_orders", "order_id"),
        ("dw.fact_order_items", "(order_id || '|' || CAST(order_item_id AS VARCHAR))"),
    ]
    for table, key_expr in pk_specs:
        total_cnt, distinct_cnt = conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT {key_expr}) FROM {table}"
        ).fetchone()
        if total_cnt != distinct_cnt:
            raise AssertionError(
                f"Trùng khóa chính ở {table}: tổng {total_cnt:,} != duy nhất {distinct_cnt:,}"
            )
    results.append(("Unique PKs: tất cả 7 bảng dw.*", "7/7 bảng duy nhất 100%"))

    # 3. Đối soát tài chính (sai số tuyệt đối < 0.01)
    finance_specs = [
        (
            "SUM(price): fact_order_items vs raw.order_items",
            "SELECT SUM(price) FROM dw.fact_order_items",
            "SELECT SUM(price) FROM raw.order_items",
        ),
        (
            "SUM(freight_value): fact_order_items vs raw.order_items",
            "SELECT SUM(freight_value) FROM dw.fact_order_items",
            "SELECT SUM(freight_value) FROM raw.order_items",
        ),
        (
            "SUM(payment_total): fact_orders vs raw.order_payments",
            "SELECT SUM(payment_total) FROM dw.fact_orders",
            "SELECT SUM(payment_value) FROM raw.order_payments",
        ),
    ]
    for label, dw_query, raw_query in finance_specs:
        dw_val = float(conn.execute(dw_query).fetchone()[0] or 0.0)
        raw_val = float(conn.execute(raw_query).fetchone()[0] or 0.0)
        diff = abs(dw_val - raw_val)
        if diff >= 0.01:
            raise AssertionError(
                f"Lệch đối soát tài chính [{label}]: dw={dw_val:,.4f}, raw={raw_val:,.4f}, diff={diff:,.4f}"
            )
        results.append((label, f"diff = {diff:.6f} (dw={dw_val:,.2f})"))

    # 4. Xác nhận kiểu VARCHAR và số 0 đầu ở dim_geolocation.zip_code_prefix
    zip_dtype = conn.execute(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = 'dw'
          AND table_name = 'dim_geolocation'
          AND column_name = 'zip_code_prefix'
        """
    ).fetchone()[0]
    if "VARCHAR" not in zip_dtype.upper():
        raise AssertionError(
            f"Kiểu dữ liệu dw.dim_geolocation.zip_code_prefix phải là VARCHAR, nhận được: {zip_dtype}"
        )

    leading_zero_cnt = conn.execute(
        "SELECT COUNT(*) FROM dw.dim_geolocation WHERE zip_code_prefix LIKE '0%'"
    ).fetchone()[0]
    if leading_zero_cnt == 0:
        raise AssertionError("Không còn mã zip bắt đầu bằng '0' trong dw.dim_geolocation!")
    results.append(
        (
            "Zip VARCHAR & leading zeros: dw.dim_geolocation",
            f"{zip_dtype}, {leading_zero_cnt:,} mã bắt đầu bằng '0'",
        )
    )

    return results


def export_dw_to_parquet(
    conn: duckdb.DuckDBPyConnection,
    processed_dir: Path,
) -> list[tuple[str, int, int, float]]:
    """Xuất toàn bộ 7 bảng `dw.*` ra file `.parquet` và thu thập thống kê."""
    summary_rows: list[tuple[str, int, int, float]] = []

    for table in DW_TABLES:
        out_path = processed_dir / f"{table}.parquet"
        conn.execute(
            f"COPY dw.{table} TO '{out_path.as_posix()}' (FORMAT PARQUET);"
        )
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
        summary_rows.append((f"dw.{table}", rows, cols, size_mb))

    return summary_rows


def print_summary(
    table_stats: list[tuple[str, int, int, float]],
    check_results: list[tuple[str, str]],
    elapsed_sec: float,
) -> None:
    """In bảng tóm tắt Star Schema, file Parquet và kết quả kiểm thử đối soát."""
    header = (
        f"{'Bảng (Table)':<22} | {'Số dòng (Rows)':>14} | "
        f"{'Số cột':>7} | {'Parquet (MB)':>12}"
    )
    sep = "-" * len(header)

    print("\n=== TÓM TẮT KHO DỮ LIỆU STAR SCHEMA (DW) & PARQUET ===")
    print(sep)
    print(header)
    print(sep)
    for table_name, rows, cols, size_mb in table_stats:
        print(f"{table_name:<22} | {rows:>14,} | {cols:>7} | {size_mb:>12.2f}")
    print(sep)
    print(
        f"{'TỔNG CỘNG (7 bảng)':<22} | "
        f"{sum(r[1] for r in table_stats):>14,} | {'-':>7} | "
        f"{sum(r[3] for r in table_stats):>12.2f}"
    )
    print(sep)

    print("\n=== KẾT QUẢ KIỂM THỬ ĐỐI SOÁT DỮ LIỆU (RECONCILIATION) ===")
    for label, detail in check_results:
        print(f"[PASS] {label:<50} -> {detail}")

    print(f"\nThư mục Parquet : {DATA_PROCESSED_DIR}")
    print(f"Tổng thời gian  : {elapsed_sec:.2f} giây")


def build_warehouse() -> None:
    """Thực thi xây dựng Star Schema, kiểm thử đối soát và xuất file Parquet."""
    sql_file = SQL_DIR / "01_build_star_schema.sql"
    if not WAREHOUSE_PATH.is_file():
        print(f"[ERROR] Không tìm thấy file DuckDB tại: {WAREHOUSE_PATH}")
        print("Vui lòng chạy `python -m src.ingest` trước.")
        sys.exit(1)
    if not sql_file.is_file():
        print(f"[ERROR] Không tìm thấy file SQL tại: {sql_file}")
        sys.exit(1)

    ensure_dirs()
    start_time = time.perf_counter()
    sql_script = sql_file.read_text(encoding="utf-8")

    with duckdb.connect(str(WAREHOUSE_PATH)) as conn:
        conn.execute(sql_script)
        check_results = run_reconciliation_checks(conn)
        table_stats = export_dw_to_parquet(conn, DATA_PROCESSED_DIR)

    elapsed_sec = time.perf_counter() - start_time
    print_summary(table_stats, check_results, elapsed_sec)


if __name__ == "__main__":
    build_warehouse()
