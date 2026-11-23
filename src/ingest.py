"""Pipeline nạp 9 bảng CSV gốc của bộ dữ liệu Olist vào schema `raw` trong DuckDB."""

import sys
import time
from pathlib import Path

import duckdb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import DATA_RAW_DIR, WAREHOUSE_PATH, ensure_dirs  # noqa: E402

RAW_TABLES: dict[str, str] = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

TABLE_CSV_OPTIONS: dict[str, str] = {
    "customers": "types={'customer_zip_code_prefix': 'VARCHAR'}",
    "geolocation": "types={'geolocation_zip_code_prefix': 'VARCHAR'}",
    "sellers": "types={'seller_zip_code_prefix': 'VARCHAR'}",
    "order_reviews": "strict_mode=False, quote='\"', escape='\"'",
}


def validate_raw_files(raw_dir: Path, tables: dict[str, str]) -> None:
    """Kiểm tra đủ 9 file CSV gốc trước khi nạp vào DuckDB."""
    missing = [
        (table_name, filename)
        for table_name, filename in tables.items()
        if not (raw_dir / filename).is_file()
    ]
    if missing:
        print(f"[ERROR] Không tìm thấy đầy đủ file CSV trong thư mục: {raw_dir}")
        print("Danh sách file bị thiếu:")
        for table_name, filename in missing:
            print(f"  - {filename} (bảng raw.{table_name})")
        sys.exit(1)


def print_summary(
    stats: list[tuple[str, int, int]],
    elapsed_sec: float,
    warehouse_path: Path,
) -> None:
    """In bảng tóm tắt số dòng, số cột, thời gian chạy và dung lượng file DuckDB."""
    total_rows = sum(rows for _, rows, _ in stats)
    size_mb = warehouse_path.stat().st_size / (1024 * 1024)

    header = f"{'Bảng (Table)':<28} | {'Số dòng (Rows)':>15} | {'Số cột (Cols)':>13}"
    separator = "-" * len(header)

    print("\n=== TÓM TẮT NẠP DỮ LIỆU TẦNG RAW (DUCKDB) ===")
    print(separator)
    print(header)
    print(separator)
    for table_name, rows, cols in stats:
        print(f"{f'raw.{table_name}':<28} | {rows:>15,} | {cols:>13}")
    print(separator)
    print(f"{'TỔNG CỘNG (9 bảng)':<28} | {total_rows:>15,} | {'-':>13}")
    print(separator)
    print(f"Tổng thời gian thực thi : {elapsed_sec:.2f} giây")
    print(f"Kho dữ liệu DuckDB     : {warehouse_path}")
    print(f"Dung lượng file        : {size_mb:.2f} MB")


def ingest_raw_tables() -> None:
    """Nạp nguyên bản toàn bộ 9 file CSV Olist vào schema `raw` của DuckDB."""
    ensure_dirs()
    validate_raw_files(DATA_RAW_DIR, RAW_TABLES)

    start_time = time.perf_counter()
    stats: list[tuple[str, int, int]] = []

    with duckdb.connect(str(WAREHOUSE_PATH)) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS raw;")

        for table_name, filename in RAW_TABLES.items():
            csv_path = (DATA_RAW_DIR / filename).as_posix()
            extra_opts = TABLE_CSV_OPTIONS.get(table_name, "")
            opts = "header=True, sample_size=-1" + (
                f", {extra_opts}" if extra_opts else ""
            )
            query = (
                f"CREATE OR REPLACE TABLE raw.{table_name} AS "
                f"SELECT * FROM read_csv_auto('{csv_path}', {opts});"
            )
            conn.execute(query)

        for table_name in RAW_TABLES:
            rows = conn.execute(f"SELECT COUNT(*) FROM raw.{table_name}").fetchone()[0]
            cols = conn.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = 'raw' AND table_name = ?
                """,
                [table_name],
            ).fetchone()[0]
            stats.append((table_name, rows, cols))

    elapsed_sec = time.perf_counter() - start_time
    print_summary(stats, elapsed_sec, WAREHOUSE_PATH)


if __name__ == "__main__":
    ingest_raw_tables()
