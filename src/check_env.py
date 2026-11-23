"""Script kiểm tra môi trường Python, thư viện lõi, DuckDB và cấu hình đường dẫn."""

import importlib
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import PROJECT_ROOT  # noqa: E402

REQUIRED_PACKAGES: tuple[str, ...] = (
    "duckdb",
    "pandas",
    "pyarrow",
    "scipy",
    "statsmodels",
    "sklearn",
    "xgboost",
    "shap",
    "matplotlib",
    "seaborn",
)


def check_packages() -> bool:
    """Kiểm tra import và in phiên bản của từng thư viện lõi."""
    print("=== 1. KIỂM TRA THƯ VIỆN CỐT LÕI ===")
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {pkg:<14} (version: {version})")
        except Exception as exc:
            print(f"[FAIL] {pkg:<14} -> {exc}")
            all_ok = False
    return all_ok


def check_duckdb() -> bool:
    """Chạy truy vấn SQL thử nghiệm trong bộ nhớ (không tạo file trên ổ đĩa)."""
    print("\n=== 2. KIỂM TRA DUCKDB ===")
    import duckdb

    with duckdb.connect(":memory:") as conn:
        row = conn.execute("SELECT 'DuckDB Ready' AS status").fetchone()

    status = row[0] if row else None
    if status != "DuckDB Ready":
        print(f"[FAIL] Kết quả DuckDB không như kỳ vọng: {status}")
        return False

    print(f"[OK] DuckDB query status: {status}")
    return True


def main() -> None:
    """Chạy toàn bộ các bước kiểm tra môi trường."""
    if not check_packages():
        print("\n[ERROR] Thiếu hoặc lỗi thư viện. Hãy kiểm tra lại môi trường ảo .venv.")
        sys.exit(1)

    if not check_duckdb():
        print("\n[ERROR] Kiểm tra DuckDB thất bại.")
        sys.exit(1)

    print("\n=== 3. KIỂM TRA CẤU HÌNH ĐƯỜNG DẪN ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print("\nMÔI TRƯỜNG ĐÃ SẴN SÀNG!")


if __name__ == "__main__":
    main()
