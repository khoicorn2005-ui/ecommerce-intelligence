"""Cấu hình đường dẫn tập trung cho dự án E-commerce Customer & Revenue Intelligence."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_WAREHOUSE_DIR: Path = DATA_DIR / "warehouse"
WAREHOUSE_PATH: Path = DATA_WAREHOUSE_DIR / "olist.duckdb"

SQL_DIR: Path = PROJECT_ROOT / "sql"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DASHBOARD_DIR: Path = PROJECT_ROOT / "dashboard"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

REQUIRED_DIRS: tuple[Path, ...] = (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_WAREHOUSE_DIR,
    SQL_DIR,
    NOTEBOOKS_DIR,
    MODELS_DIR,
    DASHBOARD_DIR,
    FIGURES_DIR,
)


def ensure_dirs() -> None:
    """Đảm bảo tất cả các thư mục làm việc chuẩn của dự án đều tồn tại."""
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
