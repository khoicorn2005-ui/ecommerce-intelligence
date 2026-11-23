# E-commerce Customer & Revenue Intelligence — Project Specification

## 1. Mục tiêu dự án
Xây dựng giải pháp phân tích dữ liệu End-to-End trên bộ dữ liệu đa bảng Olist E-commerce, đi trọn vẹn 4 cấp độ phân tích:
- Descriptive (Mô tả): Kho dữ liệu SQL Star Schema, KPI kinh doanh, Cohort Analysis đo tỷ lệ giữ chân khách hàng.
- Diagnostic (Chẩn đoán): Kiểm định thống kê (Chi-square, Levene, One-Way ANOVA, Tukey HSD) chứng minh nguyên nhân đánh giá thấp và hủy đơn.
- Predictive (Dự báo): Phân cụm khách hàng RFM + K-Means; Tự cài đặt Logistic Regression từ đầu (thuần NumPy + Gradient Descent) đối chiếu với XGBoost & SHAP; Dự báo doanh thu (SARIMAX vs LSTM/GRU); Phát hiện đơn hàng bất thường (Isolation Forest).
- Prescriptive (Đề xuất): Kết nối bảng dữ liệu sạch & kết quả dự báo (định dạng Parquet) vào Power BI để đề xuất hành động kinh doanh kèm ước tính chi phí/lợi ích.

## 2. Quy ước kỹ thuật (Bắt buộc)
- Ngôn ngữ & Môi trường: Python 3.13+ chạy trong môi trường ảo `.venv` cục bộ.
- Cơ sở dữ liệu: DuckDB lưu tại `data/warehouse/olist.duckdb` (không cài thêm server database chạy nền).
- Quản lý đường dẫn: Mọi đường dẫn tệp phải dùng `pathlib.Path` lấy từ `src/config.py`, tuyệt đối không hardcode đường dẫn ổ đĩa tuyệt đối (`D:\...`).
- Tách module gọn gàng: Logic chính viết trong `src/*.py` và `sql/*.sql`; thư mục `notebooks/` chỉ dùng cho trực quan hóa khám phá (EDA) và trình bày biểu đồ.
- Kỷ luật học máy: Chống rò rỉ dữ liệu (data leakage), chia tập theo thời gian cho chuỗi thời gian và Stratified K-Fold cho bài toán phân lớp.
- Quản lý Git: TUYỆT ĐỐI không đẩy `.venv/`, dữ liệu thô (`data/raw/*.csv`), file `.duckdb` hoặc file nặng lên GitHub (phải chặn trong `.gitignore`).

## 3. Cấu trúc thư mục chuẩn
- `data/raw/`: Chứa các file CSV gốc tải về (bị Git bỏ qua).
- `data/processed/`: Chứa các bảng Parquet sạch và kết quả dự báo cho Power BI (bị Git bỏ qua).
- `data/warehouse/`: Chứa file cơ sở dữ liệu `olist.duckdb` (bị Git bỏ qua).
- `sql/`: Chứa các file truy vấn SQL (schema, transform, cohort, KPI).
- `src/`: Mã nguồn Python tách theo từng module (`config.py`, `etl.py`, `stats_tests.py`, `models/`...).
- `notebooks/`: Các notebook EDA ngắn gọn theo từng chương.
- `models/`: Lưu trọng số/mô hình đã huấn luyện (bị Git bỏ qua nếu nặng).
- `dashboard/`: File Power BI (`.pbix`) và ảnh chụp giao diện.
- `reports/figures/`: Biểu đồ xuất ra để gắn vào báo cáo và `README.md`.

## 4. Cẩm nang học tập cá nhân (Local Only — Không đẩy lên GitHub)
- File `PROJECT_HANDBOOK.md` nằm tại thư mục gốc, được chặn trong `.gitignore` (tuyệt đối không commit lên GitHub).
- Vai trò: Là bách khoa toàn thư và từ điển kỹ thuật của riêng chủ dự án, được cập nhật nối tiếp sau mỗi chương/chặng.
- Văn phong: Tiếng Việt dễ hiểu cho người mới, giải thích rõ "tại sao làm vậy", thuật ngữ mới, công thức toán/thống kê và các câu hỏi phỏng vấn liên quan.