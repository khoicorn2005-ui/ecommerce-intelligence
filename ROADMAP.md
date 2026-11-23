# Khung sườn trọn bộ dự án: E-commerce Customer & Revenue Intelligence

Xuyên suốt dự án, nhánh `main` luôn là bản chạy sạch không lỗi; từ Chương 3 trở đi, mỗi chương sẽ được làm riêng trên một nhánh `feature/...` rồi mới kiểm thử, commit và gộp (merge) vào `main`.

## Phần 1: Chuẩn bị & Khởi tạo nền móng (Nhánh `main` — làm trong bài này)

- **Chương 1:** Chốt bài toán kinh doanh & quy ước kỹ thuật trong file `PROJECT_SPEC.md`.
- **Chương 2:** Tạo kho lưu trữ trên GitHub, thiết lập môi trường ảo `.venv`, giao Antigravity dựng khung thư mục chuẩn và đẩy commit khởi tạo đầu tiên.

## Phần 2: Xử lý dữ liệu End-to-End & Phân tích Mô tả

- **Chương 3: Thu thập, Kiểm toán & Nạp dữ liệu thô** (`feature/data-ingestion`) — Tải 9 bảng CSV thực tế của bộ **Olist E-commerce**, viết script tự động kiểm tra tính đầy đủ, trùng lặp, giá trị thiếu, ngoại lệ bằng quy tắc IQR và nạp toàn bộ vào cơ sở dữ liệu **DuckDB**.
- **Chương 4: Kho dữ liệu SQL Star Schema & Cohort Analysis** (`feature/sql-warehouse-cohort`) — Viết các câu lệnh SQL xây bảng Fact/Dimension sạch, tính các chỉ số kinh doanh cốt lõi và xây ma trận **Cohort Analysis** đo tỷ lệ giữ chân khách hàng theo tháng.

## Phần 3: Chẩn đoán Thống kê & Tự train Mô hình Học máy / Học sâu

- **Chương 5: Phân tích Chẩn đoán bằng Kiểm định Thống kê** (`feature/statistical-diagnostics`) — Chứng minh nguyên nhân đánh giá thấp/hủy đơn bằng kiểm định **Chi-square**, tiền kiểm định **Levene**, **One-Way ANOVA** và hậu kiểm **Tukey**.
- **Chương 6: Phân cụm RFM & Tự code Mô hình Phân lớp từ đầu** (`feature/customer-ml-models`) — Gom cụm khách hàng bằng **K-Means**; tự viết mô hình **Logistic Regression từ đầu bằng thuần NumPy** (tối ưu bằng Gradient Descent với hàm mất mát Cross-Entropy) để dự đoán đơn hàng rủi ro đánh giá xấu, đối chiếu với **XGBoost** (đánh giá qua Precision, Recall, F1, ROC-AUC) và giải thích mô hình bằng **SHAP**.
- **Chương 7: Dự báo Doanh thu Chuỗi thời gian & Phát hiện Bất thường** (`feature/forecast-anomaly`) — Kiểm định tính dừng **ADF**, xây mô hình **SARIMAX** đối chiếu với **LSTM/GRU** (chạy GPU miễn phí trên Google Colab), và lọc đơn hàng bất thường bằng **Isolation Forest**.

## Phần 4: Trực quan hóa Power BI, Đề xuất Hành động & Phát hành v1.0.0

- **Chương 8: Dashboard Power BI, Viết README Case Study & Phát hành** (`feature/powerbi-release`) — Xuất bảng dữ liệu sạch kèm kết quả dự báo sang `.parquet`, dựng Dashboard **Power BI** gắn kịch bản đề xuất hành động (Prescriptive), chụp ảnh Dashboard viết `README.md` chuẩn đi xin việc, gộp vào `main` và gắn Tag `v1.0.0`.