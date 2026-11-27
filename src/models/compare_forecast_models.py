"""Chương 7 — Chặng 2: Đối chiếu công bằng 4 mô hình dự báo (Seasonal Naive vs SARIMAX vs LSTM vs GRU)."""

import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    WAREHOUSE_PATH,
    ensure_dirs,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Tính thống nhất MAE, RMSE, MAPE (%) và SMAPE (%) trên 60 ngày Hold-out Test."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mape = float(np.mean(np.abs((yt - yp) / np.maximum(yt, 1e-8))) * 100.0)
    smape = float(
        np.mean(2.0 * np.abs(yp - yt) / np.maximum(np.abs(yt) + np.abs(yp), 1e-8)) * 100.0
    )
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}


def save_benchmark_to_duckdb_and_parquet(
    full_timeline_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """Ghi 2 bảng đối chiếu dự báo vào DuckDB (`dw.*`) và xuất ra Parquet."""
    comp_parquet = DATA_PROCESSED_DIR / "ml_revenue_forecast_comparison.parquet"
    metrics_parquet = DATA_PROCESSED_DIR / "mart_forecast_model_metrics.parquet"

    with duckdb.connect(str(WAREHOUSE_PATH)) as conn:
        conn.register("fc_comp_view", full_timeline_df)
        conn.execute(
            "CREATE OR REPLACE TABLE dw.ml_revenue_forecast_comparison AS SELECT * FROM fc_comp_view;"
        )
        conn.unregister("fc_comp_view")

        conn.register("fc_metrics_view", metrics_df)
        conn.execute(
            "CREATE OR REPLACE TABLE dw.mart_forecast_model_metrics AS SELECT * FROM fc_metrics_view;"
        )
        conn.unregister("fc_metrics_view")

        conn.execute(
            f"COPY dw.ml_revenue_forecast_comparison TO '{comp_parquet.as_posix()}' (FORMAT PARQUET);"
        )
        conn.execute(
            f"COPY dw.mart_forecast_model_metrics TO '{metrics_parquet.as_posix()}' (FORMAT PARQUET);"
        )

    return comp_parquet, metrics_parquet


def plot_benchmark_comparison(
    test_merged: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Vẽ biểu đồ 2x1: Diễn biến 60 ngày Test của 4 mô hình & Cột so sánh MAE + SMAPE."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig = plt.figure(figsize=(15.5, 9.8), dpi=150, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.14)

    ax_top = fig.add_subplot(gs[0, :])
    ax_mae = fig.add_subplot(gs[1, 0])
    ax_smape = fig.add_subplot(gs[1, 1], sharey=ax_mae)

    fig.suptitle(
        "Đối chiếu Công bằng 4 Mô hình Dự báo Doanh thu Ngày (60 Ngày Hold-out Test: 2018-06-22 -> 2018-08-20)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    dates = pd.to_datetime(test_merged["date"])
    ax_top.plot(
        dates,
        test_merged["actual_gmv"] / 1000.0,
        color="#0F172A",
        linewidth=2.3,
        label="Thực tế (Actual GMV)",
    )
    ax_top.plot(
        dates,
        test_merged["pred_baseline"] / 1000.0,
        color="#94A3B8",
        linestyle=":",
        linewidth=1.6,
        alpha=0.85,
        label="Seasonal Naive Baseline",
    )
    ax_top.plot(
        dates,
        test_merged["pred_sarimax"] / 1000.0,
        color="#2563EB",
        linewidth=2.1,
        label="SARIMAX (log1p + exog)",
    )
    ax_top.plot(
        dates,
        test_merged["pred_lstm"] / 1000.0,
        color="#7C3AED",
        linestyle="--",
        linewidth=2.0,
        label="PyTorch LSTM (Colab GPU)",
    )
    ax_top.plot(
        dates,
        test_merged["pred_gru"] / 1000.0,
        color="#059669",
        linestyle="-.",
        linewidth=2.0,
        label="PyTorch GRU (Colab GPU)",
    )

    ax_top.set_title(
        "Quỹ đạo Dự báo Đa bước 60 ngày Hold-out Test: Thống kê Cổ điển vs Học sâu",
        fontsize=11.8,
        fontweight="bold",
        pad=10,
    )
    ax_top.set_ylabel("Doanh thu GMV (Nghìn BRL)", fontsize=10.2, fontweight="bold")
    ax_top.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax_top.get_xticklabels(), rotation=20, ha="right", fontsize=8.8)
    ax_top.legend(loc="upper left", ncol=3, frameon=True, facecolor="white", fontsize=9)

    # --- Dưới trái: So sánh MAE (BRL) ---
    m_sorted = metrics_df.sort_values("mae", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(m_sorted))
    colors_mae = [
        "#DC2626" if is_best else "#3B82F6" for is_best in m_sorted["is_best_model"]
    ]
    bars1 = ax_mae.barh(y_pos, m_sorted["mae"], color=colors_mae, height=0.55, alpha=0.88)
    ax_mae.set_yticks(y_pos)
    ax_mae.set_yticklabels(m_sorted["model_name"], fontsize=9.8, fontweight="bold")
    ax_mae.invert_yaxis()
    ax_mae.set_xlim(0, max(m_sorted["mae"].max() * 1.38, 12000.0))
    ax_mae.set_xlabel("MAE — Sai số Tuyệt đối Trung bình (BRL, thấp hơn = tốt hơn)", fontsize=9.8, fontweight="bold")
    ax_mae.set_title("Xếp hạng theo MAE (BRL)", fontsize=11.2, fontweight="bold", pad=8)

    for bar, (_, r) in zip(bars1, m_sorted.iterrows()):
        w = bar.get_width()
        tag = " [VÔ ĐỊCH]" if r["is_best_model"] else ""
        ax_mae.text(
            w + m_sorted["mae"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:,.0f} BRL{tag}",
            va="center",
            ha="left",
            fontsize=8.8,
            fontweight="bold" if r["is_best_model"] else "normal",
            color="#991B1B" if r["is_best_model"] else "#1E293B",
        )

    # --- Dưới phải: So sánh SMAPE (%) & MAPE (%) ---
    width = 0.36
    bars_smape = ax_smape.barh(
        y_pos - width / 2,
        m_sorted["smape"],
        height=width,
        color="#0D9488",
        alpha=0.88,
        label="SMAPE (%)",
    )
    bars_mape = ax_smape.barh(
        y_pos + width / 2,
        m_sorted["mape"],
        height=width,
        color="#F59E0B",
        alpha=0.88,
        label="MAPE (%)",
    )
    ax_smape.tick_params(axis="y", labelleft=False)
    max_pct = max(m_sorted["mape"].max(), m_sorted["smape"].max())
    ax_smape.set_xlim(0, max(34.0, max_pct * 1.32))
    ax_smape.set_xlabel("Sai số Phần trăm (%, thấp hơn = tốt hơn)", fontsize=9.8, fontweight="bold")
    ax_smape.set_title("So sánh SMAPE (%) & MAPE (%)", fontsize=11.2, fontweight="bold", pad=8)

    for bar in bars_smape:
        w = bar.get_width()
        ax_smape.text(
            w + 0.4,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.2f}%",
            va="center",
            ha="left",
            fontsize=8.4,
            color="#115E59",
            fontweight="bold",
        )
    for bar in bars_mape:
        w = bar.get_width()
        ax_smape.text(
            w + 0.4,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.2f}%",
            va="center",
            ha="left",
            fontsize=8.4,
            color="#92400E",
        )
    ax_smape.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.8)

    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.14, right=0.98)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_benchmark_report(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Xuất báo cáo đối chiếu 4 mô hình dự báo (không chứa dấu | chưa thoát trong ô bảng)."""
    best_row = metrics_df.loc[metrics_df["is_best_model"]].iloc[0]
    lstm_row = metrics_df.loc[metrics_df["model_id"] == "lstm"].iloc[0]
    gru_row = metrics_df.loc[metrics_df["model_id"] == "gru"].iloc[0]
    sarimax_row = metrics_df.loc[metrics_df["model_id"] == "sarimax"].iloc[0]

    better_dl = "GRU" if gru_row["mae"] <= lstm_row["mae"] else "LSTM"
    dl_diff_mae = abs(gru_row["mae"] - lstm_row["mae"])

    lines: list[str] = [
        "# Báo cáo Đối chiếu Mô hình Dự báo Doanh thu (SARIMAX vs PyTorch LSTM & GRU) — Chương 7 (Chặng 2)",
        "",
        f"- **Thời điểm tổng hợp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- **Thiết lập thực nghiệm công bằng (Apples-to-Apples Benchmark):**",
        "  - Cùng tập **Train = 533 ngày** (`2017-01-05` – `2018-06-21`) và **Hold-out Test = 60 ngày** (`2018-06-22` – `2018-08-20`).",
        "  - Cùng chiến lược biến đổi mục tiêu `np.log1p(daily_gmv)` để ổn định phương sai và dự báo đa bước hoàn toàn tự động 60 ngày không nhìn trộm giá trị thật trên tập Test.",
        "  - Cùng nhóm thông tin lịch/sự kiện biết trước trong tương lai (`is_weekend`, `is_black_friday`, `is_black_friday_weekend`, `is_early_month` và mã hóa lượng giác chu kỳ tuần/tháng cho mạng nơ-ron).",
        "",
        "---",
        "",
        "## 1. Bảng Xếp hạng Định lượng 4 Mô hình trên 60 Ngày Hold-out Test",
        "",
        "| Hạng | Mô hình | MAE (BRL) | RMSE (BRL) | MAPE (%) | SMAPE (%) | Cải thiện MAE vs Baseline | Kết luận |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    ranked = metrics_df.sort_values(["mae", "rmse"], ascending=True).reset_index(drop=True)
    for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
        b = "**" if r["is_best_model"] else ""
        imp_str = f"{r['mae_improvement_vs_baseline_pct']:+.1f}%"
        status = "**MÔ HÌNH VÔ ĐỊCH**" if r["is_best_model"] else "Đối chứng"
        lines.append(
            f"| {rank} | {b}{r['model_name']}{b} | {b}{r['mae']:,.2f}{b} | {b}{r['rmse']:,.2f}{b} | "
            f"{b}{r['mape']:.2f}%{b} | {b}{r['smape']:.2f}%{b} | {imp_str} | {status} |"
        )

    lines.extend(
        [
            "",
            "![Đối chiếu 4 mô hình dự báo](figures/ch7_forecast_benchmark_comparison.png)",
            "",
            "---",
            "",
            "## 2. Phân tích Khoa học: Thống kê Cổ điển (SARIMAX) vs Học sâu (LSTM / GRU)",
            "",
            f"1. **Mô hình dẫn đầu thực nghiệm:** **{best_row['model_name']}** đạt sai số thấp nhất trên tập Hold-out Test (`MAE = {best_row['mae']:,.2f} BRL`, `RMSE = {best_row['rmse']:,.2f} BRL`, `SMAPE = {best_row['smape']:.2f}%`, cải thiện `{best_row['mae_improvement_vs_baseline_pct']:+.1f}%` so với Seasonal Naive).",
            f"2. **Tại sao trên cỡ mẫu ~533 ngày, SARIMAX (`MAE = {sarimax_row['mae']:,.2f} BRL`) cực kỳ cạnh tranh so với mạng học sâu?**",
            "   - Mạng hồi quy sâu (LSTM/GRU) có hàng nghìn tham số nên phát huy thế mạnh lớn nhất khi có hàng chục nghìn điểm thời gian hoặc nhiều chuỗi chéo (*panel time series*). Trên một chuỗi đơn lẻ cỡ nhỏ (~533 quan sát) và dự báo đệ quy xa tới 60 bước, sai số nhỏ ở các bước đầu của mạng nơ-ron dễ tích lũy dần (*error accumulation*), trong khi cấu trúc tự hồi quy + mùa vụ tường minh của SARIMAX giữ quỹ đạo trung bình rất vững.",
            f"3. **So sánh nội bộ Học sâu (LSTM vs GRU):** **{better_dl}** đạt MAE tốt hơn mô hình còn lại `{dl_diff_mae:,.2f} BRL` (`LSTM MAE = {lstm_row['mae']:,.2f} BRL` vs `GRU MAE = {gru_row['mae']:,.2f} BRL`). Với dữ liệu cỡ vừa, kiến trúc ít cổng hơn của GRU (2 cổng Reset/Update thay vì 3 cổng + Cell State như LSTM) giúp giảm nguy cơ quá khớp (*overfitting*) và hội tụ nhanh hơn.",
            "",
            "---",
            "",
            "## 3. Khuyến nghị Triển khai cho Dashboard Power BI (Chương 8)",
            "",
            "- **Mô hình triển khai chính thức (Production Choice):** Sử dụng đường dự báo 30 ngày tương lai kèm dải khoảng tin cậy 95% trực tiếp từ **SARIMAX** (kết hợp bảng đối chiếu 4 mô hình `mart_forecast_model_metrics.parquet` trên trang phân tích mô hình của Power BI), vì SARIMAX vừa có độ chính xác cao, giải thích được từng hệ số mùa vụ/sự kiện, chạy lại trong ~1 giây trên CPU cục bộ và cung cấp khoảng tin cậy xác suất (`95% CI`) tự nhiên cho kế hoạch dòng tiền & tồn kho.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_forecast_comparison() -> None:
    """Thực thi đối chiếu 4 mô hình dự báo sau khi có kết quả từ Google Colab."""
    sarimax_parquet = DATA_PROCESSED_DIR / "ml_revenue_forecast_sarimax.parquet"
    colab_csv = DATA_PROCESSED_DIR / "colab_dl_forecasts.csv"

    if not sarimax_parquet.is_file():
        print(f"[ERROR] Chưa tìm thấy {sarimax_parquet}. Hãy chạy `python -m src.models.forecast_sarimax` trước.")
        sys.exit(1)

    if not colab_csv.is_file():
        print("=== CHỜ KẾT QUẢ TỪ GOOGLE COLAB GPU ===")
        print(f"Chưa tìm thấy tệp dự báo Deep Learning tại: {colab_csv}")
        print("Hướng dẫn thực hiện nhanh (3 bước):")
        print("  1. Mở `notebooks/ch7_lstm_gru_colab.ipynb` trên Google Colab (chọn Runtime -> T4 GPU).")
        print("  2. Chạy toàn bộ các cell và tải lên `data/processed/ts_daily_revenue_for_colab.csv` khi được hỏi.")
        print("  3. Chép file `colab_dl_forecasts.csv` vừa tải về vào `data/processed/colab_dl_forecasts.csv`, sau đó chạy lại:")
        print("     python -m src.models.compare_forecast_models")
        return

    ensure_dirs()
    start_time = time.perf_counter()

    sarimax_all = pd.read_parquet(sarimax_parquet)
    sarimax_all["date_str"] = pd.to_datetime(sarimax_all["date"]).dt.strftime("%Y-%m-%d")

    dl_df = pd.read_csv(colab_csv)
    dl_df["date_str"] = pd.to_datetime(dl_df["date"]).dt.strftime("%Y-%m-%d")

    backtest_sarimax = sarimax_all[sarimax_all["record_type"] == "test_backtest"].copy()
    merged_test = pd.merge(
        backtest_sarimax,
        dl_df[["date_str", "actual_gmv", "pred_lstm", "pred_gru"]],
        on="date_str",
        suffixes=("", "_colab"),
        how="inner",
    ).sort_values("date_str")

    assert len(merged_test) == 60, (
        f"Số ngày khớp giữa SARIMAX và Colab phải đúng bằng 60, nhận được: {len(merged_test)}"
    )
    assert np.allclose(
        merged_test["actual_gmv"].to_numpy(),
        merged_test["actual_gmv_colab"].to_numpy(),
        atol=1e-2,
    ), "Giá trị actual_gmv giữa Colab và kho cục bộ không khớp!"

    merged_test = merged_test.rename(
        columns={
            "naive_forecast_gmv": "pred_baseline",
            "sarimax_forecast_gmv": "pred_sarimax",
        }
    )

    y_true = merged_test["actual_gmv"].to_numpy(dtype=float)
    model_specs = [
        ("baseline", "Seasonal Naive Baseline", merged_test["pred_baseline"].to_numpy(dtype=float)),
        ("sarimax", "SARIMAX (log1p + exog)", merged_test["pred_sarimax"].to_numpy(dtype=float)),
        ("lstm", "PyTorch LSTM (Colab GPU)", merged_test["pred_lstm"].to_numpy(dtype=float)),
        ("gru", "PyTorch GRU (Colab GPU)", merged_test["pred_gru"].to_numpy(dtype=float)),
    ]

    metric_rows: list[dict] = []
    base_mae = compute_metrics(y_true, model_specs[0][2])["mae"]
    for m_id, m_name, y_pred in model_specs:
        m = compute_metrics(y_true, y_pred)
        imp_pct = (base_mae - m["mae"]) / base_mae * 100.0
        metric_rows.append(
            {
                "model_id": m_id,
                "model_name": m_name,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape"],
                "smape": m["smape"],
                "mae_improvement_vs_baseline_pct": imp_pct,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    best_idx = metrics_df.sort_values(["mae", "rmse"], ascending=True).index[0]
    metrics_df["is_best_model"] = metrics_df.index == best_idx

    # Tạo bảng timeline đầy đủ (train + test_backtest có đủ 4 model + future_forecast) cho Power BI
    dl_lookup = dl_df.set_index("date_str")[["pred_lstm", "pred_gru"]]
    full_timeline = sarimax_all.rename(
        columns={
            "naive_forecast_gmv": "pred_baseline",
            "sarimax_forecast_gmv": "pred_sarimax",
        }
    ).copy()
    full_timeline["pred_lstm"] = full_timeline["date_str"].map(dl_lookup["pred_lstm"])
    full_timeline["pred_gru"] = full_timeline["date_str"].map(dl_lookup["pred_gru"])
    full_timeline["date"] = pd.to_datetime(full_timeline["date_str"]).dt.date
    full_timeline = full_timeline[
        [
            "date",
            "daily_orders",
            "actual_gmv",
            "pred_baseline",
            "pred_sarimax",
            "pred_lstm",
            "pred_gru",
            "ci_lower_95",
            "ci_upper_95",
            "record_type",
        ]
    ]

    comp_parquet, metrics_parquet = save_benchmark_to_duckdb_and_parquet(
        full_timeline, metrics_df
    )

    fig_path = FIGURES_DIR / "ch7_forecast_benchmark_comparison.png"
    report_path = REPORTS_DIR / "ch7_forecast_benchmark.md"

    plot_benchmark_comparison(merged_test, metrics_df, fig_path)
    write_benchmark_report(metrics_df, report_path)

    elapsed = time.perf_counter() - start_time
    print("\n=== BẢNG ĐỐI CHIẾU CÔNG BẰNG 4 MÔ HÌNH DỰ BÁO (60 NGÀY TEST) ===")
    header = (
        f"{'Mô hình':<26} | {'MAE (BRL)':>10} | {'RMSE (BRL)':>11} | "
        f"{'MAPE (%)':>9} | {'SMAPE (%)':>10} | {'vs Base':>8} | {'Vô địch'}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for _, r in metrics_df.sort_values(["mae", "rmse"]).iterrows():
        win = "YES" if r["is_best_model"] else ""
        print(
            f"{r['model_name']:<26} | {r['mae']:>10,.2f} | {r['rmse']:>11,.2f} | "
            f"{r['mape']:>8.2f}% | {r['smape']:>9.2f}% | {r['mae_improvement_vs_baseline_pct']:>+7.1f}% | {win}"
        )
    print("-" * len(header))
    print(f"- Parquet So sánh : {comp_parquet}")
    print(f"- Parquet Metrics : {metrics_parquet}")
    print(f"- Biểu đồ Đối chiếu: {fig_path}")
    print(f"- Báo cáo Markdown: {report_path}")
    print(f"Tổng thời gian    : {elapsed:.2f} giây")


if __name__ == "__main__":
    run_forecast_comparison()
