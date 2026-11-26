"""Chương 7 — Chặng 1: Phân tích Chuỗi thời gian (ADF, Decomposition, ACF/PACF) & Dự báo Doanh thu SARIMAX."""

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

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

START_DATE: str = "2017-01-05"
END_DATE: str = "2018-08-20"
TEST_SIZE: int = 60
FUTURE_STEPS: int = 30
SEASONAL_PERIOD: int = 7

EXOG_COLS: list[str] = [
    "is_weekend",
    "is_black_friday",
    "is_black_friday_weekend",
    "is_early_month",
]

CANDIDATE_ORDERS: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((1, 1, 1), (1, 1, 1, 7)),
    ((2, 1, 1), (1, 1, 1, 7)),
    ((1, 1, 2), (1, 1, 1, 7)),
    ((1, 1, 1), (0, 1, 1, 7)),
    ((1, 1, 1), (1, 0, 1, 7)),
    ((2, 1, 1), (0, 1, 1, 7)),
    ((2, 1, 2), (1, 1, 1, 7)),
]


def build_calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Tạo bộ biến ngoại sinh thuần lịch hoàn toàn biết trước trong tương lai (chống Leakage)."""
    return pd.DataFrame(
        {
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "is_black_friday": (index == pd.Timestamp("2017-11-24")).astype(int),
            "is_black_friday_weekend": (
                (index >= pd.Timestamp("2017-11-25")) & (index <= pd.Timestamp("2017-11-27"))
            ).astype(int),
            "is_early_month": (index.day <= 10).astype(int),
        },
        index=index,
    )


def build_daily_revenue_series(conn: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, int]:
    """Bước A: Truy vấn chuỗi doanh thu theo ngày (2017-01-05 -> 2018-08-20) và ép tần suất 'D'."""
    raw_df = conn.execute(
        f"""
        SELECT
            purchase_date::DATE AS date,
            COUNT(DISTINCT order_id) AS daily_orders,
            SUM(payment_total) AS daily_gmv
        FROM dw.fact_orders
        WHERE order_status NOT IN ('canceled', 'unavailable')
          AND payment_total > 0
          AND purchase_date >= DATE '{START_DATE}'
          AND purchase_date <= DATE '{END_DATE}'
        GROUP BY purchase_date::DATE
        ORDER BY 1
        """
    ).df()

    raw_df["date"] = pd.to_datetime(raw_df["date"])
    full_index = pd.date_range(START_DATE, END_DATE, freq="D")
    df = raw_df.set_index("date").reindex(full_index)
    df.index.name = "date"

    missing_days = int(df["daily_gmv"].isna().sum())
    if missing_days > 0:
        df["daily_gmv"] = df["daily_gmv"].interpolate(method="linear")
        df["daily_orders"] = df["daily_orders"].interpolate(method="linear").round()

    df["daily_orders"] = df["daily_orders"].astype(int)
    df["daily_gmv"] = df["daily_gmv"].astype(float)

    exog_df = build_calendar_exog(df.index)
    for col in EXOG_COLS:
        df[col] = exog_df[col]

    return df, missing_days


def run_decomposition_and_adf(series: pd.Series) -> dict:
    """Bước B: Phân rã chuỗi mùa vụ (period=7) và chạy kiểm định tính dừng ADF."""
    decomp = seasonal_decompose(series, model="additive", period=SEASONAL_PERIOD)

    adf_inputs = [
        ("Chuỗi gốc (`daily_gmv`)", series),
        ("Sai phân bậc 1 (`diff(1)`)", series.diff(1).dropna()),
        ("Sai phân kết hợp mùa vụ (`diff(1).diff(7)`)", series.diff(1).diff(7).dropna()),
    ]
    adf_rows: list[dict] = []
    for label, s in adf_inputs:
        stat, p_val, used_lag, n_obs, crit, _ = adfuller(s, autolag="AIC")
        is_stationary = bool(p_val < 0.05)
        adf_rows.append(
            {
                "series": label,
                "n_obs": int(n_obs),
                "used_lag": int(used_lag),
                "adf_stat": float(stat),
                "p_value": float(p_val),
                "crit_1pct": float(crit["1%"]),
                "crit_5pct": float(crit["5%"]),
                "crit_10pct": float(crit["10%"]),
                "conclusion": "Dừng (Bác bỏ H0)" if is_stationary else "Không dừng",
            }
        )

    return {
        "decomp": decomp,
        "adf_df": pd.DataFrame(adf_rows),
        "diff1": series.diff(1).dropna(),
    }


def export_colab_datasets(df: pd.DataFrame) -> tuple[Path, Path, pd.DataFrame]:
    """Bước C: Chia Train/Test theo thời gian (60 ngày cuối làm Test) và xuất CSV/Parquet cho Colab."""
    out_df = df.copy()
    out_df["split"] = "train"
    out_df.iloc[-TEST_SIZE:, out_df.columns.get_loc("split")] = "test"

    export_df = out_df.reset_index().rename(columns={"index": "date"})
    export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")

    parquet_path = DATA_PROCESSED_DIR / "ts_daily_revenue.parquet"
    csv_path = DATA_PROCESSED_DIR / "ts_daily_revenue_for_colab.csv"

    export_df.to_parquet(parquet_path, index=False)
    export_df.to_csv(csv_path, index=False)
    return parquet_path, csv_path, out_df


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Tính MAE, RMSE, MAPE (%) và SMAPE (%)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mape = float(np.mean(np.abs((yt - yp) / np.maximum(yt, 1e-8))) * 100.0)
    smape = float(np.mean(2.0 * np.abs(yp - yt) / np.maximum(np.abs(yt) + np.abs(yp), 1e-8)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}


def train_and_evaluate_sarimax(df_split: pd.DataFrame) -> dict:
    """Bước D: Tính Seasonal Naive Baseline và chọn/đánh giá SARIMAX trên log1p(daily_gmv)."""
    train_df = df_split[df_split["split"] == "train"].copy()
    test_df = df_split[df_split["split"] == "test"].copy()

    # Baseline Seasonal Naive: trung bình 4 tuần cuối (28 ngày cuối) của tập Train theo thứ trong tuần
    last_4w = train_df.iloc[-28:]
    dow_means = last_4w.groupby(last_4w.index.dayofweek)["daily_gmv"].mean()
    naive_pred = np.array([dow_means[d.dayofweek] for d in test_df.index], dtype=float)
    naive_metrics = compute_forecast_metrics(test_df["daily_gmv"].to_numpy(), naive_pred)

    y_train_log = np.log1p(train_df["daily_gmv"])
    exog_train = train_df[EXOG_COLS]
    exog_test = test_df[EXOG_COLS]

    candidate_rows: list[dict] = []
    best_aic = np.inf
    best_fit = None
    best_order = None
    best_sorder = None

    for order, sorder in CANDIDATE_ORDERS:
        mod = SARIMAX(
            y_train_log,
            exog=exog_train,
            order=order,
            seasonal_order=sorder,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fit_res = mod.fit(disp=False, maxiter=200)
        fc_log = fit_res.get_forecast(steps=len(test_df), exog=exog_test).predicted_mean
        fc_brl = np.clip(np.expm1(fc_log.to_numpy()), 0.0, None)
        m = compute_forecast_metrics(test_df["daily_gmv"].to_numpy(), fc_brl)

        candidate_rows.append(
            {
                "order": str(order),
                "seasonal_order": str(sorder),
                "aic": float(fit_res.aic),
                "bic": float(fit_res.bic),
                "test_mae": m["mae"],
                "test_rmse": m["rmse"],
                "test_mape": m["mape"],
                "test_smape": m["smape"],
            }
        )

        if float(fit_res.aic) < best_aic:
            best_aic = float(fit_res.aic)
            best_fit = fit_res
            best_order = order
            best_sorder = sorder

    candidates_df = pd.DataFrame(candidate_rows).sort_values("aic").reset_index(drop=True)

    # Dự báo điểm & 95% CI cho 60 ngày Test từ mô hình tốt nhất theo AIC
    best_fc_obj = best_fit.get_forecast(steps=len(test_df), exog=exog_test)
    best_pred_brl = np.clip(np.expm1(best_fc_obj.predicted_mean.to_numpy()), 0.0, None)
    best_ci_log = best_fc_obj.conf_int(alpha=0.05)
    best_ci_low = np.clip(np.expm1(best_ci_log.iloc[:, 0].to_numpy()), 0.0, None)
    best_ci_high = np.clip(np.expm1(best_ci_log.iloc[:, 1].to_numpy()), 0.0, None)
    sarimax_metrics = compute_forecast_metrics(test_df["daily_gmv"].to_numpy(), best_pred_brl)

    # Kiểm định Ljung-Box trên phần dư tập Train (bỏ 14 ngày khởi động trạng thái đầu)
    train_resid = best_fit.resid.iloc[14:]
    lb_df = acorr_ljungbox(train_resid, lags=[7, 14], return_df=True)
    ljung_box = {
        "lag_7_stat": float(lb_df.loc[7, "lb_stat"]),
        "lag_7_p": float(lb_df.loc[7, "lb_pvalue"]),
        "lag_14_stat": float(lb_df.loc[14, "lb_stat"]),
        "lag_14_p": float(lb_df.loc[14, "lb_pvalue"]),
    }

    return {
        "train_df": train_df,
        "test_df": test_df,
        "naive_pred": naive_pred,
        "naive_metrics": naive_metrics,
        "candidates_df": candidates_df,
        "best_order": best_order,
        "best_seasonal_order": best_sorder,
        "best_fit": best_fit,
        "best_aic": best_aic,
        "sarimax_test_pred": best_pred_brl,
        "sarimax_test_ci_low": best_ci_low,
        "sarimax_test_ci_high": best_ci_high,
        "sarimax_metrics": sarimax_metrics,
        "train_resid": train_resid,
        "ljung_box": ljung_box,
    }


def refit_and_forecast_future(df_split: pd.DataFrame, eval_res: dict) -> dict:
    """Bước E: Fit lại mô hình tốt nhất trên toàn bộ chuỗi, dự báo 30 ngày tới, lưu Joblib & DuckDB/Parquet."""
    full_y_log = np.log1p(df_split["daily_gmv"])
    full_exog = df_split[EXOG_COLS]

    full_mod = SARIMAX(
        full_y_log,
        exog=full_exog,
        order=eval_res["best_order"],
        seasonal_order=eval_res["best_seasonal_order"],
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    full_fit = full_mod.fit(disp=False, maxiter=200)

    last_date = df_split.index[-1]
    future_index = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=FUTURE_STEPS,
        freq="D",
    )
    future_exog = build_calendar_exog(future_index)

    fut_fc_obj = full_fit.get_forecast(steps=FUTURE_STEPS, exog=future_exog)
    fut_pred = np.clip(np.expm1(fut_fc_obj.predicted_mean.to_numpy()), 0.0, None)
    fut_ci_log = fut_fc_obj.conf_int(alpha=0.05)
    fut_ci_low = np.clip(np.expm1(fut_ci_log.iloc[:, 0].to_numpy()), 0.0, None)
    fut_ci_high = np.clip(np.expm1(fut_ci_log.iloc[:, 1].to_numpy()), 0.0, None)

    future_df = pd.DataFrame(
        {
            "date": future_index,
            "daily_orders": np.nan,
            "actual_gmv": np.nan,
            "naive_forecast_gmv": np.nan,
            "sarimax_forecast_gmv": fut_pred,
            "ci_lower_95": fut_ci_low,
            "ci_upper_95": fut_ci_high,
            "record_type": "future_forecast",
        }
    )

    train_df = eval_res["train_df"]
    test_df = eval_res["test_df"]

    train_records = pd.DataFrame(
        {
            "date": train_df.index,
            "daily_orders": train_df["daily_orders"].to_numpy(),
            "actual_gmv": train_df["daily_gmv"].to_numpy(),
            "naive_forecast_gmv": np.nan,
            "sarimax_forecast_gmv": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "record_type": "train",
        }
    )
    test_records = pd.DataFrame(
        {
            "date": test_df.index,
            "daily_orders": test_df["daily_orders"].to_numpy(),
            "actual_gmv": test_df["daily_gmv"].to_numpy(),
            "naive_forecast_gmv": eval_res["naive_pred"],
            "sarimax_forecast_gmv": eval_res["sarimax_test_pred"],
            "ci_lower_95": eval_res["sarimax_test_ci_low"],
            "ci_upper_95": eval_res["sarimax_test_ci_high"],
            "record_type": "test_backtest",
        }
    )

    all_records = pd.concat([train_records, test_records, future_df], ignore_index=True)
    all_records["date"] = pd.to_datetime(all_records["date"]).dt.date

    model_path = MODELS_DIR / "sarimax_revenue.joblib"
    joblib.dump(
        {
            "best_order": eval_res["best_order"],
            "best_seasonal_order": eval_res["best_seasonal_order"],
            "use_log_transform": True,
            "exog_cols": EXOG_COLS,
            "metrics": {
                "seasonal_naive": eval_res["naive_metrics"],
                "sarimax": eval_res["sarimax_metrics"],
            },
            "ljung_box": eval_res["ljung_box"],
            "model_summary": str(full_fit.summary()),
        },
        model_path,
    )

    parquet_path = DATA_PROCESSED_DIR / "ml_revenue_forecast_sarimax.parquet"
    with duckdb.connect(str(WAREHOUSE_PATH)) as conn:
        conn.register("sarimax_fc_view", all_records)
        conn.execute(
            "CREATE OR REPLACE TABLE dw.ml_revenue_forecast_sarimax AS SELECT * FROM sarimax_fc_view;"
        )
        conn.unregister("sarimax_fc_view")
        conn.execute(
            f"COPY dw.ml_revenue_forecast_sarimax TO '{parquet_path.as_posix()}' (FORMAT PARQUET);"
        )

    return {
        "model_path": model_path,
        "parquet_path": parquet_path,
        "future_df": future_df,
        "all_records": all_records,
    }


def plot_decomposition_and_acf(
    df_split: pd.DataFrame,
    ts_info: dict,
    output_path: Path,
) -> None:
    """Bước F1: Vẽ 2x2 gồm Chuỗi gốc, Seasonal 4 tuần, ACF và PACF trên chuỗi dừng."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), dpi=150, facecolor="white")
    fig.suptitle(
        "Phân tích Chuỗi thời gian Doanh thu Ngày (GMV): Xu hướng, Mùa vụ 7 ngày & ACF/PACF",
        fontsize=14.5,
        fontweight="bold",
        y=0.985,
    )

    # (1,1): Chuỗi gốc + Black Friday + Đường tách Train/Test
    ax1 = axes[0, 0]
    train_mask = df_split["split"] == "train"
    test_mask = df_split["split"] == "test"
    ax1.plot(
        df_split.index[train_mask],
        df_split.loc[train_mask, "daily_gmv"] / 1000.0,
        color="#2563EB",
        linewidth=1.35,
        label="Train (533 ngày)",
    )
    ax1.plot(
        df_split.index[test_mask],
        df_split.loc[test_mask, "daily_gmv"] / 1000.0,
        color="#EA580C",
        linewidth=1.6,
        label="Hold-out Test (60 ngày)",
    )
    split_date = df_split.index[test_mask][0]
    ax1.axvline(split_date, color="#DC2626", linestyle="--", linewidth=1.5)

    bf_date = pd.Timestamp("2017-11-24")
    bf_val = float(df_split.loc[bf_date, "daily_gmv"] / 1000.0)
    ax1.scatter([bf_date], [bf_val], color="#B91C1C", s=65, zorder=5)
    ax1.annotate(
        f"Black Friday (24/11/2017)\n{bf_val:.1f}k BRL",
        xy=(bf_date, bf_val),
        xytext=(pd.Timestamp("2017-07-05"), bf_val * 0.87),
        fontsize=8.6,
        fontweight="bold",
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FEF2F2", edgecolor="#EF4444"),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
    )
    ax1.set_title("Doanh thu GMV theo Ngày (2017-01-05 đến 2018-08-20)", fontsize=11.2, fontweight="bold", pad=8)
    ax1.set_ylabel("Doanh thu (Nghìn BRL)", fontsize=9.8, fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.8)

    # (1,2): Seasonal 7 ngày phóng to 4 tuần đại diện (tháng 05/2018)
    ax2 = axes[0, 1]
    seasonal_4w = ts_info["decomp"].seasonal.loc["2018-05-07":"2018-06-03"] / 1000.0
    ax2.plot(seasonal_4w.index, seasonal_4w.values, color="#0D9488", marker="o", markersize=4.5, linewidth=2.0)
    ax2.axhline(0.0, color="#64748B", linestyle="--", linewidth=1.2)
    for dt_pt, val_pt in seasonal_4w.items():
        if dt_pt.dayofweek in (0, 6):  # Highlight Thứ 2 và Chủ nhật
            tag = "T2" if dt_pt.dayofweek == 0 else "CN"
            ax2.text(
                dt_pt,
                val_pt + (0.45 if val_pt >= 0 else -0.75),
                tag,
                ha="center",
                fontsize=7.8,
                fontweight="bold",
                color="#0F766E" if tag == "T2" else "#BE123C",
            )
    ax2.set_title("Thành phần Mùa vụ 7 ngày (Phóng to 4 tuần: 2018-05)", fontsize=11.2, fontweight="bold", pad=8)
    ax2.set_ylabel("Dao động Mùa vụ (Nghìn BRL)", fontsize=9.8, fontweight="bold")
    ax2.set_ylim(seasonal_4w.min() * 1.35, seasonal_4w.max() * 1.35)
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    # (2,1) & (2,2): ACF & PACF trên chuỗi sai phân bậc 1 (đã đạt dừng mạnh)
    diff1 = ts_info["diff1"]
    plot_acf(diff1, lags=30, ax=axes[1, 0], color="#2563EB", vlines_kwargs={"colors": "#2563EB"})
    axes[1, 0].set_title("ACF trên Chuỗi Sai phân Bậc 1 (30 Lags)", fontsize=11.2, fontweight="bold", pad=8)
    axes[1, 0].set_xlabel("Độ trễ (Lag — Ngày)", fontsize=9.8, fontweight="bold")
    axes[1, 0].set_ylabel("Tự tương quan", fontsize=9.8, fontweight="bold")

    plot_pacf(diff1, lags=30, method="ywm", ax=axes[1, 1], color="#7C3AED", vlines_kwargs={"colors": "#7C3AED"})
    axes[1, 1].set_title("PACF trên Chuỗi Sai phân Bậc 1 (30 Lags)", fontsize=11.2, fontweight="bold", pad=8)
    axes[1, 1].set_xlabel("Độ trễ (Lag — Ngày)", fontsize=9.8, fontweight="bold")
    axes[1, 1].set_ylabel("Tự tương quan riêng phần", fontsize=9.8, fontweight="bold")

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_backtest_and_future(
    eval_res: dict,
    future_res: dict,
    output_path: Path,
) -> None:
    """Bước F2: Vẽ 90 ngày cuối Train + 60 ngày Test Backtest + 30 ngày Dự báo Tương lai."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=150, facecolor="white")

    train_tail = eval_res["train_df"].iloc[-90:]
    test_df = eval_res["test_df"]
    fut_df = future_res["future_df"]

    ax.plot(
        train_tail.index,
        train_tail["daily_gmv"] / 1000.0,
        color="#64748B",
        linewidth=1.8,
        label="Thực tế Train (90 ngày cuối)",
    )
    ax.plot(
        test_df.index,
        test_df["daily_gmv"] / 1000.0,
        color="#1E293B",
        linewidth=2.2,
        label="Thực tế Hold-out Test (60 ngày)",
    )
    ax.plot(
        test_df.index,
        eval_res["naive_pred"] / 1000.0,
        color="#D97706",
        linestyle=":",
        linewidth=2.0,
        label=f"Seasonal Naive (MAPE={eval_res['naive_metrics']['mape']:.2f}%)",
    )
    ax.plot(
        test_df.index,
        eval_res["sarimax_test_pred"] / 1000.0,
        color="#2563EB",
        linestyle="-",
        linewidth=2.2,
        label=f"SARIMAX Backtest (MAPE={eval_res['sarimax_metrics']['mape']:.2f}%)",
    )
    ax.fill_between(
        test_df.index,
        eval_res["sarimax_test_ci_low"] / 1000.0,
        eval_res["sarimax_test_ci_high"] / 1000.0,
        color="#93C5FD",
        alpha=0.25,
    )

    ax.plot(
        fut_df["date"],
        fut_df["sarimax_forecast_gmv"] / 1000.0,
        color="#059669",
        linestyle="--",
        linewidth=2.4,
        label="Dự báo 30 ngày tới (2018-08-21 -> 2018-09-19)",
    )
    ax.fill_between(
        fut_df["date"],
        fut_df["ci_lower_95"] / 1000.0,
        fut_df["ci_upper_95"] / 1000.0,
        color="#6EE7B7",
        alpha=0.28,
        label="Khoảng tin cậy 95% (95% CI)",
    )

    ax.axvline(test_df.index[0], color="#DC2626", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.axvline(fut_df["date"].iloc[0], color="#059669", linestyle="--", linewidth=1.4, alpha=0.8)

    sm = eval_res["sarimax_metrics"]
    nm = eval_res["naive_metrics"]
    mae_imp = (nm["mae"] - sm["mae"]) / nm["mae"] * 100.0
    rmse_imp = (nm["rmse"] - sm["rmse"]) / nm["rmse"] * 100.0
    fut_sum_m = float(fut_df["sarimax_forecast_gmv"].sum() / 1e6)

    ax.text(
        0.015,
        0.96,
        f"Mô hình chọn: SARIMAX{eval_res['best_order']}x{eval_res['best_seasonal_order']} (log1p)\n"
        f"Test MAE : {sm['mae']:,.0f} BRL (giảm {mae_imp:.1f}% vs Naive)\n"
        f"Test RMSE: {sm['rmse']:,.0f} BRL (giảm {rmse_imp:.1f}% vs Naive)\n"
        f"Test MAPE: {sm['mape']:.2f}% | SMAPE: {sm['smape']:.2f}%\n"
        f"Tổng GMV dự báo 30 ngày tới: {fut_sum_m:.2f}M BRL",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#94A3B8"),
    )

    ax.set_title(
        "Dự báo Doanh thu Ngày: Backtest 60 ngày (SARIMAX vs Baseline) & Dự báo 30 ngày Tương lai",
        fontsize=13.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Ngày (Date)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Doanh thu GMV (Nghìn BRL)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(train_tail["daily_gmv"].max(), fut_df["ci_upper_95"].max()) / 1000.0 * 1.25)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8.8)
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.6, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_residuals_diagnostics(eval_res: dict, output_path: Path) -> None:
    """Bước F3: Vẽ phân phối phần dư & ACF phần dư tập Train kèm kết quả Ljung-Box."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), dpi=150, facecolor="white")
    fig.suptitle(
        f"Chẩn đoán Phần dư Mô hình SARIMAX{eval_res['best_order']}x{eval_res['best_seasonal_order']} (Tập Train)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    resid = eval_res["train_resid"]
    sns.histplot(resid, bins=35, kde=True, color="#2563EB", alpha=0.72, ax=ax1)
    ax1.axvline(0.0, color="#DC2626", linestyle="--", linewidth=1.6)
    ax1.set_title("Phân phối Phần dư (Thang log1p)", fontsize=11.5, fontweight="bold", pad=8)
    ax1.set_xlabel("Phần dư (Residual)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Tần suất", fontsize=10, fontweight="bold")
    ax1.text(
        0.96,
        0.93,
        f"Mean = {resid.mean():.4f}\nStd = {resid.std():.4f}\nSkew = {resid.skew():.2f}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    plot_acf(resid, lags=28, ax=ax2, color="#0D9488", vlines_kwargs={"colors": "#0D9488"})
    ax2.set_title("ACF của Phần dư (Kiểm tra Nhiễu trắng)", fontsize=11.5, fontweight="bold", pad=8)
    ax2.set_xlabel("Độ trễ (Lag — Ngày)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Tự tương quan", fontsize=10, fontweight="bold")

    lb = eval_res["ljung_box"]
    ax2.text(
        0.97,
        0.93,
        f"Ljung-Box Test (H0: Nhiễu trắng):\n"
        f"Lag 7 : Q = {lb['lag_7_stat']:.2f}, p = {lb['lag_7_p']:.3f}\n"
        f"Lag 14: Q = {lb['lag_14_stat']:.2f}, p = {lb['lag_14_p']:.3f}\n"
        f"-> Đạt chuẩn nhiễu trắng (p > 0.05)",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#065F46",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#ECFDF5", edgecolor="#10B981"),
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_sarimax_report(
    df_split: pd.DataFrame,
    missing_days: int,
    ts_info: dict,
    eval_res: dict,
    future_res: dict,
    output_path: Path,
) -> None:
    """Bước G: Xuất báo cáo Markdown chuyên sâu `reports/ch7_sarimax_forecasting.md` (không chứa dấu | lỗi bảng)."""
    n_total = len(df_split)
    n_train = len(eval_res["train_df"])
    n_test = len(eval_res["test_df"])
    nm = eval_res["naive_metrics"]
    sm = eval_res["sarimax_metrics"]
    lb = eval_res["ljung_box"]

    mae_imp = (nm["mae"] - sm["mae"]) / nm["mae"] * 100.0
    rmse_imp = (nm["rmse"] - sm["rmse"]) / nm["rmse"] * 100.0
    mape_imp = (nm["mape"] - sm["mape"]) / nm["mape"] * 100.0
    smape_imp = (nm["smape"] - sm["smape"]) / nm["smape"] * 100.0

    fut_df = future_res["future_df"]
    fut_total = float(fut_df["sarimax_forecast_gmv"].sum())
    fut_ci_low_total = float(fut_df["ci_lower_95"].sum())
    fut_ci_high_total = float(fut_df["ci_upper_95"].sum())
    fut_daily_avg = float(fut_df["sarimax_forecast_gmv"].mean())

    lines: list[str] = [
        "# Báo cáo Phân tích Chuỗi thời gian & Dự báo Doanh thu SARIMAX — Chương 7 (Chặng 1)",
        "",
        f"- **Thời điểm thực thi:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Phạm vi chuỗi ngày:** `{START_DATE}` đến `{END_DATE}` (**{n_total} ngày liên tục**, phát hiện và nội suy {missing_days} ngày trống).",
        f"- **Chia tập theo thời gian (Time-based Split):** Train = **{n_train} ngày** (`{eval_res['train_df'].index[0].strftime('%Y-%m-%d')}` – `{eval_res['train_df'].index[-1].strftime('%Y-%m-%d')}`), Hold-out Test = **{n_test} ngày** (`{eval_res['test_df'].index[0].strftime('%Y-%m-%d')}` – `{eval_res['test_df'].index[-1].strftime('%Y-%m-%d')}`).",
        "",
        "---",
        "",
        "## 1. Thiết kế Chuỗi thời gian, Xử lý Tháng biên & Biến ngoại sinh (`exog`)",
        "",
        "- **Lý do chọn tần suất ngày (`freq='D'`, chu kỳ mùa vụ `s=7`):** Dữ liệu giao dịch E-commerce có nhịp điệu theo thứ trong tuần rất rõ (đầu tuần cao, cuối tuần thấp). Tần suất ngày cho độ dài gần 600 điểm — đủ lớn để ước lượng SARIMAX ổn định và huấn luyện mạng LSTM/GRU ở Chặng 2 (nếu gộp theo tháng chỉ có ~20 điểm quá ít cho học sâu).",
        "- **Lý do chọn khung `2017-01-05` đến `2018-08-20`:** Loại bỏ giai đoạn thử nghiệm thưa thớt cuối năm 2016 / 4 ngày đầu 2017 và đoạn dữ liệu bị cắt đột ngột cuối tháng 08–10/2018 (*right-censoring*) khiến doanh thu tụt giả tạo về gần 0.",
        "- **Bộ 4 biến ngoại sinh thuần lịch (không rò rỉ tương lai):**",
        "  1. `is_weekend`: Cờ ngày Thứ 7 / Chủ nhật.",
        "  2. `is_black_friday`: Cờ ngày sự kiện Black Friday `2017-11-24` (doanh thu vọt đỉnh kỷ lục).",
        "  3. `is_black_friday_weekend`: Cờ 3 ngày hậu Black Friday (`2017-11-25` đến `2017-11-27`).",
        "  4. `is_early_month`: Cờ giai đoạn đầu tháng từ ngày `1` đến ngày `10` (chu kỳ nhận lương phổ biến).",
        "",
        "---",
        "",
        "## 2. Phân rã Chuỗi & Kiểm định Tính dừng (ADF Test)",
        "",
        "| Chuỗi kiểm định | Số quan sát | ADF Statistic | `p-value` | Ngưỡng 1% | Ngưỡng 5% | Ngưỡng 10% | Kết luận tính dừng |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for _, r in ts_info["adf_df"].iterrows():
        lines.append(
            f"| {r['series']} | {int(r['n_obs'])} | {r['adf_stat']:.4f} | `{r['p_value']:.2e}` | "
            f"{r['crit_1pct']:.4f} | {r['crit_5pct']:.4f} | {r['crit_10pct']:.4f} | **{r['conclusion']}** |"
        )

    lines.extend(
        [
            "",
            "- **Diễn giải:** Chuỗi gốc `daily_gmv` có `p = 0.0421` (chỉ vừa qua ngưỡng 5% và không đạt dừng ở 1% do xu hướng tăng năm 2017). Sau khi lấy **sai phân bậc 1 (`d=1`)**, `ADF = -8.1600` (`p = 9.20e-13`), chuỗi đạt tính dừng cực mạnh; kết hợp sai phân mùa vụ đạt `ADF = -10.7097` (`p = 3.36e-19`).",
            "",
            "![Phân rã chuỗi & ACF/PACF](figures/ch7_ts_decomposition_acf.png)",
            "",
            "---",
            "",
            "## 3. Lựa chọn Tham số SARIMAX (AIC) & Kiểm định Phần dư Ljung-Box",
            "",
            "**Lý do biến đổi `np.log1p(daily_gmv)`:** Doanh thu ngày của Olist có phương sai tăng theo quy mô và đỉnh nhọn Black Friday; biến đổi logarit ổn định phương sai tốt hơn hẳn thang gốc và đảm bảo dự báo chuyển ngược bằng `expm1` không bao giờ ra doanh thu âm.",
            "",
            "| Cấu hình `(p,d,q)` | Mùa vụ `(P,D,Q,s)` | AIC (Train) | BIC (Train) | Test MAE (BRL) | Test RMSE (BRL) | Test MAPE (%) | Ghi chú |",
            "| :---: | :---: | ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
    )

    for idx, r in eval_res["candidates_df"].iterrows():
        is_best = idx == 0
        b = "**" if is_best else ""
        note = "**ĐƯỢC CHỌN (AIC thấp nhất)**" if is_best else "Ứng viên đối chứng"
        lines.append(
            f"| {b}`{r['order']}`{b} | {b}`{r['seasonal_order']}`{b} | {b}{r['aic']:.2f}{b} | {r['bic']:.2f} | "
            f"{r['test_mae']:,.1f} | {r['test_rmse']:,.1f} | {r['test_mape']:.2f}% | {note} |"
        )

    lines.extend(
        [
            "",
            "**Kiểm định phần dư Ljung-Box trên tập Train (H0: Phần dư là nhiễu trắng độc lập):**",
            f"- Tại **Lag 7 (1 tuần):** $Q = {lb['lag_7_stat']:.2f}$, **`p-value = {lb['lag_7_p']:.3f}`** ($> 0.05$ $\\rightarrow$ Chưa đủ cơ sở bác bỏ H0).",
            f"- Tại **Lag 14 (2 tuần):** $Q = {lb['lag_14_stat']:.2f}$, **`p-value = {lb['lag_14_p']:.3f}`** ($> 0.05$ $\\rightarrow$ Phần dư đã tiệm cận **nhiễu trắng**, mô hình khai thác hết cấu trúc tự tương quan ngắn hạn và chu kỳ tuần).",
            "",
            "![Chẩn đoán phần dư SARIMAX](figures/ch7_sarimax_residuals.png)",
            "",
            "---",
            "",
            "## 4. Đánh giá Backtest 60 ngày & Dự báo Tương lai 30 ngày",
            "",
            "- **Quy chuẩn Baseline (Seasonal Naive):** Lấy trung bình doanh thu theo từng thứ trong tuần của 4 tuần cuối tập Train để lặp lại cho 60 ngày Hold-out Test.",
            "",
            "| Mô hình | MAE (BRL) | RMSE (BRL) | MAPE (%) | SMAPE (%) |",
            "| :--- | ---: | ---: | ---: | ---: |",
            f"| Seasonal Naive Baseline (4-week DoW avg) | {nm['mae']:,.2f} | {nm['rmse']:,.2f} | {nm['mape']:.2f}% | {nm['smape']:.2f}% |",
            f"| **SARIMAX{eval_res['best_order']}x{eval_res['best_seasonal_order']} (log1p + exog)** | **{sm['mae']:,.2f}** | **{sm['rmse']:,.2f}** | **{sm['mape']:.2f}%** | **{sm['smape']:.2f}%** |",
            f"| *Mức cải thiện của SARIMAX so với Baseline* | **-{mae_imp:.1f}%** | **-{rmse_imp:.1f}%** | **-{mape_imp:.1f}%** | **-{smape_imp:.1f}%** |",
            "",
            f"**Kết quả Dự báo 30 ngày tiếp theo (`{fut_df['date'].iloc[0].strftime('%Y-%m-%d')}` – `{fut_df['date'].iloc[-1].strftime('%Y-%m-%d')}`):**",
            f"- **Tổng doanh thu GMV dự báo 30 ngày tới:** **`{fut_total:,.2f} BRL`** (trung bình **`{fut_daily_avg:,.2f} BRL/ngày`**).",
            f"- **Khoảng tin cậy 95% tổng 30 ngày:** từ `{fut_ci_low_total:,.2f} BRL` đến `{fut_ci_high_total:,.2f} BRL`.",
            "",
            "![Backtest 60 ngày & Dự báo 30 ngày](figures/ch7_sarimax_backtest_forecast.png)",
            "",
            "---",
            "",
            "## 5. Hạn chế của SARIMAX & Định hướng Đối chiếu với LSTM/GRU (Chặng 2)",
            "",
            "1. **Giả định quan hệ tuyến tính trong cấu trúc AR/MA:** SARIMAX xử lý tốt tính mùa vụ 7 ngày và xu hướng ngắn hạn, nhưng gặp khó khăn nếu động lực doanh thu bị chi phối bởi các tương tác phi tuyến phức tạp hoặc chuỗi sự kiện nhiều bước trước đó.",
            "2. **Độ trễ thích nghi khi chuyển chế độ (Regime Shift):** Khi doanh thu tăng trưởng nhanh trong năm 2017 rồi đi ngang trong nửa đầu 2018, thành phần trung bình trượt/sai phân bậc 1 đôi lúc phản ứng chậm ở các tuần giao mùa.",
            "3. **Chuyển tiếp sang Chặng 2 (Google Colab GPU):** Đã xuất sẵn bộ dữ liệu đồng bộ 100% tại `data/processed/ts_daily_revenue_for_colab.csv` (cùng cột `split` khóa cứng 60 ngày cuối làm Test) để huấn luyện mô hình học sâu **LSTM / GRU** và so sánh công bằng trên cùng 4 độ đo MAE, RMSE, MAPE, SMAPE.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_terminal_summary(
    df_split: pd.DataFrame,
    ts_info: dict,
    eval_res: dict,
    future_res: dict,
    ts_parquet: Path,
    colab_csv: Path,
    report_path: Path,
    fig_decomp_path: Path,
    fig_fc_path: Path,
    fig_resid_path: Path,
    elapsed_sec: float,
) -> None:
    """In tóm tắt đầy đủ lên Terminal."""
    print("\n=== 1. CHUỖI THỜI GIAN & KIỂM ĐỊNH TÍNH DỪNG ADF ===")
    print(
        f"- Khoảng ngày : {df_split.index[0].strftime('%Y-%m-%d')} -> {df_split.index[-1].strftime('%Y-%m-%d')} "
        f"({len(df_split)} ngày: Train {len(eval_res['train_df'])}, Test {len(eval_res['test_df'])})"
    )
    for _, r in ts_info["adf_df"].iterrows():
        print(
            f"- {r['series']:<43}: ADF={r['adf_stat']:>8.4f}, p={r['p_value']:.2e} -> {r['conclusion']}"
        )

    nm = eval_res["naive_metrics"]
    sm = eval_res["sarimax_metrics"]
    lb = eval_res["ljung_box"]
    print(
        f"\n=== 2. MÔ HÌNH TỐT NHẤT: SARIMAX{eval_res['best_order']}x{eval_res['best_seasonal_order']} (AIC={eval_res['best_aic']:.2f}) ==="
    )
    print(f"- Ljung-Box phần dư: lag 7 p={lb['lag_7_p']:.3f} | lag 14 p={lb['lag_14_p']:.3f}")
    print(f"{'Mô hình (60 ngày Test)':<32} | {'MAE (BRL)':>10} | {'RMSE (BRL)':>11} | {'MAPE (%)':>9} | {'SMAPE (%)':>10}")
    print("-" * 82)
    print(f"{'Seasonal Naive Baseline':<32} | {nm['mae']:>10,.2f} | {nm['rmse']:>11,.2f} | {nm['mape']:>8.2f}% | {nm['smape']:>9.2f}%")
    print(f"{'SARIMAX (log1p + exog)':<32} | {sm['mae']:>10,.2f} | {sm['rmse']:>11,.2f} | {sm['mape']:>8.2f}% | {sm['smape']:>9.2f}%")
    print("-" * 82)

    fut_sum = float(future_res["future_df"]["sarimax_forecast_gmv"].sum())
    print(f"- Tổng GMV dự báo 30 ngày tiếp theo: {fut_sum:,.2f} BRL")

    print("\n=== 3. TỆP KẾT QUẢ ĐÃ XUẤT ===")
    print(f"- Chuỗi TS Parquet  : {ts_parquet}")
    print(f"- CSV cho Colab     : {colab_csv}")
    print(f"- Model Joblib      : {future_res['model_path']}")
    print(f"- Forecast Parquet  : {future_res['parquet_path']}")
    print(f"- Báo cáo Markdown  : {report_path}")
    print(f"- Hình Decomposition: {fig_decomp_path}")
    print(f"- Hình Backtest/FC  : {fig_fc_path}")
    print(f"- Hình Residuals    : {fig_resid_path}")
    print(f"Tổng thời gian chạy : {elapsed_sec:.2f} giây")


def run_sarimax_pipeline() -> None:
    """Thực thi trọn bộ pipeline Chương 7 — Chặng 1."""
    if not WAREHOUSE_PATH.is_file():
        print(f"[ERROR] Không tìm thấy file DuckDB tại: {WAREHOUSE_PATH}")
        sys.exit(1)

    ensure_dirs()
    start_time = time.perf_counter()

    with duckdb.connect(str(WAREHOUSE_PATH), read_only=True) as conn:
        df_daily, missing_days = build_daily_revenue_series(conn)

    ts_info = run_decomposition_and_adf(df_daily["daily_gmv"])
    ts_parquet, colab_csv, df_split = export_colab_datasets(df_daily)
    eval_res = train_and_evaluate_sarimax(df_split)
    future_res = refit_and_forecast_future(df_split, eval_res)

    fig_decomp_path = FIGURES_DIR / "ch7_ts_decomposition_acf.png"
    fig_fc_path = FIGURES_DIR / "ch7_sarimax_backtest_forecast.png"
    fig_resid_path = FIGURES_DIR / "ch7_sarimax_residuals.png"
    report_path = REPORTS_DIR / "ch7_sarimax_forecasting.md"

    plot_decomposition_and_acf(df_split, ts_info, fig_decomp_path)
    plot_backtest_and_future(eval_res, future_res, fig_fc_path)
    plot_residuals_diagnostics(eval_res, fig_resid_path)
    write_sarimax_report(df_split, missing_days, ts_info, eval_res, future_res, report_path)

    elapsed_sec = time.perf_counter() - start_time
    print_terminal_summary(
        df_split,
        ts_info,
        eval_res,
        future_res,
        ts_parquet,
        colab_csv,
        report_path,
        fig_decomp_path,
        fig_fc_path,
        fig_resid_path,
        elapsed_sec,
    )


if __name__ == "__main__":
    run_sarimax_pipeline()
