"""Cài đặt mô hình Logistic Regression từ đầu bằng thuần NumPy + Gradient Descent."""

from __future__ import annotations

import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class NumpyLogisticRegression:
    """Mô hình Phân lớp Nhị phân Logistic Regression thuần NumPy (Weighted BCE + L2)."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1500,
        l2_reg: float = 0.01,
        class_weight: str | dict[int, float] | None = None,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.l2_reg = float(l2_reg)
        self.class_weight = class_weight
        self.tol = float(tol)
        self.verbose = bool(verbose)

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Hàm kích hoạt Sigmoid có kẹp ngưỡng chống tràn số mũ (overflow)."""
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Tính trọng số mẫu theo tùy chọn `class_weight` (hỗ trợ `'balanced'`)."""
        m = y.shape[0]
        if self.class_weight is None:
            return np.ones(m, dtype=np.float64)

        if self.class_weight == "balanced":
            n_neg = float(np.sum(y == 0))
            n_pos = float(np.sum(y == 1))
            if n_neg == 0 or n_pos == 0:
                return np.ones(m, dtype=np.float64)
            w_neg = m / (2.0 * n_neg)
            w_pos = m / (2.0 * n_pos)
            return np.where(y == 1, w_pos, w_neg).astype(np.float64)

        if isinstance(self.class_weight, dict):
            w0 = float(self.class_weight.get(0, 1.0))
            w1 = float(self.class_weight.get(1, 1.0))
            return np.where(y == 1, w1, w0).astype(np.float64)

        raise ValueError(f"Giá trị class_weight không hợp lệ: {self.class_weight}")

    def _compute_loss(
        self,
        y: np.ndarray,
        p: np.ndarray,
        sample_weights: np.ndarray,
        w: np.ndarray,
    ) -> float:
        """Hàm mất mát Binary Cross-Entropy có trọng số mẫu + L2 Regularization trên `w`."""
        eps = 1e-15
        m = float(y.shape[0])
        sw_sum = float(np.sum(sample_weights))
        bce = -np.sum(
            sample_weights * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        ) / sw_sum
        l2_penalty = (self.l2_reg / (2.0 * m)) * float(np.dot(w, w))
        return float(bce + l2_penalty)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NumpyLogisticRegression:
        """Huấn luyện mô hình bằng Batch Gradient Descent."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        if X_arr.ndim != 2:
            raise ValueError("Ma trận đặc trưng X phải có số chiều bằng 2 (m, n).")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("Số dòng của X và y không khớp nhau.")

        m, n_features = X_arr.shape
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0

        sample_weights = self._compute_sample_weights(y_arr)
        sw_sum = float(np.sum(sample_weights))

        self.loss_history_ = []
        prev_loss = np.inf

        for iteration in range(1, self.n_iterations + 1):
            linear_pred = X_arr @ w + b
            p = self._sigmoid(linear_pred)

            curr_loss = self._compute_loss(y_arr, p, sample_weights, w)
            self.loss_history_.append(curr_loss)

            weighted_error = sample_weights * (p - y_arr)
            dw = (X_arr.T @ weighted_error) / sw_sum + (self.l2_reg / float(m)) * w
            db = float(np.sum(weighted_error) / sw_sum)

            w -= self.learning_rate * dw
            b -= self.learning_rate * db

            if self.verbose and (iteration == 1 or iteration % 250 == 0):
                print(f"[Iter {iteration:4d}] Loss = {curr_loss:.6f}")

            if abs(prev_loss - curr_loss) < self.tol:
                self.n_iter_ = iteration
                break
            prev_loss = curr_loss
        else:
            self.n_iter_ = self.n_iterations

        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất hai lớp `[P(y=0), P(y=1)]` với kích thước `(m, 2)`."""
        if self.coef_ is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi `fit(X, y)` trước.")
        X_arr = np.asarray(X, dtype=np.float64)
        p_pos = self._sigmoid(X_arr @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Dự đoán nhãn nhị phân `0/1` theo ngưỡng xác suất `threshold`."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(np.int64)

    def score_roc_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Tính ROC-AUC thuần NumPy dựa trên thống kê hạng Mann-Whitney (xử lý đồng hạng)."""
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        scores = self.predict_proba(X)[:, 1]

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Cần có đủ cả hai lớp 0 và 1 để tính ROC-AUC.")

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        n = len(scores)

        # Gán hạng trung bình (1-based) cho các điểm có xác suất bằng nhau
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            ranks[order[i : j + 1]] = avg_rank
            i = j + 1

        sum_pos_ranks = float(np.sum(ranks[y_arr == 1]))
        auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)


def _run_self_test() -> None:
    """Kiểm chứng độc lập class `NumpyLogisticRegression` trên dữ liệu giả lập mất cân bằng."""
    print("=== KIỂM CHỨNG ĐỘC LẬP: NumpyLogisticRegression ===")
    rng = np.random.default_rng(42)
    n_samples, n_features = 2_000, 6
    X_syn = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    true_w = np.array([1.8, -1.4, 1.0, -0.7, 0.0, 0.5])
    logits = X_syn @ true_w - 2.1 + rng.normal(0.0, 0.3, size=n_samples)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_syn = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(int)

    model = NumpyLogisticRegression(
        learning_rate=0.15,
        n_iterations=1500,
        l2_reg=0.01,
        class_weight="balanced",
        tol=1e-7,
    )
    model.fit(X_syn, y_syn)
    auc = model.score_roc_auc(X_syn, y_syn)
    proba = model.predict_proba(X_syn)

    assert proba.shape == (n_samples, 2), "Kích thước predict_proba sai!"
    assert np.allclose(proba.sum(axis=1), 1.0), "Tổng xác suất 2 lớp phải bằng 1!"
    assert model.loss_history_[-1] < model.loss_history_[0], "Loss không giảm sau huấn luyện!"
    assert auc > 0.88, f"ROC-AUC trên dữ liệu giả lập quá thấp: {auc:.4f}"

    print(f"- Tỷ lệ lớp dương giả lập : {y_syn.mean() * 100:.2f}%")
    print(f"- Số vòng lặp hội tụ       : {model.n_iter_}")
    print(f"- Loss đầu -> Loss cuối    : {model.loss_history_[0]:.4f} -> {model.loss_history_[-1]:.4f}")
    print(f"- ROC-AUC thuần NumPy      : {auc:.4f}")
    print("[PASS] NumpyLogisticRegression hoạt động chính xác!")


if __name__ == "__main__":
    _run_self_test()
