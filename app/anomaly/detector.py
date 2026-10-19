"""Isolation Forest wrapper that turns a raw model score into a 0..1 risk signal."""
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class AmountAnomalyDetector:
    def __init__(self, n_estimators: int = 200, contamination: float = 0.05, seed: int = 42):
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=seed)
        self.train_scores: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "AmountAnomalyDetector":
        self.model.fit(X)
        self.train_scores = np.sort(-self.model.score_samples(X))   # higher = more anomalous
        return self

    def signal(self, x: np.ndarray) -> float:
        """Percentile of this invoice's anomaly score among training scores (0..1).
        0.99 means 'more unusual than 99% of invoices seen in training'."""
        s = -self.model.score_samples(x.reshape(1, -1))[0]
        return float(np.searchsorted(self.train_scores, s) / len(self.train_scores))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AmountAnomalyDetector":
        return joblib.load(path)
