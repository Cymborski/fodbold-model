from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression

class IsoCalibrator:
    def __init__(self):
        self.cals = [IsotonicRegression(out_of_bounds='clip') for _ in range(3)]
        self.fitted = False

    def fit(self, y_true: np.ndarray, proba: np.ndarray):
        for k in range(3):
            t = (y_true == k).astype(float)
            self.cals[k].fit(proba[:,k], t)
        self.fitted = True
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        assert self.fitted
        out = np.zeros_like(proba)
        for k in range(3):
            out[:,k] = self.cals[k].predict(proba[:,k])
        s = out.sum(axis=1, keepdims=True) + 1e-12
        return out/s
