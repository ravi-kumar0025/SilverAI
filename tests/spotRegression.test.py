import numpy as np
import pandas as pd

from src.features.market_features import SPOT_FEATURE_COLUMNS
from src.models.return_regression import predict_regression


class _Model:
    def __init__(self, value): self.value = value
    def predict(self, frame): return np.full(len(frame), self.value)


def test_regression_price_conversion_and_interval_ordering():
    frame = pd.DataFrame({column: [0.0] for column in SPOT_FEATURE_COLUMNS} | {"silver_close": [50.0]})
    artifacts = {"mean": {"model": _Model(0.01)}, "q10": {"model": _Model(-0.01)}, "q50": {"model": _Model(0.01)}, "q90": {"model": _Model(0.02)}}
    result = predict_regression(frame, artifacts).iloc[0]
    assert result.predicted_price == 50.0 * np.exp(0.01)
    assert result.price_low <= result.predicted_price <= result.price_high
