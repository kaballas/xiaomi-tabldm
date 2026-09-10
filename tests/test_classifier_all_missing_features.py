from __future__ import annotations

import numpy as np
import pandas as pd

from tabldm._sklearn.classifier import _fill_all_missing_features


def test_fill_all_missing_features_handles_pandas_extension_dtypes():
    frame = pd.DataFrame(
        {
            "observed": [1.0, 2.0],
            "text": pd.Series([pd.NA, pd.NA], dtype="string"),
            "numeric": [np.nan, np.nan],
            "boolean": pd.Series([pd.NA, pd.NA], dtype="boolean"),
        }
    )
    feature_mask = frame.isna().all(axis=0).to_numpy()

    filled = _fill_all_missing_features(frame, feature_mask)

    assert filled["observed"].tolist() == [1.0, 2.0]
    assert filled["text"].tolist() == ["__tabldm_all_missing__"] * 2
    assert filled["numeric"].tolist() == [0.0, 0.0]
    assert filled["boolean"].tolist() == [False, False]
    assert frame["text"].isna().all()
    assert frame["numeric"].isna().all()
    assert frame["boolean"].isna().all()
