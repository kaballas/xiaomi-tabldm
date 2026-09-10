import importlib
from pathlib import Path
import sys
import types
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from horse_racing_preprocessing import preprocess_episode_features


def load_reference_preprocessing():
    """Load the original sklearn pipeline without importing torch-backed tabldm."""
    root = Path(__file__).resolve().parents[1]
    package_name = "_tabldm_preprocessing_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root / "tabldm")]
    sklearn_package = types.ModuleType(f"{package_name}._sklearn")
    sklearn_package.__path__ = [str(root / "tabldm" / "_sklearn")]
    sys.modules.setdefault(package_name, package)
    sys.modules.setdefault(f"{package_name}._sklearn", sklearn_package)
    return importlib.import_module(f"{package_name}._sklearn.preprocessing")


def reference_episode_features(context, query, features, y_context):
    preprocessing = load_reference_preprocessing()
    encoder = preprocessing.TransformToNumerical()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_context = encoder.fit_transform(context.loc[:, features])
        X_query = encoder.transform(query.loc[:, features])

    generator = preprocessing.EnsembleGenerator(
        classification=True,
        n_estimators=1,
        norm_methods="none",
        feat_shuffle_method="none",
        class_shuffle_method="none",
        random_state=0,
    )
    generator.fit(X_context, y_context)
    X_all, _ = next(iter(generator.transform(X_query, mode="both").values()))
    return np.asarray(X_all[0], dtype=np.float32)


def test_fast_preprocessing_matches_original_pipeline_with_mixed_features():
    context = pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "all_missing": [np.nan] * 6,
            "context_constant": [7.0] * 6,
            "category": ["b", "a", np.nan, "b", "c", "a"],
            "boolean": [True, False, True, False, True, False],
        }
    )
    query = pd.DataFrame(
        {
            "numeric": [np.nan, 20.0],
            "all_missing": [1.0, 2.0],
            "context_constant": [8.0, 9.0],
            "category": ["unknown", np.nan],
            "boolean": [False, True],
        }
    )
    features = list(context.columns)
    y_context = np.array([0, 1, 0, 1, 0, 1])

    expected = reference_episode_features(context, query, features, y_context)
    actual = preprocess_episode_features(context, query, features)

    np.testing.assert_array_equal(actual, expected)


def test_fast_preprocessing_matches_original_outlier_clipping():
    context = pd.DataFrame(
        {
            "ordinary": np.arange(30, dtype=np.float64),
            "outlier": np.r_[np.zeros(29), 1000.0],
        }
    )
    query = pd.DataFrame({"ordinary": [-100.0, 100.0], "outlier": [-1000.0, 2000.0]})
    features = list(context.columns)
    y_context = np.arange(len(context)) % 2

    expected = reference_episode_features(context, query, features, y_context)
    actual = preprocess_episode_features(context, query, features)

    np.testing.assert_array_equal(actual, expected)
