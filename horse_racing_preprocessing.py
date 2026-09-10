"""Fast, context-fitted preprocessing for horse-racing training episodes."""

import numpy as np
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def _to_numerical(context, query):
    """Match TransformToNumerical without constructing a ColumnTransformer."""
    categorical_columns = make_column_selector(
        dtype_include=["string", "object", "category", "boolean"]
    )(context)
    numeric_columns = make_column_selector(dtype_include="number")(context)
    context_parts = []
    query_parts = []

    if numeric_columns:
        imputer = SimpleImputer()
        context_numeric = context.loc[:, numeric_columns].to_numpy(copy=False)
        query_numeric = query.loc[:, numeric_columns].to_numpy(copy=False)
        context_parts.append(imputer.fit_transform(context_numeric))
        query_parts.append(imputer.transform(query_numeric))
    if categorical_columns:
        encoder = OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        context_categorical = context.loc[:, categorical_columns].to_numpy(copy=False)
        query_categorical = query.loc[:, categorical_columns].to_numpy(copy=False)
        context_parts.append(encoder.fit_transform(context_categorical))
        query_parts.append(encoder.transform(query_categorical))

    if not context_parts:
        raise ValueError("No supported numeric or categorical feature columns were found")
    return np.concatenate(context_parts, axis=1), np.concatenate(query_parts, axis=1)


def _soft_outlier_clip(X, lower_bounds, upper_bounds):
    X = np.maximum(-np.log1p(np.abs(X)) + lower_bounds, X)
    return np.minimum(np.log1p(np.abs(X)) + upper_bounds, X)


def preprocess_episode_features(context_frame, query_frame, feature_columns):
    """Match the trainer's fixed one-member, no-shuffle/no-normalization pipeline."""
    context, query = _to_numerical(
        context_frame.loc[:, feature_columns],
        query_frame.loc[:, feature_columns],
    )

    # UniqueFeatureFilter(threshold=1), fitted on context only.
    keep = np.array(
        [np.unique(context[:, index]).size > 1 for index in range(context.shape[1])],
        dtype=bool,
    )
    if not keep.any():
        raise ValueError("Every feature is constant within the episode context")
    context = context[:, keep]
    query = query[:, keep]

    # CustomStandardScaler, again fitted only on context.
    mean = np.mean(context, axis=0)
    scale = np.std(context, axis=0) + 1e-6
    context = np.clip((context - mean) / scale, -100, 100)
    query = np.clip((query - mean) / scale, -100, 100)

    # OutlierRemover's two-pass statistics and soft clipping.
    ddof = 1 if len(context) > 1 else 0
    means = np.nanmean(context, axis=0)
    stds = np.maximum(np.nanstd(context, axis=0, ddof=ddof), 1e-6)
    clean_context = context.copy()
    outliers = (context < means - 4.0 * stds) | (context > means + 4.0 * stds)
    clean_context[outliers] = np.nan

    means = np.nanmean(clean_context, axis=0)
    stds = np.maximum(np.nanstd(clean_context, axis=0, ddof=ddof), 1e-6)
    lower_bounds = means - 4.0 * stds
    upper_bounds = means + 4.0 * stds
    context = _soft_outlier_clip(context, lower_bounds, upper_bounds)
    query = _soft_outlier_clip(query, lower_bounds, upper_bounds)

    return np.asarray(np.concatenate([context, query], axis=0), dtype=np.float32)
