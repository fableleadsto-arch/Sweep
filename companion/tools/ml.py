"""Machine-learning capabilities — Scikit-learn / XGBoost / LightGBM.

One input contract (tabular data + target column), three engines: classic
Scikit-learn, gradient boosting (XGBoost preferred, LightGBM fallback). All
imports are lazy.
"""

from __future__ import annotations

from typing import Any

from .common import as_rows, load


def _dataset(data: Any) -> tuple[list[list], list[str]]:
    """Normalize tabular data into (rows, column_names)."""
    columns, rows = as_rows(data)
    if not rows:
        raise ValueError(
            "No tabular data found. Send `data` as a CSV string, a list of "
            "row-dicts, or a dict with `rows`/`columns`."
        )
    if columns is None:
        width = max(len(r) for r in rows)
        columns = [f"col{i}" for i in range(width)]
    return rows, columns


def _split_xy(rows: list[list], columns: list[str], params: dict[str, Any]) -> tuple[list, list, str, list, list]:
    """Return (feature_rows, target_values, target_name, feature_columns, target_column)."""
    target = str(params.get("target") or columns[-1])
    if target not in columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {', '.join(columns)}")
    target_idx = columns.index(target)
    feature_cols = [c for i, c in enumerate(columns) if i != target_idx]
    X = [[r[i] for i in range(len(columns)) if i != target_idx] for r in rows]
    y = [r[target_idx] for r in rows]
    return X, y, target, feature_cols, target


def _numeric_flags(X: list[list]) -> list[bool]:
    flags = []
    for row in X:
        if not flags:
            flags = [isinstance(v, (int, float)) for v in row]
        break
    if not flags:
        width = max(len(r) for r in X) if X else 0
        flags = [True] * width
    # Widen flags if rows vary in length.
    width = max(len(r) for r in X) if X else 0
    while len(flags) < width:
        flags.append(True)
    return flags[:width]


def _fit_and_evaluate(
    X: list[list],
    y: list,
    numeric_flags: list[bool],
    mode: str,
    make_classifier,
    make_regressor,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.model_selection import train_test_split

    X = [list(r) for r in X]
    X_np = np.array([[float(v) if isinstance(v, (int, float)) else str(v) for v in r] for r in X], dtype=object)
    y_np = np.array(y, dtype=object)

    numeric_mask = np.array(numeric_flags[: X_np.shape[1]])

    is_classification = mode == "classify" or (
        mode in {"auto", "classify"}
        and not all(isinstance(v, (int, float)) for v in y_np)
    )
    if mode == "regress":
        is_classification = False

    cat_idx = [i for i in range(X_np.shape[1]) if not numeric_mask[i]]
    num_idx = [i for i in range(X_np.shape[1]) if numeric_mask[i]]

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                num_idx,
            ),
            (
                "cat",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                cat_idx,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.25, random_state=42, stratify=None
    )
    if is_classification:
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        clf = Pipeline([("pre", preprocess), ("model", make_classifier())])
        clf.fit(X_train, y_train_enc)
        pred = clf.predict(X_test)
        return {
            "task": "classification",
            "classes": [str(c) for c in le.classes_],
            "accuracy": round(float(accuracy_score(y_test_enc, pred)), 4),
            "f1_macro": round(float(f1_score(y_test_enc, pred, average="macro")), 4),
            "samples": int(len(y)),
            "test_samples": int(len(y_test)),
            "metrics": {"accuracy": round(float(accuracy_score(y_test_enc, pred)), 4), "f1_macro": round(float(f1_score(y_test_enc, pred, average="macro")), 4)},
        }
    y_train_num = np.array([float(v) for v in y_train])
    y_test_num = np.array([float(v) for v in y_test])
    reg = Pipeline([("pre", preprocess), ("model", make_regressor())])
    reg.fit(X_train, y_train_num)
    pred = reg.predict(X_test)
    return {
        "task": "regression",
        "r2": round(float(r2_score(y_test_num, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test_num, pred)), 4),
        "samples": int(len(y)),
        "test_samples": int(len(y_test)),
        "metrics": {"r2": round(float(r2_score(y_test_num, pred)), 4), "mae": round(float(mean_absolute_error(y_test_num, pred)), 4)},
    }


def run_ml(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a classic Scikit-learn model and evaluate it."""
    data = payload.get("data")
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "auto").lower()

    load("sklearn")
    load("numpy")

    rows, columns = _dataset(data)

    if mode in {"cluster", "anomaly"}:
        return _unsupervised(rows, columns, mode, params)

    X, y, target, feature_cols, _ = _split_xy(rows, columns, params)
    flags = _numeric_flags(X)

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    def make_classifier():
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    def make_regressor():
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    metrics = _fit_and_evaluate(X, y, flags, mode, make_classifier, make_regressor)
    metrics["target"] = target
    metrics["engine"] = "scikit-learn"
    summary = (
        f"Trained a scikit-learn {metrics['task']} model on {metrics['samples']} rows "
        f"(target: {target}). "
    )
    if metrics["task"] == "classification":
        summary += f"Accuracy {metrics['accuracy']:.2%}, F1 (macro) {metrics['f1_macro']:.2%}."
    else:
        summary += f"R² {metrics['r2']:.3f}, MAE {metrics['mae']:.3g}."
    return {"result": metrics, "summary": summary, "libraries_used": ["scikit-learn", "numpy"]}


def run_gradient_boost(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a gradient-boosting model (XGBoost preferred, LightGBM fallback)."""
    data = payload.get("data")
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "auto").lower()

    from .common import CapabilityUnavailable, module_available

    if module_available("xgboost"):
        engine = "xgboost"
        from xgboost import XGBClassifier, XGBRegressor

        def make_classifier():
            return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)

        def make_regressor():
            return XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)

    elif module_available("lightgbm"):
        engine = "lightgbm"
        from lightgbm import LGBMClassifier, LGBMRegressor

        def make_classifier():
            return LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

        def make_regressor():
            return LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

    else:
        raise CapabilityUnavailable(
            "Gradient boosting needs XGBoost or LightGBM. Install with "
            "`pip install -r requirements.companion-ai.txt`."
        )

    rows, columns = _dataset(data)
    X, y, target, feature_cols, _ = _split_xy(rows, columns, params)
    flags = _numeric_flags(X)

    metrics = _fit_and_evaluate(X, y, flags, mode, make_classifier, make_regressor)
    metrics["target"] = target
    metrics["engine"] = engine
    summary = (
        f"Trained a gradient-boosting ({engine}) {metrics['task']} model on "
        f"{metrics['samples']} rows (target: {target}). "
    )
    if metrics["task"] == "classification":
        summary += f"Accuracy {metrics['accuracy']:.2%}, F1 (macro) {metrics['f1_macro']:.2%}."
    else:
        summary += f"R² {metrics['r2']:.3f}, MAE {metrics['mae']:.3g}."
    return {"result": metrics, "summary": summary, "libraries_used": [engine, "numpy"]}


def _unsupervised(rows: list[list], columns: list[str], mode: str, params: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    X = [list(r) for r in rows]
    X_np = np.array(X, dtype=float)
    if mode == "cluster":
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        k = int(params.get("k") or 3)
        k = max(2, min(k, len(rows) - 1 if len(rows) > 1 else 2))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_np)
        sil = silhouette_score(X_np, labels) if len(set(labels)) > 1 and len(rows) > k else None
        return {
            "result": {
                "task": "clustering",
                "clusters": int(k),
                "labels": [int(x) for x in labels],
                "silhouette": round(float(sil), 4) if sil is not None else None,
                "inertia": round(float(model.inertia_), 4),
                "samples": len(rows),
            },
            "summary": f"K-means clustering into {k} clusters — silhouette {sil:.3f}." if sil is not None else f"K-means clustering into {k} clusters.",
            "libraries_used": ["scikit-learn", "numpy"],
        }
    from sklearn.ensemble import IsolationForest

    contamination = float(params.get("contamination") or 0.1)
    model = IsolationForest(contamination=min(0.5, max(0.01, contamination)), random_state=42)
    labels = model.fit_predict(X_np)
    outliers = [int(i) for i, l in enumerate(labels) if l == -1]
    return {
        "result": {
            "task": "anomaly-detection",
            "outlier_indices": outliers,
            "outlier_count": len(outliers),
            "contamination": round(float(contamination), 3),
            "samples": len(rows),
        },
        "summary": f"IsolationForest flagged {len(outliers)}/{len(rows)} rows as anomalies (contamination {contamination:.0%}).",
        "libraries_used": ["scikit-learn", "numpy"],
    }
