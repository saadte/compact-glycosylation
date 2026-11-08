import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC


def fit_multiclass_control(
    fitting: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    probability: bool = True,
    max_iter: int = -1,
    standardize: bool = False,
    preprocessor=None,
) -> tuple[SVC | Pipeline, dict]:
    # Fit preprocessing inside each grouped inner fold, then refit the selected model.
    if standardize and preprocessor is not None:
        raise ValueError("Use either standardization or an explicit preprocessor")
    inner = list(
        GroupKFold(3, shuffle=True, random_state=20261108).split(fitting, groups=groups)
    )
    configurations = [(c, k) for c in (0.1, 1, 10, 100) for k in ("linear", "rbf")]

    def estimator(c, kernel, *, calibrate=False):
        classifier = SVC(
            C=c, kernel=kernel, probability=calibrate, random_state=0, max_iter=max_iter
        )
        if preprocessor is not None:
            return Pipeline(
                [("preprocess", clone(preprocessor)), ("classifier", classifier)]
            )
        if standardize:
            return Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("normalize", Normalizer()),
                    ("classifier", classifier),
                ]
            )
        return classifier

    values, statuses = [], []
    for c, kernel in configurations:
        accuracy = []
        failed_fold = None
        for fold, (train, validation) in enumerate(inner):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    classifier = estimator(c, kernel).fit(fitting[train], target[train])
            except ConvergenceWarning:
                failed_fold = fold
                break
            accuracy.append(
                float(
                    np.mean(
                        classifier.predict(fitting[validation]) == target[validation]
                    )
                )
            )
        values.append(float(np.mean(accuracy)) if failed_fold is None else None)
        statuses.append(
            {"converged": failed_fold is None, "failed_inner_fold": failed_fold}
        )
    eligible = [i for i, value in enumerate(values) if value is not None]
    if not eligible:
        raise RuntimeError(f"No SVC configuration converged within max_iter={max_iter}")
    best = max(eligible, key=lambda i: values[i])
    c, kernel = configurations[best]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = estimator(c, kernel, calibrate=probability).fit(fitting, target)
    except ConvergenceWarning as error:
        raise RuntimeError(
            f"Selected outer SVC refit did not converge: C={c}, kernel={kernel}, "
            f"inner_accuracy={values[best]}, max_iter={max_iter}"
        ) from error
    return model, {
        "configuration": configurations[best],
        "inner_accuracy": values[best],
        "all_configurations": configurations,
        "all_inner_accuracy": values,
        "all_inner_fit_status": statuses,
        "probability_calibration": probability,
        "solver_max_iter": max_iter,
        "standardize": standardize,
        "preprocessor_class": type(preprocessor).__name__
        if preprocessor is not None
        else None,
    }


def fit_binary_models(estimator, fitting: np.ndarray, labels: np.ndarray) -> list:
    models = []
    for category in range(labels.shape[1]):
        unique = np.unique(labels[:, category])
        if len(unique) == 1:
            models.append(float(unique[0]))
        else:
            models.append(clone(estimator).fit(fitting, labels[:, category]))
    return models


def predict_binary_models(models: list, query: np.ndarray) -> np.ndarray:
    result = np.empty((len(query), len(models)))
    for category, model in enumerate(models):
        if isinstance(model, float):
            result[:, category] = model
        else:
            result[:, category] = model.predict_proba(query)[
                :, list(model.classes_).index(1)
            ]
    return result


def predict_binary_classes(models: list, query: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.full(len(query), int(model))
            if isinstance(model, float)
            else model.predict(query)
            for model in models
        ]
    )


def binary_probabilities(
    estimator, fitting: np.ndarray, labels: np.ndarray, query: np.ndarray
) -> np.ndarray:
    return predict_binary_models(fit_binary_models(estimator, fitting, labels), query)


def scores(labels: np.ndarray, probabilities: np.ndarray, *, predicted=None) -> dict:
    predicted = (
        probabilities >= 0.5 if predicted is None else np.asarray(predicted, dtype=bool)
    )
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite donor probabilities")
    if predicted.shape != labels.shape:
        raise ValueError("Prediction and annotation dimensions differ")
    supported = (labels.sum(axis=0) > 0) & (labels.sum(axis=0) < len(labels))
    aps = [
        float(average_precision_score(labels[:, k], probabilities[:, k]))
        if supported[k]
        else None
        for k in range(labels.shape[1])
    ]
    return {
        "samples_f1": float(
            f1_score(labels, predicted, average="samples", zero_division=0)
        ),
        "exact_match": float((labels == predicted).all(axis=1).mean()),
        "top1_accuracy": float(
            (labels[np.arange(len(labels)), probabilities.argmax(axis=1)] == 1).mean()
        ),
        "macro_ap": float(np.mean([v for v in aps if v is not None]))
        if supported.any()
        else None,
        "ap": aps,
        "mcc": [
            float(matthews_corrcoef(labels[:, k], predicted[:, k]))
            if np.unique(labels[:, k]).size > 1 and np.unique(predicted[:, k]).size > 1
            else 0.0
            for k in range(labels.shape[1])
        ],
        "label_support": labels.sum(axis=0).tolist(),
        "n": len(labels),
    }


def paired_group_interval(
    labels, candidate, control, groups, *, draws=500, seed=20261107
):
    # Resample sequence groups, conditional on the saved models and folds.
    random = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique]
    difference = []
    for _ in range(draws):
        subset = np.concatenate(
            [indices[i] for i in random.integers(len(unique), size=len(unique))]
        )
        a = f1_score(
            labels[subset], candidate[subset] >= 0.5, average="samples", zero_division=0
        )
        b = f1_score(
            labels[subset], control[subset] >= 0.5, average="samples", zero_division=0
        )
        difference.append(a - b)
    return {
        "metric": "samples_f1_difference",
        "estimate": scores(labels, candidate)["samples_f1"]
        - scores(labels, control)["samples_f1"],
        "interval95": np.quantile(difference, [0.025, 0.975]).tolist(),
        "draws": draws,
        "seed": seed,
        "conditional_on_fixed_models_and_folds": True,
    }
