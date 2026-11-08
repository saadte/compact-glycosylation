from collections.abc import Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler, normalize

REGULARIZERS = np.array([1e-5, 1e-4, 1e-3, 1e-2])


def transform_objects(raw, state):
    x = np.asarray(raw, dtype=np.float64).copy()
    x -= state["mean"]
    x /= state["scale"]
    return normalize(x)


def fit_object_kernel(raw, fitting_objects):
    raw = np.asarray(raw, dtype=np.float64)
    fitting_objects = np.asarray(fitting_objects, dtype=int)
    if (
        raw.ndim != 2
        or not np.isfinite(raw).all()
        or len(fitting_objects) < 2
        or (len(np.unique(fitting_objects)) != len(fitting_objects))
    ):
        raise ValueError(
            "Object kernel requires finite rows and distinct fitting objects"
        )
    scaler = StandardScaler().fit(raw[fitting_objects])
    state = {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "fitting_objects": fitting_objects.astype(np.uint16),
    }
    x = transform_objects(raw, state)
    distances = pdist(x[fitting_objects], metric="sqeuclidean")
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("No positive fitting-object distance for RBF bandwidth")
    state["gamma"] = float(1 / np.median(positive))
    kernel = rbf_kernel(x, gamma=state["gamma"])
    mean_error = float(np.max(np.abs(scaler.mean_ - raw[fitting_objects].mean(0))))
    expected_variance = raw[fitting_objects].var(0)
    variance_error = float(np.max(np.abs(scaler.var_ - expected_variance)))
    relative_variance_error = float(
        np.max(
            np.abs(scaler.var_ - expected_variance) / (1 + np.abs(expected_variance))
        )
    )
    if (
        mean_error > 1e-12
        or relative_variance_error > 1e-12
        or scaler.n_samples_seen_ != len(fitting_objects)
    ):
        raise ValueError("Actual object scaler does not match the fitting partition")
    audit = {
        "fitting_unique_objects": len(fitting_objects),
        "dimensions": raw.shape[1],
        "mean_max_error": mean_error,
        "variance_max_error": variance_error,
        "variance_relative_error": relative_variance_error,
        "constant_coordinates": int(np.sum(scaler.var_ == 0)),
        "gamma": state["gamma"],
        "positive_distance_count": len(positive),
    }
    return (kernel, state, audit)


def object_kernel_from_state(raw, state):
    x = transform_objects(raw, state)
    return rbf_kernel(x, gamma=float(state["gamma"]))


def ridge_path(kernel, target, *, regularizers=REGULARIZERS):
    kernel = np.asarray(kernel, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        kernel.shape != (len(target), len(target))
        or not np.isfinite(kernel).all()
        or (not np.isfinite(target).all())
        or (not np.allclose(kernel, kernel.T, atol=1e-12, rtol=0))
    ):
        raise ValueError("Ridge fitting kernel must be finite, square and symmetric")
    penalties = len(target) * np.asarray(regularizers, dtype=np.float64)
    if np.any(penalties <= 0):
        raise ValueError("Ridge penalties must be positive")
    eigenvalues, vectors = eigh(kernel, driver="evd", check_finite=True)
    if eigenvalues[0] < -1e-07:
        raise ValueError(
            "Fitting pair kernel is not positive semidefinite within numerical tolerance"
        )
    inverse = 1 / (eigenvalues[:, None] + penalties)
    ones_projection = vectors.T @ np.ones(len(target))
    target_projection = vectors.T @ target
    solve_ones = vectors @ (inverse * ones_projection[:, None])
    solve_target = vectors @ (inverse * target_projection[:, None])
    intercept = solve_target.sum(0) / solve_ones.sum(0)
    dual = solve_target - solve_ones * intercept
    residual = float(
        np.max(np.abs(kernel @ dual + penalties * dual + intercept - target[:, None]))
    )
    constraint = float(np.max(np.abs(dual.sum(0))))
    if not np.isfinite(dual).all() or residual > 1e-07 or constraint > 1e-07:
        raise ValueError(
            "Ridge path failed the direct equation or free-intercept constraint"
        )
    return (
        dual,
        intercept,
        {
            "minimum_eigenvalue": float(eigenvalues[0]),
            "equation_residual_max": residual,
            "intercept_constraint_max": constraint,
        },
    )


def _expected_precision(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    cutoff = np.sort(scores)[-k]
    above, tied = (scores > cutoff, scores == cutoff)
    return float((labels[above].sum() + (k - above.sum()) * labels[tied].mean()) / k)


def per_acceptor_ranking(
    *,
    fold_ids: Sequence[str],
    folds: Sequence[str],
    compounds: Sequence[str],
    enzymes: Sequence[str],
    labels: np.ndarray,
    scores: np.ndarray,
    minimum_enzymes: int = 5,
    precision_at: Sequence[int] = (1, 3, 5),
    chemistry_only: bool = False,
) -> dict:
    expected_folds = list(fold_ids)
    row_folds, cids, ids = (np.asarray(v) for v in (folds, compounds, enzymes))
    y, s = (np.asarray(labels), np.asarray(scores))
    arrays = (row_folds, cids, ids, y, s)
    if (
        any((a.ndim != 1 for a in arrays))
        or len({len(a) for a in arrays}) != 1
        or (not len(y))
    ):
        raise ValueError(
            "Ranking inputs must be nonempty, aligned one-dimensional arrays"
        )
    if not expected_folds or len(set(expected_folds)) != len(expected_folds):
        raise ValueError("Declared query fold IDs must be nonempty and unique")
    for values in (expected_folds, row_folds, cids, ids):
        if not all((isinstance(v, str) and v for v in values)):
            raise ValueError("Fold, compound and enzyme IDs must be nonempty strings")
    if not set(row_folds) <= set(expected_folds):
        raise ValueError("Observed query fold is absent from the declared fold IDs")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Assay labels must be binary and scores finite")
    ks = list(precision_at)
    if (
        not ks
        or any(
            (
                isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0
                for k in ks
            )
        )
        or len(set(ks)) != len(ks)
    ):
        raise ValueError("Precision cutoffs must be unique positive integers")
    if (
        isinstance(minimum_enzymes, bool)
        or not isinstance(minimum_enzymes, (int, np.integer))
        or minimum_enzymes < max(ks)
    ):
        raise ValueError(
            "Minimum enzyme support must be an integer at least max(precision_at)"
        )
    enzyme_fold = {}
    pairs = set()
    for fold, cid, enzyme in zip(row_folds, cids, ids, strict=True):
        if enzyme in enzyme_fold and enzyme_fold[enzyme] != fold:
            raise ValueError(
                "An enzyme appears in multiple independently fitted query folds"
            )
        enzyme_fold[enzyme] = fold
        if (enzyme, cid) in pairs:
            raise ValueError(
                "Duplicate measured enzyme/compound pair requires prior assay resolution"
            )
        pairs.add((enzyme, cid))
    groups, summaries = ([], [])
    metric_names = ["average_precision", "auroc", *[f"precision_at_{k}" for k in ks]]
    for fold in expected_folds:
        fold_groups = []
        for cid in sorted(set(cids[row_folds == fold])):
            selected = (row_folds == fold) & (cids == cid)
            group_y, group_s = (y[selected], s[selected])
            if chemistry_only and (not np.all(group_s == group_s[0])):
                raise ValueError(
                    "Chemistry-only scores violate exact within-compound tie invariance"
                )
            reason = None
            if len(group_y) < minimum_enzymes:
                reason = "insufficient_assayed_enzymes"
            elif len(np.unique(group_y)) != 2:
                reason = "single_label_class"
            group = {
                "fold": fold,
                "compound": cid,
                "assayed_enzymes": len(group_y),
                "positives": int(group_y.sum()),
                "eligible": reason is None,
                "exclusion_reason": reason,
                **dict.fromkeys(metric_names),
            }
            if reason is None:
                group["average_precision"] = float(
                    average_precision_score(group_y, group_s)
                )
                group["auroc"] = float(roc_auc_score(group_y, group_s))
                group.update(
                    {
                        f"precision_at_{k}": _expected_precision(group_y, group_s, k)
                        for k in ks
                    }
                )
                fold_groups.append(group)
            groups.append(group)
        summaries.append(
            {
                "fold": fold,
                "eligible_compounds": len(fold_groups),
                **{
                    metric: float(np.mean([g[metric] for g in fold_groups]))
                    if fold_groups
                    else None
                    for metric in metric_names
                },
            }
        )
    complete = all((f["eligible_compounds"] > 0 for f in summaries))
    return {
        "complete": complete,
        "measured_pairs": len(y),
        "eligible_groups": sum((g["eligible"] for g in groups)),
        "excluded_groups": sum((not g["eligible"] for g in groups)),
        "aggregate": {
            metric: float(np.mean([f[metric] for f in summaries])) if complete else None
            for metric in metric_names
        },
        "folds": summaries,
        "groups": groups,
    }


def encode_consensus(
    sequence: str, alphabet: list[str], matrix: np.ndarray
) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] != len(alphabet) or (not sequence):
        raise ValueError(
            "Ordered encoding requires a nonempty sequence and aligned matrix columns"
        )
    columns = [alphabet.index("*" if c == "-" else c) for c in sequence]
    return matrix[:, columns].ravel().copy()
