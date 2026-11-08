import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import normalize

from gtb_predict.data import DONORS, read_training
from gtb_predict.evaluation import fit_multiclass_control, scores


def validate_roles(ids: np.ndarray, groups: np.ndarray, folds: np.ndarray) -> None:
    if groups.shape != ids.shape or folds.shape != ids.shape:
        raise ValueError("Group and fold arrays must match the accession array")
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("Expected all five outer folds numbered 0 to 4")
    for group in np.unique(groups):
        if len(np.unique(folds[groups == group])) != 1:
            raise ValueError("A similarity group crosses outer folds")


def decision_scores(model, features: np.ndarray, *, calibrated: bool) -> np.ndarray:
    fill = 0.0 if calibrated else -1e6
    values = np.full((len(features), len(DONORS)), fill)
    if calibrated:
        values[:, model.classes_] = model.predict_proba(features)
    else:
        decision = model.decision_function(features)
        if decision.ndim == 1:
            decision = np.column_stack((-decision, decision))
        values[:, model.classes_] = decision
    return values


def evaluate(
    training: Path,
    roles_path: Path,
    features_path: Path,
    view: str,
    profile: str,
    output: Path,
) -> dict:
    rows, _, _ = read_training(training)
    ids = np.array([row["ID"] for row in rows])
    labels = np.array(
        [[int(donor in row["Donor"].split(",,")) for donor in DONORS] for row in rows]
    )
    if not np.all(labels.sum(axis=1) == 1):
        raise ValueError("The donor endpoint requires one annotation per row")
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    if roles["ids"] != ids.tolist():
        raise ValueError("Frozen roles and training accession order differ")
    groups = np.asarray(roles["groups"])
    folds = np.asarray(roles["outer_folds"])
    validate_roles(ids, groups, folds)
    if len(np.unique(groups)) != 171:
        raise ValueError("Expected the 171 released development similarity groups")
    with np.load(features_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["ids"], ids):
            raise ValueError("Feature and training accession order differ")
        features = saved[view]
    if (
        features.ndim != 2
        or len(features) != len(ids)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Features must be a finite accession-aligned matrix")
    features = normalize(features)
    calibrated = profile == "public_embedding"
    standardize = profile == "compact"
    max_iter = -1 if calibrated else 1_000_000
    output.mkdir(parents=True, exist_ok=True)
    decision = np.full(labels.shape, np.nan)
    native = np.zeros(labels.shape, dtype=int)
    reports = []
    for outer in range(5):
        fitting = np.flatnonzero(folds != outer)
        query = np.flatnonzero(folds == outer)
        model, selection = fit_multiclass_control(
            features[fitting],
            labels[fitting].argmax(axis=1),
            groups[fitting],
            probability=calibrated,
            max_iter=max_iter,
            standardize=standardize,
        )
        decision[query] = decision_scores(model, features[query], calibrated=calibrated)
        native[query] = np.eye(len(DONORS), dtype=int)[model.predict(features[query])]
        joblib.dump(
            {
                "model": model,
                "fit_ids": ids[fitting],
                "query_ids": ids[query],
                "selection": selection,
                "view": view,
                "profile": profile,
            },
            output / f"outer{outer}.joblib",
            compress=3,
        )
        reports.append(
            {
                "fold": outer,
                "selection": selection,
                "metrics": scores(
                    labels[query], decision[query], predicted=native[query]
                ),
            }
        )
    if not np.isfinite(decision).all():
        raise ValueError("An accession lacks an out-of-fold prediction")
    np.savez_compressed(
        output / "predictions.npz",
        ids=ids,
        labels=labels,
        groups=groups,
        folds=folds,
        decision=decision,
        native=native,
    )
    report = {
        "profile": profile,
        "view": view,
        "score_type": "calibrated probability" if calibrated else "SVC OvR decision",
        "metrics": scores(labels, decision, predicted=native),
        "folds": reports,
        "annotation_classification_not_biochemical_multispecificity": True,
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen donor features on development folds"
    )
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument(
        "--profile", choices=("public_embedding", "compact"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        args.training, args.roles, args.features, args.view, args.profile, args.output
    )


if __name__ == "__main__":
    main()
