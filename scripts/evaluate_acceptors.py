import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from gtb_predict.acceptors import (
    REGULARIZERS,
    fit_object_kernel,
    per_acceptor_ranking,
    ridge_path,
)


def load_inputs(
    inputs_path: Path, roles_path: Path, features_path: Path | None, representation: str
) -> dict:
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    roles = json.loads(roles_path.read_text(encoding="utf-8"))["enzymes"]
    rows = inputs["source_pairs"]
    enzymes = np.array([row["enzyme"] for row in rows])
    cids = np.array([row["cid"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    ids, compounds = np.unique(enzymes), np.unique(cids)
    groups = np.array([roles[e]["group"] for e in enzymes])
    folds = np.array([roles[e]["outer_fold"] for e in enzymes])
    if not np.isin(labels, (0, 1)).all() or len(set(zip(enzymes, cids))) != len(rows):
        raise ValueError(
            "Source observations must be unique binary enzyme-compound pairs"
        )
    if (
        len(rows),
        int(labels.sum()),
        len(ids),
        len(compounds),
        len(np.unique(groups)),
    ) != (3990, 793, 73, 77, 32):
        raise ValueError(
            "Expected 3990 pairs, 793 positives, 73 enzymes, 77 compounds and 32 groups"
        )
    if set(ids) != set(roles) or set(folds.tolist()) != set(range(5)):
        raise ValueError("Source enzyme identities or five outer folds differ")
    for group in np.unique(groups):
        if len(np.unique(folds[groups == group])) != 1:
            raise ValueError("A similarity component crosses outer folds")
    chemistry = np.array(
        [inputs["chemistry"][cid] for cid in compounds], dtype=np.float64
    )
    if chemistry.shape != (77, 153) or not np.isfinite(chemistry).all():
        raise ValueError(
            "Expected 153 finite published descriptors for each source compound"
        )
    protein = None
    if representation == "chemistry_only":
        if features_path is not None:
            raise ValueError("Chemistry-only evaluation does not take protein features")
    else:
        if features_path is None:
            raise ValueError("The representation requires an explicit feature file")
        with np.load(features_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["ids"], ids):
                raise ValueError(
                    "Protein feature IDs must match sorted source enzyme IDs"
                )
            protein = np.asarray(saved[representation], dtype=np.float64)
        if (
            protein.ndim != 2
            or len(protein) != len(ids)
            or not protein.shape[1]
            or not np.isfinite(protein).all()
        ):
            raise ValueError(
                "Protein features must be a finite nonempty source-aligned matrix"
            )
        if representation == "ordered_blosum_reference" and protein.shape[1] != 4935:
            raise ValueError("Ordered BLOSUM requires the original 4935 coordinates")
    return {
        "enzymes": enzymes,
        "cids": cids,
        "labels": labels,
        "groups": groups,
        "folds": folds,
        "ids": ids,
        "compounds": compounds,
        "chemistry": chemistry,
        "protein": protein,
        "protein_rows": np.searchsorted(ids, enzymes),
        "chemical_rows": np.searchsorted(compounds, cids),
    }


def role_kernels(data: dict, fitting: np.ndarray) -> tuple:
    chemical, chemical_state, _ = fit_object_kernel(
        data["chemistry"],
        np.unique(data["chemical_rows"][fitting]),
    )
    protein, protein_state = None, None
    if data["protein"] is not None:
        protein, protein_state, _ = fit_object_kernel(
            data["protein"],
            np.unique(data["protein_rows"][fitting]),
        )
    return chemical, protein, {"chemical": chemical_state, "protein": protein_state}


def pair_kernel(
    data: dict, kernels: tuple, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    chemical, protein, _ = kernels
    gram = chemical[np.ix_(data["chemical_rows"][left], data["chemical_rows"][right])]
    if protein is not None:
        gram = (
            gram
            * protein[np.ix_(data["protein_rows"][left], data["protein_rows"][right])]
        )
    return gram


def predict(
    data: dict,
    kernels: tuple,
    query: np.ndarray,
    fitting: np.ndarray,
    dual: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    if kernels[1] is None:
        # Evaluate each compound once so chemistry-only scores remain exact ties.
        _, first, inverse = np.unique(
            data["chemical_rows"][query], return_index=True, return_inverse=True
        )
        return (pair_kernel(data, kernels, query[first], fitting) @ dual + intercept)[
            inverse
        ]
    return pair_kernel(data, kernels, query, fitting) @ dual + intercept


def endpoint(
    data: dict,
    rows: np.ndarray,
    scores: np.ndarray,
    folds: np.ndarray,
    fold_ids: list[str],
) -> dict:
    return per_acceptor_ranking(
        fold_ids=fold_ids,
        folds=folds.astype(str),
        compounds=data["cids"][rows].astype(str),
        enzymes=data["enzymes"][rows].astype(str),
        labels=data["labels"][rows],
        scores=scores,
        chemistry_only=data["protein"] is None,
    )


def evaluate(data: dict, output: Path, representation: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    predictions = np.full(len(data["labels"]), np.nan)
    reports = []
    with threadpool_limits(limits=2):
        for outer in range(5):
            fitting = np.flatnonzero(data["folds"] != outer)
            query = np.flatnonzero(data["folds"] == outer)
            inner_predictions = np.full((len(fitting), len(REGULARIZERS)), np.nan)
            inner_folds = np.full(len(fitting), -1)
            splits = GroupKFold(3, shuffle=True, random_state=20261108).split(
                fitting,
                groups=data["groups"][fitting],
            )
            for inner, (train, validation) in enumerate(splits):
                fit_rows, query_rows = fitting[train], fitting[validation]
                kernels = role_kernels(data, fit_rows)
                dual, bias, _ = ridge_path(
                    pair_kernel(data, kernels, fit_rows, fit_rows),
                    data["labels"][fit_rows],
                )
                inner_predictions[validation] = predict(
                    data, kernels, query_rows, fit_rows, dual, bias
                )
                inner_folds[validation] = inner
            objective = []
            for column in range(len(REGULARIZERS)):
                result = endpoint(
                    data,
                    fitting,
                    inner_predictions[:, column],
                    inner_folds,
                    ["0", "1", "2"],
                )
                if not result["complete"]:
                    raise ValueError(
                        "A declared inner fold lacks fixed-compound ranking support"
                    )
                objective.append(result["aggregate"]["average_precision"])
            selected = int(np.flatnonzero(np.array(objective) == max(objective))[-1])
            kernels = role_kernels(data, fitting)
            dual, bias, audit = ridge_path(
                pair_kernel(data, kernels, fitting, fitting),
                data["labels"][fitting],
                regularizers=REGULARIZERS[[selected]],
            )
            predictions[query] = predict(
                data, kernels, query, fitting, dual[:, 0], bias[0]
            )
            state = {
                f"{kind}_{key}": value
                for kind, values in kernels[2].items()
                if values is not None
                for key, value in values.items()
            }
            np.savez_compressed(
                output / f"outer{outer}.npz",
                fitting_rows=fitting,
                query_rows=query,
                dual=dual[:, 0],
                intercept=bias[0],
                regularizer=REGULARIZERS[selected],
                **state,
            )
            np.savez_compressed(
                output / f"inner{outer}.npz",
                fitting_rows=fitting,
                folds=inner_folds,
                predictions=inner_predictions,
                regularizers=REGULARIZERS,
            )
            reports.append(
                {
                    "fold": outer,
                    "inner_fixed_acceptor_ap": objective,
                    "selected_lambda": float(REGULARIZERS[selected]),
                    "solver": audit,
                }
            )
    metrics = endpoint(
        data,
        np.arange(len(predictions)),
        predictions,
        data["folds"],
        [str(i) for i in range(5)],
    )
    if (metrics["eligible_groups"], metrics["excluded_groups"]) != (152, 213):
        raise ValueError("The original fixed-compound ranking support differs")
    np.savez_compressed(
        output / "predictions.npz",
        scores=predictions,
        **{
            key: data[key]
            for key in (
                "enzymes",
                "cids",
                "labels",
                "groups",
                "folds",
                "ids",
                "compounds",
            )
        },
    )
    report = {
        "representation": representation,
        "metrics": metrics,
        "folds": reports,
        "score_type": "uncalibrated joint kernel ridge",
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate source-only fixed-compound enzyme ranking"
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument(
        "--representation",
        required=True,
        choices=(
            "mlm_6m__whole",
            "random_6m__whole",
            "esm2_8m__whole",
            "ordered_blosum_reference",
            "chemistry_only",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_inputs(args.inputs, args.roles, args.features, args.representation)
    evaluate(data, args.output, args.representation)


if __name__ == "__main__":
    main()
