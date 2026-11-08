from pathlib import Path

import numpy as np
from sklearn.feature_selection import chi2

AMINO_ACIDS = "ARNDCEQGHILKMFPSTWYV"
RAW_PROPERTIES = np.array(
    [
        [
            0.61,
            0.60,
            0.06,
            0.46,
            1.07,
            0.47,
            0,
            0.07,
            0.61,
            2.22,
            1.53,
            1.15,
            1.18,
            2.02,
            1.95,
            0.05,
            0.05,
            2.65,
            1.88,
            1.32,
        ],
        [
            52.6,
            109.1,
            75.7,
            68.4,
            68.3,
            84.7,
            89.7,
            36.3,
            91.9,
            102,
            102,
            105.1,
            97.7,
            113.9,
            73.6,
            54.9,
            71.2,
            135.4,
            116.2,
            85.1,
        ],
        [
            93.7,
            250.4,
            146.3,
            142.6,
            135.2,
            182.9,
            177.7,
            52.6,
            188.1,
            182.2,
            173.7,
            215.2,
            197.6,
            228.6,
            0,
            109.5,
            142.1,
            271.6,
            239.9,
            157.2,
        ],
        [
            8.1,
            10.5,
            11.6,
            13,
            5.5,
            12.3,
            10.5,
            9,
            10.4,
            5.2,
            4.9,
            11.3,
            5.7,
            5.2,
            8,
            9.2,
            8.6,
            5.4,
            6.2,
            5.9,
        ],
        [0, 1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
).T
PROPERTIES = (RAW_PROPERTIES - RAW_PROPERTIES.min(axis=0)) / np.ptp(
    RAW_PROPERTIES, axis=0
)


def structural_features(
    pdb_path: Path, index_path: Path, ss_path: Path, expected_length: int
) -> np.ndarray:
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    structure = PDBParser(QUIET=True).get_structure("protein", str(pdb_path))
    ShrakeRupley().compute(structure, level="R")
    residues = list(structure[0].get_residues())
    sasa = np.array([residue.sasa for residue in residues])
    ss = np.array([int(i) for i in ss_path.read_text().strip()])
    if len(ss) != len(residues):
        raise ValueError(
            f"Structure/secondary-structure length mismatch: {pdb_path.name}"
        )
    low_confidence = np.array(
        [any(atom.bfactor < 70 for atom in residue) for residue in residues]
    )
    sasa[low_confidence] = 0
    ss[low_confidence] = 0
    index = np.array([int(i) for i in index_path.read_text().split()])
    if not np.array_equal(index, np.arange(index[0], index[-1] + 1)):
        raise ValueError("Released Rossmann crop is not a contiguous zero-based range")
    sasa, ss = sasa[index], ss[index]
    if len(sasa) != expected_length or np.ptp(sasa) == 0 or not ss.any():
        raise ValueError(f"Invalid or unsupported structural features: {pdb_path.name}")
    return np.column_stack(((sasa - sasa.min()) / np.ptp(sasa), ss / 3))


def encode_alignment(
    sequence: str, positions: np.ndarray, structure: np.ndarray
) -> np.ndarray:
    values = np.zeros((len(sequence), 7))
    for column, residue in enumerate(sequence):
        if residue in "-X*":
            continue
        if residue not in AMINO_ACIDS:
            raise ValueError(f"Unrecognized residue {residue}")
        values[column, :5] = PROPERTIES[AMINO_ACIDS.index(residue)]
        values[column, 5:] = structure[positions[column]]
    return values.reshape(-1)


def fitting_columns(aligned_sequences: list[str], features: np.ndarray) -> np.ndarray:
    gaps = (np.array([list(s) for s in aligned_sequences]) == "-").mean(axis=0)
    columns = np.flatnonzero(np.repeat(gaps <= 0.5, 7))
    selected = features[:, columns]
    centered = selected - selected.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    dropped = (np.triu(np.abs(correlation), k=1) > 0.9).any(axis=0)
    return columns[~dropped]


def select_columns(features: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    active = labels.sum(axis=0) > 0
    if active.sum() < 2:
        raise ValueError(
            "Chi-square selection requires at least two observed donor categories"
        )
    scores, _ = chi2(features, labels[:, active])
    # SelectKBest treats nonfinite constant-feature scores as lowest and keeps
    # the last columns on ties via a stable sort.
    scores = np.nan_to_num(scores, nan=-np.inf)
    return np.sort(np.argsort(scores, kind="stable")[-min(count, len(scores)) :])
