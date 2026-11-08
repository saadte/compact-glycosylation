import numpy as np
from scipy import sparse

from .features import AMINO_ACIDS, PROPERTIES

ALPHABET = AMINO_ACIDS + "X"


def residue_codes(sequence: str) -> np.ndarray:
    # Ambiguous residues share code 20; gaps are not accepted.
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO*"):
        raise ValueError("Missing sequence or unsupported residue")
    return np.array(
        [ALPHABET.index(a) if a in AMINO_ACIDS else 20 for a in sequence],
        dtype=np.int16,
    )


def spectrum(sequences: list[str], *, k: int) -> sparse.csr_matrix:
    # Windows containing ambiguous residues contribute no count.
    if k not in (1, 2, 3):
        raise ValueError("Supported spectrum orders are1,2,3")
    rows, columns, data = [], [], []
    for row, sequence in enumerate(sequences):
        codes = residue_codes(sequence)
        counts = {}
        for start in range(len(codes) - k + 1):
            word = codes[start : start + k]
            if np.all(word < 20):
                index = 0
                for value in word:
                    index = index * 20 + int(value)
                counts[index] = counts.get(index, 0) + 1
        for column, value in counts.items():
            rows.append(row)
            columns.append(column)
            data.append(value)
    return sparse.csr_matrix(
        (np.array(data, dtype=np.float64), (rows, columns)),
        shape=(len(sequences), 20**k),
    )


def physicochemical(sequences: list[str]) -> np.ndarray:
    output = []
    for sequence in sequences:
        codes = residue_codes(sequence)
        known = codes < 20
        if not known.any():
            raise ValueError("A sequence has no standard residues")
        values = PROPERTIES[codes[known]]
        features = [np.log1p(len(codes)), float((~known).mean())]
        features.extend(values.mean(axis=0))
        features.extend(values.std(axis=0))
        for part in np.array_split(codes, 3):
            features.extend(np.bincount(part, minlength=21) / max(1, len(part)))
        for lag in (1, 2, 4, 8):
            valid = (
                known[:-lag] & known[lag:]
                if len(codes) > lag
                else np.zeros(0, dtype=bool)
            )
            if valid.any():
                products = (
                    PROPERTIES[codes[:-lag][valid]] * PROPERTIES[codes[lag:][valid]]
                )
                features.extend(products.mean(axis=0))
            else:
                features.extend(np.zeros(5))
        output.append(features)
    return np.array(output, dtype=np.float64)


def project_to_reference(reference: str, sequence: str) -> np.ndarray:
    # Gap code 21 marks uncovered positions; insertions are omitted.
    from Bio.Align import PairwiseAligner

    residue_codes(reference)
    query = residue_codes(sequence)
    aligner = PairwiseAligner(
        mode="global",
        match_score=2,
        mismatch_score=-1,
        open_gap_score=-4,
        extend_gap_score=-0.2,
    )
    alignment = aligner.align(reference, sequence)[0]
    projected = np.full(len(reference), 21, dtype=np.int16)
    for (a, b), (c, d) in zip(*alignment.aligned):
        if b - a != d - c:
            raise ValueError("Unequal aligned pairwise blocks")
        projected[a:b] = query[c:d]
    return projected


def position_matrix(projected: np.ndarray) -> sparse.csr_matrix:
    if projected.ndim != 2 or np.any((projected < 0) | (projected > 21)):
        raise ValueError("Expected row-by-position residue codes0–21")
    rows = np.repeat(np.arange(len(projected)), projected.shape[1])
    columns = (np.arange(projected.shape[1])[None, :] * 22 + projected).reshape(-1)
    return sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(projected), projected.shape[1] * 22),
    )
