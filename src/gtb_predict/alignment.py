import subprocess
from pathlib import Path

import numpy as np


def write_fasta(path: Path, ids: list[str], sequences: dict[str, str]) -> None:
    path.write_text(
        "".join(f">{name}\n{sequences[name]}\n" for name in ids), encoding="utf-8"
    )


def read_alignment(path: Path) -> dict[str, str]:
    records = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            if name in records:
                raise ValueError("Duplicate alignment identifier")
            records[name] = ""
        elif line:
            if name is None:
                raise ValueError("Alignment sequence precedes header")
            records[name] += line.strip()
    if not records or len({len(s) for s in records.values()}) != 1:
        raise ValueError("Missing or ragged alignment")
    return records


def fit_alignment(ids: list[str], sequences: dict[str, str], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    source, target = directory / "input.fasta", directory / "reference.fasta"
    write_fasta(source, ids, sequences)
    subprocess.run(
        [
            "clustalo",
            "-i",
            str(source),
            "-o",
            str(target),
            "--outfmt=fasta",
            "--threads=2",
            "--force",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = read_alignment(target)
    if set(records) != set(ids):
        raise ValueError("Fitting alignment accession mismatch")
    return target


def project_query(
    reference: dict[str, str], combined: dict[str, str], query: str
) -> tuple[str, np.ndarray]:
    # Project onto the full fitting profile and retain original query positions.
    ids = list(reference)
    matrix = np.array([list(combined[name]) for name in ids])
    keep = np.any(matrix != "-", axis=0)
    for name in ids:
        if "".join(np.array(list(combined[name]))[keep]) != reference[name]:
            raise ValueError("Clustal profile changed fitting residue columns")
    query_array = np.array(list(combined[query]))
    positions = np.cumsum(query_array != "-") - 1
    positions[query_array == "-"] = -1
    return "".join(query_array[keep]), positions[keep]


def align_query(
    reference_path: Path, query: str, sequence: str, directory: Path
) -> tuple[str, np.ndarray]:
    source, target = (
        directory / f"query_{query}.fasta",
        directory / f"aligned_{query}.fasta",
    )
    source.write_text(f">{query}\n{sequence}\n")
    subprocess.run(
        [
            "clustalo",
            "--profile1",
            str(reference_path),
            "--profile2",
            str(source),
            "-o",
            str(target),
            "--outfmt=fasta",
            "--threads=1",
            "--force",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return project_query(read_alignment(reference_path), read_alignment(target), query)


def identity_matrix(records: dict[str, str], ids: list[str]) -> np.ndarray:
    # Identity uses the shorter ungapped sequence as its denominator.
    array = np.array([list(records[name]) for name in ids], dtype="S1")
    lengths = (array != b"-").sum(axis=1)
    result = np.empty((len(ids), len(ids)))
    for i, row in enumerate(array):
        matches = ((array == row) & (array != b"-") & (row != b"-")).sum(axis=1)
        result[i] = matches / np.minimum(lengths, lengths[i])
    return result


def connected_groups(similarity: np.ndarray, threshold: float) -> np.ndarray:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Expected a square similarity matrix")
    if not np.isfinite(similarity).all() or not np.allclose(similarity, similarity.T):
        raise ValueError("Expected finite symmetric sequence similarities")
    parents = list(range(len(similarity)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in zip(*np.where(np.triu(similarity >= threshold, k=1))):
        parents[find(int(right))] = find(int(left))
    roots = [find(i) for i in range(len(parents))]
    return np.unique(roots, return_inverse=True)[1]
