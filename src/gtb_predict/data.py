import csv
import io
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "bb6b16c6bea4d6987bfa388edb77f734d00eb445"
BASE_URL = (
    "https://raw.githubusercontent.com/vbharadwcomosci/GTB_substrate_prediction/"
    + UPSTREAM_COMMIT
    + "/Data/"
)
TRAINING_FILES = {
    "train.csv": "CSV_Files/train.csv",
    "full.fasta": "Align_Files/train/fasta_long.fasta",
    "domain.fasta": "Align_Files/train/fasta_short.fasta",
}
DONORS = (
    "GDP-alpha-D-mannose",
    "GDP-beta-L-fucose",
    "UDP-alpha-D-galactose",
    "UDP-alpha-D-glucose",
    "UDP-alpha-D-glucuronate",
    "UDP-alpha-D-xylose",
    "UDP-beta-L-rhamnose",
    "Eight",
)


def fetch_text(url: str, *, max_bytes: int = 2_000_000) -> str:
    with urllib.request.urlopen(url, timeout=90) as response:
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"Source exceeds {max_bytes} bytes: {url}")
    return payload.decode("utf-8")


def parse_fasta(text: str) -> dict[str, str]:
    sequences = {}
    accession = None
    for line in text.splitlines():
        if line.startswith(">"):
            fields = line[1:].split()[0].split("|")
            accession = fields[1] if len(fields) == 3 else fields[0]
            if accession in sequences:
                raise ValueError(f"Duplicate FASTA accession: {accession}")
            sequences[accession] = ""
        elif line.strip():
            if accession is None:
                raise ValueError("Sequence before FASTA header")
            sequences[accession] += line.strip().upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")
    if not sequences or any(not s or set(s) - allowed for s in sequences.values()):
        raise ValueError("Missing sequence or unsupported FASTA characters")
    return sequences


def read_training(directory: Path) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    with (directory / "train.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["ID"] for row in rows]
    if len(rows) != 413 or len(set(ids)) != 413:
        raise ValueError("Expected exactly 413 unique released training accessions")
    if any(set(row["Donor"].split(",,")) - set(DONORS) for row in rows):
        raise ValueError("Unrecognized training donor")
    sequences = []
    for filename in ("full.fasta", "domain.fasta"):
        entries = parse_fasta((directory / filename).read_text(encoding="utf-8"))
        if set(ids) != set(entries):
            raise ValueError(f"Training accession mismatch: {filename}")
        sequences.append(entries)
    return rows, sequences[0], sequences[1]


def prepare_training(directory: Path) -> dict:
    # Download only the released training partition and its matching sequences.
    directory.mkdir(parents=True, exist_ok=True)
    csv_text = fetch_text(BASE_URL + TRAINING_FILES["train.csv"])
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    ids = [row["ID"] for row in rows]
    if len(ids) != 413 or len(set(ids)) != 413:
        raise ValueError("Unexpected upstream training partition")
    (directory / "train.csv").write_text(csv_text, encoding="utf-8")
    for filename in ("full.fasta", "domain.fasta"):
        entries = parse_fasta(fetch_text(BASE_URL + TRAINING_FILES[filename]))
        missing = set(ids) - entries.keys()
        if missing:
            raise ValueError(
                f"Missing training sequences in {filename}: {sorted(missing)}"
            )
        text = "".join(f">{accession}\n{entries[accession]}\n" for accession in ids)
        (directory / filename).write_text(text, encoding="utf-8")
    read_training(directory)
    return {"training_rows": len(rows), "commit": UPSTREAM_COMMIT}
