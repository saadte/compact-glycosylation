import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .sequence import residue_codes


def token_batch(sequences: list[str], *, device: str) -> torch.Tensor:
    # Token codes: 0 padding, 1 mask, 2 reserved, 3-22 amino acids, 23 unknown.
    result = np.zeros((len(sequences), max(map(len, sequences))), dtype=np.int64)
    for i, sequence in enumerate(sequences):
        result[i, : len(sequence)] = residue_codes(sequence) + 3
    return torch.as_tensor(result, device=device)


class ResidualConvolution(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        hidden = width // 2
        self.norm = nn.LayerNorm(width)
        self.inward = nn.Linear(width, hidden * 2)
        self.convolution = nn.Conv1d(
            hidden, hidden, 5, padding=2 * dilation, dilation=dilation, groups=hidden
        )
        self.outward = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        hidden = F.glu(self.inward(self.norm(values)), dim=-1) * valid
        hidden = F.gelu(self.convolution(hidden.transpose(1, 2)).transpose(1, 2))
        return (values + self.dropout(self.outward(hidden))) * valid


class CompactProtein(nn.Module):
    def __init__(self, width: int, layers: int, dropout: float = 0.1):
        super().__init__()
        self.width, self.layers = width, layers
        self.embedding = nn.Embedding(24, width, padding_idx=0)
        self.position = nn.Linear(2, width, bias=False)
        self.blocks = nn.ModuleList(
            [ResidualConvolution(width, 2 ** (i % 6), dropout) for i in range(layers)]
        )
        self.norm = nn.LayerNorm(width)
        self.reconstruct = nn.Linear(width, 20)
        self.coherence = nn.Sequential(
            nn.Linear(2 * width, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * width, 8))

    def residues(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = tokens != 0
        lengths = valid.sum(dim=1, keepdim=True)
        if torch.any(lengths == 0):
            raise ValueError("Empty protein input")
        position = (
            torch.arange(tokens.shape[1], device=tokens.device)[None, :] / lengths
        )
        coordinates = torch.stack([position, 1 - position], dim=-1)
        mask = valid[:, :, None]
        hidden = (self.embedding(tokens) + self.position(coordinates)) * mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.norm(hidden) * mask, valid

    def representation(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        lengths = valid.sum(dim=1, keepdim=True)
        position = torch.arange(valid.shape[1], device=valid.device)[None, :]
        pooled = [(hidden * valid[:, :, None]).sum(dim=1) / lengths]
        for j in range(3):
            part = (
                valid
                & (3 * position >= j * lengths)
                & (3 * position < (j + 1) * lengths)
            )
            pooled.append(
                (hidden * part[:, :, None]).sum(dim=1)
                / part.sum(dim=1, keepdim=True).clamp_min(1)
            )
        return torch.cat(pooled, dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, valid = self.residues(tokens)
        return self.classifier(self.representation(hidden, valid))


def pretraining_loss(
    model: CompactProtein,
    tokens: torch.Tensor,
    *,
    random: np.random.Generator,
    coherence_weight: float,
) -> tuple[torch.Tensor, dict]:
    # Mask 15% of canonical residues in spans of 3-7, with 80/10/10 corruption.
    original = tokens.detach().cpu().numpy()
    canonical = (original >= 3) & (original <= 22)
    selected_array = np.zeros(original.shape, dtype=bool)
    for i in range(len(original)):
        if not canonical[i].any():
            raise ValueError("A pretraining input has no canonical residues")
        count = max(1, round(int(canonical[i].sum()) * 0.15))
        while selected_array[i].sum() < count:
            available = np.flatnonzero(canonical[i] & ~selected_array[i])
            start = int(random.choice(available))
            span = int(random.integers(3, 8))
            proposed = np.arange(start, min(original.shape[1], start + span))
            proposed = proposed[canonical[i, proposed] & ~selected_array[i, proposed]]
            needed = count - int(selected_array[i].sum())
            selected_array[i, proposed[:needed]] = True
    selected = torch.as_tensor(selected_array, device=tokens.device)
    if not selected.any():
        raise ValueError("No canonical masked targets")
    labels = (tokens - 3).masked_fill(~selected, -100)
    noise = torch.as_tensor(random.random(tokens.shape), device=tokens.device)
    corrupted = tokens.clone()
    corrupted[selected & (noise < 0.8)] = 1
    mutations = torch.as_tensor(
        random.integers(3, 23, tokens.shape), device=tokens.device
    )
    mutate = selected & (noise >= 0.8) & (noise < 0.9)
    corrupted[mutate] = mutations[mutate]
    hidden, valid = model.residues(corrupted)
    mlm = F.cross_entropy(model.reconstruct(hidden).reshape(-1, 20), labels.reshape(-1))
    coherence = torch.zeros((), device=tokens.device)
    if coherence_weight:
        lengths = valid.sum(dim=1, keepdim=True)
        position = torch.arange(valid.shape[1], device=tokens.device)[None, :]
        left_mask, right_mask = (
            valid & (2 * position < lengths),
            valid & (2 * position >= lengths),
        )
        left = (hidden * left_mask[:, :, None]).sum(1) / left_mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        right = (hidden * right_mask[:, :, None]).sum(1) / right_mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        logits = torch.cat(
            [
                model.coherence(torch.cat([left, right], 1)).flatten(),
                model.coherence(torch.cat([left, right.roll(1, 0)], 1)).flatten(),
            ]
        )
        coherence = F.binary_cross_entropy_with_logits(
            logits,
            torch.cat(
                [
                    torch.ones(len(tokens), device=tokens.device),
                    torch.zeros(len(tokens), device=tokens.device),
                ]
            ),
        )
    return mlm + coherence_weight * coherence, {
        "masked_cross_entropy": float(mlm.detach()),
        "coherence_loss": float(coherence.detach()),
        "input_tokens": int(valid.sum()),
        "masked_tokens": int(selected.sum()),
    }
