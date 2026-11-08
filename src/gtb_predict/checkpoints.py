from pathlib import Path

import torch
from torch import nn

from .compact import CompactProtein
from .low_rank import LowRankLinear

HEADS = ("classifier.", "reconstruct.", "coherence.")


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    torch.save(payload, temporary)
    temporary.replace(path)


def strip_heads(model: CompactProtein) -> None:
    for name in ("reconstruct", "coherence", "classifier"):
        setattr(model, name, nn.Identity())


def load_encoder(path: Path, *, device: str) -> CompactProtein:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = CompactProtein(saved["width"], saved["layers"])
    strip_heads(model)
    if "rank" in saved:
        for block in model.blocks:
            block.inward = LowRankLinear(
                model.width, model.width, saved["rank"], bias=True
            )
            block.outward = LowRankLinear(
                model.width // 2, model.width, saved["rank"], bias=True
            )
    state = saved["encoder"] if "encoder" in saved else saved["model"]
    model.load_state_dict(
        {key: value for key, value in state.items() if not key.startswith(HEADS)}
    )
    return model.to(device).eval()


def save_random_encoder(path: Path, *, seed: int) -> None:
    # Initialize all original heads before stripping them to preserve the control state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CompactProtein(512, 16)
    strip_heads(model)
    save_checkpoint(
        path,
        {
            "model": cpu_state(model),
            "width": 512,
            "layers": 16,
            "seed": seed,
            "configuration": "random_6m",
            "heads_stripped": True,
        },
    )
