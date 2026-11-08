import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, *, bias: bool):
        super().__init__()
        if not 0 < rank <= min(in_features, out_features):
            raise ValueError(
                "Rank must be positive and no larger than either dimension"
            )
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(values))


def factorize_linear(
    layer: nn.Linear, *, rank: int, output_second_moment: torch.Tensor | None = None
) -> LowRankLinear:
    # Calibrated moments are E[(Wx)(Wx)^T], excluding bias and padding.
    # Decomposition uses float64; returned factors retain the original device and dtype.
    if not 0 < rank <= min(layer.in_features, layer.out_features):
        raise ValueError("Rank must be positive and no larger than either dimension")
    weight = layer.weight.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(weight).all():
        raise ValueError("Projection weights must be finite")
    if output_second_moment is None:
        u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
        scale = singular[:rank].sqrt()
        up = u[:, :rank] * scale
        down = scale[:, None] * vh[:rank]
    else:
        moment = output_second_moment.detach().to(device="cpu", dtype=torch.float64)
        expected = (layer.out_features, layer.out_features)
        if moment.shape != expected or not torch.isfinite(moment).all():
            raise ValueError("Output moment has invalid dimensions or nonfinite values")
        magnitude = float(moment.abs().max())
        tolerance = 1e-6 * max(magnitude, 1e-12)
        if float((moment - moment.T).abs().max()) > tolerance:
            raise ValueError("Output moment must be symmetric")
        eigenvalues, eigenvectors = torch.linalg.eigh((moment + moment.T) / 2)
        if float(eigenvalues[0]) < -tolerance * layer.out_features:
            raise ValueError("Output moment must be positive semidefinite")
        up = eigenvectors[:, -rank:]
        down = up.T @ weight
    result = LowRankLinear(
        layer.in_features, layer.out_features, rank, bias=layer.bias is not None
    ).to(device=layer.weight.device, dtype=layer.weight.dtype)
    with torch.no_grad():
        result.down.weight.copy_(down)
        result.up.weight.copy_(up)
        if layer.bias is not None:
            result.up.bias.copy_(layer.bias)
    return result
