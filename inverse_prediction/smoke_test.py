"""Validate parsing, scaling, and a real-size inverse optimization step."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from inverse_model import InverseRegressor, ParameterBounds, model_parameter_count, parse_parameters


def main() -> None:
    assert parse_parameters("c3000_phi18.00_logk-2.300000.png") == (3000.0, 18.0, -2.3)
    assert parse_parameters("3000_18.00_-2.30.png") == (3000.0, 18.0, -2.3)
    bounds = ParameterBounds()
    raw = torch.tensor([[3000.0, 18.0, -2.3], [7000.0, 27.0, -0.3]])
    normalized = bounds.normalize(raw)
    assert torch.allclose(normalized, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert torch.allclose(bounds.denormalize(normalized), raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InverseRegressor(256, 512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    images = torch.rand(4, 3, 256, 512, device=device)
    targets = torch.rand(4, 3, device=device)
    predictions = model(images)
    loss = F.mse_loss(predictions, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert predictions.shape == (4, 3)
    assert torch.isfinite(loss)
    print(
        f"Smoke test passed on {device}: output={tuple(predictions.shape)}, "
        f"parameters={model_parameter_count(model):,}, loss={float(loss.detach()):.6f}"
    )


if __name__ == "__main__":
    main()
