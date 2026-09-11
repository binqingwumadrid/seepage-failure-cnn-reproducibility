"""Fast shape, loss, metric, parser, and checkpoint-free inference smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from forward_model import (
    CompositeImageLoss,
    ForwardDecoder,
    ParameterBounds,
    parse_parameters,
    per_sample_metrics,
    tensor_to_image,
)


def main() -> None:
    assert parse_parameters("c3000_phi18.00_logk-2.300000.png") == (3000.0, 18.0, -2.3)
    bounds = ParameterBounds()
    raw = torch.tensor([[3000.0, 18.0, -2.3], [7000.0, 27.0, -0.3]])
    normalized = bounds.normalize(raw)
    assert torch.allclose(normalized, torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))

    model = ForwardDecoder(image_height=16, image_width=32)
    prediction = model(torch.zeros(2, 3))
    target = torch.rand_like(prediction)
    assert prediction.shape == (2, 3, 16, 32)
    loss = CompositeImageLoss()(prediction, target)
    loss.backward()
    metrics = per_sample_metrics(prediction.detach(), target)
    assert all(values.shape == (2,) for values in metrics.values())
    assert torch.isfinite(loss)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "prediction.png"
        tensor_to_image(prediction[0]).save(output)
        assert output.is_file()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
