"""RealBasicVSR runs its per-frame parts (optical flow, cleaning) in small groups to bound memory."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neuralframe.inference.archs import basicvsr  # noqa: E402


@pytest.mark.parametrize("frames", [1, 4, 5, 11])
def test_grouped_processing_matches_whole_chunk(frames: int, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    net = basicvsr.RealBasicVSRNet(mid_channels=16, num_propagation_blocks=2, num_cleaning_blocks=2).eval()
    x = torch.rand(1, frames, 3, 32, 48)
    with torch.no_grad():
        monkeypatch.setattr(basicvsr, "FRAME_GROUP", 4)
        grouped = net(x)
        monkeypatch.setattr(basicvsr, "FRAME_GROUP", 10**6)
        whole = net(x)
    assert grouped.shape == (1, frames, 3, 128, 192)
    torch.testing.assert_close(grouped, whole, atol=1e-6, rtol=0)


def test_cleaning_stops_on_mean_residue_over_all_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    net = basicvsr.RealBasicVSRNet(mid_channels=8, num_propagation_blocks=1, num_cleaning_blocks=1,
                                   dynamic_refine_thres=0.0).eval()
    calls: list[int] = []
    original = net.image_cleaning.forward

    def counting(x: torch.Tensor) -> torch.Tensor:
        calls.append(x.shape[0])
        return original(x)

    monkeypatch.setattr(net.image_cleaning, "forward", counting)
    monkeypatch.setattr(basicvsr, "FRAME_GROUP", 4)
    with torch.no_grad():
        net.clean(torch.rand(1, 6, 3, 16, 16))
    # Threshold 0 never stops early: three passes over 6 frames in groups of 4 and 2.
    assert calls == [4, 2, 4, 2, 4, 2]
