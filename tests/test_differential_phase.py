import numpy as np

from scripts.pipeline import _circular_differential_phase


def test_circular_differential_phase_unwraps_over_time() -> None:
    sample = np.arange(2000, dtype=np.float32)
    expected = 0.02 * sample + 0.4 * np.sin(2 * np.pi * sample / 73)
    left = 3.0 + 0.01 * sample
    right = left + expected
    phase = np.column_stack([left, np.zeros_like(left), right]).astype(np.float32)

    recovered = _circular_differential_phase(phase, gauge_points=2)[:, 0]

    recovered -= recovered[0] - expected[0]
    np.testing.assert_allclose(recovered, expected, atol=2e-5)


def test_circular_differential_phase_rejects_invalid_gauge() -> None:
    phase = np.zeros((10, 3), dtype=np.float32)

    for gauge_points in (0, 3):
        try:
            _circular_differential_phase(phase, gauge_points)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid gauge length was accepted")
