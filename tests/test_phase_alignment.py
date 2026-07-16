import numpy as np

from scripts.pipeline import PhaseBlockAligner


def test_block_alignment_recovers_continuity() -> None:
    samples_per_block = 500
    sample_count = 1500
    sample = np.arange(sample_count, dtype=np.float32)
    true_phase = np.column_stack(
        [
            0.012 * sample + 0.3 * np.sin(2 * np.pi * sample / 80),
            -0.008 * sample + 0.2 * np.sin(2 * np.pi * sample / 55),
        ]
    ).astype(np.float32)
    aligner = PhaseBlockAligner(edge_samples=32, apply_unwrap=False)
    recovered = []
    for start in range(0, sample_count, samples_per_block):
        block = true_phase[start : start + samples_per_block].copy()
        block -= block[0]
        aligned, _ = aligner.align(block)
        recovered.append(aligned)
    recovered_phase = np.vstack(recovered)

    boundary_steps = np.diff(recovered_phase, axis=0)[
        [samples_per_block - 1, 2 * samples_per_block - 1]
    ]
    assert np.max(np.abs(boundary_steps)) < 0.05
    assert np.sqrt(np.mean((recovered_phase - true_phase) ** 2)) < 0.02


def test_default_alignment_preserves_already_unwrapped_values() -> None:
    block = np.array([[0.0], [4.0], [8.0], [12.0]], dtype=np.float32)
    aligned, _ = PhaseBlockAligner(apply_unwrap=False).align(block)
    np.testing.assert_allclose(aligned, block)
