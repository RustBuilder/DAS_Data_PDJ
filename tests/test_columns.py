import numpy as np

from scripts.pipeline import _parse_columns


def test_parse_paired_amplitude_phase_columns() -> None:
    header = [
        "id",
        "amp_loc_540",
        "phase_loc_540",
        "amp_loc_541",
        "phase_loc_541",
    ]
    amplitude, phase, locations = _parse_columns(header)
    assert amplitude == ["amp_loc_540", "amp_loc_541"]
    assert phase == ["phase_loc_540", "phase_loc_541"]
    np.testing.assert_array_equal(locations, [540, 541])
