from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import signal
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REFERENCE_URLS = {
    "daspy_basic_processing": (
        "https://daspy-tutorial.readthedocs.io/en/latest/Basic%20Processing.html"
    ),
    "daspy_denoising": (
        "https://daspy-tutorial.readthedocs.io/en/latest/Denoising.html"
    ),
    "iq_phase_demodulation": (
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6749300/"
    ),
    "differential_unwrap_integral": (
        "https://opg.optica.org/jlt/abstract.cfm?uri=jlt-39-22-7274"
    ),
    "incremental_phase_unwrap": (
        "https://www.mdpi.com/1424-8220/25/10/3218"
    ),
}

GAUGE_LENGTH_POINTS = 10


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ProcessingConfig:
    sample_rate_hz: float
    block_rows: int
    block_duration_s: float
    phase_block_unwrap: bool
    phase_unwrap_discont_rad: float
    align_edge_samples: int
    bandpass_low_hz: float
    bandpass_high_hz: float
    filter_order: int
    common_mode_removal: bool
    gauge_length_points: int
    csv_chunk_rows: int
    filter_channel_batch: int
    plot_max_time_points: int
    hdf5_compression: str
    hdf5_compression_level: int

    @classmethod
    def from_environment(cls, project_root: Path) -> "ProcessingConfig":
        load_dotenv(project_root / ".env", override=False)
        return cls(
            sample_rate_hz=float(os.getenv("SAMPLE_RATE_HZ", "10000")),
            block_rows=int(os.getenv("BLOCK_ROWS", "5000")),
            block_duration_s=float(os.getenv("BLOCK_DURATION_S", "0.5")),
            phase_block_unwrap=_env_bool("PHASE_BLOCK_UNWRAP", False),
            phase_unwrap_discont_rad=float(
                os.getenv("PHASE_UNWRAP_DISCONT_RAD", str(math.pi))
            ),
            align_edge_samples=int(os.getenv("ALIGN_EDGE_SAMPLES", "64")),
            bandpass_low_hz=float(os.getenv("BANDPASS_LOW_HZ", "5")),
            bandpass_high_hz=float(os.getenv("BANDPASS_HIGH_HZ", "2000")),
            filter_order=int(os.getenv("FILTER_ORDER", "4")),
            common_mode_removal=_env_bool("COMMON_MODE_REMOVAL", True),
            gauge_length_points=GAUGE_LENGTH_POINTS,
            csv_chunk_rows=int(os.getenv("CSV_CHUNK_ROWS", "5000")),
            filter_channel_batch=int(os.getenv("FILTER_CHANNEL_BATCH", "12")),
            plot_max_time_points=int(os.getenv("PLOT_MAX_TIME_POINTS", "3000")),
            hdf5_compression=os.getenv("HDF5_COMPRESSION", "gzip"),
            hdf5_compression_level=int(
                os.getenv("HDF5_COMPRESSION_LEVEL", "4")
            ),
        )


class PhaseBlockAligner:
    """Align block-relative phase using robust one-sample boundary prediction."""

    def __init__(
        self,
        edge_samples: int = 64,
        apply_unwrap: bool = False,
        unwrap_discont_rad: float = math.pi,
    ) -> None:
        self.edge_samples = edge_samples
        self.apply_unwrap = apply_unwrap
        self.unwrap_discont_rad = unwrap_discont_rad
        self.previous_tail: np.ndarray | None = None
        self.previous_last: np.ndarray | None = None

    def align(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phase = np.asarray(block, dtype=np.float32)
        if phase.ndim != 2 or phase.shape[0] < 2:
            raise ValueError("phase block must have shape (time, location)")

        if self.apply_unwrap:
            phase = np.unwrap(
                phase,
                discont=self.unwrap_discont_rad,
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            phase = phase.copy()

        if self.previous_last is None or self.previous_tail is None:
            offset = np.zeros(phase.shape[1], dtype=np.float32)
            aligned = phase
        else:
            edge = min(self.edge_samples, len(self.previous_tail), len(phase))
            previous_steps = np.diff(self.previous_tail[-edge:], axis=0)
            current_steps = np.diff(phase[:edge], axis=0)
            previous_slope = np.median(previous_steps, axis=0)
            current_slope = np.median(current_steps, axis=0)
            expected_step = 0.5 * (previous_slope + current_slope)
            offset = self.previous_last + expected_step - phase[0]
            aligned = phase + offset

        keep = min(self.edge_samples, len(aligned))
        self.previous_tail = aligned[-keep:].copy()
        self.previous_last = aligned[-1].copy()
        return aligned.astype(np.float32, copy=False), offset.astype(
            np.float32, copy=False
        )


def _circular_differential_phase(
    phase: np.ndarray,
    gauge_points: int,
) -> np.ndarray:
    if phase.ndim != 2:
        raise ValueError("phase must have shape (time, location)")
    if not 0 < gauge_points < phase.shape[1]:
        raise ValueError("gauge_points must be within the phase location axis")
    difference = phase[:, gauge_points:] - phase[:, :-gauge_points]
    wrapped = np.remainder(difference + math.pi, 2.0 * math.pi) - math.pi
    return np.unwrap(wrapped, axis=0).astype(np.float32, copy=False)


def _load_metadata(dataset_dir: Path) -> dict[str, Any]:
    metadata_path = dataset_dir / "capture_info.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return handle.readline().rstrip("\r\n").split(",")


def _parse_columns(header: Iterable[str]) -> tuple[list[str], list[str], np.ndarray]:
    amplitude_columns: list[str] = []
    phase_columns: list[str] = []
    locations: list[int] = []
    for column in header:
        if column.startswith("amp_loc_"):
            amplitude_columns.append(column)
            locations.append(int(column.removeprefix("amp_loc_")))
        elif column.startswith("phase_loc_"):
            phase_columns.append(column)
    expected_phase = [f"phase_loc_{location}" for location in locations]
    if phase_columns != expected_phase:
        raise ValueError("Amplitude and phase columns are not paired by location")
    return amplitude_columns, phase_columns, np.asarray(locations, dtype=np.int32)


def _distance_axis(metadata: dict[str, Any], count: int) -> np.ndarray:
    selection = metadata["selection"]
    start = float(selection["distance_start_m"])
    spacing = float(selection["sample_spacing_m"])
    return (start + spacing * np.arange(count)).astype(np.float32)


def _compression_kwargs(config: ProcessingConfig) -> dict[str, Any]:
    if config.hdf5_compression.lower() == "gzip":
        return {
            "compression": "gzip",
            "compression_opts": config.hdf5_compression_level,
            "shuffle": True,
        }
    if config.hdf5_compression.lower() in {"lzf", "szip"}:
        return {"compression": config.hdf5_compression.lower(), "shuffle": True}
    return {}


def _create_matrix_dataset(
    group: h5py.Group,
    name: str,
    shape: tuple[int, int],
    config: ProcessingConfig,
) -> h5py.Dataset:
    chunk_rows = min(config.block_rows, shape[0])
    return group.create_dataset(
        name,
        shape=shape,
        dtype="float32",
        chunks=(chunk_rows, shape[1]),
        **_compression_kwargs(config),
    )


def _import_csv_to_hdf5(
    csv_path: Path,
    metadata: dict[str, Any],
    h5_path: Path,
    config: ProcessingConfig,
) -> dict[str, Any]:
    header = _read_header(csv_path)
    amplitude_columns, phase_columns, locations = _parse_columns(header)
    rows = int(metadata["rows"])
    columns = int(metadata["columns"])
    if columns != len(header):
        raise ValueError(f"Metadata columns={columns}, CSV header={len(header)}")
    if config.csv_chunk_rows != config.block_rows:
        raise ValueError("CSV_CHUNK_ROWS must equal BLOCK_ROWS for reset alignment")

    distance_m = _distance_axis(metadata, len(locations))
    gauge_points = config.gauge_length_points
    if not 0 < gauge_points < len(locations):
        raise ValueError(
            f"GAUGE_LENGTH_POINTS={gauge_points} must be between 1 and "
            f"{len(locations) - 1}"
        )
    differential_left_locations = locations[:-gauge_points]
    differential_right_locations = locations[gauge_points:]
    differential_center_locations = (
        differential_left_locations.astype(np.float64)
        + differential_right_locations.astype(np.float64)
    ) / 2.0
    differential_distance_m = (
        distance_m[:-gauge_points].astype(np.float64)
        + distance_m[gauge_points:].astype(np.float64)
    ) / 2.0
    dtype_map = {column: np.float32 for column in header}
    dtype_map["id"] = np.int64
    point_phase_aligner = PhaseBlockAligner(
        edge_samples=config.align_edge_samples,
        apply_unwrap=config.phase_block_unwrap,
        unwrap_discont_rad=config.phase_unwrap_discont_rad,
    )
    differential_phase_aligner = PhaseBlockAligner(
        edge_samples=config.align_edge_samples,
        apply_unwrap=False,
    )
    phase_min = math.inf
    phase_max = -math.inf
    expected_phase_quantum = math.pi / 1024
    phase_quantum_estimates: list[float] = []
    point_offsets: list[np.ndarray] = []
    differential_offsets: list[np.ndarray] = []
    block_starts: list[int] = []
    expected_id = 0

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema_version"] = 1
        h5.attrs["source_csv"] = str(csv_path)
        h5.attrs["source_metadata"] = str(csv_path.parent / "capture_info.json")
        h5.attrs["sample_rate_hz"] = float(metadata["acquisition"]["csv_row_rate_hz"])
        h5.attrs["block_rows"] = int(metadata["acquisition"]["block_rows"])
        h5.attrs["phase_unit"] = "radian"
        h5.attrs["amplitude_unit"] = "sdk_native_unspecified"
        h5.attrs["phase_source"] = "BlockIQ.phase(0), SDK-demodulated"
        h5.attrs["phase_alignment"] = "robust_boundary_continuity"
        h5.attrs["das_measurement"] = "circular_spatial_differential_phase"
        h5.attrs["gauge_length_points"] = gauge_points
        h5.attrs["gauge_length_m"] = float(
            gauge_points * metadata["selection"]["sample_spacing_m"]
        )
        h5.attrs["additional_block_unwrap"] = config.phase_block_unwrap

        axes = h5.create_group("axes")
        axes.create_dataset(
            "time_s",
            data=np.arange(rows, dtype=np.float64)
            / float(metadata["acquisition"]["csv_row_rate_hz"]),
        )
        axes.create_dataset("location_index", data=locations)
        axes.create_dataset("distance_m", data=distance_m)
        axes.create_dataset(
            "differential_left_location_index", data=differential_left_locations
        )
        axes.create_dataset(
            "differential_right_location_index", data=differential_right_locations
        )
        axes.create_dataset(
            "differential_center_location_index",
            data=differential_center_locations.astype(np.float32),
        )
        axes.create_dataset(
            "differential_center_distance_m",
            data=differential_distance_m.astype(np.float32),
        )

        amplitude_group = h5.create_group("amplitude")
        phase_group = h5.create_group("phase")
        amplitude_raw = _create_matrix_dataset(
            amplitude_group, "raw_sdk_native", (rows, len(locations)), config
        )
        phase_raw = _create_matrix_dataset(
            phase_group, "raw_block_relative_rad", (rows, len(locations)), config
        )
        phase_aligned = _create_matrix_dataset(
            phase_group, "aligned_rad", (rows, len(locations)), config
        )
        differential_phase_aligned = _create_matrix_dataset(
            phase_group,
            "differential_aligned_rad",
            (rows, len(differential_center_locations)),
            config,
        )
        differential_phase_aligned.attrs["processing"] = (
            "right-minus-left circular gauge difference; temporal unwrap inside "
            "each BlockIQ block; robust additive alignment between blocks"
        )

        reader = pd.read_csv(
            csv_path,
            dtype=dtype_map,
            chunksize=config.csv_chunk_rows,
            engine="c",
            memory_map=True,
        )
        total_chunks = math.ceil(rows / config.csv_chunk_rows)
        write_row = 0
        for chunk in tqdm(
            reader,
            total=total_chunks,
            desc=f"Import {csv_path.stem}",
            unit="block",
        ):
            row_count = len(chunk)
            ids = chunk["id"].to_numpy(dtype=np.int64, copy=False)
            expected = np.arange(expected_id, expected_id + row_count, dtype=np.int64)
            if not np.array_equal(ids, expected):
                raise ValueError(f"Non-contiguous ids near row {write_row}")

            amplitude = chunk[amplitude_columns].to_numpy(
                dtype=np.float32, copy=False
            )
            raw_phase = chunk[phase_columns].to_numpy(dtype=np.float32, copy=False)
            aligned_phase, point_offset = point_phase_aligner.align(raw_phase)
            differential_phase = _circular_differential_phase(
                raw_phase,
                gauge_points,
            )
            aligned_differential_phase, differential_offset = (
                differential_phase_aligner.align(differential_phase)
            )
            row_slice = slice(write_row, write_row + row_count)
            amplitude_raw[row_slice] = amplitude
            phase_raw[row_slice] = raw_phase
            phase_aligned[row_slice] = aligned_phase
            differential_phase_aligned[row_slice] = aligned_differential_phase

            phase_min = min(phase_min, float(np.min(raw_phase)))
            phase_max = max(phase_max, float(np.max(raw_phase)))
            absolute_phase = np.abs(raw_phase)
            quantum_candidates = absolute_phase[
                (absolute_phase >= 0.5 * expected_phase_quantum)
                & (absolute_phase <= 1.5 * expected_phase_quantum)
            ]
            if quantum_candidates.size:
                phase_quantum_estimates.append(float(np.median(quantum_candidates)))
            point_offsets.append(point_offset)
            differential_offsets.append(differential_offset)
            block_starts.append(write_row)
            write_row += row_count
            expected_id += row_count

        if write_row != rows:
            raise ValueError(f"Expected {rows} rows, imported {write_row}")
        phase_group.create_dataset(
            "block_offsets_rad", data=np.stack(point_offsets).astype(np.float32)
        )
        phase_group.create_dataset(
            "differential_block_offsets_rad",
            data=np.stack(differential_offsets).astype(np.float32),
        )
        phase_group.create_dataset(
            "block_start_rows", data=np.asarray(block_starts, dtype=np.int64)
        )

    return {
        "locations": locations,
        "distance_m": distance_m,
        "phase_min_rad": phase_min,
        "phase_max_rad": phase_max,
        "observed_phase_quantization_rad": float(
            np.median(phase_quantum_estimates)
            if phase_quantum_estimates
            else expected_phase_quantum
        ),
        "block_offsets_rad": np.stack(point_offsets).astype(np.float32),
        "differential_block_offsets_rad": np.stack(differential_offsets).astype(
            np.float32
        ),
        "gauge_length_points": gauge_points,
        "gauge_length_m": float(
            gauge_points * metadata["selection"]["sample_spacing_m"]
        ),
    }


def _detrend_in_batches(data: np.ndarray, batch_size: int) -> np.ndarray:
    output = np.empty_like(data, dtype=np.float32)
    for start in range(0, data.shape[1], batch_size):
        stop = min(start + batch_size, data.shape[1])
        output[:, start:stop] = signal.detrend(
            data[:, start:stop], axis=0, type="linear"
        ).astype(np.float32, copy=False)
    return output


def _remove_common_mode(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    common = np.median(data, axis=1).astype(np.float32, copy=False)
    denominator = float(np.dot(common, common))
    if denominator <= np.finfo(np.float32).eps:
        return data.copy(), np.zeros(data.shape[1], dtype=np.float32)
    coefficients = (common @ data / denominator).astype(np.float32, copy=False)
    denoised = data - common[:, None] * coefficients[None, :]
    return denoised.astype(np.float32, copy=False), coefficients


def _bandpass_in_batches(
    data: np.ndarray,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int,
    batch_size: int,
) -> np.ndarray:
    nyquist = 0.5 * sample_rate_hz
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"Bandpass must satisfy 0 < low < high < {nyquist:g} Hz"
        )
    sos = signal.butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    output = np.empty_like(data, dtype=np.float32)
    for start in tqdm(
        range(0, data.shape[1], batch_size),
        desc="Zero-phase bandpass",
        unit="batch",
    ):
        stop = min(start + batch_size, data.shape[1])
        filtered = signal.sosfiltfilt(sos, data[:, start:stop], axis=0)
        output[:, start:stop] = filtered.astype(np.float32, copy=False)
    return output


def _preprocess_matrix(
    data: np.ndarray,
    config: ProcessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    detrended = _detrend_in_batches(data, config.filter_channel_batch)
    if config.common_mode_removal:
        denoised, coefficients = _remove_common_mode(detrended)
    else:
        denoised = detrended
        coefficients = np.zeros(data.shape[1], dtype=np.float32)
    filtered = _bandpass_in_batches(
        denoised,
        sample_rate_hz=config.sample_rate_hz,
        low_hz=config.bandpass_low_hz,
        high_hz=config.bandpass_high_hz,
        order=config.filter_order,
        batch_size=config.filter_channel_batch,
    )
    return filtered, coefficients


def _write_preprocessed_data(
    h5_path: Path,
    config: ProcessingConfig,
) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r+") as h5:
        amplitude_raw = h5["amplitude/raw_sdk_native"][:]
        amplitude_baseline = np.median(amplitude_raw, axis=0).astype(np.float32)
        safe_baseline = np.where(
            np.abs(amplitude_baseline) > np.finfo(np.float32).eps,
            amplitude_baseline,
            1.0,
        )
        amplitude_relative = (
            (amplitude_raw - amplitude_baseline[None, :])
            / safe_baseline[None, :]
        ).astype(np.float32, copy=False)
        del amplitude_raw
        amplitude_preprocessed, amplitude_common_coefficients = _preprocess_matrix(
            amplitude_relative, config
        )
        del amplitude_relative

        amplitude_dataset = _create_matrix_dataset(
            h5["amplitude"],
            "preprocessed_relative",
            amplitude_preprocessed.shape,
            config,
        )
        amplitude_dataset[:] = amplitude_preprocessed
        amplitude_dataset.attrs["unit"] = "relative_change"
        amplitude_dataset.attrs["processing"] = (
            "median baseline normalization; linear detrend; optional spatial "
            "median common-mode projection removal; zero-phase Butterworth bandpass"
        )
        h5["amplitude"].create_dataset(
            "baseline_sdk_native", data=amplitude_baseline
        )
        h5["amplitude"].create_dataset(
            "common_mode_coefficients", data=amplitude_common_coefficients
        )

        gauge_points = config.gauge_length_points
        differential_quality = np.minimum(
            np.abs(amplitude_baseline[:-gauge_points]),
            np.abs(amplitude_baseline[gauge_points:]),
        ).astype(np.float32, copy=False)
        h5["amplitude"].create_dataset(
            "differential_endpoint_quality", data=differential_quality
        )
        h5["amplitude/differential_endpoint_quality"].attrs["definition"] = (
            "minimum absolute median SDK amplitude at the two gauge endpoints"
        )

        phase_aligned = h5["phase/differential_aligned_rad"][:]
        phase_preprocessed, phase_common_coefficients = _preprocess_matrix(
            phase_aligned, config
        )
        del phase_aligned
        phase_dataset = _create_matrix_dataset(
            h5["phase"],
            "differential_preprocessed_rad",
            phase_preprocessed.shape,
            config,
        )
        phase_dataset[:] = phase_preprocessed
        phase_dataset.attrs["unit"] = "radian"
        phase_dataset.attrs["processing"] = (
            "circular gauge difference; temporal unwrap; block alignment; linear "
            "detrend; optional spatial median common-mode projection removal; "
            "zero-phase Butterworth bandpass"
        )
        h5["phase"].create_dataset(
            "common_mode_coefficients", data=phase_common_coefficients
        )

    return {
        "amplitude_baseline": amplitude_baseline,
        "differential_endpoint_quality": differential_quality,
        "amplitude_common_coefficients": amplitude_common_coefficients,
        "phase_common_coefficients": phase_common_coefficients,
    }


def _channel_metrics(
    h5_path: Path,
    config: ProcessingConfig,
) -> tuple[pd.DataFrame, int]:
    nperseg = min(32768, int(config.sample_rate_hz * 4))
    with h5py.File(h5_path, "r") as h5:
        phase_dataset = h5["phase/differential_preprocessed_rad"]
        amplitude_dataset = h5["amplitude/preprocessed_relative"]
        locations = h5["axes/differential_center_location_index"][:]
        left_locations = h5["axes/differential_left_location_index"][:]
        right_locations = h5["axes/differential_right_location_index"][:]
        distance_m = h5["axes/differential_center_distance_m"][:]
        amplitude_baseline = h5["amplitude/baseline_sdk_native"][:]
        endpoint_quality = h5["amplitude/differential_endpoint_quality"][:]
        center_indices = np.arange(phase_dataset.shape[1]) + (
            config.gauge_length_points // 2
        )
        metrics: list[dict[str, float | int]] = []
        for start in tqdm(
            range(0, phase_dataset.shape[1], config.filter_channel_batch),
            desc="Channel spectra",
            unit="batch",
        ):
            stop = min(start + config.filter_channel_batch, phase_dataset.shape[1])
            phase_data = phase_dataset[:, start:stop]
            amplitude_data = amplitude_dataset[:, center_indices[start:stop]]
            frequencies, phase_psd = signal.welch(
                phase_data,
                fs=config.sample_rate_hz,
                nperseg=nperseg,
                axis=0,
                detrend="constant",
                scaling="density",
            )
            _, amplitude_psd = signal.welch(
                amplitude_data,
                fs=config.sample_rate_hz,
                nperseg=nperseg,
                axis=0,
                detrend="constant",
                scaling="density",
            )
            analysis_mask = (
                (frequencies >= config.bandpass_low_hz)
                & (frequencies <= config.bandpass_high_hz)
            )
            for local_index in range(stop - start):
                phase_column_psd = phase_psd[:, local_index]
                amplitude_column_psd = amplitude_psd[:, local_index]
                dominant_phase_index = np.argmax(phase_column_psd[analysis_mask])
                dominant_amplitude_index = np.argmax(
                    amplitude_column_psd[analysis_mask]
                )
                analysis_frequencies = frequencies[analysis_mask]
                global_index = start + local_index
                metrics.append(
                    {
                        "column_index": global_index,
                        "location_index": float(locations[global_index]),
                        "gauge_left_location_index": int(
                            left_locations[global_index]
                        ),
                        "gauge_right_location_index": int(
                            right_locations[global_index]
                        ),
                        "distance_m": float(distance_m[global_index]),
                        "amplitude_baseline_sdk_native": float(
                            amplitude_baseline[center_indices[global_index]]
                        ),
                        "differential_endpoint_quality": float(
                            endpoint_quality[global_index]
                        ),
                        "amplitude_preprocessed_rms": float(
                            np.sqrt(np.mean(amplitude_data[:, local_index] ** 2))
                        ),
                        "phase_preprocessed_rms_rad": float(
                            np.sqrt(np.mean(phase_data[:, local_index] ** 2))
                        ),
                        "phase_analysis_band_power": float(
                            np.trapezoid(
                                phase_column_psd[analysis_mask],
                                frequencies[analysis_mask],
                            )
                        ),
                        "phase_dominant_frequency_hz": float(
                            analysis_frequencies[dominant_phase_index]
                        ),
                        "amplitude_dominant_frequency_hz": float(
                            analysis_frequencies[dominant_amplitude_index]
                        ),
                    }
                )
    table = pd.DataFrame(metrics)
    representative_index = int(table["phase_analysis_band_power"].idxmax())
    return table, representative_index


def _downsample_for_image(data: np.ndarray, max_points: int) -> np.ndarray:
    if data.shape[0] <= max_points:
        return data
    factor = math.ceil(data.shape[0] / max_points)
    usable = (data.shape[0] // factor) * factor
    reduced = data[:usable].reshape(-1, factor, data.shape[1]).mean(axis=1)
    if usable < data.shape[0]:
        reduced = np.vstack([reduced, data[usable:].mean(axis=0, keepdims=True)])
    return reduced.astype(np.float32, copy=False)


def _robust_limits(data: np.ndarray, symmetric: bool) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    if symmetric:
        limit = float(np.percentile(np.abs(finite), 99.5))
        if limit <= 0:
            limit = 1.0
        return -limit, limit
    low, high = np.percentile(finite, [0.5, 99.5])
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def _plot_waterfall(
    data: np.ndarray,
    distance_m: np.ndarray,
    duration_s: float,
    title: str,
    colorbar_label: str,
    output_path: Path,
    symmetric: bool,
    max_points: int,
) -> None:
    image = _downsample_for_image(data, max_points)
    vmin, vmax = _robust_limits(image, symmetric=symmetric)
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
    plotted = ax.imshow(
        image.T,
        origin="lower",
        aspect="auto",
        extent=[0.0, duration_s, float(distance_m[0]), float(distance_m[-1])],
        cmap="RdBu_r" if symmetric else "viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set(title=title, xlabel="Time (s)", ylabel="Distance (m)")
    fig.colorbar(plotted, ax=ax, label=colorbar_label, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _welch_1d(data: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(32768, int(sample_rate_hz * 4), len(data))
    return signal.welch(
        data,
        fs=sample_rate_hz,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )


def _plot_dataset_outputs(
    h5_path: Path,
    output_dirs: dict[str, Path],
    representative_index: int,
    config: ProcessingConfig,
) -> None:
    with h5py.File(h5_path, "r") as h5:
        time_s = h5["axes/time_s"][:]
        distance_m = h5["axes/distance_m"][:]
        locations = h5["axes/location_index"][:]
        differential_distance_m = h5["axes/differential_center_distance_m"][:]
        differential_locations = h5[
            "axes/differential_center_location_index"
        ][:]
        duration_s = float(time_s[-1] + 1.0 / config.sample_rate_hz)

        datasets = {
            "amplitude_raw": h5["amplitude/raw_sdk_native"][:],
            "amplitude_preprocessed": h5["amplitude/preprocessed_relative"][:],
            "phase_raw": h5["phase/raw_block_relative_rad"][:],
            "phase_aligned": h5["phase/aligned_rad"][:],
            "phase_differential_aligned": h5[
                "phase/differential_aligned_rad"
            ][:],
            "phase_preprocessed": h5[
                "phase/differential_preprocessed_rad"
            ][:],
        }

        _plot_waterfall(
            datasets["amplitude_raw"],
            distance_m,
            duration_s,
            "Original SDK-native amplitude",
            "SDK-native amplitude",
            output_dirs["original"] / "amplitude_raw_waterfall.png",
            symmetric=False,
            max_points=config.plot_max_time_points,
        )
        _plot_waterfall(
            datasets["phase_raw"],
            distance_m,
            duration_s,
            "Original block-relative IQ phase (0.5 s resets visible)",
            "Phase (rad)",
            output_dirs["original"] / "phase_raw_waterfall.png",
            symmetric=True,
            max_points=config.plot_max_time_points,
        )
        _plot_waterfall(
            datasets["phase_aligned"],
            distance_m,
            duration_s,
            "Diagnostic point phase after 0.5 s block alignment",
            "Phase (rad)",
            output_dirs["recovered"] / "phase_aligned_waterfall.png",
            symmetric=True,
            max_points=config.plot_max_time_points,
        )
        _plot_waterfall(
            datasets["phase_differential_aligned"],
            differential_distance_m,
            duration_s,
            (
                f"Recovered {config.gauge_length_points}-point circular "
                "differential phase"
            ),
            "Differential phase (rad)",
            output_dirs["recovered"]
            / "differential_phase_recovered_waterfall.png",
            symmetric=True,
            max_points=config.plot_max_time_points,
        )
        _plot_waterfall(
            datasets["phase_preprocessed"],
            differential_distance_m,
            duration_s,
            "Preprocessed DAS differential phase",
            "Differential phase (rad)",
            output_dirs["recovered"] / "phase_preprocessed_waterfall.png",
            symmetric=True,
            max_points=config.plot_max_time_points,
        )
        _plot_waterfall(
            datasets["amplitude_preprocessed"],
            distance_m,
            duration_s,
            "Preprocessed relative-amplitude variation",
            "Relative amplitude",
            output_dirs["recovered"] / "amplitude_preprocessed_waterfall.png",
            symmetric=True,
            max_points=config.plot_max_time_points,
        )

        display_step = max(1, len(time_s) // 25000)
        display_slice = slice(None, None, display_step)
        center_point_index = representative_index + config.gauge_length_points // 2
        location = float(differential_locations[representative_index])
        distance = float(differential_distance_m[representative_index])
        point_location = int(locations[center_point_index])

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=180, sharex=True)
        axes[0].plot(
            time_s[display_slice],
            datasets["amplitude_raw"][display_slice, center_point_index],
            linewidth=0.45,
        )
        axes[0].set_ylabel("SDK amplitude")
        axes[0].set_title(
            f"Original SDK signals near loc {point_location} ({distance:.1f} m)"
        )
        axes[1].plot(
            time_s[display_slice],
            datasets["phase_raw"][display_slice, center_point_index],
            linewidth=0.45,
        )
        axes[1].set(xlabel="Time (s)", ylabel="Phase (rad)")
        fig.tight_layout()
        fig.savefig(output_dirs["original"] / "raw_timeseries.png", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=180, sharex=True)
        axes[0].plot(
            time_s[display_slice],
            datasets["amplitude_preprocessed"][display_slice, center_point_index],
            linewidth=0.45,
        )
        axes[0].set_ylabel("Relative amplitude")
        axes[0].set_title(
            f"DAS gauge centered at loc {location:g} ({distance:.1f} m)"
        )
        axes[1].plot(
            time_s[display_slice],
            datasets["phase_preprocessed"][display_slice, representative_index],
            linewidth=0.45,
        )
        axes[1].set(xlabel="Time (s)", ylabel="Differential phase (rad)")
        fig.tight_layout()
        fig.savefig(
            output_dirs["recovered"] / "recovered_timeseries.png",
            bbox_inches="tight",
        )
        plt.close(fig)

        raw_phase_f, raw_phase_psd = _welch_1d(
            datasets["phase_differential_aligned"][:, representative_index],
            config.sample_rate_hz,
        )
        processed_phase_f, processed_phase_psd = _welch_1d(
            datasets["phase_preprocessed"][:, representative_index],
            config.sample_rate_hz,
        )
        raw_amp_f, raw_amp_psd = _welch_1d(
            datasets["amplitude_raw"][:, center_point_index],
            config.sample_rate_hz,
        )
        processed_amp_f, processed_amp_psd = _welch_1d(
            datasets["amplitude_preprocessed"][:, center_point_index],
            config.sample_rate_hz,
        )

        eps = np.finfo(np.float64).tiny
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=180, sharex=True)
        axes[0].semilogy(raw_amp_f, np.maximum(raw_amp_psd, eps), label="Raw")
        axes[0].semilogy(
            processed_amp_f,
            np.maximum(processed_amp_psd, eps),
            label="Preprocessed relative",
        )
        axes[0].set(ylabel="Amplitude PSD", title="Amplitude spectrum")
        axes[0].legend()
        axes[1].semilogy(
            raw_phase_f,
            np.maximum(raw_phase_psd, eps),
            label="Recovered differential",
        )
        axes[1].semilogy(
            processed_phase_f,
            np.maximum(processed_phase_psd, eps),
            label="Recovered/preprocessed",
        )
        axes[1].set(
            xlim=(0, config.bandpass_high_hz),
            xlabel="Frequency (Hz)",
            ylabel="Phase PSD (rad²/Hz)",
            title="DAS differential-phase spectrum",
        )
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(
            output_dirs["comparison"] / "raw_vs_recovered_spectrum.png",
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=180, sharex=True)
        raw_phase = datasets["phase_raw"][display_slice, center_point_index]
        aligned_phase = datasets["phase_aligned"][display_slice, center_point_index]
        processed_phase = datasets["phase_preprocessed"][
            display_slice, representative_index
        ]
        axes[0].plot(time_s[display_slice], raw_phase, linewidth=0.45, label="Raw")
        axes[0].plot(
            time_s[display_slice], aligned_phase, linewidth=0.55, label="Aligned"
        )
        axes[0].set(ylabel="Phase (rad)", title="Phase reset recovery")
        axes[0].legend()
        axes[1].plot(
            time_s[display_slice], processed_phase, linewidth=0.45, color="tab:red"
        )
        axes[1].set(
            xlabel="Time (s)",
            ylabel="Differential phase (rad)",
            title=(
                f"Gauge-differenced, detrended, common-mode suppressed, "
                f"{config.bandpass_low_hz:g}–"
                f"{config.bandpass_high_hz:g} Hz zero-phase filtered"
            ),
        )
        fig.tight_layout()
        fig.savefig(
            output_dirs["comparison"] / "phase_alignment_and_preprocessing.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def _write_representative_csv(
    h5_path: Path,
    output_path: Path,
    representative_index: int,
) -> None:
    with h5py.File(h5_path, "r") as h5:
        gauge_points = int(h5.attrs["gauge_length_points"])
        center_point_index = representative_index + gauge_points // 2
        left_point_index = representative_index
        right_point_index = representative_index + gauge_points
        frame = pd.DataFrame(
            {
                "time_s": h5["axes/time_s"][:],
                "amplitude_raw_sdk_native": h5["amplitude/raw_sdk_native"][
                    :, center_point_index
                ],
                "amplitude_preprocessed_relative": h5[
                    "amplitude/preprocessed_relative"
                ][:, center_point_index],
                "phase_left_raw_block_relative_rad": h5[
                    "phase/raw_block_relative_rad"
                ][:, left_point_index],
                "phase_right_raw_block_relative_rad": h5[
                    "phase/raw_block_relative_rad"
                ][:, right_point_index],
                "phase_center_aligned_rad": h5["phase/aligned_rad"][
                    :, center_point_index
                ],
                "differential_phase_aligned_rad": h5[
                    "phase/differential_aligned_rad"
                ][:, representative_index],
                "differential_phase_preprocessed_rad": h5[
                    "phase/differential_preprocessed_rad"
                ][
                    :, representative_index
                ],
            }
        )
    frame.to_csv(output_path, index=False, float_format="%.9g")


def _prepare_output_dirs(dataset_output: Path) -> dict[str, Path]:
    output_dirs = {
        "original": dataset_output / "00_original",
        "recovered": dataset_output / "01_recovered",
        "comparison": dataset_output / "02_comparison",
        "data": dataset_output / "03_data",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return output_dirs


def _validate_capture_config(
    metadata: dict[str, Any], config: ProcessingConfig
) -> None:
    actual_rate = float(metadata["acquisition"]["csv_row_rate_hz"])
    actual_block_rows = int(metadata["acquisition"]["block_rows"])
    if not math.isclose(actual_rate, config.sample_rate_hz, rel_tol=0, abs_tol=1e-6):
        raise ValueError(
            f"SAMPLE_RATE_HZ={config.sample_rate_hz}, metadata={actual_rate}"
        )
    if actual_block_rows != config.block_rows:
        raise ValueError(
            f"BLOCK_ROWS={config.block_rows}, metadata={actual_block_rows}"
        )
    actual_duration = actual_block_rows / actual_rate
    if not math.isclose(
        actual_duration, config.block_duration_s, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError(
            f"BLOCK_DURATION_S={config.block_duration_s}, metadata={actual_duration}"
        )


def _update_global_summary(output_root: Path, summary_row: dict[str, Any]) -> None:
    summary_path = output_root / "processing_summary.csv"
    if summary_path.exists():
        table = pd.read_csv(summary_path)
        table["dataset"] = table["dataset"].astype(str)
        table = table[table["dataset"] != summary_row["dataset"]]
        table = pd.concat([table, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        table = pd.DataFrame([summary_row])
    table = table.reindex(columns=list(summary_row))
    table["dataset_sort"] = pd.to_numeric(table["dataset"], errors="coerce")
    table = table.sort_values("dataset_sort").drop(columns="dataset_sort")
    table.to_csv(summary_path, index=False, float_format="%.9g")

    markdown_lines = [
        "# 三组 BlockIQ DAS 差分相位处理结果总览",
        "",
        "## 自动选出的代表标距通道",
        "",
        (
            "主分析信号为 SDK `BlockIQ.phase(0)` 形成的圆周空间差分相位；"
            "单点相位仅保留作采集与块复位诊断。代表通道按预处理差分相位在 "
            "`5–2000 Hz` 内的总功率自动选择。"
        ),
        "",
        "| 数据编号 | 标距左端 | 标距右端 | 标距 | 中心距离 | 主峰 | 分析带功率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        markdown_lines.append(
            f"| {row.dataset} | {int(row.gauge_left_location_index)} | "
            f"{int(row.gauge_right_location_index)} | {row.gauge_length_m:g} m | "
            f"{row.representative_distance_m:.1f} m | "
            f"{row.phase_dominant_frequency_hz:.3f} Hz | "
            f"{row.phase_analysis_band_power:.3f} |"
        )
    markdown_lines.extend(
        [
            "",
            "## 数据语义",
            "",
            (
                "- CSV 的 `amp_loc_*` / `phase_loc_*` 来自 "
                "`PD_FDM_IQ_AMP_PHASE` 模式下的 `BlockIQ.amp(0)` / "
                "`BlockIQ.phase(0)`，是 SDK 偏振合成和 IQ 解调后的幅相，"
                "不是原始 ADC 或独立 I/Q。"
            ),
            (
                "- 差分相位按 `wrap(phase_right - phase_left)`、块内时间解缠、"
                "0.5 秒块间连续对齐、去趋势、空间共模抑制和零相位带通处理。"
            ),
            (
                "- 主峰是 Welch 功率谱的最大离散频点；数据编号 "
                "`113/619/985` 是采集时间戳尾号，不是激励频率。"
            ),
            "- 未换算应变或声压；完整处理定义见项目根目录 `README.md`。",
            "",
        ]
    )
    (output_root / "RESULTS_SUMMARY.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )


def run_dataset(
    dataset: str,
    dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    config = ProcessingConfig.from_environment(project_root)
    env_name = f"DATASET_{dataset}_DIR"
    if dataset_dir is None:
        configured_path = os.getenv(env_name)
        if not configured_path:
            raise ValueError(f"{env_name} is not configured")
        dataset_dir = configured_path
    source_dir = Path(dataset_dir).resolve()
    metadata = _load_metadata(source_dir)
    _validate_capture_config(metadata, config)

    csv_filename = metadata["files"][0]["filename"]
    csv_path = source_dir / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    output_root = Path(os.getenv("OUTPUT_ROOT", project_root / "results")).resolve()
    dataset_output = output_root / dataset
    output_dirs = _prepare_output_dirs(dataset_output)
    h5_path = output_dirs["data"] / f"das_{dataset}_processed.h5"

    import_diagnostics = _import_csv_to_hdf5(
        csv_path=csv_path,
        metadata=metadata,
        h5_path=h5_path,
        config=config,
    )
    preprocessing_diagnostics = _write_preprocessed_data(h5_path, config)
    channel_table, representative_index = _channel_metrics(
        h5_path,
        config=config,
    )
    channel_table.to_csv(
        output_dirs["data"] / "channel_summary.csv",
        index=False,
        float_format="%.9g",
    )
    _write_representative_csv(
        h5_path,
        output_dirs["data"] / "representative_timeseries.csv",
        representative_index,
    )
    _plot_dataset_outputs(
        h5_path=h5_path,
        output_dirs=output_dirs,
        representative_index=representative_index,
        config=config,
    )

    representative = channel_table.iloc[representative_index]
    point_offsets = import_diagnostics.pop("block_offsets_rad")
    differential_offsets = import_diagnostics.pop(
        "differential_block_offsets_rad"
    )
    import_diagnostics["locations"] = import_diagnostics["locations"].tolist()
    import_diagnostics["distance_m"] = import_diagnostics["distance_m"].tolist()
    quantum = import_diagnostics["observed_phase_quantization_rad"]
    report = {
        "dataset": dataset,
        "source_directory": str(source_dir),
        "source_csv": str(csv_path),
        "output_hdf5": str(h5_path),
        "capture_metadata": metadata,
        "processing_config": asdict(config),
        "phase_interpretation": {
            "source": "BlockIQ.phase(0)",
            "source_semantics": (
                "SDK-demodulated polarization-diversity logical-channel phase; "
                "not raw ADC samples and not independent I/Q branches"
            ),
            "input_unit": "radian",
            "degree_to_radian_conversion_required": False,
            "observed_quantization_step_rad": quantum,
            "pi_over_1024_rad": math.pi / 1024,
            "step_relative_error_vs_pi_over_1024": abs(
                quantum - math.pi / 1024
            )
            / (math.pi / 1024),
            "input_exceeds_plus_minus_pi_within_block": (
                import_diagnostics["phase_min_rad"] < -math.pi
                or import_diagnostics["phase_max_rad"] > math.pi
            ),
            "additional_block_unwrap_applied": config.phase_block_unwrap,
            "reset_period_s": config.block_duration_s,
            "alignment_method": (
                "Each block receives an additive per-location offset so its first "
                "sample continues the robust median slope estimated on both sides "
                "of the boundary."
            ),
            "scientific_limitation": (
                "Block resets remove the absolute inter-block phase constant. The "
                "continuous result is reconstructed under a local continuity "
                "assumption; the missing absolute constant is not directly observed."
            ),
        },
        "das_phase_measurement": {
            "primary_signal": "circular_spatial_differential_phase",
            "gauge_length_points": config.gauge_length_points,
            "gauge_length_m": import_diagnostics["gauge_length_m"],
            "definition": (
                "wrap(phase_right - phase_left) to [-pi, pi), unwrap along time "
                "inside each 0.5 s block, then align blocks by robust boundary "
                "continuity"
            ),
            "single_point_phase_role": "preserved for SDK and reset diagnostics",
            "spectrum_and_channel_selection_use": (
                "preprocessed differential phase"
            ),
        },
        "amplitude_interpretation": {
            "input_unit": "sdk_native_unspecified",
            "raw_amplitude_preserved": True,
            "iq_branch_reconstruction_performed": False,
            "reason": (
                "capture_info.json states SDK polarization combining was already "
                "applied and independent physical I/Q branches are unavailable."
            ),
        },
        "phase_import_diagnostics": import_diagnostics,
        "block_offset_diagnostics": {
            "block_count": int(point_offsets.shape[0]),
            "point_phase_median_abs_offset_rad": float(
                np.median(np.abs(point_offsets))
            ),
            "point_phase_max_abs_offset_rad": float(
                np.max(np.abs(point_offsets))
            ),
            "differential_phase_median_abs_offset_rad": float(
                np.median(np.abs(differential_offsets))
            ),
            "differential_phase_max_abs_offset_rad": float(
                np.max(np.abs(differential_offsets))
            ),
        },
        "common_mode_coefficients": {
            "amplitude_median_abs": float(
                np.median(
                    np.abs(
                        preprocessing_diagnostics["amplitude_common_coefficients"]
                    )
                )
            ),
            "phase_median_abs": float(
                np.median(
                    np.abs(preprocessing_diagnostics["phase_common_coefficients"])
                )
            ),
        },
        "representative_channel": {
            "column_index": int(representative["column_index"]),
            "location_index": float(representative["location_index"]),
            "gauge_left_location_index": int(
                representative["gauge_left_location_index"]
            ),
            "gauge_right_location_index": int(
                representative["gauge_right_location_index"]
            ),
            "distance_m": float(representative["distance_m"]),
            "differential_endpoint_quality": float(
                representative["differential_endpoint_quality"]
            ),
            "phase_dominant_frequency_hz": float(
                representative["phase_dominant_frequency_hz"]
            ),
            "amplitude_dominant_frequency_hz": float(
                representative["amplitude_dominant_frequency_hz"]
            ),
            "phase_analysis_band_power": float(
                representative["phase_analysis_band_power"]
            ),
        },
        "strain_conversion": {
            "performed": False,
            "reason": (
                "The capture metadata does not provide all required calibrated "
                "parameters: optical wavelength, effective refractive index, "
                "and photoelastic coefficient."
            ),
        },
        "method_references": REFERENCE_URLS,
    }
    report_path = output_dirs["data"] / "processing_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_row = {
        "dataset": dataset,
        "representative_location_index": int(representative["location_index"]),
        "gauge_left_location_index": int(
            representative["gauge_left_location_index"]
        ),
        "gauge_right_location_index": int(
            representative["gauge_right_location_index"]
        ),
        "gauge_length_m": import_diagnostics["gauge_length_m"],
        "representative_distance_m": float(representative["distance_m"]),
        "phase_dominant_frequency_hz": float(
            representative["phase_dominant_frequency_hz"]
        ),
        "phase_analysis_band_power": float(
            representative["phase_analysis_band_power"]
        ),
        "output_directory": str(dataset_output),
    }
    _update_global_summary(output_root, summary_row)
    return report


def print_report_summary(report: dict[str, Any]) -> None:
    representative = report["representative_channel"]
    print(
        f"Dataset {report['dataset']} complete: gauge "
        f"{representative['gauge_left_location_index']}–"
        f"{representative['gauge_right_location_index']}, centered at loc "
        f"{representative['location_index']:g} / "
        f"{representative['distance_m']:.1f} m, dominant phase frequency "
        f"{representative['phase_dominant_frequency_hz']:.3f} Hz"
    )
    print(f"HDF5: {report['output_hdf5']}")
