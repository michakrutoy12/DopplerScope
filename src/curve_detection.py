"""Automatic extraction of frequency ridges from an audio spectrogram.

The detector is deliberately signal-processing based (no ML model required):
1. resample audio if useful and calculate an STFT;
2. enhance narrow ridges relative to their local frequency background;
3. detect candidate peaks frame by frame;
4. link peaks over short dropouts using dynamic programming;
5. score whole trajectories for duration, continuity, movement and smoothness;
6. suppress each accepted/rejected trajectory along its full interpolated path
   so it is not returned again as several fragments.

This is a candidate *moving-ridge* detector, not an aircraft classifier.
ADS-B/Doppler-model matching should make the final aircraft decision.

Dependencies:
    numpy, scipy
Optional (plotting / audio loading):
    matplotlib, soundfile

Example:
    import soundfile as sf
    from curve_detection import detect_curves, plot_detection

    audio, sr = sf.read("recording.wav")
    curves, debug = detect_curves(
        audio, sr,
        min_frequency=50,
        max_frequency=300,
        max_curves=8,
        return_debug=True,
    )
    for i, curve in enumerate(curves):
        print(i, curve["duration"], curve["coverage"], curve["mean_confidence"])
    plot_detection(debug, curves)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, resample_poly, stft

_CurveData = dict[str, Any]
_CurveMetrics = dict[str, float]


@dataclass
class _Peak:
    """A spectral peak in one STFT frame."""

    frame: int
    bin_index: int
    frequency: float
    contrast_db: float
    prominence_db: float
    confidence: float
    reward: float
    dp_score: float = 0.0
    previous: _Peak | None = None
    active: bool = True


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert mono or interleaved multichannel audio to finite mono float32."""
    x = np.asarray(audio)

    if x.ndim == 2:
        # soundfile returns (samples, channels). This also handles the usual
        # channels-first case when the first dimension is clearly the smaller one.
        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            x = np.mean(x, axis=0)
        else:
            x = np.mean(x, axis=1)
    elif x.ndim != 1:
        raise ValueError("audio must be a 1-D mono array or a 2-D multichannel array")

    # Copy so DC removal never mutates the caller's audio buffer.
    x = np.array(x, dtype=np.float32, copy=True)
    if x.size == 0:
        raise ValueError("audio is empty")
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, copy=False)

    # Remove DC. It can otherwise dominate the lowest FFT bins.
    x -= np.mean(x, dtype=np.float64).astype(np.float32)
    return x


def _resample_for_band(
    audio: np.ndarray,
    sample_rate: int,
    max_frequency: float,
) -> tuple[np.ndarray, int]:
    """Downsample while keeping enough Nyquist headroom for the requested band."""
    target_rate = min(int(sample_rate), max(4000, int(np.ceil(max_frequency * 4.0))))
    if target_rate >= sample_rate:
        return audio, int(sample_rate)

    divisor = gcd(int(sample_rate), int(target_rate))
    up = target_rate // divisor
    down = int(sample_rate) // divisor
    y = resample_poly(audio, up, down).astype(np.float32, copy=False)
    return y, target_rate


def _odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _make_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    min_frequency: float,
    max_frequency: float,
    nperseg: int,
    hop_length: int,
    background_width_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return time, frequency, dB spectrogram and locally whitened spectrogram."""
    if len(audio) < 256:
        raise ValueError("audio is too short; provide at least 256 samples")

    nperseg = min(int(nperseg), len(audio))
    # Keep an FFT-friendly window size. For very short clips, use the largest
    # power of two that fits the audio.
    if nperseg < 256:
        nperseg = 2 ** int(np.floor(np.log2(len(audio))))
    else:
        nperseg = 2 ** int(np.floor(np.log2(nperseg)))
    nperseg = max(256, nperseg)

    hop_length = max(1, min(int(hop_length), nperseg - 1))
    noverlap = nperseg - hop_length

    frequencies, times, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )

    magnitude = np.abs(spectrum).astype(np.float32, copy=False)
    spec_db_full = (20.0 * np.log10(magnitude + 1e-10)).astype(np.float32)

    df = float(frequencies[1] - frequencies[0])
    background_bins = _odd_at_least(round(background_width_hz / max(df, 1e-9)), 5)

    # A narrow tonal ridge occupies only a small part of this frequency window,
    # so the median approximates its local spectral background.
    background = median_filter(
        spec_db_full,
        size=(background_bins, 1),
        mode="nearest",
    )
    enhanced_full = spec_db_full - background
    enhanced_full = gaussian_filter1d(enhanced_full, sigma=0.7, axis=0, mode="nearest")

    band_mask = (frequencies >= min_frequency) & (frequencies <= max_frequency)
    if not np.any(band_mask):
        raise ValueError(
            "the requested frequency band contains no FFT bins; "
            "increase nperseg or widen the frequency range"
        )

    return (
        times.astype(np.float32),
        frequencies[band_mask].astype(np.float32),
        spec_db_full[band_mask].astype(np.float32),
        enhanced_full[band_mask].astype(np.float32),
    )


def _detect_frame_peaks(
    frequencies: np.ndarray,
    enhanced: np.ndarray,
    min_contrast_db: float,
    min_prominence_db: float,
    max_peaks_per_frame: int,
    peak_distance_bins: int,
) -> list[list[_Peak]]:
    """Find and score local spectral maxima independently in each frame."""
    peaks_by_frame: list[list[_Peak]] = []

    for frame in range(enhanced.shape[1]):
        column = enhanced[:, frame]
        peak_bins, properties = find_peaks(
            column,
            height=min_contrast_db,
            prominence=min_prominence_db,
            distance=max(1, int(peak_distance_bins)),
        )

        frame_peaks: list[_Peak] = []
        prominences = properties.get("prominences", np.zeros(len(peak_bins)))

        for bin_index, prominence in zip(peak_bins, prominences):
            contrast = float(column[bin_index])
            prominence = float(prominence)
            excess = max(0.0, contrast - min_contrast_db)

            # Sub-bin peak interpolation: fit a parabola through the peak and
            # its two neighbors. This estimates frequency between FFT bins.
            # It improves numerical resolution but does not create information
            # that the recording/SNR/window length do not contain.
            offset = 0.0
            if 0 < bin_index < len(column) - 1:
                left = float(column[bin_index - 1])
                center = float(column[bin_index])
                right = float(column[bin_index + 1])
                denominator = left - 2.0 * center + right
                if abs(denominator) > 1e-12:
                    offset = 0.5 * (left - right) / denominator
                    offset = float(np.clip(offset, -0.5, 0.5))
            refined_frequency = float(
                frequencies[bin_index]
                + offset * (frequencies[1] - frequencies[0])
            )

            # Confidence is a convenient 0..1 quality indicator, not a
            # calibrated probability that the ridge belongs to an aircraft.
            confidence = float(np.clip(0.25 + excess / 12.0, 0.0, 1.0))
            reward = float(
                0.5 + 0.40 * min(excess, 12.0) + 0.15 * min(prominence, 10.0)
            )

            frame_peaks.append(
                _Peak(
                    frame=frame,
                    bin_index=int(bin_index),
                    frequency=refined_frequency,
                    contrast_db=contrast,
                    prominence_db=prominence,
                    confidence=confidence,
                    reward=reward,
                )
            )

        # Limit graph size in frames containing many unrelated tonal peaks.
        frame_peaks.sort(key=lambda peak: peak.reward, reverse=True)
        peaks_by_frame.append(frame_peaks[:max_peaks_per_frame])

    return peaks_by_frame


def _best_path(
    peaks_by_frame: list[list[_Peak]],
    frame_step_seconds: float,
    frequency_resolution: float,
    max_slope_hz_s: float,
    max_missing_frames: int,
    transition_weight: float,
    gap_penalty: float,
    curvature_weight: float,
) -> list[_Peak]:
    """Find the best ridge using frequency-indexed dynamic programming."""

    active_by_frame = []
    frequency_arrays = []
    all_active = []

    for frame_peaks in peaks_by_frame:
        active = [p for p in frame_peaks if p.active]
        active.sort(key=lambda p: p.frequency)

        active_by_frame.append(active)
        frequency_arrays.append(
            np.fromiter((p.frequency for p in active), dtype=np.float64)
        )
        all_active.extend(active)

    if not all_active:
        return []

    for frame_index, frame_peaks in enumerate(active_by_frame):
        for peak in frame_peaks:
            peak.dp_score = peak.reward
            peak.previous = None

            for frame_gap in range(1, max_missing_frames + 2):
                previous_frame_index = frame_index - frame_gap

                if previous_frame_index < 0:
                    break

                dt = frame_gap * frame_step_seconds
                allowed_jump = (
                    max_slope_hz_s * dt
                    + 2.0 * frequency_resolution
                )

                previous_peaks = active_by_frame[previous_frame_index]

                if not previous_peaks:
                    continue

                previous_freqs = frequency_arrays[previous_frame_index]

                lo = np.searchsorted(
                    previous_freqs,
                    peak.frequency - allowed_jump
                )

                hi = np.searchsorted(
                    previous_freqs,
                    peak.frequency + allowed_jump
                )

                for previous in previous_peaks[lo:hi]:
                    jump = abs(peak.frequency - previous.frequency)
                    jump_ratio = jump / allowed_jump

                    cost = transition_weight * jump_ratio**2
                    cost += gap_penalty * (frame_gap - 1)

                    if previous.previous is not None:
                        older = previous.previous

                        old_dt = (
                            previous.frame - older.frame
                        ) * frame_step_seconds

                        if old_dt > 0:
                            old_slope = (
                                previous.frequency - older.frequency
                            ) / old_dt

                            new_slope = (
                                peak.frequency - previous.frequency
                            ) / dt

                            slope_change = (
                                abs(new_slope - old_slope)
                                / max(max_slope_hz_s, 1e-6)
                            )

                            cost += curvature_weight * min(
                                slope_change**2, 4.0
                            )

                    candidate_score = (
                        previous.dp_score + peak.reward - cost
                    )

                    if candidate_score > peak.dp_score:
                        peak.dp_score = candidate_score
                        peak.previous = previous

    end_peak = max(all_active, key=lambda p: p.dp_score)

    path = []
    current = end_peak

    while current is not None:
        path.append(current)
        current = current.previous

    path.reverse()
    return path


def _path_quality(path: list[_Peak], frame_step_seconds: float) -> _CurveMetrics:
    """Measure duration, coverage, movement and smoothness of a candidate ridge."""
    if not path:
        return {
            "duration": 0.0,
            "coverage": 0.0,
            "mean_confidence": 0.0,
            "frequency_excursion": 0.0,
            "motion_fraction": 0.0,
            "smoothness": 0.0,
            "score": 0.0,
        }

    first_frame = path[0].frame
    last_frame = path[-1].frame
    span_frames = max(1, last_frame - first_frame + 1)
    duration = (last_frame - first_frame) * frame_step_seconds

    frequencies = np.asarray(
        [peak.frequency for peak in path], dtype=np.float64
    )
    differences = np.diff(frequencies)
    excursion = float(
        np.percentile(frequencies, 95) - np.percentile(frequencies, 5)
    )
    coverage = len(path) / span_frames
    mean_confidence = float(np.mean([peak.confidence for peak in path]))

    # Ignore tiny frame-to-frame changes caused by FFT-bin jitter. A real
    # Doppler ridge should exhibit measurable movement over a meaningful part
    # of its duration, though it need not be monotonic.
    motion_fraction = (
        float(np.mean(np.abs(differences) >= 0.20)) if differences.size else 0.0
    )

    if differences.size >= 2:
        slopes = differences / frame_step_seconds
        slope_changes = np.diff(slopes)
        scale = max(float(np.percentile(np.abs(slopes), 75)), 0.5)
        smoothness = float(
            1.0 / (1.0 + np.median(np.abs(slope_changes)) / scale)
        )
    else:
        smoothness = 0.0

    # Average reward is retained as a signal-strength indicator. The caller
    # uses motion/coverage gates separately, so a loud stationary tone cannot
    # win merely by lasting a long time.
    score = float(np.mean([peak.reward for peak in path]))

    return {
        "duration": float(duration),
        "coverage": float(coverage),
        "mean_confidence": mean_confidence,
        "frequency_excursion": excursion,
        "motion_fraction": motion_fraction,
        "smoothness": smoothness,
        "score": score,
    }


def _suppress_path(
    path: list[_Peak],
    peaks_by_frame: list[list[_Peak]],
    suppression_band_hz: float,
) -> None:
    """Suppress the selected ridge as a continuous time-frequency tube.

    The old implementation removed only peaks in the selected frame and its
    immediate neighbours. If a ridge had a weak section, later passes could
    rediscover its remaining pieces as separate curves. Here the path is
    interpolated over its complete time span and a narrow tube around it is
    removed, including short gaps.
    """
    if not path:
        return

    path_frames = np.asarray([peak.frame for peak in path], dtype=np.float64)
    path_freqs = np.asarray([peak.frequency for peak in path], dtype=np.float64)
    first_frame = int(path_frames[0])
    last_frame = int(path_frames[-1])

    for frame_index in range(first_frame, last_frame + 1):
        expected_frequency = float(np.interp(frame_index, path_frames, path_freqs))
        for candidate in peaks_by_frame[frame_index]:
            if (
                abs(candidate.frequency - expected_frequency)
                <= suppression_band_hz
            ):
                candidate.active = False


def _endpoint_slope(
    time: np.ndarray,
    frequency: np.ndarray,
    *,
    at_end: bool,
) -> float:
    """Estimate local frequency slope near a fragment endpoint."""

    if len(time) < 3:
        return 0.0

    edge = float(time[-1] if at_end else time[0])
    window = 7.0

    if at_end:
        mask = (time >= edge - window) & (time <= edge)
    else:
        mask = (time >= edge) & (time <= edge + window)

    t = time[mask]
    f = frequency[mask]

    if len(t) < 3:
        count = min(6, len(time))

        if at_end:
            t = time[-count:]
            f = frequency[-count:]
        else:
            t = time[:count]
            f = frequency[:count]

    dt = np.diff(t)
    valid = dt > 1e-6

    slopes = np.diff(f)[valid] / dt[valid]

    if slopes.size == 0:
        return 0.0

    return float(np.median(slopes))


def _merge_pair_cost(
    first: _CurveData,
    second: _CurveData,
    *,
    max_gap_s: float,
    frequency_tolerance_hz: float,
    slope_tolerance_hz_s: float,
    max_overlap_s: float,
) -> float | None:
    """Return a cost for joining two fragments, or None if they are incompatible."""
    ta = np.asarray(first["time"], dtype=np.float64)
    fa = np.asarray(first["frequency"], dtype=np.float64)
    tb = np.asarray(second["time"], dtype=np.float64)
    fb = np.asarray(second["frequency"], dtype=np.float64)

    if len(ta) < 2 or len(tb) < 2:
        return None

    # Chronological ordering.
    if (tb[0], tb[-1]) < (ta[0], ta[-1]):
        ta, tb = tb, ta
        fa, fb = fb, fa

    gap = float(tb[0] - ta[-1])
    slope_a = _endpoint_slope(ta, fa, at_end=True)
    slope_b = _endpoint_slope(tb, fb, at_end=False)
    slope_error = abs(slope_a - slope_b)

    if gap >= 0:
        if gap > max_gap_s:
            return None

        predicted_b = fa[-1] + slope_a * gap
        predicted_a = fb[0] - slope_b * gap

        frequency_error = 0.5 * (
            abs(predicted_b - fb[0])
            + abs(predicted_a - fa[-1])
        )

        allowed_error = (
            frequency_tolerance_hz
            + 0.20 * (abs(slope_a) + abs(slope_b)) * gap
        )

        if frequency_error > allowed_error:
            return None

    else:
        # Fragments overlap in time.
        overlap = -gap
        if overlap > max_overlap_s:
            return None

        left = max(float(ta[0]), float(tb[0]))
        right = min(float(ta[-1]), float(tb[-1]))

        if right <= left:
            return None

        sample_t = np.linspace(
            left,
            right,
            min(25, max(5, int((right - left) / 0.5) + 1)),
        )

        difference = np.abs(
            np.interp(sample_t, ta, fa)
            - np.interp(sample_t, tb, fb)
        )

        median_difference = float(np.median(difference))
        high_difference = float(np.percentile(difference, 90))

        if median_difference > max(
            3.0, frequency_tolerance_hz * 0.20
        ) or high_difference > max(7.0, frequency_tolerance_hz * 0.45):
            return None

        frequency_error = median_difference

    if slope_error > slope_tolerance_hz_s:
        return None

    gap_cost = max(0.0, gap) / max(max_gap_s, 1e-6)

    frequency_cost = frequency_error / max(frequency_tolerance_hz, 1e-6)
    slope_cost = slope_error / max(slope_tolerance_hz_s, 1e-6)
    total_cost = frequency_cost + 0.35 * slope_cost + 0.15 * gap_cost

    return total_cost if total_cost <= 1.35 else None


def _combine_curve_points(
    a: _CurveData,
    b: _CurveData,
    frame_step_seconds: float,
) -> _CurveData:
    """Combine two fragments and remove duplicate observations."""
    time = np.concatenate(
        (
            np.asarray(a["time"], dtype=np.float64),
            np.asarray(b["time"], dtype=np.float64),
        )
    )
    frequency = np.concatenate(
        (
            np.asarray(a["frequency"], dtype=np.float64),
            np.asarray(b["frequency"], dtype=np.float64),
        )
    )
    confidence = np.concatenate(
        (
            np.asarray(a["confidence"], dtype=np.float64),
            np.asarray(b["confidence"], dtype=np.float64),
        )
    )

    order = np.argsort(time, kind="stable")
    time = time[order]
    frequency = frequency[order]
    confidence = confidence[order]
    merge_dt = max(1e-4, frame_step_seconds * 0.35)

    out_time = []
    out_frequency = []
    out_confidence = []
    i = 0

    while i < len(time):
        j = i + 1
        while j < len(time) and time[j] - time[j - 1] <= merge_dt:
            j += 1

        weights = np.maximum(confidence[i:j], 0.05)
        out_time.append(float(np.average(time[i:j], weights=weights)))
        out_frequency.append(
            float(np.average(frequency[i:j], weights=weights))
        )
        out_confidence.append(float(np.max(confidence[i:j])))
        i = j

    return {
        "time": np.asarray(out_time, dtype=np.float32),
        "frequency": np.asarray(out_frequency, dtype=np.float32),
        "confidence": np.asarray(out_confidence, dtype=np.float32),
        "fragment_count": (
            int(a.get("fragment_count", 1))
            + int(b.get("fragment_count", 1))
        ),
    }


def _merge_fragments(
    fragments: list[_CurveData],
    frame_step_seconds: float,
    *,
    max_gap_s: float = 15.0,
    frequency_tolerance_hz: float = 24.0,
    slope_tolerance_hz_s: float = 7.0,
    max_overlap_s: float = 45.0,
) -> list[_CurveData]:
    """Merge compatible fragments into continuous trajectories."""
    curves = []

    for fragment in fragments:
        item = {
            "time": np.asarray(fragment["time"], dtype=np.float32),
            "frequency": np.asarray(fragment["frequency"], dtype=np.float32),
            "confidence": np.asarray(fragment["confidence"], dtype=np.float32),
            "fragment_count": int(fragment.get("fragment_count", 1)),
        }

        if len(item["time"]) >= 2:
            curves.append(item)

    if len(curves) < 2:
        return curves

    order = sorted(
        range(len(curves)),
        key=lambda i: (
            float(curves[i]["time"][0]),
            float(curves[i]["time"][-1]),
            float(np.median(curves[i]["frequency"])),
        )
    )

    candidates = []
    for pos, i in enumerate(order):
        for j in order[pos + 1:]:
            cost = _merge_pair_cost(
                curves[i],
                curves[j],
                max_gap_s=max_gap_s,
                frequency_tolerance_hz=frequency_tolerance_hz,
                slope_tolerance_hz_s=slope_tolerance_hz_s,
                max_overlap_s=max_overlap_s,
            )

            if cost is not None:
                candidates.append((cost, i, j))

    candidates.sort(key=lambda item: item[0])

    successor = {}
    predecessor = {}
    # Each fragment can have only one previous and one next fragment.
    for _, i, j in candidates:
        if i not in successor and j not in predecessor:
            successor[i] = j
            predecessor[j] = i

    merged = []
    visited = set()
    for start_index in range(len(curves)):
        if start_index in predecessor or start_index in visited:
            continue

        chain = [start_index]
        visited.add(start_index)

        while chain[-1] in successor:
            next_index = successor[chain[-1]]
            if next_index in visited:
                break

            chain.append(next_index)
            visited.add(next_index)

        combined = curves[chain[0]]

        for next_index in chain[1:]:
            combined = _combine_curve_points(
                combined,
                curves[next_index],
                frame_step_seconds,
            )

        merged.append(combined)

    # Safety: retain any fragment not included in a chain.
    for index, curve in enumerate(curves):
        if index not in visited:
            merged.append(curve)

    return merged


def _quality_from_arrays(
    curve: _CurveData,
    frame_step_seconds: float,
) -> _CurveMetrics:
    """Recalculate curve quality after merging."""
    time = np.asarray(curve["time"], dtype=np.float64)
    frequency = np.asarray(curve["frequency"], dtype=np.float64)
    confidence = np.asarray(curve["confidence"], dtype=np.float64)

    if len(time) < 2:
        return {
            "duration": 0.0,
            "coverage": 0.0,
            "mean_confidence": 0.0,
            "frequency_excursion": 0.0,
            "motion_fraction": 0.0,
            "smoothness": 0.0,
            "score": 0.0,
        }

    duration = float(time[-1] - time[0])

    span_frames = max(
        1, int(round(duration / max(frame_step_seconds, 1e-9))) + 1
    )
    coverage = min(1.0, len(time) / span_frames)
    differences = np.diff(frequency)
    excursion = float(np.percentile(frequency, 95) - np.percentile(frequency, 5))
    motion_fraction = (
        float(np.mean(np.abs(differences) >= 0.20))
        if differences.size
        else 0.0
    )

    if differences.size >= 2:
        slopes = differences / np.maximum(np.diff(time), 1e-6)
        slope_changes = np.diff(slopes)
        scale = max(float(np.percentile(np.abs(slopes), 75)), 0.5)
        smoothness = float(1.0 / (1.0 + np.median(np.abs(slope_changes)) / scale))
    else:
        smoothness = 0.0

    mean_confidence = float(np.mean(confidence))

    score = (
        mean_confidence
        * coverage
        * np.log1p(max(0.0, duration))
    )

    return {
        "duration": duration,
        "coverage": float(coverage),
        "mean_confidence": mean_confidence,
        "frequency_excursion": excursion,
        "motion_fraction": motion_fraction,
        "smoothness": smoothness,
        "score": float(score),
    }


def detect_curves(
    audio: Any,
    sample_rate: int,
    *,
    min_frequency: float = 200.0,
    max_frequency: float = 2000.0,
    max_curves: int = 8,
    nperseg: int = 4096,
    hop_length: int = 512,
    background_width_hz: float = 15.0,
    min_contrast_db: float = 4.5,
    min_prominence_db: float = 1.0,
    max_peaks_per_frame: int = 20,
    peak_distance_bins: int = 2,
    max_slope_hz_s: float = 8.0,
    max_missing_frames: int = 6,
    transition_weight: float = 2.0,
    gap_penalty: float = 1.0,
    curvature_weight: float = 0.5,
    suppression_band_hz: float = 5.0,
    min_duration_s: float = 5.0,
    min_points: int = 20,
    min_coverage: float = 0.30,
    min_excursion_hz: float = 8.0,
    min_motion_fraction: float = 0.08,
    min_mean_confidence: float = 0.48,
    merge_fragments: bool = True,
    merge_max_gap_s: float = 15.0,
    merge_frequency_tolerance_hz: float = 24.0,
    merge_slope_tolerance_hz_s: float = 7.0,
    max_fragments: int | None = None,
    return_debug: bool = False,
) -> list[_CurveData] | tuple[list[_CurveData], dict[str, Any]]:
    """Detect and merge candidate moving spectral ridges."""
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")

    if not (
        np.isfinite(min_frequency)
        and np.isfinite(max_frequency)
        and 0 <= min_frequency < max_frequency
    ):
        raise ValueError("Invalid frequency range")

    if not isinstance(max_curves, (int, np.integer)) or max_curves < 1:
        raise ValueError("max_curves must be a positive integer")

    if max_frequency >= int(sample_rate) / 2:
        raise ValueError("max_frequency must be below Nyquist frequency")

    if max_missing_frames < 0:
        raise ValueError("max_missing_frames cannot be negative")

    if max_peaks_per_frame < 1 or peak_distance_bins < 1:
        raise ValueError("peak limits must be positive")

    if max_slope_hz_s <= 0 or min_duration_s < 0 or min_points < 2:
        raise ValueError("Invalid detection parameters")

    if not (
        0 <= min_coverage <= 1
        and 0 <= min_motion_fraction <= 1
        and 0 <= min_mean_confidence <= 1
    ):
        raise ValueError("Quality thresholds must be between 0 and 1")

    if min_excursion_hz < 0 or merge_max_gap_s < 0:
        raise ValueError("Invalid merge parameters")

    if (
        merge_frequency_tolerance_hz <= 0
        or merge_slope_tolerance_hz_s <= 0
    ):
        raise ValueError("Merge tolerances must be positive")

    # Prepare audio and calculate its spectrogram.
    x = _to_mono_float32(audio)
    x, effective_rate = _resample_for_band(x, int(sample_rate), max_frequency)

    if max_frequency >= effective_rate / 2:
        raise ValueError("Frequency range exceeds resampled Nyquist")

    times, frequencies, spec_db, enhanced = _make_spectrogram(
        x,
        effective_rate,
        min_frequency,
        max_frequency,
        nperseg,
        hop_length,
        background_width_hz,
    )

    if len(times) < 2 or len(frequencies) < 3:
        raise ValueError("Not enough spectrogram data")

    frame_step_seconds = float(np.median(np.diff(times)))
    frequency_resolution = float(np.median(np.diff(frequencies)))

    peaks_by_frame = _detect_frame_peaks(
        frequencies,
        enhanced,
        min_contrast_db,
        min_prominence_db,
        max_peaks_per_frame,
        peak_distance_bins,
    )

    # Extract fragments first. A fragment can become a valid curve
    # only after it has been joined with another fragment.
    if max_fragments is None:
        max_fragments = min(48, max(16, max_curves * 3))

    if max_fragments < 1:
        raise ValueError("max_fragments must be positive")

    fragments: list[_CurveData] = []
    min_fragment_points = max(6, min_points // 4)
    min_fragment_duration = max(0.75, min_duration_s / 5.0)

    for _ in range(max_fragments):
        path = _best_path(
            peaks_by_frame,
            frame_step_seconds,
            frequency_resolution,
            max_slope_hz_s,
            max_missing_frames,
            transition_weight,
            gap_penalty,
            curvature_weight,
        )

        if not path:
            break

        quality = _path_quality(path, frame_step_seconds)

        if (
            len(path) >= min_fragment_points
            and quality["duration"] >= min_fragment_duration
        ):
            fragments.append(
                {
                    "time": np.asarray(
                        [times[peak.frame] for peak in path],
                        dtype=np.float32,
                    ),
                    "frequency": np.asarray(
                        [peak.frequency for peak in path],
                        dtype=np.float32,
                    ),
                    "confidence": np.asarray(
                        [peak.confidence for peak in path],
                        dtype=np.float32,
                    ),
                    "fragment_count": 1,
                }
            )

        # Remove the entire selected ridge, including short gaps.
        _suppress_path(path, peaks_by_frame, suppression_band_hz)

    fragments_before_merge = len(fragments)

    # Merge fragments that have compatible frequency and slope.
    if merge_fragments and len(fragments) > 1:

        fragments = _merge_fragments(
            fragments,
            frame_step_seconds,
            max_gap_s=merge_max_gap_s,
            frequency_tolerance_hz=merge_frequency_tolerance_hz,
            slope_tolerance_hz_s=merge_slope_tolerance_hz_s,
        )

    # Recalculate quality after merging.
    curves: list[_CurveData] = []
    for fragment in fragments:
        quality = _quality_from_arrays(fragment, frame_step_seconds)

        accepted = (
            len(fragment["time"]) >= min_points
            and quality["duration"] >= min_duration_s
            and quality["coverage"] >= min_coverage
            and quality["frequency_excursion"] >= min_excursion_hz
            and quality["motion_fraction"] >= min_motion_fraction
            and quality["mean_confidence"] >= min_mean_confidence
        )

        if accepted:
            curves.append(
                {
                    "time": fragment["time"],
                    "frequency": fragment["frequency"],
                    "confidence": fragment["confidence"],
                    "fragment_count": fragment.get("fragment_count", 1),
                    **quality,
                }
            )

    # Long and coherent curves rank above short, bright fragments.
    curves.sort(key=lambda curve: curve["score"], reverse=True)
    curves = curves[:max_curves]

    if not return_debug:
        return curves

    debug = {
        "time": times,
        "frequency": frequencies,
        "spectrogram_db": spec_db,
        "enhanced_db": enhanced,
        "effective_sample_rate": effective_rate,
        "frame_step_seconds": frame_step_seconds,
        "frequency_resolution": frequency_resolution,
        "fragments_before_merge": fragments_before_merge,
        "curves_after_merge": len(fragments),
    }

    return curves, debug


def plot_detection(
    debug: dict[str, Any],
    curves: list[_CurveData],
    *,
    use_enhanced: bool = False,
    figsize: tuple[float, float] = (14.0, 7.0),
) -> tuple[Any, Any]:
    """Plot the detected curves over the original or background-suppressed STFT."""
    import matplotlib.pyplot as plt

    matrix = (
        debug["enhanced_db"] if use_enhanced else debug["spectrogram_db"]
    )
    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(
        debug["time"],
        debug["frequency"],
        matrix,
        shading="auto",
        cmap="magma",
    )
    colorbar_label = (
        "Local contrast (dB)" if use_enhanced else "Relative level (dB)"
    )
    fig.colorbar(mesh, ax=ax, label=colorbar_label)

    for index, curve in enumerate(curves, start=1):
        ax.plot(
            curve["time"],
            curve["frequency"],
            linewidth=1.8,
            marker=".",
            markersize=3,
            label=f"Curve {index} (conf={curve['mean_confidence']:.2f})",
        )

    ax.set_xlabel("Time from recording start (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Automatically detected spectral ridges")
    if curves:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def save_curves_csv(curves: list[_CurveData], filename: str | Path) -> None:
    """Save all detected observations in a long-form CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["curve_id", "time", "frequency", "confidence"])
        for curve_id, curve in enumerate(curves):
            for time, frequency, confidence in zip(
                curve["time"], curve["frequency"], curve["confidence"]
            ):
                writer.writerow(
                    [
                        curve_id,
                        f"{float(time):.6f}",
                        f"{float(frequency):.6f}",
                        f"{float(confidence):.4f}",
                    ]
                )


def save_best_curve_csv(
    curves: list[_CurveData],
    filename: str | Path,
) -> None:
    """Save observations from the highest-ranked curve to CSV."""
    curve = curves[0]
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "frequency_hz"])
        for time, frequency in zip(curve["time"], curve["frequency"]):
            writer.writerow([f"{float(time):.6f}", f"{float(frequency):.6f}"])


if __name__ == "__main__":
    # Quick command-line example:
    #   python curve_detection.py recording.wav
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python curve_detection.py recording.wav")

    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit("Install soundfile first: pip install soundfile") from exc

    audio, sr = sf.read(sys.argv[1], always_2d=False)
    detected, diagnostic = detect_curves(audio, sr, return_debug=True)
    print(f"Detected {len(detected)} candidate curve(s)")
    for index, curve in enumerate(detected, start=1):
        print(
            f"{index}: {curve['duration']:.1f}s, "
            f"coverage={curve['coverage']:.2f}, "
            f"confidence={curve['mean_confidence']:.2f}, "
            f"excursion={curve['frequency_excursion']:.1f}Hz, "
            f"motion={curve['motion_fraction']:.2f}"
        )

    save_best_curve_csv(detected, "data/detected_curves.csv")
    plot_detection(diagnostic, detected)
    import matplotlib.pyplot as plt

    plt.show()