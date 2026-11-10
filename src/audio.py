import numpy as np
from scipy.interpolate import CubicSpline, interp1d


def embed_thick_doppler(
    time_points,
    freq_points,
    sample_rate=44100,
    tone_amplitude=0.0002,
    noise_amplitude=0.0003,
    line_thickness=3.0,
    harmonics=4,
    modulation_depth=0.08,
    modulation_rate=2.5,
    fade_fraction=0.15,
    control_rate=200,
    seed=None,
):
    """Generate a synthetic engine sound from a Doppler frequency curve."""
    time_points = np.asarray(time_points, dtype=np.float64)
    freq_points = np.asarray(freq_points, dtype=np.float64)

    if time_points.ndim != 1 or freq_points.ndim != 1:
        raise ValueError("time_points and freq_points must be one-dimensional")
    if len(time_points) < 2:
        raise ValueError("At least two points are required")
    if len(time_points) != len(freq_points):
        raise ValueError("time_points and freq_points must have the same length")
    if not np.all(np.isfinite(time_points)) or not np.all(np.isfinite(freq_points)):
        raise ValueError("Input points must contain only finite values")
    if np.any(np.diff(time_points) <= 0):
        raise ValueError("time_points must be strictly increasing")
    if sample_rate <= 0 or control_rate <= 0 or harmonics < 1:
        raise ValueError("sample_rate, control_rate, and harmonics must be positive")
    if not 0 < fade_fraction <= 1:
        raise ValueError("fade_fraction must be in the range (0, 1]")

    rng = np.random.default_rng(seed)
    start_time = time_points[0]
    duration = time_points[-1] - start_time
    sample_count = int(duration * sample_rate)
    if sample_count < 2:
        raise ValueError("The curve must span at least two audio samples")

    audio_time = np.arange(sample_count) / sample_rate
    absolute_time = audio_time + start_time

    base_frequency = CubicSpline(
        time_points,
        freq_points,
        bc_type="natural",
    )(absolute_time)

    control_count = int(duration * control_rate) + 2
    control_time = np.linspace(0, duration, control_count)
    frequency_jitter = interp1d(
        control_time,
        rng.normal(0, line_thickness, control_count),
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(audio_time)
    frequency = base_frequency + frequency_jitter

    normalized_time = audio_time / duration
    fade_in = np.clip(normalized_time / fade_fraction, 0, 1)
    fade_out = np.clip((1 - normalized_time) / fade_fraction, 0, 1)
    envelope = np.minimum(fade_in, fade_out) ** 0.7

    modulation = 1 + modulation_depth * np.sin(
        2 * np.pi * modulation_rate * audio_time
    )
    amplitude_variation = interp1d(
        control_time,
        rng.normal(0, 0.025, control_count),
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(audio_time)
    amplitude_envelope = envelope * modulation * (1 + amplitude_variation)

    phase = np.empty(sample_count, dtype=np.float64)
    phase[0] = 0
    phase[1:] = (
        2
        * np.pi
        / sample_rate
        * np.cumsum((frequency[:-1] + frequency[1:]) / 2)
    )

    audio = np.zeros(sample_count, dtype=np.float32)
    for harmonic in range(1, harmonics + 1):
        amplitude = tone_amplitude / harmonic**1.3
        audio += (amplitude * np.sin(harmonic * phase)).astype(np.float32)

    audio += rng.normal(0, noise_amplitude, sample_count).astype(np.float32)
    audio *= amplitude_envelope.astype(np.float32)

    return absolute_time, audio
