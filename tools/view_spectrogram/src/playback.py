import numpy as np
from scipy.signal import butter


def build_playback_filter(low, high, sample_rate):
    nyquist = sample_rate / 2.0
    if not (0 <= low < high <= nyquist):
        raise ValueError(
            f"Enter frequencies satisfying 0 <= Low < High <= {nyquist:.1f} Hz."
        )

    if low == 0 and high >= nyquist:
        return None
    if low == 0:
        return butter(
            4, high, btype="lowpass", fs=sample_rate, output="sos"
        ).astype(np.float32)
    if high >= nyquist:
        return butter(
            4, low, btype="highpass", fs=sample_rate, output="sos"
        ).astype(np.float32)
    return butter(
        4, [low, high], btype="bandpass", fs=sample_rate, output="sos"
    ).astype(np.float32)


def format_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:04.1f}"
