import math

import numpy as np
import soundfile as sf
from scipy.fft import rfft
from scipy.signal import get_window


# Keep the rendered mesh near plot pixel density; zooming recomputes detail.
MAX_DISPLAY_COLUMNS = 2048
MAX_ANALYSIS_FRAMES = 60000
MAX_FREQUENCY_ROWS = 900
MAX_BATCH_READ_SECONDS = 2.0
FFT_BATCH_LIMIT = 128


def calculate_visible_spectrogram(
    file_path,
    sample_rate,
    total_samples,
    start_time,
    end_time,
    fft_size,
    hop_length,
    min_frequency,
    max_frequency,
    channel_mode,
    cancel_event=None,
):
    """Calculate a bounded, sample-aligned spectrogram for a visible interval."""
    nyquist = sample_rate / 2.0
    if (
        not math.isfinite(min_frequency)
        or not math.isfinite(max_frequency)
        or min_frequency < 0
        or min_frequency >= max_frequency
        or max_frequency > nyquist
    ):
        raise ValueError("Frequency range must be within 0 Hz and Nyquist.")

    hop_length = min(int(hop_length), int(fft_size))
    window = get_window("hann", fft_size, fftbins=True).astype(np.float32)
    window_sum = float(np.sum(window))

    first_sample = max(0, int(math.floor(start_time * sample_rate)))
    last_sample = min(total_samples, int(math.ceil(end_time * sample_rate)))
    sample_count = last_sample - first_sample
    if sample_count <= 0:
        return None

    hop_multiplier = max(
        1,
        int(math.ceil(sample_count / (MAX_ANALYSIS_FRAMES * hop_length))),
    )
    effective_hop = hop_length * hop_multiplier

    first_center = (first_sample // effective_hop) * effective_hop
    last_center = ((last_sample + effective_hop - 1) // effective_hop) * effective_hop
    centers = np.arange(
        first_center, last_center + 1, effective_hop, dtype=np.int64
    )
    if centers.size == 0:
        centers = np.array([(first_sample + last_sample) // 2], dtype=np.int64)
    frame_count = int(centers.size)
    display_columns = min(MAX_DISPLAY_COLUMNS, frame_count)
    bin_spacing = sample_rate / fft_size
    last_available_bin = fft_size // 2
    first_bin = min(
        last_available_bin,
        max(0, int(math.ceil(min_frequency / bin_spacing))),
    )
    last_bin = min(
        last_available_bin + 1,
        int(math.floor(max_frequency / bin_spacing)) + 1,
    )
    if last_bin <= first_bin:
        first_bin = min(
            last_available_bin,
            max(0, int(round((min_frequency + max_frequency) / (2 * bin_spacing)))),
        )
        last_bin = first_bin + 1
    selected_bin_count = last_bin - first_bin

    frequency_rows = min(MAX_FREQUENCY_ROWS, selected_bin_count)
    row_edges = np.linspace(
        0, selected_bin_count, frequency_rows + 1, dtype=np.int32
    )
    output = np.full(
        (frequency_rows, display_columns), -160.0, dtype=np.float32
    )
    batch_size = int(MAX_BATCH_READ_SECONDS * sample_rate / effective_hop)
    batch_size = max(1, min(FFT_BATCH_LIMIT, batch_size))

    with sf.SoundFile(file_path, mode="r") as audio_file:
        batch_start = 0
        while batch_start < frame_count:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Spectrogram calculation was superseded.")

            batch_end = min(frame_count, batch_start + batch_size)
            batch_centers = centers[batch_start:batch_end]
            frame_starts = batch_centers - fft_size // 2
            requested_start = int(frame_starts[0])
            requested_end = int(frame_starts[-1] + fft_size)
            read_start = min(total_samples, max(0, requested_start))
            read_end = max(read_start, min(total_samples, requested_end))

            audio_file.seek(read_start)
            data = audio_file.read(
                read_end - read_start, dtype="float32", always_2d=True
            )
            if channel_mode == "Left":
                audio = data[:, 0]
            elif channel_mode == "Right":
                audio = data[:, min(1, data.shape[1] - 1)]
            elif data.shape[1] == 1:
                audio = data[:, 0]
            else:
                audio = np.mean(data, axis=1, dtype=np.float32)

            buffer = np.zeros(requested_end - requested_start, dtype=np.float32)
            if audio.size:
                insert_start = read_start - requested_start
                buffer[insert_start:insert_start + audio.size] = audio

            local_starts = (frame_starts - requested_start).astype(np.int64)
            all_windows = np.lib.stride_tricks.sliding_window_view(buffer, fft_size)
            frames = all_windows[local_starts]
            spectrum = np.abs(rfft(frames * window[None, :], axis=1))
            spectrum *= np.float32(2.0 / window_sum)
            spectrum[:, 0] *= 0.5
            if fft_size % 2 == 0:
                spectrum[:, -1] *= 0.5

            spectrum = spectrum[:, first_bin:last_bin]
            reduced_frequency = np.maximum.reduceat(
                spectrum, row_edges[:-1], axis=1
            )
            db = 20.0 * np.log10(
                np.maximum(reduced_frequency, np.float32(1e-8))
            ).astype(np.float32, copy=False)

            column_ids = (
                np.arange(batch_start, batch_end, dtype=np.int64)
                * display_columns // frame_count
            )
            if display_columns == frame_count:
                output[:, batch_start:batch_end] = db.T
            else:
                columns, group_starts = np.unique(
                    column_ids, return_index=True
                )
                pooled = np.maximum.reduceat(db, group_starts, axis=0)
                output[:, columns] = np.maximum(output[:, columns], pooled.T)
            batch_start = batch_end

    frame_edge_indices = np.ceil(
        np.arange(display_columns + 1, dtype=np.float64)
        * frame_count / display_columns
    ).astype(np.int64)
    time_edges = (
        centers[0] - effective_hop / 2.0
        + frame_edge_indices * effective_hop
    ) / sample_rate
    frequency_edges = np.linspace(
        min_frequency, max_frequency, frequency_rows + 1, dtype=np.float64
    )
    return output, time_edges, frequency_edges, effective_hop
