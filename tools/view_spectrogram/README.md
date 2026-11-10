# Audio spectrogram viewer

The audio spectrogram viewer is a desktop application for inspecting recordings
and exporting a selected spectrogram region for manual frequency tracing. It
does not estimate the aircraft position itself; its exported snapshot is the
input to the [curve tool](../curve_tool/README.md).

Run commands from the DopplerScope project root.

## Installation

Install the viewer dependencies:

```sh
python3 -m pip install numpy scipy soundfile sounddevice PySide6 pyqtgraph
```

`sounddevice` sends audio to the system's playback device. On some systems it
may also require a PortAudio runtime provided by the operating system.

## Start the application

```sh
python3 tools/view_spectrogram/main.py
```

Choose **Open audio…** and select a recording. The file dialog supports WAV,
FLAC, OGG, AIFF, AIF, and AU files through SoundFile/libsndfile.

## Workflow

1. Open a recording.
2. Set a useful time and frequency range. Use zoom and pan to inspect the
   signal, or use **Select range** to select a time segment.
3. Adjust the FFT size, hop length, and channel to refine the analysis.
4. Choose **Export visible data…** to save the currently displayed segment.
5. Open the saved snapshot in
   [`tools/curve_tool`](../curve_tool/README.md), trace the signal, and export
   its time/frequency points as CSV.

The spectrogram is recalculated for the visible interval when the view or
analysis settings change. This keeps the displayed result responsive for long
recordings. The playback filter affects playback only; it does not change the
spectrogram or the exported data.

## Controls

### Analysis and display

- **FFT size:** window sizes from 256 to 32768 samples. Larger windows provide
  finer frequency resolution but reduce time resolution.
- **Hop length:** frame spacing from 16 to 4096 samples. Smaller hops provide
  denser time sampling; the effective hop is limited to the FFT size.
- **Channel:** analyze the mono mix, left channel, or right channel.
- **Minimum / maximum frequency:** set the displayed frequency interval in
  hertz. The maximum cannot exceed the recording's Nyquist frequency.
- **Dynamic range:** set the displayed dB range.
- **From / To:** enter the visible time interval in seconds. Use **Show
  segment** to apply it or **Full recording** to return to the full duration.

### Graph navigation

- Scroll to zoom around the pointer.
- Drag to pan.
- Right-drag to zoom to a rectangle.
- Click **Select range**, then drag across the graph to choose a time segment.
- Drag the red playhead to seek in the audio.

### Playback

Use **Play**, **Pause**, and **Stop**, or drag the seek slider. The optional
filter is specified by low and high cutoff frequencies in hertz:

- low cutoff `0` and high cutoff at or above Nyquist: no filter;
- low cutoff `0`: low-pass filter;
- high cutoff at Nyquist: high-pass filter;
- otherwise: band-pass filter.

Applying a playback filter does not modify the source file.

## Exported snapshot

**Export visible data…** writes a pickle file (default name
`spectrogram_segment.pkl`) containing a dictionary with:

- `spectrogram_db`: two-dimensional dB matrix;
- `time_edges`: time coordinates for the matrix columns, in seconds;
- `frequency_edges`: frequency coordinates for the matrix rows, in hertz;
- `metadata`: source filename, sample rate, FFT and hop settings, selected
  channel, dynamic range, and requested/displayed time and frequency ranges.

The snapshot records the visible analysis region, not the original audio and
not the manually traced curve. The curve tool uses the matrix and edge arrays
to place frequency points in the correct time/frequency coordinates.

Pickle files can execute code when opened. Only load snapshots created by this
project or obtained from a source you trust.

## Implementation

The application entry point is `main.py`. The `src/` package separates the Qt
interface and playback worker (`qt_viewer.py`), spectrogram calculation
(`spectrogram.py`), and playback filter/time helpers (`playback.py`).
