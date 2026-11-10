# Spectrogram curve tool

The curve tool is a desktop application for manually tracing a measured
frequency line on a spectrogram snapshot. It exports the trace as a simple CSV
that the DopplerScope optimizer can read.

Run commands from the DopplerScope project root.

## Installation

Install the curve tool dependencies:

```sh
python3 -m pip install numpy PySide6 pyqtgraph
```

The spectrogram snapshot must first be exported by the
[audio spectrogram viewer](../view_spectrogram/README.md).

## Start the application

```sh
python3 tools/curve_tool/main.py
```

Choose **Open snapshot…** and select a spectrogram snapshot. The application
opens pickle snapshots produced by the viewer.

## Trace a curve

1. Adjust **Dynamic range** to make the signal easier to see.
2. Leave **Add points** enabled and click along the observed frequency line.
   The horizontal axis is time in seconds and the vertical axis is frequency in
   hertz.
3. Drag any point to refine its position. Points are constrained to the
   snapshot's time and frequency bounds.
4. To remove a point, turn off **Add points**, select the point, and click
   **Delete selected**. Use **Clear curve** to remove all points.
5. Add at least two points and choose **Export curve…**.

Mouse controls:

- Scroll to zoom around the pointer.
- Drag the plot to pan.
- Right-drag to zoom to a rectangle.
- Click to add points while **Add points** is enabled.
- Drag a point to move it.

Use **Zoom +**, **Zoom −**, and **Fit** for plot navigation.

## Export format and location

The default filename is:

```text
data/observed_curve.csv
```

The save dialog can be used to select another destination. The exported CSV
has a header and two columns; points are sorted by time before writing:

```csv
time_s,frequency_hz
12.5,612.3
13.0,608.1
```

`time_s` is in seconds and `frequency_hz` is in hertz. The optimizer shifts
observation times so the first sample is at zero before fitting the curve to a
flight track.

## Use the curve in DopplerScope

If you saved to the default location, run from the project root:

```sh
python3 main.py optimize
```

For a curve saved elsewhere, pass its path:

```sh
python3 main.py optimize --observations path/to/curve.csv
```

See the main [DopplerScope README](../../README.md) for optimizer settings and
terrain handling.

## Snapshot safety

Spectrogram snapshots use Python pickle format. Pickle files can execute code
when opened; only load snapshots created by this project or obtained from a
source you trust.
