# DopplerScope

DopplerScope is a Python project for estimating where an aircraft recording was made from its Doppler shift.

The basic idea is to take an aircraft's flight track, find the aircraft's sound in a recording, trace its frequency over time, and compare that curve with the Doppler curves predicted for different observer locations.

There are two main ways to use the project:

- **Estimate a location:** load a recording, trace the Doppler line in its spectrogram, and run the optimizer.
- **Simulate a recording:** use a known flight track and observer location to generate a Doppler curve, then optionally turn it into a WAV file.

## Installation

The project requires Python 3.

Install the main dependencies:

```sh
python3 -m pip install numpy pandas scipy matplotlib "xarray[netcdf]"
```

For audio-based curve detection in `estimate`, and for the spectrogram viewer and curve tool:

```sh
python3 -m pip install PySide6 pyqtgraph soundfile sounddevice
```

For the ADS-B track recorder:

```sh
python3 -m pip install requests
```

### Terrain data

The `simulate` and `estimate` commands use `data/GEBCO_2026.nc` by default. Download all `height_data.tar.gz.*` parts from the same GitHub Release into `data/`, then extract them:

```sh
cd data
cat height_data.tar.gz.* | tar -xvzf -
cd ..
```

The resulting file should be `data/GEBCO_2026.nc`.

## Quick start

### 1. Record or get a flight track

You can use your own ADS-B track or record one with:

```sh
python3 tools/record_track.py
```

The recorder uses the `adsb.lol` API, lets you select an aircraft, and saves its positions to a CSV file. Press `Ctrl+C` to stop recording.

The default track path used by some commands is `data/49d665_track.csv`. That file is not included in the repository, so pass the path to your own track if needed.

### 2. Open an audio recording

```sh
python3 tools/view_spectrogram/main.py
```

Open a recording and find a visible aircraft-related frequency line. The viewer supports WAV, FLAC, OGG, AIFF, and AU files.

You can zoom and pan, select a time range, adjust the frequency range and display contrast, and listen to the recording with optional playback filters. Export the visible part of the spectrogram using **Export visible data…**.

### 3. Extract the Doppler curve

You can have `estimate` detect a curve from an audio recording and use it directly:

```sh
python3 main.py estimate \
  --audio path/to/recording.wav \
  --track data/test_track.csv \
  --curve-min-frequency 200 \
  --curve-max-frequency 2000
```

The detector ranks candidate curves by quality and uses the highest-ranked one by default. If it finds multiple curves, use `--curve-index` to choose another (indices start at 0). To save the selected curve for review or reuse, add `--curve-output data/detected_curve.csv`. Adjust the frequency range to cover the line in your recording.

You can also trace the curve manually. Open a spectrogram snapshot in the curve tool:

```sh
python3 tools/curve_tool/main.py
```

Add points along the frequency line, then export the result as a CSV with two columns:

```csv
time_s,frequency_hz
12.5,612.3
13.0,608.1
```

The times are in seconds and the frequencies are in hertz.

### 4. Estimate the observer position

Run the optimizer with the extracted or manually measured curve and the aircraft track. For an existing curve CSV:

```sh
python3 main.py estimate \
  --observations data/observed_curve.csv \
  --track data/test_track.csv
```

The result includes the estimated latitude and longitude, the offset from the reference point, the estimated time offset, source frequency, and fit variance.

If you want to use the bundled example curve, pass `--observations data/observed_curve.csv`. Otherwise, use the CSV exported from the curve tool.

## Why does it work?

The method relies on a simple property of the Doppler effect: the frequency heard by an observer depends on how quickly the aircraft is moving towards or away from them.

When the aircraft approaches, the observed frequency is higher. After it passes and moves away, the observed frequency becomes lower. The important part is that this change depends on the observer's position, because moving the observer changes the aircraft's radial velocity throughout the flight.

For a sound with source frequency `f_source`, the received frequency is modelled as:

```text
f_received = f_source * c / (c + v_radial)
```

Here, `c` is the speed of sound (343 m/s in the current model), and `v_radial` is the aircraft's velocity component along the line between the aircraft and the observer. Positive radial velocity means the aircraft is moving away.

The optimizer works backwards from the measured curve:

1. It takes a possible observer position and a possible time offset between the recording and the flight track.
2. For every measured point, it finds the corresponding aircraft position and velocity.
3. It accounts for the time sound needs to travel from the aircraft to the observer. In other words, the sound was emitted earlier than the moment it was recorded.
4. Using the aircraft's radial velocity, it reverses the Doppler shift and estimates the frequency that the aircraft must have emitted.
5. It repeats this for all points in the curve and checks how consistent those estimated source frequencies are.

The actual emitted frequency is not expected to jump around just because the aircraft moves. So, if a candidate observer position is correct, reversing the Doppler shift should give roughly the same source frequency for the whole curve.

That gives the optimizer a score: the variance of the estimated source frequencies. A lower variance means the candidate position explains the measured curve better.

The search first checks a coarse grid of possible east/west positions, north/south positions, and time offsets. It then refines the best candidates with Powell optimization. The mean of the inferred source frequencies is reported as the estimated source frequency.

This is why the method can estimate a position without knowing the exact emitted frequency beforehand: the frequency is estimated together with the observer position, while the optimizer looks for the position that makes it most consistent.

The result still depends on the quality of the inputs. A noisy or incorrectly traced spectrogram line, an inaccurate flight track, or a source whose frequency changes significantly during the recording can make the estimate less reliable. Terrain data is also useful when the observer is not at sea level.

## Command-line usage

Run all commands from the project root. Use `--help` to see the available options:

```sh
python3 main.py --help
python3 main.py estimate --help
```

### `track`

Load a flight CSV, show its duration and number of points, and optionally plot the interpolated 3D path.

```sh
python3 main.py track [--track PATH] [--origin-lat DEGREES] \
  [--origin-lon DEGREES] [--plot]
```

| Option | Default | Description |
| --- | --- | --- |
| `--track` | `data/49d665_track.csv` | Flight track CSV. |
| `--origin-lat` | `51.364946` | Reference latitude. |
| `--origin-lon` | `0.856076` | Reference longitude. |
| `--plot` | Off | Display the 3D flight path. |

### `simulate`

Generate a Doppler curve for a known observer position.

```sh
python3 main.py simulate \
  --observer-lat DEGREES --observer-lon DEGREES \
  --source-frequency HZ [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--track` | `data/49d665_track.csv` | Flight track CSV. |
| `--origin-lat` | `51.364946` | Reference latitude. |
| `--origin-lon` | `0.856076` | Reference longitude. |
| `--observer-lat` | Required | Observer latitude. |
| `--observer-lon` | Required | Observer longitude. |
| `--observer-alt` | `0` m | Observer altitude when terrain is disabled. |
| `--source-frequency` | Required | Emitted frequency in hertz. |
| `--dt` | `0.1` s | Simulation sample interval. |
| `--elevation` | `data/GEBCO_2026.nc` | Terrain dataset. |
| `--no-elevation` | Off | Disable terrain data. |
| `--map-width` | `10000` m | Terrain grid width. |
| `--map-height` | `10000` m | Terrain grid height. |
| `--map-step` | `100` m | Terrain grid spacing. |
| `--output` | `data/simulated_curve.csv` | Output CSV path. |

Example:

```sh
python3 main.py simulate \
  --track data/test_track.csv \
  --observer-lat 51.364946 --observer-lon 0.856076 \
  --source-frequency 600
```

### `estimate`

Estimate the observer position from a measured Doppler curve.

```sh
python3 main.py estimate [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--track` | `data/49d665_track.csv` | Flight track CSV. |
| `--observations` | `data/observed_curve.csv` | Measured curve CSV. Legacy tuple pickles are also accepted. Mutually exclusive with `--audio`. |
| `--audio` | — | Audio file to detect a curve from instead of loading `--observations` (requires `soundfile`). |
| `--curve-min-frequency` | `200` Hz | Lowest frequency searched in the audio. |
| `--curve-max-frequency` | `2000` Hz | Highest frequency searched in the audio. |
| `--curve-max-curves` | `8` | Maximum number of candidate curves to detect. |
| `--curve-index` | `0` | Zero-based candidate index to use; 0 is the highest-ranked curve. |
| `--curve-output` | — | Optional CSV path for saving the selected detected curve. |
| `--origin-lat` | `51.364946` | Latitude of the reference point around which the observer position is searched. |
| `--origin-lon` | `0.856076` | Longitude of the reference point around which the observer position is searched. |
| `--width` | `10000` m | East/west search width. |
| `--height` | `10000` m | North/south search height. |
| `--elevation` | `data/GEBCO_2026.nc` | Terrain dataset. |
| `--no-elevation` | Off | Disable terrain data and use zero observer elevation. |
| `--grid-step` | `100` m | Terrain grid spacing. |
| `--delta-min` | `0` s | Minimum time offset. |
| `--delta-max` | Automatic | Maximum time offset that fits the observations in the track. |
| `--coarse-xy-step` | `2000` m | Coarse search spacing. |
| `--coarse-time-step` | `30` s | Coarse time-offset spacing. |
| `--refine-starts` | `3` | Number of coarse candidates to refine. |
| `--min-valid-fraction` | `1.0` | Minimum fraction of observations that must fit the track. |

For each new recording, set both `--origin-lat` and `--origin-lon` to a reference point near the observer: the optimizer searches around that point. The listed defaults are example coordinates, not suitable assumptions for every recording.

Example without terrain data:

```sh
python3 main.py estimate \
  --observations data/observed_curve.csv \
  --track data/test_track.csv \
  --no-elevation
```

### `synthesize`

Create a synthetic engine-like WAV file from a time/frequency curve.

```sh
python3 main.py synthesize [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--curve` | `data/simulated_curve.csv` | Input curve CSV. |
| `--output` | `data/doppler.wav` | Output WAV file. |
| `--sample-rate` | `44100` | Sample rate in Hz. |
| `--tone-amplitude` | `0.0002` | Fundamental tone amplitude. |
| `--noise-amplitude` | `0.0003` | Noise amplitude. |
| `--line-thickness` | `3.0` Hz | Frequency jitter standard deviation. |
| `--harmonics` | `4` | Number of harmonics. |
| `--modulation-depth` | `0.08` | Amplitude modulation depth. |
| `--modulation-rate` | `2.5` Hz | Amplitude modulation rate. |
| `--fade-fraction` | `0.15` | Fade-in/fade-out fraction. |
| `--control-rate` | `200` samples/s | Random modulation update rate. |
| `--seed` | Random | Optional seed for repeatable output. |

Example:

```sh
python3 main.py simulate \
  --observer-lat 51.364946 --observer-lon 0.856076 \
  --source-frequency 600

python3 main.py synthesize --curve data/simulated_curve.csv
```

## Other tools

### Spectrogram viewer

```sh
python3 tools/view_spectrogram/main.py
```

The viewer includes FFT and hop-length settings, channel selection, frequency and dynamic-range controls, time selection, zooming, panning, and playback. Playback can use low-pass, high-pass, or band-pass filtering.

### Curve tool

```sh
python3 tools/curve_tool/main.py
```

The curve tool opens a spectrogram snapshot and lets you add, move, and delete points before exporting the curve as CSV.

**Note:** pickle files can execute code when opened. Only open snapshots and legacy observation pickles from sources you trust.

### Flight track recorder

```sh
python3 tools/record_track.py
```

The recorder polls the `adsb.lol` API, lets you choose an aircraft, and saves its positions until you stop it with `Ctrl+C`. The search location and radius are configured in the script.

Track CSV files contain these columns:

```text
timestamp, icao, flight, lat, lon, alt_baro, gs, track
```

The loader converts timestamps to elapsed seconds, altitude from feet to meters, and ground speed from knots to meters per second.

## Data files

Recordings, generated files, local flight tracks, spectrogram snapshots, and terrain data are not included in the repository.

| File | Description |
| --- | --- |
| `data/test_track.csv` | Example flight track. |
| `data/observed_curve.csv` | Example measured Doppler curve. |
| `data/GEBCO_2026.nc` | Terrain dataset downloaded separately. |

## Project structure

```text
main.py
src/
  audio.py
  doppler.py
  geo.py
tools/
  record_track.py
  view_spectrogram/
  curve_tool/
data/
```

Main modules:

- `src/geo.py` — flight-track loading, interpolation, and terrain data.
- `src/doppler.py` — Doppler simulation and observer-position optimization.
- `src/audio.py` — synthetic audio generation.
