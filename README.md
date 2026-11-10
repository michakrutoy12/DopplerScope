# DopplerScope

DopplerScope is a small Python toolkit for studying the Doppler shift of sound
from aircraft. It combines recorded flight tracks, audio spectrograms, manually
traced frequency curves, terrain elevation, and a numerical observer-position
optimizer.

The project has two ways to work with a Doppler curve:

1. **Measure one:** open a recording in the spectrogram viewer, export a
   spectrogram snapshot, trace the signal in the curve tool, and optimize the
   observer position from the resulting CSV.
2. **Generate one:** calculate a Doppler curve from a flight track and a
   specified observer location, then optionally synthesize a WAV from that
   curve.

All commands below are run from the project root.

## Requirements and installation

Use Python 3 and install the dependencies for the features you need. The
repository does not currently include a dependency lockfile or a
`requirements.txt`.

The command-line application uses NumPy, pandas, SciPy, Matplotlib, and Xarray.
The `netcdf` extra installs Xarray's common NetCDF backends:

```sh
python3 -m pip install numpy pandas scipy matplotlib "xarray[netcdf]"
```

For the audio viewer and curve tool, install the GUI and audio packages:

```sh
python3 -m pip install PySide6 pyqtgraph soundfile sounddevice
```

The aircraft track recorder also requires `requests`:

```sh
python3 -m pip install requests
```

The terrain-backed commands expect the included `data/GEBCO_2026.nc` file to
be readable by Xarray and to contain an `elevation` variable on `lat` and
`lon` coordinates.

## Quick start: measure a curve and estimate the observer

### 1. Open a recording and export its spectrogram

```sh
python3 tools/view_spectrogram/main.py
```

Choose **Open audio…** and select a recording. The viewer supports WAV, FLAC,
OGG, AIFF, and AU files through SoundFile. It calculates the visible
spectrogram on demand so you can change the time and frequency range without
rendering the entire recording at full resolution.

Use **Export visible data…** to save the currently displayed time segment as a
pickle snapshot. The snapshot contains the spectrogram in dB, time and
frequency bin edges, and analysis settings.

### 2. Trace the Doppler line

```sh
python3 tools/curve_tool/main.py
```

Choose **Open snapshot…** and select the snapshot from the previous step. Add
points along the signal in time order; drag points to refine their positions.
At least two points are required for export. **Export curve…** saves the
measurement as:

```text
tools/curve_tool/observed_curve.csv
```

The CSV has a header and two columns:

```csv
time_s,frequency_hz
12.5,612.3
13.0,608.1
```

Times are seconds and frequencies are hertz. The points are sorted by time on
export. The curve tool can also save to another path.

### 3. Estimate the observer position

```sh
python3 main.py optimize
```

The optimizer loads the exported curve, the default flight track
(`data/49d665_track.csv`), and the terrain dataset
(`data/GEBCO_2026.nc`). It prints the estimated latitude and longitude,
horizontal offsets from the reference point, terrain elevation, time offset,
estimated source frequency, fit variance, and optimizer status.

To use another curve or track:

```sh
python3 main.py optimize \
  --observations path/to/curve.csv \
  --track data/47a3b0_track.csv
```

The bundled `data/observed_curve.csv` is an example curve; select it explicitly
with `--observations data/observed_curve.csv` if you want to use it instead of
the curve-tool export.

## Command-line reference

Show all commands:

```sh
python3 main.py --help
```

Show options for one command:

```sh
python3 main.py optimize --help
```

Explicit relative paths are resolved from the current working directory.
Default input and output paths are anchored to the project directory.

### `track`

Load a flight CSV, report its duration and point count, and optionally display
its interpolated 3D path.

```sh
python3 main.py track [--track PATH] [--origin-lat DEGREES] \
  [--origin-lon DEGREES] [--plot]
```

| Option | Default | Description |
| --- | --- | --- |
| `--track` | `data/49d665_track.csv` | Flight track CSV. |
| `--origin-lat` | `51.364946` | Reference latitude in decimal degrees. |
| `--origin-lon` | `0.856076` | Reference longitude in decimal degrees. |
| `--plot` | Off | Show the 3D interpolated flight path. |

### `simulate`

Calculate the Doppler-shifted frequency and sound arrival time for an observer
at a known location. Save the result to a CSV with `time_s` and
`frequency_hz` columns.

```sh
python3 main.py simulate \
  --observer-lat DEGREES --observer-lon DEGREES \
  --source-frequency HZ [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--track` | `data/49d665_track.csv` | Flight track CSV. |
| `--origin-lat` | `51.364946` | Reference latitude in decimal degrees. |
| `--origin-lon` | `0.856076` | Reference longitude in decimal degrees. |
| `--observer-lat` | Required | Observer latitude in decimal degrees. |
| `--observer-lon` | Required | Observer longitude in decimal degrees. |
| `--observer-alt` | `0` m | Observer altitude when terrain loading is disabled. |
| `--source-frequency` | Required | Emitted frequency in hertz. |
| `--dt` | `0.1` s | Interval between simulated samples. |
| `--elevation` | `data/GEBCO_2026.nc` | NetCDF elevation data, loaded with Xarray. |
| `--no-elevation` | Off | Skip terrain loading and use `--observer-alt`. |
| `--map-width` | `10000` m | East-west terrain grid width around the reference point. |
| `--map-height` | `10000` m | North-south terrain grid height around the reference point. |
| `--map-step` | `100` m | Terrain grid spacing. |
| `--output` | `data/simulated_curve.csv` | Output curve CSV. |

Example:

```sh
python3 main.py simulate \
  --observer-lat 51.364946 --observer-lon 0.856076 \
  --source-frequency 600
```

### `optimize`

Find the horizontal observer position and receive-time offset that minimize
the variance of the source frequencies inferred from the measured curve.
Optimization uses a coarse grid search followed by bounded Powell refinement.
By default, every observation must fit within the flight track.

```sh
python3 main.py optimize [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--track` | `data/49d665_track.csv` | Flight track CSV. |
| `--observations` | `tools/curve_tool/observed_curve.csv` | Observed curve CSV; legacy tuple pickles are also accepted. |
| `--origin-lat` | `51.364946` | Reference latitude in decimal degrees. |
| `--origin-lon` | `0.856076` | Reference longitude in decimal degrees. |
| `--width` | `10000` m | East-west search width, centered on the reference point. |
| `--height` | `10000` m | North-south search height, centered on the reference point. |
| `--elevation` | `data/GEBCO_2026.nc` | NetCDF elevation data, loaded with Xarray. |
| `--no-elevation` | Off | Skip terrain data and assume zero observer elevation. |
| `--grid-step` | `100` m | Terrain grid spacing. Keep the terrain grid large enough to cover the search area. |
| `--delta-min` | `0` s | Minimum receive-time offset. |
| `--delta-max` | Automatic | Maximum offset that fits the final observation in the flight duration. |
| `--coarse-xy-step` | `2000` m | Coarse search spacing on each horizontal axis. |
| `--coarse-time-step` | `30` s | Coarse search spacing for the time offset. |
| `--refine-starts` | `3` | Number of best coarse-grid candidates to refine. |
| `--min-valid-fraction` | `1.0` | Minimum fraction of observations that must fit the flight track; range `(0, 1]`. |

For example, to optimize from the bundled example curve without a terrain map:

```sh
python3 main.py optimize \
  --observations data/observed_curve.csv \
  --no-elevation
```

#### Terrain handling

Terrain is not rendered or exported as a separate step. For `simulate` and
`optimize`, the application opens the NetCDF file with `xarray.open_dataset`,
interpolates the `elevation` variable over a local latitude/longitude grid,
converts that grid to east/north meter coordinates, and passes both arrays
directly to the Doppler calculation. The grid is centered on the reference
latitude and longitude. Candidates outside the grid or over missing elevation
values cannot be evaluated.

Use `--no-elevation` to skip this process. In simulation the observer altitude
then comes from `--observer-alt`; in optimization it defaults to zero.

### `synthesize`

Generate a synthetic engine-like sound from a time/frequency CSV curve. The
CSV must have a header followed by at least two rows; the first two columns are
read as time in seconds and frequency in hertz.

```sh
python3 main.py synthesize [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--curve` | `data/simulated_curve.csv` | Input time/frequency CSV. |
| `--output` | `data/doppler.wav` | Output WAV file. |
| `--sample-rate` | `44100` samples/s | Output sample rate. |
| `--tone-amplitude` | `0.0002` | Fundamental tone amplitude. |
| `--noise-amplitude` | `0.0003` | Added noise amplitude. |
| `--line-thickness` | `3.0` Hz | Standard deviation of frequency jitter. |
| `--harmonics` | `4` | Number of harmonics in the sound. |
| `--modulation-depth` | `0.08` | Periodic amplitude modulation depth. |
| `--modulation-rate` | `2.5` Hz | Periodic amplitude modulation rate. |
| `--fade-fraction` | `0.15` | Fraction of the clip used for fade-in and fade-out; range `(0, 1]`. |
| `--control-rate` | `200` samples/s | Rate of the low-frequency random modulation. |
| `--seed` | Random | Optional seed for repeatable noise and modulation. |

Typical workflow:

```sh
python3 main.py simulate \
  --observer-lat 51.364946 --observer-lon 0.856076 \
  --source-frequency 600
python3 main.py synthesize --curve data/simulated_curve.csv
```

## Audio tools

### Spectrogram viewer

Run `python3 tools/view_spectrogram/main.py` to view an audio recording. The
viewer provides:

- FFT size, hop length, and channel selection;
- minimum and maximum display frequencies and dynamic range;
- time-range selection, zoom, pan, and a draggable playback position;
- playback controls and an optional low-, high-, or band-pass playback filter;
- export of the visible spectrogram region as a pickle snapshot.

Scroll to zoom around the pointer, drag to pan, and right-drag to zoom to a
rectangle. Use **Select range** to choose a time interval and **Show segment**
or **Full recording** to change the displayed range.

### Curve tool

Run `python3 tools/curve_tool/main.py` to open an exported spectrogram snapshot
and trace the observed frequency over time. The tool supports contrast
adjustment, zoom and pan, point insertion and dragging, point deletion, and
curve clearing. It exports time-sorted CSV points.

Pickle files can execute code when opened. Only load spectrogram snapshots and
legacy observation pickles from sources you trust.

## Flight track recorder

`tools/record_track.py` polls the public ADS-B Exchange-compatible
`adsb.lol` API for aircraft and appends the selected aircraft's positions to a
CSV. It currently uses a fixed search location and radius configured at the
top of the script, asks you to select an aircraft, and records until
interrupted with **Ctrl+C**. Its polling interval is set by the final
`record_track(5)` call.

Run it from the project root:

```sh
python3 tools/record_track.py
```

The recorder needs network access and the `requests` package. Track files use
the columns `timestamp`, `icao`, `flight`, `lat`, `lon`, `alt_baro`, `gs`, and
`track`. The loader in `src/geo.py` converts timestamps to elapsed seconds,
barometric altitude from feet to meters, and ground speed from knots to meters
per second.

## Data files

| File | Purpose |
| --- | --- |
| `data/49d665_track.csv`, `data/47a3b0_track.csv` | Example aircraft tracks. |
| `data/GEBCO_2026.nc` | Terrain elevation used by default for simulation and optimization. |
| `data/observed_curve.csv` | Example measured time/frequency curve. |
| `data/dermo.wav` | Example audio recording. |
| `data/result_noisy_doppler.wav` | Example audio with a noisy Doppler signal. |
| `data/test_segment.pkl` | Example spectrogram snapshot for the curve tool. |

## Project layout

```text
main.py                         Command-line entry point
src/
  audio.py                      Synthetic Doppler audio generation
  doppler.py                    Doppler simulation and observer optimization
  geo.py                        Flight-track interpolation and terrain grids
tools/
  record_track.py               ADS-B track recorder
  view_spectrogram/             Audio spectrogram viewer
  curve_tool/                   Manual curve extraction from a spectrogram
data/                           Example tracks, audio, curves, and terrain
```

The core functions are:

- `src.geo.load_flight`: loads and interpolates a flight track in local
  east/north coordinates.
- `src.geo.get_height_map`: interpolates terrain elevation with Xarray.
- `src.doppler.simulate_doppler`: calculates arrival times and Doppler-shifted
  frequencies for a known observer.
- `src.doppler.optimize_observer`: estimates observer position, terrain
  elevation, time offset, and source frequency from observations.
- `src.audio.embed_thick_doppler`: generates audio from a frequency curve.

## Doppler model

The flight track is converted to a local tangent plane relative to the
reference point. East-west and north-south positions are measured in meters;
aircraft altitude is also represented in meters. The simulator calculates the
aircraft's radial velocity relative to the observer and applies:

```text
f_received = f_source × c / (c + v_radial)
```

where `c` is the fixed speed of sound (343 m/s). Radial velocity is positive
when the aircraft is moving away from the observer. Arrival time includes
propagation delay:

```text
t_arrival = t_emission + distance / c
```

For optimization, measured times are shifted so the first observation is at
zero. The receive-time offset then positions this curve relative to the flight.
At each candidate observer position and time offset, Doppler correction
produces an estimate of the source frequency for every valid observation. The
optimizer minimizes the variance of those estimates; their mean is reported as
the estimated source frequency.

## Notes and limitations

- The optimizer searches a local east/north plane centered at the reference
  latitude and longitude; `x` is east/west and `y` is north/south, in meters.
- The optimizer's time offset is bounded by `--delta-min` and `--delta-max`.
  Make sure the observed curve and selected bounds can fit within the flight
  track.
- Terrain-aware search requires the full search area to lie within the
  interpolated terrain grid and contain usable elevation values.
- The simulator uses a constant speed of sound of 343 m/s.
- Track interpolation uses cubic splines. Input tracks should have enough
  distinct, valid timestamps and positions for interpolation.
- `main.py` is the supported command-line entry point. The audio and curve
  applications are separate GUI tools under `tools/`.
