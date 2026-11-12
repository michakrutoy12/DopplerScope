import argparse
import csv
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import xarray as xr

from src.audio import embed_thick_doppler
from src.doppler import optimize_observer, simulate_doppler
from src.geo import EARTH_RADIUS, get_height_map, load_flight, plot_track


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRACK = PROJECT_ROOT / "data" / "test_track.csv"
DEFAULT_ELEVATION = PROJECT_ROOT / "data" / "GEBCO_2026.nc"
DEFAULT_OBSERVATIONS = (
	PROJECT_ROOT / "data" / "observed_curve.csv"
)
DEFAULT_CURVE = PROJECT_ROOT / "data" / "simulated_curve.csv"


def _coordinate_map(width, height, step):
	if width <= 0 or height <= 0 or step <= 0:
		raise ValueError("Map width, height, and step must be positive")

	x_values = np.linspace(-width / 2, width / 2, int(width // step))
	y_values = np.linspace(-height / 2, height / 2, int(height // step))
	if len(x_values) < 2 or len(y_values) < 2:
		raise ValueError("Map dimensions must contain at least two cells per axis")

	x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
	return np.stack((x_grid, y_grid), axis=-1)


def _load_height_map(path, origin_lat, origin_lon, width, height, step):
	with xr.open_dataset(path) as dataset:
		height_map = get_height_map(
			dataset["elevation"],
			origin_lat,
			origin_lon,
			width,
			height,
			step,
		)
	return height_map, _coordinate_map(width, height, step)


def _load_observations(path):
	if path.suffix.lower() in {".pkl", ".pickle"}:
		with path.open("rb") as file:
			observations = pickle.load(file)
		if not isinstance(observations, (tuple, list)) or len(observations) != 2:
			raise ValueError("Pickle file must contain (time, frequency) arrays")
		obs_time, obs_freq = observations
	else:
		data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
		if data.shape[1] < 2:
			raise ValueError("Observation CSV must contain time and frequency columns")
		obs_time, obs_freq = data[:, 0], data[:, 1]

	obs_time = np.asarray(obs_time, dtype=float)
	obs_freq = np.asarray(obs_freq, dtype=float)
	if (
		obs_time.ndim != 1
		or obs_freq.ndim != 1
		or len(obs_time) != len(obs_freq)
		or len(obs_time) < 2
	):
		raise ValueError("Observations must contain at least two time/frequency pairs")
	if not np.all(np.isfinite(obs_time)) or not np.all(np.isfinite(obs_freq)):
		raise ValueError("Observations must contain only finite values")
	if np.any(np.diff(obs_time) <= 0):
		raise ValueError("Observation times must be strictly increasing")

	return obs_time, obs_freq


def _write_curve(path, times, frequencies):
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(("time_s", "frequency_hz"))
		writer.writerows(zip(times, frequencies))


def command_track(args):
	flight = load_flight(args.track, args.origin_lat, args.origin_lon)
	start = datetime.fromtimestamp(flight.t[0], timezone.utc).isoformat()
	end = datetime.fromtimestamp(flight.t[-1], timezone.utc).isoformat()
	print(f"Flight time range (UTC): {start}–{end}")
	print(f"Track points: {len(flight.t)}")
	if args.plot:
		plot_track(flight)


def command_simulate(args):
	flight = load_flight(args.track, args.origin_lat, args.origin_lon)
	map_options = (None, None)
	if not args.no_elevation:
		map_options = _load_height_map(
			args.elevation,
			args.origin_lat,
			args.origin_lon,
			args.map_width,
			args.map_height,
			args.map_step,
		)
	times, frequencies = simulate_doppler(
		flight,
		args.observer_lat,
		args.observer_lon,
		args.source_frequency,
		observer_alt=args.observer_alt,
		dt=args.dt,
		coordinate_map=map_options[1],
		height_map=map_options[0],
	)
	_write_curve(args.output, times, frequencies)
	print(f"Saved {len(times)} Doppler samples to {args.output}")


def command_optimize(args):
	flight = load_flight(args.track, args.origin_lat, args.origin_lon)
	obs_time, obs_freq = _load_observations(args.observations)

	coordinate_map = None
	height_map = None
	if not args.no_elevation:
		height_map, coordinate_map = _load_height_map(
			args.elevation,
			args.origin_lat,
			args.origin_lon,
			args.width,
			args.height,
			args.grid_step,
		)

	delta_min = args.delta_min
	if delta_min is None:
		delta_min = flight.t[0] - obs_time[0]
	delta_max = args.delta_max
	if delta_max is None:
		delta_max = flight.t[-1] - obs_time[-1]
	estimate = optimize_observer(
		flight,
		obs_time,
		obs_freq,
		x_bounds=(-args.width / 2, args.width / 2),
		y_bounds=(-args.height / 2, args.height / 2),
		delta_bounds=(delta_min, delta_max),
		coordinate_map=coordinate_map,
		height_map=height_map,
		coarse_xy_step=args.coarse_xy_step,
		coarse_time_step=args.coarse_time_step,
		refine_starts=args.refine_starts,
		min_valid_fraction=args.min_valid_fraction,
	)

	latitude = args.origin_lat + np.degrees(estimate.y / EARTH_RADIUS)
	longitude = args.origin_lon + np.degrees(
		estimate.x / (EARTH_RADIUS * np.cos(np.radians(args.origin_lat)))
	)
	print("=== Doppler optimizer report ===")
	print(f"Observer: {latitude:.8f}°, {longitude:.8f}°")
	print(f"Offset east/west (x): {estimate.x:.3f} m")
	print(f"Offset north/south (y): {estimate.y:.3f} m")
	print(f"Observer terrain elevation: {estimate.observer_alt:.3f} m")
	recording_start = datetime.fromtimestamp(
		estimate.delta_t, timezone.utc
	).isoformat()
	print(f"Audio recording start (UTC): {recording_start}")
	print(f"Estimated source frequency: {estimate.estimated_frequency_hz:.6f} Hz")
	print(f"Frequency variance: {estimate.variance:.9g} Hz²")
	print(f"Success: {estimate.success} ({estimate.message})")
	print(f"Function evaluations: {estimate.evaluations}")


def command_synthesize(args):
	data = np.loadtxt(args.curve, delimiter=",", skiprows=1, ndmin=2)
	if data.shape[1] < 2 or data.shape[0] < 2:
		raise ValueError("Curve CSV must contain at least two time/frequency pairs")
	times, audio = embed_thick_doppler(
		data[:, 0],
		data[:, 1],
		sample_rate=args.sample_rate,
		tone_amplitude=args.tone_amplitude,
		noise_amplitude=args.noise_amplitude,
		line_thickness=args.line_thickness,
		harmonics=args.harmonics,
		modulation_depth=args.modulation_depth,
		modulation_rate=args.modulation_rate,
		fade_fraction=args.fade_fraction,
		control_rate=args.control_rate,
		seed=args.seed,
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	wav.write(args.output, args.sample_rate, audio)
	print(
		f"Saved {len(audio)} audio samples "
		f"({times[0]:.3f}–{times[-1]:.3f} s) to {args.output}"
	)


def build_parser():
	parser = argparse.ArgumentParser(
		description="DopplerScope: flight tracks, Doppler curves, and observer estimation."
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	track = subparsers.add_parser("track", help="Load and optionally plot a flight track")
	track.add_argument(
		"--track",
		type=Path,
		default=DEFAULT_TRACK,
		help=f"Flight track CSV (default: {DEFAULT_TRACK})",
	)
	track.add_argument(
		"--origin-lat",
		type=float,
		default=51.364946,
		help="Reference latitude in decimal degrees (default: 51.364946)",
	)
	track.add_argument(
		"--origin-lon",
		type=float,
		default=0.856076,
		help="Reference longitude in decimal degrees (default: 0.856076)",
	)
	track.add_argument("--plot", action="store_true", help="Show the 3D flight path")
	track.set_defaults(func=command_track)

	simulate = subparsers.add_parser("simulate", help="Simulate and save a Doppler curve")
	simulate.add_argument(
		"--track",
		type=Path,
		default=DEFAULT_TRACK,
		help=f"Flight track CSV (default: {DEFAULT_TRACK})",
	)
	simulate.add_argument(
		"--origin-lat",
		type=float,
		default=51.364946,
		help="Reference latitude in decimal degrees (default: 51.364946)",
	)
	simulate.add_argument(
		"--origin-lon",
		type=float,
		default=0.856076,
		help="Reference longitude in decimal degrees (default: 0.856076)",
	)
	simulate.add_argument(
		"--observer-lat",
		type=float,
		required=True,
		help="Observer latitude in decimal degrees",
	)
	simulate.add_argument(
		"--observer-lon",
		type=float,
		required=True,
		help="Observer longitude in decimal degrees",
	)
	simulate.add_argument(
		"--observer-alt",
		type=float,
		default=0.0,
		help="Observer altitude above sea level in meters when terrain is disabled (default: 0)",
	)
	simulate.add_argument(
		"--source-frequency",
		type=float,
		required=True,
		help="Emitted sound frequency in hertz",
	)
	simulate.add_argument(
		"--dt",
		type=float,
		default=0.1,
		help="Time interval between simulated samples in seconds (default: 0.1)",
	)
	simulate.add_argument(
		"--elevation",
		type=Path,
		default=DEFAULT_ELEVATION,
		help=f"NetCDF terrain elevation file loaded with Xarray (default: {DEFAULT_ELEVATION})",
	)
	simulate.add_argument(
		"--no-elevation",
		action="store_true",
		help="Skip terrain loading and use --observer-alt instead",
	)
	simulate.add_argument(
		"--map-width",
		type=float,
		default=10000,
		help="Terrain grid width centered on the reference point, in meters (default: 10000)",
	)
	simulate.add_argument(
		"--map-height",
		type=float,
		default=10000,
		help="Terrain grid height centered on the reference point, in meters (default: 10000)",
	)
	simulate.add_argument(
		"--map-step",
		type=float,
		default=100,
		help="Terrain grid cell spacing in meters (default: 100)",
	)
	simulate.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_CURVE,
		help=f"Output CSV path (default: {DEFAULT_CURVE})",
	)
	simulate.set_defaults(func=command_simulate)

	optimize = subparsers.add_parser("estimate", help="Estimate observer position from observations")
	optimize.add_argument(
		"--track",
		type=Path,
		default=DEFAULT_TRACK,
		help=f"Flight track CSV (default: {DEFAULT_TRACK})",
	)
	optimize.add_argument(
		"--observations",
		type=Path,
		default=DEFAULT_OBSERVATIONS,
		help=f"Observed curve CSV or legacy pickle file (default: {DEFAULT_OBSERVATIONS})",
	)
	optimize.add_argument(
		"--origin-lat",
		type=float,
		default=51.364946,
		help="Reference latitude in decimal degrees (default: 51.364946)",
	)
	optimize.add_argument(
		"--origin-lon",
		type=float,
		default=0.856076,
		help="Reference longitude in decimal degrees (default: 0.856076)",
	)
	optimize.add_argument(
		"--width",
		type=float,
		default=10000,
		help="East-west search/map width centered on reference point, in meters (default: 10000)",
	)
	optimize.add_argument(
		"--height",
		type=float,
		default=10000,
		help="North-south search/map height centered on reference point, in meters (default: 10000)",
	)
	optimize.add_argument(
		"--elevation",
		type=Path,
		default=DEFAULT_ELEVATION,
		help=f"NetCDF terrain elevation file loaded with Xarray (default: {DEFAULT_ELEVATION})",
	)
	optimize.add_argument(
		"--no-elevation",
		action="store_true",
		help="Run without loading the terrain height map",
	)
	optimize.add_argument(
		"--grid-step",
		type=float,
		default=100,
		help="Terrain grid cell spacing in meters (default: 100)",
	)
	optimize.add_argument(
		"--delta-min",
		type=float,
		help="Minimum audio-start Unix timestamp (default: flight start)",
	)
	optimize.add_argument(
		"--delta-max",
		type=float,
		help="Maximum audio-start Unix timestamp (default: latest time that fits the flight)",
	)
	optimize.add_argument(
		"--coarse-xy-step",
		type=float,
		default=2000,
		help="Coarse search spacing for each horizontal axis in meters (default: 2000)",
	)
	optimize.add_argument(
		"--coarse-time-step",
		type=float,
		default=30,
		help="Coarse search spacing for audio-start time in seconds (default: 30)",
	)
	optimize.add_argument(
		"--refine-starts",
		type=int,
		default=3,
		help="Number of best coarse candidates to refine (default: 3)",
	)
	optimize.add_argument(
		"--min-valid-fraction",
		type=float,
		default=1.0,
		help="Minimum fraction of observations that must fit the flight track (0, 1] (default: 1.0)",
	)
	optimize.set_defaults(func=command_optimize)

	synthesize = subparsers.add_parser(
		"synthesize", help="Generate a WAV sound from a time/frequency CSV curve"
	)
	synthesize.add_argument(
		"--curve",
		type=Path,
		default=DEFAULT_CURVE,
		help=f"Input time/frequency CSV (default: {DEFAULT_CURVE})",
	)
	synthesize.add_argument(
		"--output",
		type=Path,
		default=PROJECT_ROOT / "data" / "doppler.wav",
		help=f"Output WAV path (default: {PROJECT_ROOT / 'data' / 'doppler.wav'})",
	)
	synthesize.add_argument(
		"--sample-rate",
		type=int,
		default=44100,
		help="Output sample rate in samples per second (default: 44100)",
	)
	synthesize.add_argument(
		"--tone-amplitude",
		type=float,
		default=0.0002,
		help="Amplitude of the fundamental tone (default: 0.0002)",
	)
	synthesize.add_argument(
		"--noise-amplitude",
		type=float,
		default=0.0003,
		help="Amplitude of generated background noise (default: 0.0003)",
	)
	synthesize.add_argument(
		"--line-thickness",
		type=float,
		default=3.0,
		help="Standard deviation of frequency variation in hertz (default: 3.0)",
	)
	synthesize.add_argument(
		"--harmonics",
		type=int,
		default=4,
		help="Number of harmonics to add to the tone (default: 4)",
	)
	synthesize.add_argument(
		"--modulation-depth",
		type=float,
		default=0.08,
		help="Depth of periodic amplitude modulation (default: 0.08)",
	)
	synthesize.add_argument(
		"--modulation-rate",
		type=float,
		default=2.5,
		help="Periodic amplitude modulation rate in hertz (default: 2.5)",
	)
	synthesize.add_argument(
		"--fade-fraction",
		type=float,
		default=0.15,
		help="Fraction of clip duration used for fade-in and fade-out (default: 0.15)",
	)
	synthesize.add_argument(
		"--control-rate",
		type=int,
		default=200,
		help="Rate of low-frequency random modulation in samples per second (default: 200)",
	)
	synthesize.add_argument(
		"--seed",
		type=int,
		help="Random seed for repeatable noise and modulation (default: random)",
	)
	synthesize.set_defaults(func=command_synthesize)

	return parser


def main(argv=None):
	parser = build_parser()
	args = parser.parse_args(argv)
	args.func(args)


if __name__ == "__main__":
	main()
