from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from src.geo import EARTH_RADIUS

SOUND_SPEED = 343.0


@dataclass(frozen=True)
class ObserverEstimate:
	x: float
	y: float
	observer_alt: float
	delta_t: float
	variance: float
	estimated_frequency_hz: float
	success: bool
	message: str
	evaluations: int


def _make_height_interpolator(coordinate_map, height_map):
	if (coordinate_map is None) != (height_map is None):
		raise ValueError("coordinate_map and height_map must be provided together")
	if coordinate_map is None:
		return None

	coordinate_map = np.asarray(coordinate_map, dtype=float)
	height_map = np.asarray(height_map, dtype=float)
	if coordinate_map.ndim != 3 or coordinate_map.shape[-1] != 2:
		raise ValueError("coordinate_map must have shape (ny, nx, 2)")
	if height_map.shape != coordinate_map.shape[:2]:
		raise ValueError("height_map shape must match coordinate_map")

	x_axis = coordinate_map[0, :, 0]
	y_axis = coordinate_map[:, 0, 1]
	expected_x = np.broadcast_to(x_axis[None, :], coordinate_map.shape[:2])
	expected_y = np.broadcast_to(y_axis[:, None], coordinate_map.shape[:2])
	if (
		len(x_axis) < 2
		or len(y_axis) < 2
		or not np.all(np.diff(x_axis) > 0)
		or not np.all(np.diff(y_axis) > 0)
		or not np.allclose(coordinate_map[..., 0], expected_x)
		or not np.allclose(coordinate_map[..., 1], expected_y)
	):
		raise ValueError("coordinate_map must be a regular, increasing XY grid")

	return RegularGridInterpolator(
		(y_axis, x_axis),
		height_map,
		bounds_error=False,
		fill_value=np.nan,
	)


def simulate_doppler(
	flight,
	observer_lat,
	observer_lon,
	source_frequency,
	observer_alt=0.0,
	dt=0.1,
	coordinate_map=None,
	height_map=None,
):
	t_emit = np.arange(flight.t[0], flight.t[-1], dt)

	observer_lat_rad = np.radians(observer_lat)
	observer_lon_rad = np.radians(observer_lon)
	observer_x = EARTH_RADIUS * np.cos(flight.origin_lat) * (
		observer_lon_rad - flight.origin_lon
	)
	observer_y = EARTH_RADIUS * (observer_lat_rad - flight.origin_lat)

	height_interpolator = _make_height_interpolator(coordinate_map, height_map)
	if height_interpolator is not None:
		observer_alt = float(height_interpolator([[observer_y, observer_x]])[0])
		if not np.isfinite(observer_alt):
			raise ValueError("Observer coordinate is outside the height map or has no elevation")

	observer = np.array([observer_x, observer_y, observer_alt])

	positions = flight.get_position(t_emit)
	radial_velocity = flight.get_radial_velocity(t_emit, observer)

	distances = np.linalg.norm(positions - observer, axis=1)

	frequency = source_frequency * SOUND_SPEED / (SOUND_SPEED + radial_velocity)
	time = t_emit + distances / SOUND_SPEED

	return time, frequency

def _validate_optimizer_inputs(
	obs_time,
	obs_freq,
	bounds,
	coarse_xy_step,
	coarse_time_step,
	refine_starts,
	min_valid_fraction,
):
	if (
		obs_time.ndim != 1
		or obs_freq.ndim != 1
		or len(obs_time) != len(obs_freq)
		or len(obs_time) < 2
	):
		raise ValueError("obs_time and obs_freq must be 1D arrays of equal length >= 2")
	if not np.all(np.isfinite(obs_time)) or not np.all(np.isfinite(obs_freq)):
		raise ValueError("Observations must contain only finite values")
	if np.any(obs_freq <= 0):
		raise ValueError("Observed frequencies must be positive")
	if any(low >= high for low, high in bounds):
		raise ValueError("Each optimization bound must have low < high")
	if coarse_xy_step <= 0 or coarse_time_step <= 0 or refine_starts < 1:
		raise ValueError("Grid steps and refine_starts must be positive")
	if not 0 < min_valid_fraction <= 1:
		raise ValueError("min_valid_fraction must be in (0, 1]")


def _evaluate_candidate(
	flight,
	obs_time,
	obs_freq,
	x,
	y,
	delta_t,
	height_interpolator,
	min_valid_count,
):
	observer_altitude = 0.0
	if height_interpolator is not None:
		observer_altitude = float(height_interpolator([[y, x]])[0])
		if not np.isfinite(observer_altitude):
			return np.inf, np.nan

	flight_start = float(flight.t[0])
	flight_end = float(flight.t[-1])
	receive_time = flight_start + obs_time + delta_t
	emission_time = receive_time.copy()
	observer = np.array([x, y, observer_altitude])
	valid = np.ones(len(obs_time), dtype=bool)

	for _ in range(6):
		position = flight.get_position(np.clip(emission_time, flight_start, flight_end))
		distance = np.linalg.norm(position - observer, axis=-1)
		emission_time = receive_time - distance / SOUND_SPEED
		valid &= (emission_time >= flight_start) & (emission_time <= flight_end)
		emission_time = np.clip(emission_time, flight_start, flight_end)

	position, velocity = flight.get_position(emission_time), flight.get_velocity(emission_time)
	direction = position - observer
	distance = np.maximum(np.linalg.norm(direction, axis=-1), 1e-9)
	radial_velocity = np.sum(velocity * (direction / distance[:, None]), axis=-1)
	source_frequency = obs_freq * (SOUND_SPEED + radial_velocity) / SOUND_SPEED
	valid &= np.isfinite(source_frequency)

	if np.count_nonzero(valid) < min_valid_count:
		return np.inf, np.nan

	valid_frequency = source_frequency[valid]
	return float(np.var(valid_frequency)), float(np.mean(valid_frequency))


def _coarse_search(
	flight,
	obs_time,
	obs_freq,
	bounds,
	xy_step,
	time_step,
	height_interpolator,
	min_valid_count,
):
	x_bounds, y_bounds, time_bounds = bounds

	def values(bound, step):
		grid = np.arange(bound[0], bound[1], step, dtype=float)
		values = np.append(grid, bound[1])
		if bound[0] <= 0 <= bound[1]:
			values = np.append(values, 0.0)
		return np.unique(values)

	x_values = values(x_bounds, xy_step)
	y_values = values(y_bounds, xy_step)
	time_offsets = values(time_bounds, time_step)
	candidates = []

	for x in x_values:
		for y in y_values:
			for delta_t in time_offsets:
				variance, _ = _evaluate_candidate(
					flight,
					obs_time,
					obs_freq,
					x,
					y,
					delta_t,
					height_interpolator,
					min_valid_count,
				)
				if np.isfinite(variance):
					candidates.append((variance, x, y, delta_t))

	if not candidates:
		raise ValueError("No candidate has enough valid observations; check the time-offset bounds")

	candidates.sort(key=lambda candidate: candidate[0])
	evaluations = len(x_values) * len(y_values) * len(time_offsets)
	return candidates, evaluations


def _refine_candidates(
	starts,
	bounds,
	flight,
	obs_time,
	obs_freq,
	height_interpolator,
	min_valid_count,
	evaluations,
	coarse_xy_step,
	coarse_time_step,
):
	x_bounds, y_bounds, time_bounds = bounds
	global_bounds = (x_bounds, y_bounds, time_bounds)
	local_steps = (coarse_xy_step, coarse_xy_step, coarse_time_step)
	frequency_scale = min(
		max(1.0, float(np.max(np.abs(obs_freq)))),
		np.finfo(float).max ** 0.25,
	)
	invalid_objective = frequency_scale**2 * 1e6
	best = starts[0]
	success = False
	message = "Using best coarse-grid candidate"

	def objective(parameters):
		x, y = parameters[:2] * 1000
		delta_t = parameters[2]
		variance, _ = _evaluate_candidate(
			flight,
			obs_time,
			obs_freq,
			x,
			y,
			delta_t,
			height_interpolator,
			min_valid_count,
		)
		return variance if np.isfinite(variance) else invalid_objective

	for _, x, y, delta_t in starts:
		start = (x, y, delta_t)
		local_bounds = tuple(
			(
				max(bound[0], value - step),
				min(bound[1], value + step),
			)
			for bound, value, step in zip(global_bounds, start, local_steps)
		)
		optimizer_bounds = (
			(local_bounds[0][0] / 1000, local_bounds[0][1] / 1000),
			(local_bounds[1][0] / 1000, local_bounds[1][1] / 1000),
			local_bounds[2],
		)
		result = minimize(
			objective,
			x0=np.array([x / 1000, y / 1000, delta_t]),
			method="Powell",
			bounds=optimizer_bounds,
			options={"xtol": 1e-4, "ftol": 1e-10, "maxiter": 250},
		)
		evaluations += result.nfev

		if np.isfinite(result.fun) and result.fun < best[0]:
			best = (
				float(result.fun),
				float(result.x[0] * 1000),
				float(result.x[1] * 1000),
				float(result.x[2]),
			)
			success = bool(result.success)
			message = str(result.message)

	return best, success, message, evaluations


def optimize_observer(
	flight,
	obs_time,
	obs_freq,
	x_bounds=(-5000.0, 5000.0),
	y_bounds=(-5000.0, 5000.0),
	delta_bounds=(0.0, 300.0),
	coordinate_map=None,
	height_map=None,
	coarse_xy_step=2000.0,
	coarse_time_step=30.0,
	refine_starts=3,
	min_valid_fraction=1.0,
):
	"""Estimate observer position and time offset by minimizing Doppler variance."""
	obs_time = np.asarray(obs_time, dtype=float)
	obs_freq = np.asarray(obs_freq, dtype=float)
	bounds = (x_bounds, y_bounds, delta_bounds)

	_validate_optimizer_inputs(
		obs_time,
		obs_freq,
		bounds,
		coarse_xy_step,
		coarse_time_step,
		refine_starts,
		min_valid_fraction,
	)
	height_interpolator = _make_height_interpolator(coordinate_map, height_map)
	min_valid_count = max(2, int(np.ceil(len(obs_time) * min_valid_fraction)))

	candidates, evaluations = _coarse_search(
		flight,
		obs_time,
		obs_freq,
		bounds,
		coarse_xy_step,
		coarse_time_step,
		height_interpolator,
		min_valid_count,
	)
	best, success, message, evaluations = _refine_candidates(
		candidates[:refine_starts],
		bounds,
		flight,
		obs_time,
		obs_freq,
		height_interpolator,
		min_valid_count,
		evaluations,
		coarse_xy_step,
		coarse_time_step,
	)
	variance, estimated_frequency = _evaluate_candidate(
		flight,
		obs_time,
		obs_freq,
		best[1],
		best[2],
		best[3],
		height_interpolator,
		min_valid_count,
	)
	altitude = 0.0
	if height_interpolator is not None:
		altitude = float(height_interpolator([[best[2], best[1]]])[0])

	return ObserverEstimate(
		x=best[1],
		y=best[2],
		observer_alt=altitude,
		delta_t=best[3],
		variance=variance,
		estimated_frequency_hz=estimated_frequency,
		success=success,
		message=message,
		evaluations=evaluations,
	)
