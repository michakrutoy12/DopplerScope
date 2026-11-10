import numpy as np
from src.geo import EARTH_RADIUS, load_flight, get_height_map
from src.doppler import optimize_observer, simulate_doppler
import pickle
import xarray as xr

def optimize(flight, height_map_arr, origin_position, sector_size, grid_step):
	origin_lat, origin_lon = origin_position
	sector_width, sector_height = sector_size
	
	height_map = get_height_map(height_map_arr, origin_lat, origin_lon, sector_width, sector_height, grid_step)

	x_meters = np.linspace(
		-sector_width / 2,
		sector_width / 2,
		sector_width // grid_step,
	)
	y_meters = np.linspace(
		-sector_height / 2,
		sector_height / 2,
		sector_height // grid_step,
	)
	x_grid, y_grid = np.meshgrid(x_meters, y_meters, indexing='xy')
	coordinate_map = np.stack((x_grid, y_grid), axis=-1)

	with open('data/observed_curve.pkl', 'rb') as file:
		obs_time, obs_freq = pickle.load(file)

	obs_time = np.asarray(obs_time, dtype=float)
	obs_freq = np.asarray(obs_freq, dtype=float)
	obs_time -= obs_time[0]

	x_bounds = (-sector_width / 2, sector_width / 2)
	y_bounds = (-sector_height / 2, sector_height / 2)
	delta_bounds = (0.0, flight.t[-1] - obs_time[-1])

	estimate = optimize_observer(
		flight,
		obs_time,
		obs_freq,
		x_bounds=x_bounds,
		y_bounds=y_bounds,
		delta_bounds=delta_bounds,
		coordinate_map=coordinate_map,
		height_map=height_map
	)

	latitude = origin_lat + np.degrees(estimate.y / EARTH_RADIUS)
	longitude = origin_lon + np.degrees(
		estimate.x / (EARTH_RADIUS * np.cos(np.radians(origin_lat)))
	)

	print("=== Doppler optimizer report ===")
	print("Input")
	print(f"  Flight track: data/49d665_track.csv")
	print(f"  Flight time range: {flight.t[0]:.3f}–{flight.t[-1]:.3f} s")
	print(f"  Observations: {len(obs_time)} points")
	print(
		f"  Observation time range: {obs_time.min():.3f}–"
		f"{obs_time.max():.3f} s"
	)
	print(
		f"  Observed frequency range: {obs_freq.min():.3f}–"
		f"{obs_freq.max():.3f} Hz"
	)
	print(
		f"  Search bounds: x={x_bounds} m, y={y_bounds} m, "
		f"delta_t={delta_bounds} s"
	)
	print(f"  Height map: {height_map.shape[1]} × {height_map.shape[0]} cells")

	print("\nEstimate")
	print(f"  Latitude: {latitude:.8f}°")
	print(f"  Longitude: {longitude:.8f}°")
	print(f"  Offset east/west (x): {estimate.x:.3f} m")
	print(f"  Offset north/south (y): {estimate.y:.3f} m")
	print(f"  Observer terrain elevation: {estimate.observer_alt:.3f} m")
	print(f"  Time offset (delta_t): {estimate.delta_t:.6f} s")
	print(
		f"  Estimated source frequency: "
		f"{estimate.estimated_frequency_hz:.6f} Hz"
	)
	print(f"  Frequency variance: {estimate.variance:.9g} Hz²")
	print(f"  Success: {estimate.success}")
	print(f"  Optimizer message: {estimate.message}")
	print(f"  Function evaluations: {estimate.evaluations}")
	print("\n=== End report ===")

if __name__ == "__main__":
	origin_lat, origin_lon = 51.364946, 0.856076
	sector_width = 10000
	sector_height = 10000
	grid_step = 100

	flight = load_flight('data/49d665_track.csv', origin_lat, origin_lon)

	ds = xr.open_dataset('data/GEBCO_2026.nc', chunks={'lat': 1000, 'lon': 1000})
	optimize(flight, ds['elevation'], (origin_lat, origin_lon), (sector_width, sector_height), grid_step)