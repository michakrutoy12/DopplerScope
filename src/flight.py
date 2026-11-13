import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import xarray as xr

EARTH_RADIUS = 6371000.0

def get_height_map(data, center_lat, center_lon, width_m, height_m, step_m):
    meters_to_lat_deg = np.degrees(1.0 / EARTH_RADIUS)

    lat_rad = np.radians(center_lat)
    meters_to_lon_deg = np.degrees(1.0 / (EARTH_RADIUS * np.cos(lat_rad)))

    nx = int(width_m // step_m)
    ny = int(height_m // step_m)

    x_values = np.linspace(-width_m / 2, width_m / 2, nx)
    y_values = np.linspace(-height_m / 2, height_m / 2, ny)

    lons = center_lon + x_values * meters_to_lon_deg
    lats = center_lat + y_values * meters_to_lat_deg

    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")

    interp_lats = xr.DataArray(lat_mesh, dims=["y", "x"])
    interp_lons = xr.DataArray(lon_mesh, dims=["y", "x"])

    height_map = data.interp(lat=interp_lats, lon=interp_lons, method="linear").values

    return height_map


class InterpolatedFlight:
	def __init__(self, df, origin_lat, origin_lon, start_time):
		self.df = df
		self.start_time = float(start_time)
		self.t = df.index.to_numpy(dtype=float) + self.start_time

		self.origin_lat = np.radians(origin_lat)
		self.origin_lon = np.radians(origin_lon)

		lat = np.radians(df['lat'].to_numpy(dtype=float))
		lon = np.radians(df['lon'].to_numpy(dtype=float))
		alt = df['alt_baro'].to_numpy(dtype=float)

		x = EARTH_RADIUS * np.cos(self.origin_lat) * (
			lon - self.origin_lon
		)
		y = EARTH_RADIUS * (lat - self.origin_lat)

		self.x_spline = CubicSpline(self.t, x)
		self.y_spline = CubicSpline(self.t, y)
		self.z_spline = CubicSpline(self.t, alt)

		gs = df['gs'].to_numpy(dtype=float) * 0.514444
		track = np.radians(df['track'].to_numpy(dtype=float))

		vx = gs * np.sin(track)
		vy = gs * np.cos(track)

		self.vx_spline = CubicSpline(self.t, vx)
		self.vy_spline = CubicSpline(self.t, vy)

		self.vz_spline = self.z_spline.derivative()

	def get_position(self, timestamp):
		t = np.asarray(timestamp, dtype=float)

		if np.any(t < self.t[0]) or np.any(t > self.t[-1]):
			raise ValueError("Timestamp outside flight duration")

		return np.stack([
			self.x_spline(t),
			self.y_spline(t),
			self.z_spline(t)
		], axis=-1)

	def get_velocity(self, timestamp):
		t = np.asarray(timestamp, dtype=float)

		if np.any(t < self.t[0]) or np.any(t > self.t[-1]):
			raise ValueError("Timestamp outside flight duration")

		return np.stack([
			self.vx_spline(t),
			self.vy_spline(t),
			self.vz_spline(t)
		], axis=-1)


def load_flight(filename, origin_lat, origin_lon):
	df = pd.read_csv(filename)

	df['timestamp'] = pd.to_datetime(
		df['timestamp'], utc=True
	)

	df = df.set_index('timestamp')
	df = df.sort_index()

	df = df[~df.index.duplicated(keep='first')]

	start_time = df.index[0]
	df.index = (
		df.index - start_time
	).total_seconds()

	df['alt_baro'] = df['alt_baro'] * 0.3048

	df = df[['lat', 'lon', 'alt_baro', 'gs', 'track']].dropna()

	return InterpolatedFlight(
		df,
		origin_lat,
		origin_lon,
		start_time.timestamp()
	)

def plot_track(interpolated_df):
	t_vals = np.linspace(interpolated_df.t[0], interpolated_df.t[-1], 500)
	pos, _ = interpolated_df.get_state(t_vals)
	
	x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(projection='3d')

	ax.plot(x, y, z, label='Trajectory (meters)', color='red')
	ax.set_xlabel('X (meters)')
	ax.set_ylabel('Y (meters)')
	ax.set_zlabel('Altitude (meters)')
	plt.title('3D Trajectory')
	plt.legend()
	plt.show()