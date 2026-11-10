import requests
import csv
import time
import os
from datetime import datetime, timezone

BASE_URL = "https://api.adsb.lol/v2"

LAT = 51.0
LON = 2.0
RADIUS = 50


def get_json(url):
    while True:
        response = requests.get(url, timeout=20)

        if response.status_code == 429:
            wait = response.headers.get(
                "Retry-After",
                response.headers.get("x-ratelimit-reset", 30)
            )

            try:
                wait = max(1, int(float(wait)))
            except (ValueError, TypeError):
                wait = 30

            print(f"Rate limited. Waiting {wait} seconds...")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()


def get_aircraft_in_sector(lat, lon, radius):
    url = f"{BASE_URL}/lat/{lat}/lon/{lon}/dist/{radius}"
    return get_json(url).get("ac", [])


def get_aircraft_position(icao):
    url = f"{BASE_URL}/icao/{icao}"

    data = get_json(url)
    aircraft = data.get("ac", [])

    return aircraft[0] if aircraft else None


def record_track(freq):
    if freq <= 0:
        raise ValueError("Frequency must be greater than zero")

    aircraft = get_aircraft_in_sector(LAT, LON, RADIUS)

    if not aircraft:
        print("No aircraft found.")
        return

    for i, plane in enumerate(aircraft):
        print(
            f"[{i}] {plane.get('hex', 'Unknown')} | "
            f"ICAO: {plane.get('flight')} | "
            f"Altitude: {plane.get('alt_baro')} | "
            f"Speed: {plane.get('gs')}"
        )

    try:
        index = int(input("\nSelect aircraft number: "))
    except ValueError:
        print("Invalid input.")
        return

    if index < 0 or index >= len(aircraft):
        print("Invalid selection.")
        return

    target = aircraft[index]
    icao = target["hex"].lower()

    os.makedirs("data", exist_ok=True)

    filename = f"data/{icao}_track.csv"

    print(f"\nRecording {icao} to {filename}")
    print(f"Polling interval: {freq} seconds")
    print("Press Ctrl+C to stop.\n")

    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "timestamp",
                "icao",
                "flight",
                "lat",
                "lon",
                "alt_baro",
                "gs",
                "track"
            ])

        try:
            while True:
                plane = get_aircraft_position(icao)

                if plane is None:
                    print("Aircraft not currently visible.")
                    time.sleep(freq)
                    continue

                lat = plane.get("lat")
                lon = plane.get("lon")

                if lat is None or lon is None:
                    print("Position unavailable.")
                    time.sleep(freq)
                    continue

                timestamp = datetime.now(timezone.utc).isoformat()

                writer.writerow([
                    timestamp,
                    icao,
                    plane.get("flight"),
                    lat,
                    lon,
                    plane.get("alt_baro"),
                    plane.get("gs"),
                    plane.get("track")
                ])

                file.flush()

                print(
                    f"{timestamp} | "
                    f"{lat:.5f}, {lon:.5f} | "
                    f"ALT: {plane.get('alt_baro')}"
                )

                time.sleep(freq)

        except KeyboardInterrupt:
            print("\nRecording stopped.")


if __name__ == "__main__":
    record_track(5)