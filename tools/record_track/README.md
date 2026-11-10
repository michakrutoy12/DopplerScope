# Aircraft track recorder

[`../record_track.py`](../record_track.py) records the positions of a
user-selected aircraft using the public `adsb.lol` API. It writes timestamped
position samples to a CSV flight track that can be used by DopplerScope.

Run commands from the project root.

## Installation

Install the HTTP client:

```sh
python3 -m pip install requests
```

The recorder needs an internet connection and access to the `adsb.lol` API.

## Run

```sh
python3 tools/record_track.py
```

The script queries aircraft in its configured search area, prints the current
results, and prompts for the aircraft index to record. Recording continues
until you press **Ctrl+C**. The polling interval is configured in the script;
it is currently five seconds.

The search latitude, longitude, and radius are constants near the top of
`record_track.py`, not command-line arguments. The script retries requests
after an HTTP 429 rate-limit response using the server's retry header when
available.

## Output

The recorder appends to:

```text
data/<icao>_track.csv
```

Each row contains these columns, in order:

| Column | Meaning |
| --- | --- |
| `timestamp` | UTC timestamp of the sample. |
| `icao` | Aircraft ICAO address. |
| `flight` | Callsign, when provided by the API. |
| `lat` | Latitude in decimal degrees. |
| `lon` | Longitude in decimal degrees. |
| `alt_baro` | Barometric altitude in feet, as reported by the API. |
| `gs` | Ground speed in knots, as reported by the API. |
| `track` | Ground track heading in degrees. |

When creating a new output file, the recorder writes a header row. Existing
files are appended to; make sure an existing file has the same columns before
reusing it.

The flight loader in `src/geo.py` converts timestamps to elapsed seconds,
barometric altitude to meters, and ground speed to meters per second. It
expects a CSV header with the column names above. Use the resulting track with
the root command-line application:

```sh
python3 main.py track --track data/<icao>_track.csv
```

## Limitations

- The recorder is interactive and currently has no command-line options.
- Search area and polling interval must be edited in the script.
- Samples are appended rather than replacing an existing track.
- Aircraft may disappear from the API temporarily or have incomplete position
  data; in those cases the recorder skips that sample and continues polling.
