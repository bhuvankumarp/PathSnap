"""Raw-data loading for PathSnap trips.

Each trip folder <id>/ holds:
  <id>.track : lon lat t(seconds)     -- GPS observations
  <id>.nodes : lon lat                -- road nodes (row index = node_id)
  <id>.arcs  : from_node to_node      -- road segments (row index = arc_id)
  <id>.route : arc_id                 -- reference route, chronological
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
EARTH_R = 6_371_008.8  # mean Earth radius (m)


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray   # (T,3) lon, lat, t
    nodes: np.ndarray   # (N,2) lon, lat
    arcs: np.ndarray    # (A,2) from, to
    route: np.ndarray   # (R,) arc ids
    # projected (metres, local equirectangular around trip centroid)
    lon0: float = 0.0
    lat0: float = 0.0
    track_xy: np.ndarray = None
    nodes_xy: np.ndarray = None

    def project(self, lon, lat):
        """Local equirectangular projection -> metres. Trips span a few km,
        so distortion is well under 0.1% (vs. degree-Euclidean, which is wrong
        by ~33% in x at 47.6N)."""
        x = np.radians(lon - self.lon0) * EARTH_R * np.cos(np.radians(self.lat0))
        y = np.radians(lat - self.lat0) * EARTH_R
        return np.column_stack([x, y])

    def unproject(self, xy):
        lon = np.degrees(xy[:, 0] / (EARTH_R * np.cos(np.radians(self.lat0)))) + self.lon0
        lat = np.degrees(xy[:, 1] / EARTH_R) + self.lat0
        return lon, lat


def list_trips(partition: str):
    return sorted(p.name for p in (DATA_ROOT / partition).iterdir() if p.is_dir())


def _load(path, dtype, ncols):
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols) if ncols > 1 else (0,), dtype=dtype)
    a = np.loadtxt(path, dtype=dtype, ndmin=2 if ncols > 1 else 1)
    return a


def load_trip(partition: str, trip_id: str) -> Trip:
    d = DATA_ROOT / partition / trip_id
    track = _load(d / f"{trip_id}.track", float, 3)
    nodes = _load(d / f"{trip_id}.nodes", float, 2)
    arcs = _load(d / f"{trip_id}.arcs", np.int64, 2)
    route = _load(d / f"{trip_id}.route", np.int64, 1)
    tr = Trip(trip_id, track, nodes, arcs, route)
    tr.lon0, tr.lat0 = np.nanmean(track[:, 0]), np.nanmean(track[:, 1])
    tr.track_xy = tr.project(track[:, 0], track[:, 1])
    tr.nodes_xy = tr.project(nodes[:, 0], nodes[:, 1])
    return tr
