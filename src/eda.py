"""Exploratory data analysis for PathSnap (train + student_test partitions).

Outputs (results/eda/):
  inventory.csv        one row per trip: counts, gaps, extent, connectivity, route validity
  summary.txt          partition-level summary
  *.png                distribution plots
  map_<trip>.html      Folium maps (GPS points, road network, reference route)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent))
from data_io import list_trips, load_trip

OUT = Path(__file__).resolve().parents[1] / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def trip_stats(part, tid):
    tr = load_trip(part, tid)
    t = tr.track[:, 2]
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(tr.track_xy, axis=0), axis=1)
    speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=dt > 0)
    N, A = len(tr.nodes), len(tr.arcs)
    g = coo_matrix((np.ones(A), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(N, N))
    n_wcc, lab = connected_components(g, directed=True, connection="weak")
    deg = np.bincount(tr.arcs.ravel(), minlength=N)
    # route continuity: consecutive route arcs share a node (to of i == from of i+1)
    r = tr.route
    valid_ids = (r >= 0) & (r < A)
    cont = np.nan
    if valid_ids.all() and len(r) > 1:
        cont = np.mean(tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0])
    seg_len = np.linalg.norm(tr.nodes_xy[tr.arcs[:, 0]] - tr.nodes_xy[tr.arcs[:, 1]], axis=1)
    rev = set(map(tuple, tr.arcs[:, ::-1]))
    bidir = np.mean([tuple(a) in rev for a in tr.arcs])
    return dict(
        partition=part, trip=tid,
        n_gps=len(t), n_nodes=N, n_arcs=A, n_route_arcs=len(r),
        duration_s=t[-1] - t[0], nan_gps=int(np.isnan(tr.track).any(1).sum()),
        non_monotonic_t=int((dt <= 0).sum()), frac_dt_1s=float(np.mean(dt == 1)),
        max_gap_s=float(dt.max()), n_gaps_gt5s=int((dt > 5).sum()),
        path_len_km=float(step.sum() / 1000), median_speed_ms=float(np.nanmedian(speed)),
        max_speed_ms=float(np.nanmax(speed)), n_speed_gt50ms=int((speed > 50).sum()),
        extent_x_km=float(np.ptp(tr.track_xy[:, 0]) / 1000), extent_y_km=float(np.ptp(tr.track_xy[:, 1]) / 1000),
        lon_min=tr.track[:, 0].min(), lon_max=tr.track[:, 0].max(),
        lat_min=tr.track[:, 1].min(), lat_max=tr.track[:, 1].max(),
        weak_components=int(n_wcc), largest_comp_frac=float(np.bincount(lab).max() / N),
        isolated_nodes=int((deg == 0).sum()), median_arc_len_m=float(np.median(seg_len)),
        frac_bidirectional_arcs=float(bidir),
        route_ids_valid=bool(valid_ids.all()), route_continuity=float(cont),
        route_repeats=int(len(r) - len(np.unique(r))),
    )


def folium_map(part, tid):
    tr = load_trip(part, tid)
    m = folium.Map(location=[tr.lat0, tr.lon0], zoom_start=14, tiles="OpenStreetMap")
    net = folium.FeatureGroup("Road network", show=True)
    for a, b in tr.arcs[::1]:
        folium.PolyLine([(tr.nodes[a, 1], tr.nodes[a, 0]), (tr.nodes[b, 1], tr.nodes[b, 0])],
                        color="#9aa0a6", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    rt = folium.FeatureGroup("Reference route", show=True)
    for a in tr.route:
        u, v = tr.arcs[a]
        folium.PolyLine([(tr.nodes[u, 1], tr.nodes[u, 0]), (tr.nodes[v, 1], tr.nodes[v, 0])],
                        color="#1e8e3e", weight=5, opacity=0.85).add_to(rt)
    rt.add_to(m)
    gps = folium.FeatureGroup("GPS observations", show=True)
    for lon, lat, t in tr.track[::3]:
        folium.CircleMarker((lat, lon), radius=2, color="#1a73e8", fill=True, weight=0,
                            fill_opacity=0.8, tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((tr.track[0, 1], tr.track[0, 0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((tr.track[-1, 1], tr.track[-1, 0]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    m.fit_bounds([[tr.track[:, 1].min(), tr.track[:, 0].min()], [tr.track[:, 1].max(), tr.track[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:999;background:white;padding:8px;'
        'border:1px solid #ccc;font:12px sans-serif">'
        f'<b>{part} / trip {tid}</b><br>'
        '<span style="color:#1a73e8">&#9679;</span> GPS (every 3rd)<br>'
        '<span style="color:#1e8e3e">&#9644;</span> Reference route<br>'
        '<span style="color:#9aa0a6">&#9644;</span> Road network</div>'))
    m.save(OUT / f"map_{part}_{tid}.html")


def main():
    rows = [trip_stats(p, t) for p in ("train", "student_test") for t in list_trips(p)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inventory.csv", index=False)

    with open(OUT / "summary.txt", "w") as f:
        for p, g in df.groupby("partition"):
            f.write(f"=== {p}: {len(g)} trips\n")
            f.write(g.drop(columns=["trip", "partition"]).describe().T.round(3).to_string() + "\n\n")
        f.write("Anomalies:\n")
        f.write(df[(df.max_gap_s > 10) | (df.n_speed_gt50ms > 0) | (~df.route_ids_valid) | (df.nan_gps > 0)]
                [["partition", "trip", "max_gap_s", "n_speed_gt50ms", "nan_gps", "route_ids_valid", "route_continuity"]]
                .to_string() + "\n")

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for a, col, title in zip(ax.ravel(),
                             ["n_gps", "path_len_km", "median_speed_ms", "max_gap_s", "n_arcs", "route_continuity"],
                             ["GPS points / trip", "Trip length (km)", "Median speed (m/s)",
                              "Max time gap (s)", "Road arcs / trip", "Route continuity"]):
        for p, g in df.groupby("partition"):
            a.hist(g[col].dropna(), bins=20, alpha=0.6, label=p)
        a.set_title(title)
    ax[0, 0].legend()
    fig.tight_layout(); fig.savefig(OUT / "trip_distributions.png", dpi=110)

    # sampling-interval distribution (train)
    dts = np.concatenate([np.diff(load_trip("train", t).track[:, 2]) for t in list_trips("train")])
    fig, a = plt.subplots(figsize=(6, 4))
    vals, cnts = np.unique(np.clip(dts, 0, 10), return_counts=True)
    a.bar(vals, cnts / cnts.sum()); a.set_yscale("log")
    a.set_xlabel("dt between GPS fixes (s, clipped at 10)"); a.set_ylabel("fraction")
    a.set_title(f"Sampling interval, train ({np.mean(dts == 1):.1%} exactly 1 s)")
    fig.tight_layout(); fig.savefig(OUT / "sampling_interval.png", dpi=110)

    # trajectory overview plot
    fig, a = plt.subplots(figsize=(7, 7))
    for t in list_trips("train")[:20]:
        tr = load_trip("train", t)
        a.plot(tr.track[:, 0], tr.track[:, 1], lw=0.8)
    a.set_xlabel("longitude"); a.set_ylabel("latitude"); a.set_title("GPS trajectories (first 20 train trips)")
    fig.tight_layout(); fig.savefig(OUT / "trajectories_overview.png", dpi=110)

    for p, t in [("train", "00000002"), ("train", "00000005"), ("student_test", "00000007")]:
        folium_map(p, t)
    print(open(OUT / "summary.txt").read())


if __name__ == "__main__":
    main()
