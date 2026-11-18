"""HMM decoder experiments.

1. Out-of-fold (GroupKFold-5) probabilities of the tuned LightGBM on the 80 train trips.
2. Grid search of decoder parameters (ALPHA, BETA) on OOF train probabilities only.
3. Student-test: classifier-only vs classifier+HMM (pointwise top-1, continuity, route-level).
4. Folium maps for every student-test trip.
Outputs: results/hmm/*.csv, *.png, maps/*.html ; models/hmm_config.json
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import folium
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from data_io import load_trip, list_trips
from candidates import FEATURES
import hmm
from pipeline import predict_trip, score_trip
from tune_lgbm import make

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hmm"; (OUT / "maps").mkdir(parents=True, exist_ok=True)
GRID_TRIPS = 30   # train trips used for the decoder grid (speed)


def decode_eval(df, part, tids, alpha, beta):
    tops, conts, n = [], [], []
    for tid in tids:
        g = df[df.trip == tid]
        tr = load_trip(part, tid)
        G = hmm.RoadGraph(tr.nodes_xy, tr.arcs)
        dec, _ = hmm.decode(g, tr.track_xy, tr.track[:, 2], G, alpha, beta)
        ref = g.groupby("obs").ref_arc.first()
        tops.append((dec.set_index("obs").arc.reindex(ref.index) == ref).mean())
        conts.append(hmm.continuity(dec, tr.track_xy, tr.track[:, 2], G)); n.append(len(ref))
    w = np.array(n) / sum(n)
    return float(np.dot(w, tops)), float(np.dot(w, conts))


def trip_map(tr, res, path):
    m = folium.Map(location=[tr.lat0, tr.lon0], zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    mid = (tr.nodes_xy[tr.arcs[:, 0]] + tr.nodes_xy[tr.arcs[:, 1]]) / 2
    near = np.unique(np.concatenate(cKDTree(mid).query_ball_point(tr.track_xy[::5], 150)).astype(int))

    def arcs_layer(name, ids, color, w, show=True, dash=None):
        fg = folium.FeatureGroup(name, show=show)
        for a in ids:
            u, v = tr.arcs[a]
            folium.PolyLine([(tr.nodes[u, 1], tr.nodes[u, 0]), (tr.nodes[v, 1], tr.nodes[v, 0])],
                            color=color, weight=w, opacity=0.85, dash_array=dash, tooltip=f"arc {a}").add_to(fg)
        fg.add_to(m)
    arcs_layer("Road network (within 150 m)", near, "#9aa0a6", 1)
    arcs_layer("Reference route", np.unique(tr.route), "#1e8e3e", 9)
    arcs_layer("Classifier-only (per-fix argmax)", np.unique(res["clf"].arc), "#d93025", 4, show=False)
    arcs_layer("Predicted route (classifier + HMM)", res["route"], "#7b1fa2", 4)
    gps = folium.FeatureGroup("GPS observations")
    for lon, lat, t in tr.track[::5]:
        folium.CircleMarker((lat, lon), radius=2.5, color="#1a73e8", weight=0, fill=True, fill_opacity=0.9,
                            tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((tr.track[0, 1], tr.track[0, 0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((tr.track[-1, 1], tr.track[-1, 0]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    m.fit_bounds([[tr.track[:, 1].min(), tr.track[:, 0].min()], [tr.track[:, 1].max(), tr.track[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:#fff;padding:8px 10px;'
        'border:1px solid #bbb;border-radius:4px;font:12px sans-serif;line-height:1.6">'
        f'<b>Trip {tr.trip_id}</b><br><span style="color:#1a73e8">&#9679;</span> GPS (every 5th fix)<br>'
        '<span style="color:#1e8e3e">&#9644;</span> Reference route<br>'
        '<span style="color:#7b1fa2">&#9644;</span> Predicted route (HMM)<br>'
        '<span style="color:#d93025">&#9644;</span> Classifier-only (toggle)<br>'
        '<span style="color:#9aa0a6">&#9644;</span> Road network</div>'))
    m.save(path)


def main():
    tr_df = pd.read_pickle(ROOT / "features" / "train.pkl")
    bundle = joblib.load(ROOT / "models" / "lgbm_tuned.joblib")
    params = bundle["params"]

    # 1. OOF probabilities
    oof = np.zeros(len(tr_df))
    for a, b in GroupKFold(5).split(tr_df, tr_df.y, tr_df.trip):
        oof[b] = make(params).fit(tr_df[FEATURES].iloc[a], tr_df.y.iloc[a]).predict_proba(tr_df[FEATURES].iloc[b])[:, 1]
    tr_df["p"] = oof

    # 2. decoder grid on OOF train
    tids = sorted(tr_df.trip.unique())[:GRID_TRIPS]
    sub = tr_df[tr_df.trip.isin(tids)]
    base = sub.loc[sub.groupby(["trip", "obs"]).p.idxmax()]
    clf_top1 = float(base.y.mean())
    rows = []
    for alpha in (0.5, 1.0, 2.0):
        for beta in (1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0):
            t0 = time.time()
            t1, c = decode_eval(sub, "train", tids, alpha, beta)
            rows.append(dict(alpha=alpha, beta=beta, oof_top1=t1, oof_continuity=c, seconds=time.time() - t0))
            print(rows[-1], flush=True)
    grid = pd.DataFrame(rows); grid["oof_clf_only_top1"] = clf_top1
    grid.to_csv(OUT / "decoder_grid.csv", index=False)
    best = grid.sort_values("oof_top1", ascending=False).iloc[0]
    cfg = {"alpha": float(best.alpha), "beta": float(best.beta), "unreachable_logp": hmm.UNREACHABLE_LOGP,
           "gap_break_s": hmm.GAP_BREAK_S, "max_speed": hmm.MAX_SPEED, "stride": 5,
           "selected_on": f"OOF probabilities, first {GRID_TRIPS} train trips"}
    (ROOT / "models" / "hmm_config.json").write_text(json.dumps(cfg, indent=2))
    fig, a = plt.subplots(figsize=(6, 4))
    for al, g in grid.groupby("alpha"):
        a.plot(g.beta, g.oof_top1, "o-", label=f"alpha={al}")
    a.axhline(clf_top1, ls="--", c="k", lw=1, label="classifier only")
    a.set_xscale("log"); a.set_xlabel("BETA (m)"); a.set_ylabel("OOF top-1 (train)"); a.legend()
    a.set_title("Decoder parameter search"); fig.tight_layout(); fig.savefig(OUT / "decoder_grid.png", dpi=110)

    # 3. student-test comparison with the deployed pipeline
    model, feats = bundle["model"], bundle["features"]
    res_rows = []
    for tid in list_trips("student_test"):
        tr = load_trip("student_test", tid)
        res = predict_trip(tr, model, feats, cfg)
        sc = score_trip(tr, res)
        res_rows.append(dict(trip=tid, **sc, continuity_clf=res["stats"]["continuity_clf"],
                             continuity_hmm=res["stats"]["continuity_hmm"],
                             unreachable_steps=res["stats"]["unreachable_steps"],
                             gap_breaks=res["stats"]["gap_breaks"], seconds=res["stats"]["total_s"]))
        print(res_rows[-1], flush=True)
        trip_map(tr, res, OUT / "maps" / f"student_test_{tid}.html")
    df = pd.DataFrame(res_rows)
    w = df.n_scored_obs / df.n_scored_obs.sum()
    tot = {c: float((df[c] * w).sum()) for c in df.columns if c not in ("trip", "n_scored_obs")}
    df.loc[len(df)] = dict(trip="ALL (obs-weighted)", n_scored_obs=int(df.n_scored_obs.sum()), **tot)
    df.to_csv(OUT / "student_test_clf_vs_hmm.csv", index=False)
    print(df.round(4).to_string())

    d = df.iloc[:-1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(d)); w_ = 0.4
    ax[0].bar(x - w_ / 2, d.top1_clf, w_, label="classifier only"); ax[0].bar(x + w_ / 2, d.top1_hmm, w_, label="+ HMM")
    ax[0].set_xticks(x, d.trip, rotation=45); ax[0].set_ylim(0.5, 1); ax[0].set_title("Top-1 per student-test trip"); ax[0].legend()
    ax[1].bar(x - w_ / 2, d.continuity_clf, w_, label="classifier only"); ax[1].bar(x + w_ / 2, d.continuity_hmm, w_, label="+ HMM")
    ax[1].set_xticks(x, d.trip, rotation=45); ax[1].set_ylim(0, 1); ax[1].set_title("Route continuity per trip"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "clf_vs_hmm.png", dpi=110)


if __name__ == "__main__":
    main()
