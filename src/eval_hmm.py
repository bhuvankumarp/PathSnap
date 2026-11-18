"""Classifier-only vs classifier + HMM/Viterbi on student_test (LightGBM, untuned).
Writes results/hmm_comparison.csv and a Folium map comparing routes for one trip."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent))
from candidates import FEATURES
from data_io import load_trip
from hmm_viterbi import viterbi_trip
from train_models import top1, continuity

ROOT = Path(__file__).resolve().parents[1]


def main():
    tr = pd.read_pickle(ROOT / "features" / "train.pkl")
    te = pd.read_pickle(ROOT / "features" / "student_test.pkl")
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf.fit(tr[FEATURES], tr.y)
    te = te.assign(p=clf.predict_proba(te[FEATURES])[:, 1])
    rows, arcs_by = [], {}
    for tid, g in te.groupby("trip"):
        trip = load_trip("student_test", tid); arcs_by[tid] = trip.arcs
        ref = g.groupby("obs").ref_arc.first()
        _, best = top1(g, g.p.to_numpy())
        clf_acc = (best.set_index("obs").arc == ref).mean()
        hmm = viterbi_trip(g, g.p.to_numpy(), trip.arcs, trip.track[:, 2]).set_index("obs")
        hmm_acc = (hmm.arc == ref.loc[hmm.index]).mean()
        hb = hmm.reset_index().assign(trip=tid)
        rows.append(dict(trip=tid, n_obs=len(ref), clf_top1=clf_acc, hmm_top1=hmm_acc,
                         clf_continuity=continuity(best, arcs_by), hmm_continuity=continuity(hb, arcs_by)))
        print(rows[-1], flush=True)
        if tid == "00000007":
            m = folium.Map(location=[trip.lat0, trip.lon0], zoom_start=15, tiles="OpenStreetMap")
            def layer(name, arc_ids, color, w):
                fg = folium.FeatureGroup(name)
                for a in arc_ids:
                    u, v = trip.arcs[a]
                    folium.PolyLine([(trip.nodes[u, 1], trip.nodes[u, 0]), (trip.nodes[v, 1], trip.nodes[v, 0])],
                                    color=color, weight=w, opacity=0.8).add_to(fg)
                fg.add_to(m)
            layer("Reference route", np.unique(trip.route), "#1e8e3e", 8)
            layer("Classifier-only", np.unique(best.arc), "#d93025", 4)
            layer("Classifier + HMM", np.unique(hmm.arc), "#9334e6", 3)
            gps = folium.FeatureGroup("GPS")
            for lon, lat, _ in trip.track[::3]:
                folium.CircleMarker((lat, lon), radius=2, color="#1a73e8", weight=0, fill=True, fill_opacity=0.8).add_to(gps)
            gps.add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)
            m.save(ROOT / "results" / "map_hmm_vs_clf_00000007.html")
    df = pd.DataFrame(rows)
    w = df.n_obs / df.n_obs.sum()
    df.loc[len(df)] = dict(trip="ALL (obs-weighted)", n_obs=df.n_obs.sum(),
                           clf_top1=(df.clf_top1 * w).sum(), hmm_top1=(df.hmm_top1 * w).sum(),
                           clf_continuity=(df.clf_continuity * w).sum(), hmm_continuity=(df.hmm_continuity * w).sum())
    df.to_csv(ROOT / "results" / "hmm_comparison.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
