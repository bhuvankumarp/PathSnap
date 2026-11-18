"""End-to-end inference for one trip + separate server-side scoring.

predict_trip() never sees the .route file: it uses only the GPS track and the road network.
score_trip() is the only function that reads the reference route.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from data_io import Trip
from candidates import build_features, align_labels, OFFROUTE_M
import hmm

ROOT = Path(__file__).resolve().parents[1]
STRIDE = 5


def load_artifacts():
    bundle = joblib.load(ROOT / "models" / "lgbm_tuned.joblib")
    cfg = json.loads((ROOT / "models" / "hmm_config.json").read_text())
    return bundle["model"], bundle["features"], cfg


def predict_trip(tr: Trip, model, features, cfg, stride=STRIDE):
    t0 = time.time()
    obs = np.arange(0, len(tr.track), stride)
    cand = build_features(tr, obs, with_labels=False)
    cand["p"] = model.predict_proba(cand[features])[:, 1]
    t_clf = time.time() - t0
    clf_only = cand.loc[cand.groupby("obs").p.idxmax(), ["obs", "arc", "proj_frac"]].reset_index(drop=True)
    g = hmm.RoadGraph(tr.nodes_xy, tr.arcs)
    times = tr.track[:, 2]
    dec, stats = hmm.decode(cand, tr.track_xy, times, g, alpha=cfg["alpha"], beta=cfg["beta"])
    route = hmm.full_route(dec, tr.track_xy, times, g)
    stats.update(n_fixes=len(tr.track), n_matched_obs=len(dec), n_no_candidate=int(len(obs) - len(dec)),
                 classifier_s=round(t_clf, 3), total_s=round(time.time() - t0, 3),
                 continuity_hmm=hmm.continuity(dec, tr.track_xy, times, g),
                 continuity_clf=hmm.continuity(clf_only, tr.track_xy, times, g))
    return dict(hmm=dec, clf=clf_only, route=route, stats=stats, cand=cand)


def score_trip(tr: Trip, pred, stride=STRIDE):
    """Server-side evaluation against the reference route (never returned to the model)."""
    labels, _, resid = align_labels(tr)
    obs = np.arange(0, len(tr.track), stride)
    obs = obs[resid[obs] <= OFFROUTE_M]                     # same off-route rule as training
    ref = pd.Series(labels[obs], index=obs)
    out = {"n_scored_obs": int(len(obs))}
    for k in ("clf", "hmm"):
        p = pred[k].set_index("obs").arc.reindex(obs)        # missing prediction counts as wrong
        out[f"top1_{k}"] = float((p.to_numpy() == ref.to_numpy()).mean())
    ref_set, pr_set = set(map(int, tr.route)), set(pred["route"])
    tp = len(ref_set & pr_set)
    L = np.linalg.norm(tr.nodes_xy[tr.arcs[:, 1]] - tr.nodes_xy[tr.arcs[:, 0]], axis=1)
    lt = L[list(ref_set & pr_set)].sum() if tp else 0.0
    out.update(route_precision=tp / max(len(pr_set), 1), route_recall=tp / max(len(ref_set), 1),
               route_f1=2 * tp / max(len(pr_set) + len(ref_set), 1),
               route_len_recall=float(lt / L[list(ref_set)].sum()))
    return out
