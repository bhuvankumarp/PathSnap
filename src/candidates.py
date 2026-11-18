"""Per-observation labels, candidate generation and feature engineering.

Labels: the .route file is a chronological arc sequence, not one arc per GPS
point. We align points to route arcs with a monotonic DP (route index may only
stay or advance over time), minimising total point-to-segment distance. The
aligned arc is the observation's reference road segment.

Candidates: all arcs within RADIUS metres of the point, keep the K nearest.
Each (observation, candidate) pair becomes one row; target = is_reference.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from data_io import Trip

RADIUS = 60.0   # metres
K = 8           # max candidates per observation
OFFROUTE_M = 50.0  # alignment residual above which a fix is treated as off-route


def point_seg_dist(P, A, B):
    """Distances from points P (n,2) to segments A->B (n,2). Returns dist, t in [0,1]."""
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    t = np.where(L2 > 0, np.einsum("ij,ij->i", P - A, AB) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0, 1)
    proj = A + AB * t[:, None]
    return np.linalg.norm(P - proj, axis=1), t


def align_labels(tr: Trip):
    """Monotonic DP alignment of GPS points to the reference route arcs."""
    P = tr.track_xy
    R = tr.route
    A = tr.nodes_xy[tr.arcs[R, 0]]
    B = tr.nodes_xy[tr.arcs[R, 1]]
    T, n = len(P), len(R)
    cost = np.empty((T, n))
    for k in range(n):
        cost[:, k], _ = point_seg_dist(P, np.broadcast_to(A[k], P.shape), np.broadcast_to(B[k], P.shape))
    D = np.empty_like(cost)
    back = np.empty((T, n), dtype=np.int32)
    D[0] = cost[0]
    for i in range(1, T):
        prev = D[i - 1]
        # running min / argmin over j<=k (route index can only stay or advance)
        cm = np.minimum.accumulate(prev)
        cm_idx = np.maximum.accumulate(np.where(prev == cm, np.arange(n), 0)).astype(np.int32)
        D[i] = cost[i] + cm
        back[i] = cm_idx
    k = int(np.argmin(D[-1]))
    idx = np.empty(T, dtype=np.int32)
    for i in range(T - 1, -1, -1):
        idx[i] = k
        k = back[i, k]
    return R[idx], idx, cost[np.arange(T), idx]


class ArcIndex:
    """KD-tree over points sampled every ~5 m along each arc."""

    def __init__(self, tr: Trip, step=5.0):
        A = tr.nodes_xy[tr.arcs[:, 0]]
        B = tr.nodes_xy[tr.arcs[:, 1]]
        L = np.linalg.norm(B - A, axis=1)
        n = np.maximum(1, np.ceil(L / step).astype(int))
        arc_rep = np.repeat(np.arange(len(L)), n + 1)
        frac = np.concatenate([np.linspace(0, 1, k + 1) for k in n])
        pts = A[arc_rep] + (B - A)[arc_rep] * frac[:, None]
        self.tree = cKDTree(pts)
        self.arc_rep = arc_rep
        self.A, self.B, self.L = A, B, L
        self.step = step

    def query(self, P, radius=RADIUS, k=K):
        lists = self.tree.query_ball_point(P, r=radius + self.step)
        out = []
        for i, idx in enumerate(lists):
            arcs = np.unique(self.arc_rep[idx]) if idx else np.empty(0, int)
            if len(arcs):
                d, t = point_seg_dist(np.broadcast_to(P[i], (len(arcs), 2)), self.A[arcs], self.B[arcs])
                keep = d <= radius
                arcs, d, t = arcs[keep], d[keep], t[keep]
                o = np.argsort(d)[:k]
                arcs, d, t = arcs[o], d[o], t[o]
            else:
                d = t = np.empty(0)
            out.append((arcs, d, t))
        return out


def gps_heading(xy, w=2):
    """Heading of travel at each point (radians, math convention) from a +-w window."""
    a = np.vstack([np.repeat(xy[:1], w, 0), xy, np.repeat(xy[-1:], w, 0)])
    v = a[2 * w:] - a[:-2 * w]
    return np.arctan2(v[:, 1], v[:, 0]), np.linalg.norm(v, axis=1)


def build_features(tr: Trip, obs_idx=None, with_labels=True):
    """Return DataFrame with one row per (observation, candidate)."""
    P = tr.track_xy
    T = len(P)
    if obs_idx is None:
        obs_idx = np.arange(T)
    t = tr.track[:, 2]
    head, disp = gps_heading(P)
    dt_win = np.clip(np.r_[t[2:], [t[-1]] * 2][:T] - np.r_[[t[0]] * 2, t[:-2]][:T], 1, None)
    speed = disp / dt_win
    gap_prev = np.r_[0, np.diff(t)]
    deg = np.bincount(tr.arcs.ravel(), minlength=len(tr.nodes))

    idx = ArcIndex(tr)
    labels = None
    if with_labels:
        labels, _, resid = align_labels(tr)
        # points far from every reference arc are off-route (detours / off-network):
        # their label is unreliable, so they are excluded from training & scoring.
        obs_idx = obs_idx[resid[obs_idx] <= OFFROUTE_M]
    cands = idx.query(P[obs_idx])

    rows = []
    for j, (arcs, d, frac) in zip(obs_idx, cands):
        if len(arcs) == 0:
            continue
        ah = np.arctan2(*(idx.B[arcs] - idx.A[arcs])[:, ::-1].T)
        hd = np.abs((ah - head[j] + np.pi) % (2 * np.pi) - np.pi)  # 0..pi, direction-aware
        hd_undir = np.minimum(hd, np.pi - hd)
        moving = disp[j] > 2.0
        r = pd.DataFrame({
            "trip": tr.trip_id, "obs": j, "arc": arcs,
            "dist": d, "dist_rank": np.arange(len(arcs)), "dist_minus_min": d - d.min(),
            "dist_ratio": d / (d.min() + 1.0), "n_cand": len(arcs),
            "head_diff": np.where(moving, hd, np.pi / 2), "head_diff_undir": np.where(moving, hd_undir, np.pi / 4),
            "moving": moving.astype(int), "speed": speed[j], "gap_prev": gap_prev[j],
            "arc_len": idx.L[arcs], "proj_frac": frac,
            "proj_end": np.minimum(frac, 1 - frac),
            "deg_from": deg[tr.arcs[arcs, 0]], "deg_to": deg[tr.arcs[arcs, 1]],
        })
        if with_labels:
            r["y"] = (arcs == labels[j]).astype(int)
            r["ref_arc"] = labels[j]
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


FEATURES = ["dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand", "head_diff",
            "head_diff_undir", "moving", "speed", "gap_prev", "arc_len", "proj_frac",
            "proj_end", "deg_from", "deg_to"]
