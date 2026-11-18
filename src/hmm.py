"""HMM / Viterbi route reconstruction.

Hidden states : for observation k, the candidate arcs C_k (the same candidate sets the
                classifier scored), each with its projection point on the arc.
Emission      : log P_e(c | z_k) = ALPHA * log p_clf(c), the classifier probability that
                candidate c is the reference arc (ALPHA tempers over-confident scores).
Transition    : Newson & Krumm (2009) style:
                log P_t(c_i -> c_j) = -|d_net(c_i, c_j) - d_gps(z_{k-1}, z_k)| / BETA
                d_net = shortest road-network distance from the projection on c_i to the
                projection on c_j, following arc direction (Dijkstra, distance-limited).
Unreachable   : if no path within the search limit (disconnected component, missing arc)
                the transition gets UNREACHABLE_LOGP instead of -inf, so decoding never
                dies; the break is reported as a continuity failure.
GPS gaps      : search limit grows with elapsed time; for gaps > GAP_BREAK_S the chain is
                split (transition term set to 0) because the vehicle could be anywhere.
"""
import heapq
from collections import defaultdict
import numpy as np
import pandas as pd

ALPHA = 1.0
BETA = 30.0               # metres
UNREACHABLE_LOGP = -50.0
GAP_BREAK_S = 120.0
MAX_SPEED = 50.0          # m/s, bounds the Dijkstra search radius
BACKTRACK_TOL = 15.0      # metres of apparent backwards movement on the same arc treated as noise


class RoadGraph:
    def __init__(self, nodes_xy, arcs):
        self.arcs = arcs
        self.xy = nodes_xy
        self.L = np.linalg.norm(nodes_xy[arcs[:, 1]] - nodes_xy[arcs[:, 0]], axis=1)
        self.out = defaultdict(list)
        for a, (u, v) in enumerate(arcs):
            self.out[int(u)].append((int(v), a))
        self._cache = {}

    def dijkstra(self, src, limit):
        """Node distances (and predecessor arcs) from node src up to `limit` metres. Cached."""
        c = self._cache.get(src)
        if c is not None and c[0] >= limit:
            return c[1], c[2]
        dist, pred = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for v, a in self.out[u]:
                nd = d + self.L[a]
                if nd < dist.get(v, np.inf) and nd <= limit:
                    dist[v] = nd; pred[v] = a
                    heapq.heappush(pq, (nd, v))
        self._cache[src] = (limit, dist, pred)
        return dist, pred

    def net_dist(self, ai, fi, aj, fj, limit):
        """Directed network distance between point at fraction fi on arc ai and fj on arc aj."""
        if ai == aj:
            fwd = (fj - fi) * self.L[ai]
            if fwd >= -BACKTRACK_TOL:
                return abs(fwd)
        to_i, from_j = int(self.arcs[ai, 1]), int(self.arcs[aj, 0])
        head = (1 - fi) * self.L[ai]
        dist, _ = self.dijkstra(to_i, limit)
        d = dist.get(from_j)
        return None if d is None else head + d + fj * self.L[aj]

    def path_arcs(self, ai, aj, limit):
        """Arc sequence from arc ai to arc aj (exclusive of ai, inclusive of aj) or None."""
        if ai == aj:
            return []
        to_i, from_j = int(self.arcs[ai, 1]), int(self.arcs[aj, 0])
        dist, pred = self.dijkstra(to_i, limit)
        if from_j not in dist:
            return None
        seq, n = [], from_j
        while n != to_i:
            a = pred[n]; seq.append(a); n = int(self.arcs[a, 0])
        return seq[::-1] + [aj]


def decode(cands: pd.DataFrame, track_xy, times, graph: RoadGraph, alpha=ALPHA, beta=BETA):
    """cands: rows (obs, arc, proj_frac, p) for one trip. Returns (obs->arc DataFrame, stats)."""
    groups = [(int(o), g.arc.to_numpy(), g.proj_frac.to_numpy(), np.log(np.clip(g.p.to_numpy(), 1e-9, 1)))
              for o, g in cands.sort_values(["obs", "p"], ascending=[True, False]).groupby("obs", sort=True)]
    if not groups:
        return pd.DataFrame(columns=["obs", "arc", "proj_frac"]), {"steps": 0, "unreachable_steps": 0, "gap_breaks": 0}
    score = alpha * groups[0][3]
    backs, n_unreach, n_breaks = [], 0, 0
    for k in range(1, len(groups)):
        o0, a0, f0, _ = groups[k - 1]
        o1, a1, f1, em = groups[k]
        dt = max(1.0, times[o1] - times[o0])
        d_gps = float(np.linalg.norm(track_xy[o1] - track_xy[o0]))
        if dt > GAP_BREAK_S:
            n_breaks += 1
            bi = np.full(len(a1), int(np.argmax(score)))
            score = score.max() + alpha * em
            backs.append(bi); continue
        limit = max(200.0, min(MAX_SPEED * dt, 3 * d_gps + 300.0))
        trans = np.full((len(a0), len(a1)), UNREACHABLE_LOGP)
        for i in range(len(a0)):
            for j in range(len(a1)):
                d = graph.net_dist(int(a0[i]), f0[i], int(a1[j]), f1[j], limit)
                if d is not None:
                    trans[i, j] = -abs(d - d_gps) / beta
        tot = score[:, None] + trans
        bi = tot.argmax(0)
        best = tot[bi, np.arange(len(a1))]
        if best.max() <= UNREACHABLE_LOGP + score.max() + 1e-9:
            n_unreach += 1
        score = best + alpha * em
        backs.append(bi)
    path = [int(np.argmax(score))]
    for bi in reversed(backs):
        path.append(int(bi[path[-1]]))
    path.reverse()
    out = pd.DataFrame({"obs": [g[0] for g in groups], "arc": [int(g[1][s]) for g, s in zip(groups, path)],
                        "proj_frac": [float(g[2][s]) for g, s in zip(groups, path)]})
    return out, {"steps": len(groups), "unreachable_steps": n_unreach, "gap_breaks": n_breaks}


def continuity(pred: pd.DataFrame, track_xy, times, graph: RoadGraph):
    """Route continuity: share of consecutive predictions (obs k-1 -> k) that are joined by a
    drivable network path whose length agrees with the GPS displacement
    (|d_net - d_gps| <= max(30 m, 0.5 * d_gps)). Jumps to a parallel street or to the
    opposite-direction twin arc fail this test. Gaps > GAP_BREAK_S are skipped.
    pred needs columns obs, arc, proj_frac."""
    o, a, f = pred.obs.to_numpy(), pred.arc.to_numpy(), pred.proj_frac.to_numpy()
    ok = tot = 0
    for k in range(1, len(o)):
        dt = max(1.0, times[o[k]] - times[o[k - 1]])
        if dt > GAP_BREAK_S:
            continue
        d_gps = float(np.linalg.norm(track_xy[o[k]] - track_xy[o[k - 1]]))
        limit = max(200.0, min(MAX_SPEED * dt, 3 * d_gps + 300.0))
        d = graph.net_dist(int(a[k - 1]), f[k - 1], int(a[k]), f[k], limit)
        tot += 1
        ok += d is not None and abs(d - d_gps) <= max(30.0, 0.5 * d_gps)
    return ok / max(tot, 1)


def full_route(pred: pd.DataFrame, track_xy, times, graph: RoadGraph):
    """Stitch decoded arcs into one connected arc sequence (for maps / UI)."""
    a = pred.arc.to_numpy(); o = pred.obs.to_numpy()
    if len(a) == 0:
        return []
    route = [int(a[0])]
    for k in range(1, len(a)):
        if a[k] == route[-1]:
            continue
        dt = max(1.0, times[o[k]] - times[o[k - 1]])
        d_gps = float(np.linalg.norm(track_xy[o[k]] - track_xy[o[k - 1]]))
        seg = graph.path_arcs(route[-1], int(a[k]), max(200.0, min(MAX_SPEED * dt, 3 * d_gps + 300.0)))
        route.extend(seg if seg is not None else [int(a[k])])
    return route
