"""HMM / Viterbi route reconstruction (first version).

Hidden states  : candidate arcs of each observation (same candidate sets the classifier scores).
Emission       : log P(candidate is reference | features) from the classifier.
Transition     : log of exp(-BETA * hops) where hops = directed hop distance in the road graph
                 from arc i to arc j (0 = same arc, 1 = j starts where i ends, ...), searched up
                 to MAX_HOPS. Unreachable within MAX_HOPS -> fixed penalty (keeps the decoder alive
                 through network gaps / disconnected components instead of failing).
Large GPS gaps : allowed hop budget grows with elapsed time between observations.
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

BETA = 1.0
MAX_HOPS = 6
UNREACHABLE_LOGP = -15.0


class ArcGraph:
    def __init__(self, arcs):
        self.arcs = arcs
        self.out = defaultdict(list)   # node -> outgoing arc ids
        for a, (u, v) in enumerate(arcs):
            self.out[u].append(a)
        self.cache = {}

    def hops_from(self, a, max_h):
        key = (a, max_h)
        if key in self.cache:
            return self.cache[key]
        dist = {a: 0}
        frontier = [a]
        for h in range(1, max_h + 1):
            nxt = []
            for x in frontier:
                for y in self.out[self.arcs[x, 1]]:
                    if y not in dist:
                        dist[y] = h; nxt.append(y)
            frontier = nxt
            if not frontier:
                break
        self.cache[key] = dist
        return dist


def viterbi_trip(df_trip: pd.DataFrame, prob: np.ndarray, arcs: np.ndarray, times: np.ndarray):
    """df_trip rows = (obs, arc) candidates sorted by obs. Returns DataFrame obs -> predicted arc."""
    g = ArcGraph(arcs)
    d = df_trip.assign(p=np.clip(prob, 1e-6, 1 - 1e-6)).sort_values(["obs", "dist"])
    groups = [(o, grp.arc.to_numpy(), np.log(grp.p.to_numpy())) for o, grp in d.groupby("obs", sort=True)]
    score = groups[0][2].copy()
    backs = []
    for k in range(1, len(groups)):
        o_prev, a_prev, _ = groups[k - 1]
        o, a_cur, em = groups[k]
        dt = max(1.0, times[o] - times[o_prev])
        max_h = int(min(40, MAX_HOPS + dt // 10))   # bigger hop budget across GPS gaps
        trans = np.full((len(a_prev), len(a_cur)), UNREACHABLE_LOGP)
        for i, ap in enumerate(a_prev):
            hd = g.hops_from(int(ap), max_h)
            for j, ac in enumerate(a_cur):
                h = hd.get(int(ac))
                if h is not None:
                    trans[i, j] = -BETA * h
        tot = score[:, None] + trans
        bi = tot.argmax(0)
        score = tot[bi, np.arange(len(a_cur))] + em
        backs.append(bi)
    path = [int(score.argmax())]
    for bi in reversed(backs):
        path.append(int(bi[path[-1]]))
    path.reverse()
    return pd.DataFrame({"obs": [g_[0] for g_ in groups],
                         "arc": [g_[1][s] for g_, s in zip(groups, path)]})
