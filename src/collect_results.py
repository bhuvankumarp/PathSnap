"""Post-processing: recompute network-based route continuity for every model's student-test
predictions (the same metric the HMM experiments use) and add it to model_comparison.csv."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_io import load_trip
import hmm
from train_models import slug

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    te = pd.read_pickle(ROOT / "features" / "student_test.pkl")[["trip", "obs", "arc", "proj_frac", "dist"]]
    trips = {t: load_trip("student_test", t) for t in te.trip.unique()}
    graphs = {t: hmm.RoadGraph(tr.nodes_xy, tr.arcs) for t, tr in trips.items()}
    mc = pd.read_csv(RES / "model_comparison.csv")

    def cont(best):
        ok = n = 0
        for t, g in best.groupby("trip"):
            tr = trips[t]; g = g.sort_values("obs")
            c = hmm.continuity(g, tr.track_xy, tr.track[:, 2], graphs[t])
            ok += c * (len(g) - 1); n += len(g) - 1
        return ok / n

    vals = []
    for m in mc.model:
        if m.startswith("Nearest"):
            d = te.assign(p=-te.dist)
        else:
            p = pd.read_pickle(RES / "predictions" / f"{slug(m)}.pkl")
            d = te.merge(p[["trip", "obs", "arc", "p"]], on=["trip", "obs", "arc"])
        best = d.loc[d.groupby(["trip", "obs"]).p.idxmax()]
        vals.append(cont(best)); print(m, round(vals[-1], 4), flush=True)
    mc["test_route_continuity"] = vals
    mc = mc.rename(columns={"test_continuity": "test_adjacent_arc_rate"})
    mc.to_csv(RES / "model_comparison.csv", index=False)
    print(mc.round(4).to_string())


if __name__ == "__main__":
    main()
