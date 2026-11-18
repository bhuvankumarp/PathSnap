"""Build candidate-feature tables for train and student_test.

Every STRIDE-th GPS fix is used (1 Hz fixes are highly autocorrelated; stride 5
keeps ~20% of rows while preserving trip-level variety).
Writes features/<partition>.pkl and results/candidates_report.csv.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_io import list_trips, load_trip
from candidates import build_features, align_labels, OFFROUTE_M

ROOT = Path(__file__).resolve().parents[1]
STRIDE = 5


def main():
    (ROOT / "features").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    report = []
    for part in ("train", "student_test"):
        dfs = []
        t0 = time.time()
        for tid in list_trips(part):
            tr = load_trip(part, tid)
            _, _, resid = align_labels(tr)
            obs = np.arange(0, len(tr.track), STRIDE)
            df = build_features(tr, obs)
            dfs.append(df)
            offroute = float(np.mean(resid > OFFROUTE_M))
            n_obs = int(np.sum(resid[obs] <= OFFROUTE_M))
            got = df.groupby("obs").y.max()
            nearest = df[df.dist_rank == 0].set_index("obs").y
            report.append(dict(partition=part, trip=tid, n_obs=n_obs, offroute_frac=offroute,
                               obs_with_cands=len(got),
                               cand_miss_rate=1 - got.sum() / n_obs,
                               nearest_top1=nearest.sum() / n_obs,
                               mean_cands=df.groupby("obs").size().mean(),
                               align_resid_median_m=float(np.median(resid)),
                               align_resid_p95_m=float(np.percentile(resid, 95))))
            print(part, tid, report[-1]["cand_miss_rate"].round(3), report[-1]["nearest_top1"].round(3), flush=True)
        pd.concat(dfs, ignore_index=True).to_pickle(ROOT / "features" / f"{part}.pkl")
        print(part, "done in", round(time.time() - t0, 1), "s")
    rep = pd.DataFrame(report)
    rep.to_csv(ROOT / "results" / "candidates_report.csv", index=False)
    print(rep.groupby("partition")[["offroute_frac", "cand_miss_rate", "nearest_top1", "mean_cands",
                                    "align_resid_median_m", "align_resid_p95_m"]].mean().round(4))


if __name__ == "__main__":
    main()
