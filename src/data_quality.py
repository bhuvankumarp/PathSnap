"""Write results/eda/data_quality.md: the data-quality record (issues, decisions, assumptions),
with every number read from results/eda/inventory.csv and results/candidates_report.csv.
Run after eda.py and build_dataset.py."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
inv = pd.read_csv(ROOT / "results" / "eda" / "inventory.csv", dtype={"trip": str})
cand = pd.read_csv(ROOT / "results" / "candidates_report.csv", dtype={"trip": str})
tr = cand[cand.partition == "train"]
gap = inv.loc[inv.max_gap_s.idxmax()]
jump = inv.loc[inv.n_speed_gt50ms.idxmax()]
off = tr.loc[tr.offroute_frac.idxmax()]

md = f"""# Data-quality record

Partitions analysed: train ({(inv.partition == 'train').sum()} trips) and student_test ({(inv.partition == 'student_test').sum()} trips).
Evaluator-test data was not available and was not used.

| # | Issue found | Evidence | Decision |
|---|---|---|---|
| 1 | Coordinates are degrees and trips span several continents | longitude {inv.lon_min.min():.1f}° to {inv.lon_max.max():.1f}°, latitude {inv.lat_min.min():.1f}° to {inv.lat_max.max():.1f}° | Project each trip to metres (local equirectangular around its centroid) before any distance |
| 2 | Missing values | {int(inv.nan_gps.sum())} NaN coordinates, {int(inv.non_monotonic_t.sum())} non-increasing timestamps | No imputation needed |
| 3 | Irregular sampling / time gaps | {100 * inv.frac_dt_1s.mean():.1f}% of steps are exactly 1 s; {int(inv.n_gaps_gt5s.sum())} gaps > 5 s; {(inv.max_gap_s > 60).sum()} trips have a gap > 60 s; largest {gap.max_gap_s:,.0f} s (trip {gap.trip}) | `gap_prev` feature; HMM widens its search with elapsed time and splits the chain for gaps > 120 s |
| 4 | GPS jumps | {int(inv.n_speed_gt50ms.sum())} steps imply > 50 m/s across {(inv.n_speed_gt50ms > 0).sum()} trips (worst trip {jump.trip}: {int(jump.n_speed_gt50ms)}) | Speed is smoothed over a ±2-fix window; the HMM penalises transitions whose road distance disagrees with GPS distance |
| 5 | Off-route driving | {100 * tr.offroute_frac.mean():.1f}% of train fixes are > 50 m from every reference arc (worst trip {off.trip}: {100 * off.offroute_frac:.1f}%) | Treated as unreliable labels; excluded from training and scoring |
| 6 | Reference route is an arc list, not per-fix labels | routes have {int(inv.n_route_arcs.min())}–{int(inv.n_route_arcs.max())} arcs vs {int(inv.n_gps.min())}–{int(inv.n_gps.max())} fixes; route continuity {100 * inv.route_continuity.mean():.2f}% | Per-fix labels from a monotonic DP alignment (route index never goes backwards) |
| 7 | Opposite-direction twin arcs | {100 * inv.frac_bidirectional_arcs.mean():.1f}% of arcs have a reverse twin | Direction-aware heading-difference feature |
| 8 | Road network is fragmented | median {inv.weak_components.median():.0f} weakly connected components per trip; largest holds {100 * inv.largest_comp_frac.mean():.1f}% of nodes on average; {int(inv.isolated_nodes.sum())} isolated nodes | Unreachable HMM transitions get a finite penalty instead of −∞ |
| 9 | Highly correlated consecutive fixes (1 Hz) | median {inv.n_gps.median():.0f} fixes per trip | Every 5th fix used for features, training and decoding |
| 10 | Candidate recall | reference arc missing from candidates for {100 * tr.cand_miss_rate.mean():.2f}% (train) / {100 * cand[cand.partition == 'student_test'].cand_miss_rate.mean():.2f}% (student-test) of fixes | 60 m radius, ≤ 8 candidates kept |

Assumption: the reference route is treated as the best available label, not error-free ground truth
(median alignment residual {tr.align_resid_median_m.mean():.1f} m, 95th percentile {tr.align_resid_p95_m.mean():.1f} m on train).
"""
(ROOT / "results" / "eda" / "data_quality.md").write_text(md, encoding="utf-8")
print("wrote", ROOT / "results" / "eda" / "data_quality.md")
