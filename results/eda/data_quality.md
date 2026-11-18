# Data-quality record

Partitions analysed: train (80 trips) and student_test (10 trips).
Evaluator-test data was not available and was not used.

| # | Issue found | Evidence | Decision |
|---|---|---|---|
| 1 | Coordinates are degrees and trips span several continents | longitude -100.4° to 137.2°, latitude -34.6° to 61.7° | Project each trip to metres (local equirectangular around its centroid) before any distance |
| 2 | Missing values | 0 NaN coordinates, 0 non-increasing timestamps | No imputation needed |
| 3 | Irregular sampling / time gaps | 94.3% of steps are exactly 1 s; 1260 gaps > 5 s; 23 trips have a gap > 60 s; largest 153,064 s (trip 00000090) | `gap_prev` feature; HMM widens its search with elapsed time and splits the chain for gaps > 120 s |
| 4 | GPS jumps | 1531 steps imply > 50 m/s across 42 trips (worst trip 00000032: 887) | Speed is smoothed over a ±2-fix window; the HMM penalises transitions whose road distance disagrees with GPS distance |
| 5 | Off-route driving | 2.5% of train fixes are > 50 m from every reference arc (worst trip 00000054: 89.4%) | Treated as unreliable labels; excluded from training and scoring |
| 6 | Reference route is an arc list, not per-fix labels | routes have 59–2266 arcs vs 262–19313 fixes; route continuity 99.94% | Per-fix labels from a monotonic DP alignment (route index never goes backwards) |
| 7 | Opposite-direction twin arcs | 92.2% of arcs have a reverse twin | Direction-aware heading-difference feature |
| 8 | Road network is fragmented | median 19 weakly connected components per trip; largest holds 96.8% of nodes on average; 0 isolated nodes | Unreachable HMM transitions get a finite penalty instead of −∞ |
| 9 | Highly correlated consecutive fixes (1 Hz) | median 1496 fixes per trip | Every 5th fix used for features, training and decoding |
| 10 | Candidate recall | reference arc missing from candidates for 1.16% (train) / 0.29% (student-test) of fixes | 60 m radius, ≤ 8 candidates kept |

Assumption: the reference route is treated as the best available label, not error-free ground truth
(median alignment residual 8.2 m, 95th percentile 24.4 m on train).
