# PathSnap — Map-Matching & Route Reconstruction

Noisy 1 Hz GPS tracks → road-network arcs.

```
GPS track + road network
   └─ candidate generation (≤8 arcs within 60 m, KD-tree)          src/candidates.py
       └─ 15 features per (fix, candidate)
           └─ LightGBM candidate scorer (Optuna-tuned)             src/tune_lgbm.py
               └─ HMM / Viterbi with road-network transitions      src/hmm.py
                   └─ connected predicted route
FastAPI on laptop 1 (api/main.py)  ⇄  Leaflet UI on laptop 2 (ui/index.html)
```

## Repository layout
| Path | What it holds |
|---|---|
| `data/train/` | 80 trips: fitting, GroupKFold CV, decoder tuning |
| `data/student_test/` | 10 trips: development comparison |
| `data/evaluator_test/` | **put the 10 evaluator trips here on laptop 1** (git-ignored) |
| `src/data_io.py` | raw-file loading, per-trip metric projection |
| `src/eda.py` | inventory, anomaly checks, plots, Folium maps → `results/eda/` |
| `src/candidates.py` | route→fix label alignment (monotonic DP), candidates, features |
| `src/build_dataset.py` | feature tables → `features/` (stride 5) |
| `src/train_models.py` | 7 models + baseline, GroupKFold CV, confusion matrices → `results/` |
| `src/tune_lgbm.py` | Optuna TPE search, tuned vs untuned → `results/hpo/`, `models/lgbm_tuned.joblib` |
| `src/hmm.py` | road graph, Dijkstra, Viterbi decoder, continuity metric |
| `src/hmm_experiments.py` | decoder tuning on OOF train probs, classifier vs HMM, Folium maps → `results/hmm/`, `models/hmm_config.json` |
| `src/hmm_viterbi.py`, `src/eval_hmm.py` | HMM v1 (hop-count transitions, untuned LightGBM), kept as an earlier experiment → `results/hmm_comparison.csv` |
| `src/make_progress_page.py` | first-review progress page → `results/progress.html` |
| `src/pipeline.py` | inference (`predict_trip`, no labels) and server-side scoring (`score_trip`) |
| `src/collect_results.py` | network-based continuity for every model |
| `src/data_quality.py` | data-quality record → `results/eda/data_quality.md` |
| `src/make_report.py` | draft PDF report from result files → `report/` |
| `src/make_review_site.py` | first-review walkthrough page → `review1.html` (`run_review.bat`) |
| `screenshots/` | UI and API screenshots for the report |
| `api/main.py` | FastAPI service |
| `ui/index.html` | evaluator UI |

## Team and ownership
| Member | Main areas | Files |
|---|---|---|
| Bhuvan | labels, candidates, core features, final model, Bayesian optimisation, HMM/Viterbi, server-side evaluation, integration | `src/data_io.py`, `candidates.py`, `build_dataset.py`, `tune_lgbm.py`, `hmm.py`, `hmm_experiments.py`, `pipeline.py`, `collect_results.py`, `hmm_viterbi.py`, `eval_hmm.py`, `models/`, `results/hpo/`, `results/hmm/` |
| Vivek | metrics, cross-validation harness, baseline models, comparison plots, FastAPI backend | `src/train_models.py`, `api/main.py`, `results/model_comparison.*`, `results/confusion/` |
| Shrika | EDA, data-quality record, Folium maps, report figures | `src/eda.py`, `data_quality.py`, `make_report.py`, `make_review_site.py`, `make_progress_page.py`, `results/eda/`, `report/` |
| Bhoomika | evaluator UI, run instructions, UI and API screenshots | `ui/index.html`, `README.md`, `run_*.bat`, `screenshots/` |

Everyone reads the HMM section of the report before evaluation day.

## Data
Raw data is not committed (≈94 MB, distributed by the organisers). Unzip `Student folder.zip`
and copy its `train/` and `student_test/` folders into `data/`. Each trip folder holds
`<id>.track`, `<id>.nodes`, `<id>.arcs`, `<id>.route`.

## Install
```
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Reproduce all results (≈30 min on a laptop)
```
python src/eda.py
python src/build_dataset.py
python src/train_models.py
python src/tune_lgbm.py
python src/hmm_experiments.py
python src/collect_results.py
python src/make_report.py
```
Seeds are fixed (42). Logs: `results/train_models.log`, `results/hpo/tune.log`.

## Deployment (two laptops, same Wi-Fi / hotspot)

**Laptop 1 — API**
1. Copy this repo (with `models/`) and place the evaluator trips in `data/evaluator_test/<trip_id>/`.
2. Find its IP: `ipconfig` (Windows) / `ip a` (Linux) / `ipconfig getifaddr en0` (macOS).
3. Start the API (it serves `evaluator_test` automatically when that folder exists; override with
   `PATHSNAP_PARTITION=student_test`):
   ```
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   (or double-click `run_api.bat`, which also prints the laptop's IP)
   Allow Python through the firewall if Windows asks. Every request is logged in this console and
   in `logs/api.log` — show this to the evaluator.

**Laptop 2 — UI**
```
cd ui
python -m http.server 5500
```
(or double-click `run_ui.bat`). Open `http://localhost:5500`, enter `http://<laptop-1-IP>:8000`, press **Connect**, pick a trip.

### API
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status, model, HMM config |
| GET | `/trips` | – | trip ids in the served partition |
| POST | `/predict` | `{"trip_id": "00000007"}` | GPS points, predicted route geometry, per-fix matches, metrics, timing |

The reference routes never leave laptop 1: `/predict` returns accuracy numbers only.

## Key decisions
- **Distance**: local equirectangular projection per trip (trips span several continents).
- **Labels**: `.route` is an ordered arc list; each fix gets the route arc chosen by a monotonic DP.
  Fixes > 50 m from every route arc are off-route and excluded from training and scoring.
- **Stride 5**: every 5th fix is matched (1 Hz fixes are highly autocorrelated).
- **Directed arcs**: ~92% of arcs have an opposite twin at identical distance, so direction-aware
  heading difference is the key feature after distance.
- **No leakage**: GroupKFold by trip; decoder tuned on out-of-fold train probabilities; evaluator data
  is git-ignored and only read by the API.
