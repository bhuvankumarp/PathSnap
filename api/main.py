"""PathSnap inference API (runs on Laptop 1).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Trips are read from data/<PATHSNAP_PARTITION>/ (default: evaluator_test if present, else
student_test). The reference .route files stay on this machine: they are used only by
score_trip() after prediction, and the response carries accuracy numbers, never the
reference route itself.
"""
import os, sys, time, logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_io import DATA_ROOT, list_trips, load_trip   # noqa: E402
from pipeline import load_artifacts, predict_trip, score_trip   # noqa: E402

(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / "api.log")])
log = logging.getLogger("pathsnap")

PARTITION = os.environ.get("PATHSNAP_PARTITION") or (
    "evaluator_test" if (DATA_ROOT / "evaluator_test").exists() else "student_test")
MODEL, FEATURES, CFG = load_artifacts()
log.info("Loaded LightGBM + HMM config %s | serving partition '%s' (%d trips)",
         CFG, PARTITION, len(list_trips(PARTITION)))

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, 1000 * (time.time() - t0))
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "partition": PARTITION, "model": "LightGBM (Optuna-tuned) + HMM/Viterbi",
            "hmm": CFG}


@app.get("/trips")
def trips():
    return {"partition": PARTITION, "trips": list_trips(PARTITION)}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(PARTITION):
        raise HTTPException(404, f"Unknown trip '{req.trip_id}' in partition '{PARTITION}'")
    log.info("PREDICT request trip=%s", req.trip_id)
    tr = load_trip(PARTITION, req.trip_id)
    res = predict_trip(tr, MODEL, FEATURES, CFG)          # GPS + network only
    metrics = score_trip(tr, res)                          # server-side reference comparison
    log.info("PREDICT trip=%s top1_hmm=%.4f top1_clf=%.4f route_f1=%.4f obs=%d time=%.2fs",
             req.trip_id, metrics["top1_hmm"], metrics["top1_clf"], metrics["route_f1"],
             metrics["n_scored_obs"], res["stats"]["total_s"])

    def seg(a):
        u, v = tr.arcs[a]
        return [[float(tr.nodes[u, 1]), float(tr.nodes[u, 0])], [float(tr.nodes[v, 1]), float(tr.nodes[v, 0])]]

    step = max(1, len(tr.track) // 2000)
    return {
        "trip_id": req.trip_id,
        "gps": [[float(lat), float(lon)] for lon, lat, _ in tr.track[::step]],
        "predicted_route": [seg(a) for a in res["route"]],
        "classifier_only": [seg(a) for a in sorted(set(res["clf"].arc.astype(int)))],
        "matches": [{"obs": int(o), "t": float(tr.track[o, 2]), "arc": int(a)}
                    for o, a in zip(res["hmm"].obs, res["hmm"].arc)],
        "metrics": metrics,
        "stats": res["stats"],
    }
