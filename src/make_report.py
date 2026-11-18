"""Build report/PathSnap_Report_DRAFT.pdf from the result files.

All numbers and figures are read from results/ (nothing typed by hand). Narrative text is a
DRAFT scaffold: the team must rewrite it in its own words, check every claim against the code,
and fill in the AI-assistance disclosure before submission.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak)

sys.path.insert(0, str(Path(__file__).parent))
from data_io import load_trip
from train_models import slug

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report"; OUT.mkdir(exist_ok=True)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ss = getSampleStyleSheet()
# Unicode TTF so ±, °, →, − render (Helvetica lacks them); falls back to Helvetica if absent.
for cand_dir in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")):
    reg, bold = cand_dir / "arial.ttf", cand_dir / "arialbd.ttf"
    if not reg.exists():
        reg, bold = cand_dir / "DejaVuSans.ttf", cand_dir / "DejaVuSans-Bold.ttf"
    if reg.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("UF", str(reg))); pdfmetrics.registerFont(TTFont("UF-Bold", str(bold)))
        pdfmetrics.registerFontFamily("UF", normal="UF", bold="UF-Bold", italic="UF", boldItalic="UF-Bold")
        for st in ss.byName.values():
            if not hasattr(st, "fontName"):
                continue
            st.fontName = "UF-Bold" if "Bold" in st.fontName or st.name.startswith(("Heading", "Title")) else "UF"
        break
H1, H2, H3, P = ss["Heading1"], ss["Heading2"], ss["Heading3"], ss["BodyText"]
NOTE = ParagraphStyle("note", parent=P, textColor=colors.HexColor("#8a1c1c"), fontSize=9)
SMALL = ParagraphStyle("small", parent=P, fontSize=8, leading=10)


def tbl(df, pct=(), dec=3, col_w=None):
    d = df.copy()
    for c in d.columns:
        if c in pct:
            d[c] = d[c].map(lambda v: "N/A" if pd.isna(v) else f"{100 * v:.1f}%")
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "N/A" if pd.isna(v) else f"{v:.{dec}f}")
    if col_w is None:
        col_w = [17.6 * cm / len(d.columns)] * len(d.columns)
    data = [[Paragraph(f"<b>{c.replace('_', ' ')}</b>", SMALL) for c in d.columns]] + \
           [[Paragraph(str(v), SMALL) for v in row] for row in d.values]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eee9")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def img(p, w=17):
    p = Path(p)
    if not p.exists():
        return Paragraph(f"[missing figure: {p.name}]", NOTE)
    from PIL import Image as PI
    W, Hh = PI.open(p).size
    return Image(str(p), width=w * cm, height=w * cm * Hh / W)


def static_map(tid):
    """Exported (static) view of the same layers as the Folium map, for the PDF."""
    tr = load_trip("student_test", tid)
    import joblib
    from pipeline import predict_trip
    b = joblib.load(ROOT / "models" / "lgbm_tuned.joblib")
    cfg = json.loads((ROOT / "models" / "hmm_config.json").read_text())
    res = predict_trip(tr, b["model"], b["features"], cfg)
    fig, a = plt.subplots(figsize=(8, 7))
    def draw(ids, c, lw, lab):
        for k, x in enumerate(ids):
            u, v = tr.arcs[x]
            a.plot(tr.nodes[[u, v], 0], tr.nodes[[u, v], 1], c=c, lw=lw, label=lab if k == 0 else None, solid_capstyle="round")
    xmin, xmax = tr.track[:, 0].min(), tr.track[:, 0].max(); ymin, ymax = tr.track[:, 1].min(), tr.track[:, 1].max()
    mid = (tr.nodes[tr.arcs[:, 0]] + tr.nodes[tr.arcs[:, 1]]) / 2
    inb = np.flatnonzero((mid[:, 0] > xmin - .003) & (mid[:, 0] < xmax + .003) & (mid[:, 1] > ymin - .003) & (mid[:, 1] < ymax + .003))
    draw(inb, "#cfd4d1", 0.6, "Road network")
    draw(np.unique(tr.route), "#1e8e3e", 6, "Reference route")
    draw(np.unique(res["clf"].arc), "#d93025", 2.2, "Classifier-only arcs")
    draw(res["route"], "#7b1fa2", 2.2, "Predicted route (HMM)")
    a.scatter(tr.track[::5, 0], tr.track[::5, 1], s=4, c="#1a73e8", label="GPS (every 5th fix)", zorder=5)
    a.set_xlim(xmin - .002, xmax + .002); a.set_ylim(ymin - .002, ymax + .002)
    a.set_aspect(1 / np.cos(np.radians(tr.lat0))); a.legend(loc="best", fontsize=8)
    a.set_title(f"Student-test trip {tid}: GPS, reference and predicted routes"); a.set_xlabel("lon"); a.set_ylabel("lat")
    p = OUT / f"static_map_{tid}.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return p


def main():
    inv = pd.read_csv(R / "eda" / "inventory.csv")
    cand = pd.read_csv(R / "candidates_report.csv")
    mc = pd.read_csv(R / "model_comparison.csv")
    mc["config"] = mc["config"].fillna("no training: pick the nearest arc")
    hp = json.loads((R / "hpo" / "best_params.json").read_text())
    tvu = pd.read_csv(R / "hpo" / "tuned_vs_untuned.csv")
    trials = pd.read_csv(R / "hpo" / "trials.csv")
    grid = pd.read_csv(R / "hmm" / "decoder_grid.csv")
    hmmc = pd.read_csv(R / "hmm" / "student_test_clf_vs_hmm.csv")
    cfg = json.loads((ROOT / "models" / "hmm_config.json").read_text())
    allr = hmmc.iloc[-1]
    s = []
    A = s.append

    A(Paragraph("PathSnap: Map-Matching and Route Reconstruction", ss["Title"]))
    A(Paragraph("UE24CS352A Machine Learning Hackathon — Technical Report (DRAFT)", P))
    A(Paragraph("Team: ____________________ &nbsp; Members / SRNs: ____________________ &nbsp; Repository: ____________________", P))
    A(Paragraph("DRAFT: every table and figure is generated from the repository's result files. The narrative "
                "text is a starting scaffold and must be rewritten by the team in its own words, checked against "
                "the code, and accompanied by the AI-assistance disclosure in Section 8.", NOTE))

    # ---------------- 2. EDA
    A(Paragraph("2. Dataset understanding and exploratory data analysis", H1))
    part = inv.groupby("partition").agg(trips=("trip", "count"), gps_fixes=("n_gps", "sum"),
                                        median_fixes=("n_gps", "median"), median_km=("path_len_km", "median"),
                                        share_dt_1s=("frac_dt_1s", "mean"), gaps_over_5s=("n_gaps_gt5s", "sum"),
                                        nan_rows=("nan_gps", "sum"), median_arcs=("n_arcs", "median"),
                                        median_components=("weak_components", "median"),
                                        bidirectional=("frac_bidirectional_arcs", "mean"),
                                        route_continuity=("route_continuity", "mean")).reset_index()
    A(Paragraph("Partitions used: train (80 trips) for all fitting, cross-validation and decoder tuning; "
                "student_test (10 trips) for development comparison. Evaluator-test data was never available "
                "to or used by the pipeline during development.", P))
    A(tbl(part, pct=("share_dt_1s", "bidirectional", "route_continuity"), dec=1))
    A(Spacer(1, 6))
    A(Paragraph("<b>2.1 Preprocessing.</b> Files are read as headerless whitespace tables (track: lon, lat, t; "
                "nodes: lon, lat; arcs: from, to; route: arc id). Coordinates are projected per trip to metres with a "
                "local equirectangular projection around the trip centroid; trips lie on several continents "
                f"(longitude {inv.lon_min.min():.0f}° to {inv.lon_max.max():.0f}°), so degree distances are not usable. "
                "Reference routes are converted to one label per GPS fix by a monotonic dynamic-programming alignment "
                "(route index may stay or advance, total point-to-segment distance minimised). Fixes farther than 50 m "
                "from every route arc are marked off-route and excluded from training and scoring. Every 5th fix is "
                "used (1 Hz fixes are strongly autocorrelated).", P))
    A(Paragraph("<b>2.2 Missing values and anomalies.</b> "
                f"NaN coordinates: {int(inv.nan_gps.sum())}. Non-increasing timestamps: {int(inv.non_monotonic_t.sum())}. "
                f"Gaps over 5 s: {int(inv.n_gaps_gt5s.sum())}; trips with a gap over 60 s: {(inv.max_gap_s > 60).sum()} "
                f"(largest {inv.max_gap_s.max():,.0f} s). Steps implying more than 50 m/s: {int(inv.n_speed_gt50ms.sum())} "
                f"in {(inv.n_speed_gt50ms > 0).sum()} trips. Isolated road nodes: {int(inv.isolated_nodes.sum())}. "
                f"Mean share of off-route train fixes: {100 * cand[cand.partition == 'train'].offroute_frac.mean():.1f}%.", P))
    A(img(R / "eda" / "trip_distributions.png"))
    A(img(R / "eda" / "sampling_interval.png", 10))
    A(img(R / "eda" / "trajectories_overview.png", 11))
    A(Paragraph("Folium maps of selected trips (GPS, road network, reference route, layer control, legend) are in "
                "results/eda/map_*.html. Insert a screenshot of one here: [SCREENSHOT]", NOTE))
    A(Paragraph("<b>2.3 Candidates and features.</b> For each fix, up to 8 arcs within 60 m (KD-tree over points "
                "sampled every 5 m along arcs, then exact point-to-segment distance). 15 features per candidate: "
                "distance, distance rank, distance minus nearest, distance ratio, candidate count, direction-aware "
                "heading difference, undirected heading difference, moving flag, speed, time since previous fix, arc "
                "length, projection fraction, distance to arc end, start/end node degree. Heading is essential because "
                f"about {100 * inv.frac_bidirectional_arcs.mean():.0f}% of arcs have an opposite-direction twin at "
                "identical distance.", P))
    cp = cand.groupby("partition")[["offroute_frac", "cand_miss_rate", "nearest_top1", "mean_cands",
                                    "align_resid_median_m", "align_resid_p95_m"]].mean().reset_index()
    A(tbl(cp, pct=("offroute_frac", "cand_miss_rate", "nearest_top1"), dec=2))
    A(PageBreak())

    # ---------------- 3. Models
    A(Paragraph("3. Model experimentation and comparison", H1))
    A(Paragraph("Task framing: each (fix, candidate arc) row is a binary example (is this the reference arc?). "
                "Macro F1 and log loss are computed on candidate rows (threshold 0.5); Top-1 accuracy is "
                "observation-level (highest-probability candidate equals the reference arc; fixes whose reference arc "
                "is not among the candidates count as wrong). CV = 5-fold GroupKFold on the 80 train trips, "
                "grouped by trip. Route continuity = share of consecutive predictions joined by a drivable network "
                "path whose length agrees with the GPS displacement (|d_net − d_gps| ≤ max(30 m, 0.5·d_gps)). "
                "Primary selection metric: CV Top-1 on the train trips, because the per-fix ranking of candidates "
                "is what the decoder consumes; student-test confirms the choice but is not used to make it.", P))
    cols = ["model", "family", "config", "cv_macro_f1", "cv_top1", "test_macro_f1", "test_top1",
            "test_route_continuity", "test_log_loss", "infer_ms_per_obs"]
    A(tbl(mc[[c for c in cols if c in mc]], pct=("cv_top1", "test_top1", "test_route_continuity"),
          col_w=[2.4 * cm, 2 * cm, 3.4 * cm] + [1.45 * cm] * 7))
    A(Paragraph("Nearest-segment is a non-learned reference: CV metrics and log loss are N/A because it has no "
                "training stage and outputs no probabilities.", SMALL))
    A(img(R / "model_comparison.png"))

    why = {
        "LogReg": "Linear baseline with calibrated probabilities; tests whether a linear combination of distance and heading suffices.",
        "GaussianNB": "Generative probabilistic baseline; very fast; assumes conditionally independent Gaussian features, which is violated (distance, rank and ratio are strongly correlated).",
        "DecisionTree": "Captures thresholds such as 'heading difference > 90°' and interactions; interpretable but high variance.",
        "RandomForest": "Bagged trees reduce the variance of a single tree; robust to unscaled features.",
        "LightGBM": "Gradient boosting handles non-linear interactions, unscaled features and many rows efficiently; produces usable probabilities for the HMM emission.",
        "MLP(64,32)": "Neural network learns smooth non-linear decision surfaces; requires scaling and has an epoch-wise training history.",
        "KNN(k=25)": "Instance-based: a candidate is labelled like similar past candidates; sensitive to scaling and expensive at prediction time.",
    }
    OBJECTIVE = {
        "LogReg": "binary cross-entropy with L2 penalty (lbfgs).",
        "GaussianNB": "maximum-likelihood estimates of per-class feature means and variances (no iterative loss).",
        "DecisionTree": "greedy splits minimising Gini impurity.",
        "RandomForest": "each tree grown by Gini impurity on a bootstrap sample; probabilities averaged.",
        "LightGBM": "binary cross-entropy minimised by gradient boosting of trees.",
        "MLP(64,32)": "binary cross-entropy minimised with Adam; early stopping on a 10% validation split.",
        "KNN(k=25)": "none (lazy learner); prediction is the distance-weighted vote of 25 neighbours.",
    }
    for _, r in mc.dropna(subset=["cv_macro_f1"]).iterrows():
        A(Paragraph(f"3.2 {r.model} ({r.family})", H2))
        gap = r.cv_top1 - r.test_top1
        A(Paragraph(f"<b>A. Why suitable / strengths / limitations.</b> {why.get(r.model, '')} [Team: expand with your own observations.]", P))
        A(Paragraph(f"<b>B. Configuration.</b> {r.config}. Objective: {OBJECTIVE.get(r.model, '')} "
                    f"Training time on all 80 trips: {r.fit_s:.1f} s.", P))
        A(Paragraph(f"<b>C. Results.</b> Student-test Top-1 {100 * r.test_top1:.1f}%, macro F1 {r.test_macro_f1:.3f}, "
                    f"precision {r.test_precision:.3f}, recall {r.test_recall:.3f}, log loss {r.test_log_loss:.3f}, "
                    f"route continuity {100 * r.test_route_continuity:.1f}%. CV macro F1 {r.cv_macro_f1:.3f} ± {r.cv_macro_f1_std:.3f}. "
                    f"CV Top-1 minus student-test Top-1 = {100 * gap:+.1f} points; fold std {r.cv_macro_f1_std:.3f}. "
                    "[Team: state whether this indicates over- or underfitting and why.]", P))
        A(img(R / "confusion" / f"cm_{slug(r.model)}.png", 6))
        lc = R / f"loss_curve_{slug(r.model)}.png"
        if lc.exists():
            A(img(lc, 11))
    A(PageBreak())

    # ---------------- 4. HPO
    A(Paragraph("4. Bayesian hyperparameter optimisation (LightGBM)", H1))
    A(Paragraph(f"Sampler: {hp['sampler']} (Optuna). Objective: mean observation-level Top-1 over {hp['cv']} on the 80 "
                f"train trips. Trials: {hp['n_trials']} (trial 0 = the untuned configuration). Search time: "
                f"{hp['search_seconds'] / 60:.1f} min. Search space: n_estimators 100–1000 (step 50); learning_rate "
                "0.01–0.2 (log); num_leaves 15–255 (log); min_child_samples 10–300 (log); subsample 0.5–1; "
                "colsample_bytree 0.5–1; reg_lambda and reg_alpha 1e-3–10 (log). Student-test was not used during the "
                "search.", P))
    A(tbl(pd.DataFrame([hp["best_params"]]).T.reset_index().rename(columns={"index": "parameter", 0: "best value"}), dec=4))
    A(Spacer(1, 6))
    A(tbl(tvu.rename(columns={"test_continuity": "test_adjacent_arc_rate"}),
          pct=("cv_top1", "test_top1", "test_adjacent_arc_rate")))
    u, t = tvu.iloc[0], tvu.iloc[1]
    A(Paragraph(f"Tuning changed CV Top-1 by {100 * (t.cv_top1 - u.cv_top1):+.2f} points and student-test Top-1 by "
                f"{100 * (t.test_top1 - u.test_top1):+.2f} points; inference time per fix changed from "
                f"{u.infer_ms_per_obs:.4f} to {t.infer_ms_per_obs:.4f} ms. The tuned model was kept because it was "
                "selected on CV (choosing by student-test would leak). [Team: discuss whether the difference is "
                "within fold-to-fold noise (see CV std in Section 3).]", P))
    A(img(R / "hpo" / "optimisation.png"))
    A(Paragraph(f"Full trial record: results/hpo/trials.csv ({len(trials)} rows).", SMALL))
    A(PageBreak())

    # ---------------- 5. HMM
    A(Paragraph("5. HMM and Viterbi route reconstruction", H1))
    A(Paragraph("<b>Hidden states:</b> the candidate arcs of each matched fix, with their projection points. "
                f"<b>Emission:</b> α·log p(candidate is reference) from the tuned LightGBM (α = {cfg['alpha']}). "
                "<b>Transition:</b> −|d_net − d_gps| / β, where d_net is the directed shortest road distance between "
                f"the two projection points (Dijkstra, distance-limited) and d_gps the straight-line GPS displacement "
                f"(β = {cfg['beta']} m). <b>Missing candidates:</b> fixes with no arc within 60 m get no state and are "
                f"skipped; they count as wrong in scoring. <b>Disconnected segments:</b> unreachable pairs receive a "
                f"fixed log-probability of {cfg['unreachable_logp']} instead of −∞, so decoding continues across "
                f"network gaps. <b>Large GPS gaps:</b> the search radius grows with elapsed time (≤ {cfg['max_speed']} m/s); "
                f"gaps over {cfg['gap_break_s']:.0f} s split the chain. <b>Recurrence:</b> δ_k(j) = max_i [δ_(k−1)(i) + "
                "log P_t(i→j)] + log P_e(j), with back-pointers for the arg-max path; the decoded arcs are stitched "
                "into one connected route with the same shortest paths.", P))
    A(Paragraph(f"Decoder parameters (α, β) were chosen on out-of-fold train probabilities ({cfg['selected_on']}):", P))
    A(tbl(grid, pct=("oof_top1", "oof_continuity", "oof_clf_only_top1")))
    A(img(R / "hmm" / "decoder_grid.png", 11))
    A(Paragraph("Student-test, classifier-only vs classifier + HMM (same tuned model):", P))
    show = ["trip", "n_scored_obs", "top1_clf", "top1_hmm", "continuity_clf", "continuity_hmm",
            "route_precision", "route_recall", "route_f1", "unreachable_steps", "gap_breaks"]
    A(tbl(hmmc[show], pct=("top1_clf", "top1_hmm", "continuity_clf", "continuity_hmm", "route_precision",
                           "route_recall", "route_f1"), dec=1))
    better = (hmmc.top1_hmm[:-1] > hmmc.top1_clf[:-1]).sum(); worse = (hmmc.top1_hmm[:-1] < hmmc.top1_clf[:-1]).sum()
    A(Paragraph(f"Overall Top-1: {100 * allr.top1_clf:.1f}% → {100 * allr.top1_hmm:.1f}%; continuity "
                f"{100 * allr.continuity_clf:.1f}% → {100 * allr.continuity_hmm:.1f}%. HMM improved Top-1 on {better} trips "
                f"and reduced it on {worse}. [Team: inspect the trips where it did not help, using the maps, and explain.]", P))
    A(img(R / "hmm" / "clf_vs_hmm.png"))
    A(PageBreak())

    # ---------------- 6. Maps
    A(Paragraph("6. Geographical visualisation", H1))
    A(Paragraph("Interactive Folium maps for all 10 student-test trips: results/hmm/maps/*.html (layers: GPS, road "
                "network within 150 m, reference route, predicted route, classifier-only arcs; layer control and "
                "legend). Static export of one trip:", P))
    worst = hmmc.iloc[:-1].sort_values("top1_hmm").iloc[0].trip
    A(img(static_map(str(worst).zfill(8)), 15))
    A(Paragraph("[Insert a screenshot of the interactive Folium map here.]", NOTE))

    # ---------------- 7. API/UI
    A(Paragraph("7. API, UI and deployment", H1))
    A(Paragraph("Laptop 1 runs the FastAPI service (api/main.py: GET /health, GET /trips, POST /predict). It loads "
                "models/lgbm_tuned.joblib and models/hmm_config.json, runs candidate generation → LightGBM → Viterbi, "
                "then scores the result against its own reference routes, which are never sent to the UI. Every "
                "request is logged to the console and logs/api.log. Laptop 2 serves ui/index.html (Leaflet), which "
                "lists the trips returned by /trips and draws the returned GPS points, predicted route and metrics.", P))
    A(Paragraph("[Insert screenshots: UI with a matched trip; API console log on laptop 1.]", NOTE))

    # ---------------- 8. integrity
    A(Paragraph("8. Academic integrity and AI-assistance disclosure", H1))
    A(Paragraph("[Team: describe which parts were produced with AI assistance, as required by the event policy, and "
                "confirm that every member can explain the code, experiments and results.]", NOTE))

    SimpleDocTemplate(str(OUT / "PathSnap_Report_DRAFT.pdf"), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="PathSnap Technical Report (draft)").build(s)
    print("wrote", OUT / "PathSnap_Report_DRAFT.pdf")


if __name__ == "__main__":
    main()
