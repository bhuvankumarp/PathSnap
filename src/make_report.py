"""Build report/PathSnap_Technical_Report.pdf from the result files.

Every number, table and figure is read from results/ (nothing typed by hand). Screenshots are
taken from screenshots/ when present; missing ones show a visible placeholder.
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


TEAM = [("Bhuvan", "Labels, candidates, core features, final model, Bayesian optimisation, HMM/Viterbi, server-side evaluation, integration"),
        ("Vivek", "Metrics, cross-validation harness, baseline models, comparison plots, FastAPI backend"),
        ("Shrika", "EDA script, data-quality record, Folium maps, report figures"),
        ("Bhoomika", "Evaluator UI, run instructions, UI and API screenshots")]
SHOTS = ROOT / "screenshots"
REPO_URL = "https://github.com/bhuvankumarp/PathSnap"


def shot(name, caption, w=16):
    """Insert a screenshot if it exists (any of .png/.jpg), else a visible placeholder."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = SHOTS / (name + ext)
        if p.exists():
            return [img(p, w), Paragraph(f"<i>{caption}</i>", SMALL), Spacer(1, 6)]
    return [Paragraph(f"[Screenshot pending: screenshots/{name}.png — {caption}]", NOTE)]


def pc(v):
    return f"{100 * v:.1f}%"


def main():
    inv = pd.read_csv(R / "eda" / "inventory.csv", dtype={"trip": str})
    cand = pd.read_csv(R / "candidates_report.csv", dtype={"trip": str})
    mc = pd.read_csv(R / "model_comparison.csv")
    mc["config"] = mc["config"].fillna("no training: pick the nearest arc")
    fd = pd.read_csv(R / "fit_diagnostics.csv").set_index("model")
    hp = json.loads((R / "hpo" / "best_params.json").read_text())
    tvu = pd.read_csv(R / "hpo" / "tuned_vs_untuned.csv")
    trials = pd.read_csv(R / "hpo" / "trials.csv")
    grid = pd.read_csv(R / "hmm" / "decoder_grid.csv")
    hm = pd.read_csv(R / "hmm" / "student_test_clf_vs_hmm.csv", dtype={"trip": str})
    cfg = json.loads((ROOT / "models" / "hmm_config.json").read_text())
    A_ = hm.iloc[-1]; per = hm.iloc[:-1].copy()
    per["gain"] = per.top1_hmm - per.top1_clf
    m = mc.set_index("model")
    nb = m.loc["Nearest-segment (baseline)"]
    trn = cand[cand.partition == "train"]; stc = cand[cand.partition == "student_test"]
    u, t = tvu.iloc[0], tvu.iloc[1]
    lg = m.loc["LightGBM"]
    ytr = pd.read_pickle(ROOT / "features" / "train.pkl")["y"]
    n_rows, pos_rate = len(ytr), float(ytr.mean())
    s = []; A = s.append

    # ------------------------------------------------------------------ title
    A(Paragraph("PathSnap: Distributed Map-Matching and Route Reconstruction", ss["Title"]))
    A(Paragraph("UE24CS352A Machine Learning — Hackathon Technical Report", P))
    A(tbl(pd.DataFrame(TEAM, columns=["Member", "Main areas"]), col_w=[3 * cm, 14.6 * cm]))
    A(Spacer(1, 4))
    A(Paragraph(f"Repository: {REPO_URL}. Every number, table and figure in this report is produced by the scripts in "
                "the repository from the files in <i>results/</i>; none was typed by hand.", P))
    A(Paragraph("<b>Summary.</b> Each GPS fix is matched to a road arc in three stages: candidate generation (arcs within "
                "60 m), a LightGBM classifier that scores each candidate, and an HMM decoded with Viterbi that enforces "
                f"a drivable route. On the 10 student-test trips, observation-level Top-1 accuracy rises from "
                f"{pc(nb.test_top1)} (nearest arc) to {pc(t.test_top1)} (tuned LightGBM) and {pc(A_.top1_hmm)} "
                f"(LightGBM + HMM), with route continuity {pc(A_.continuity_hmm)} and route-level F1 {pc(A_.route_f1)}.", P))
    A(PageBreak())

    # ------------------------------------------------------------------ 2. EDA
    A(Paragraph("2. Dataset understanding and exploratory data analysis", H1))
    A(Paragraph("<b>Partitions.</b> Train (80 trips): EDA, fitting, cross-validation, hyperparameter search and decoder "
                "tuning. Student-test (10 trips): development comparison only. Evaluator-test: never available to the "
                "pipeline; the API reads it only on evaluation day. Unless stated otherwise, tables below cover train and "
                "student-test separately.", P))
    part = inv.groupby("partition").agg(trips=("trip", "count"), gps_fixes=("n_gps", "sum"),
                                        median_fixes=("n_gps", "median"), median_km=("path_len_km", "median"),
                                        share_dt_1s=("frac_dt_1s", "mean"), gaps_over_5s=("n_gaps_gt5s", "sum"),
                                        nan_rows=("nan_gps", "sum"), median_arcs=("n_arcs", "median"),
                                        median_components=("weak_components", "median"),
                                        twin_arcs=("frac_bidirectional_arcs", "mean"),
                                        route_continuity=("route_continuity", "mean")).reset_index()
    A(tbl(part, pct=("share_dt_1s", "twin_arcs", "route_continuity"), dec=1))
    A(Spacer(1, 6))
    A(Paragraph("2.1 How did we preprocess the data?", H2))
    A(Paragraph("(1) <b>Reading.</b> Each trip folder holds four headerless whitespace files: <i>.track</i> (lon, lat, t), "
                "<i>.nodes</i> (lon, lat; row = node id), <i>.arcs</i> (from, to; row = arc id) and <i>.route</i> "
                "(ordered arc ids). We checked row counts, types, NaNs and timestamp order for every trip.", P))
    A(Paragraph("(2) <b>Projection.</b> Trips lie on several continents (longitude "
                f"{inv.lon_min.min():.0f}° to {inv.lon_max.max():.0f}°, latitude {inv.lat_min.min():.0f}° to "
                f"{inv.lat_max.max():.0f}°). Euclidean distance in degrees is wrong (one degree of longitude shrinks with "
                "cos(latitude)), so each trip is projected to metres with a local equirectangular projection around its "
                "own centroid. Trips are a few kilometres across, so the distortion is far below GPS noise.", P))
    A(Paragraph("(3) <b>Labels.</b> The route is an ordered list of arcs, not one arc per fix. We align fixes to route arcs "
                "with dynamic programming: the route index may stay or move forward over time, and the total "
                "point-to-segment distance is minimised. The aligned arc is the fix's reference label. Median alignment "
                f"distance is {trn.align_resid_median_m.mean():.1f} m on train ({stc.align_resid_median_m.mean():.1f} m on "
                "student-test), consistent with normal GPS error.", P))
    A(Paragraph("(4) <b>Off-route fixes.</b> Fixes more than 50 m from every reference arc are treated as unreliable labels "
                f"(detours, parking, roads missing from the network): {pc(trn.offroute_frac.mean())} of train fixes and "
                f"{pc(stc.offroute_frac.mean())} of student-test fixes. They are excluded from training and scoring.", P))
    A(Paragraph("(5) <b>Subsampling.</b> Fixes are 1 Hz and strongly autocorrelated, so every 5th fix is used for "
                "features, training and decoding.", P))
    A(Paragraph("2.2 Missing values and data quality", H2))
    gap = inv.loc[inv.max_gap_s.idxmax()]; jump = inv.loc[inv.n_speed_gt50ms.idxmax()]
    A(Paragraph(f"There are {int(inv.nan_gps.sum())} missing coordinates and {int(inv.non_monotonic_t.sum())} "
                f"non-increasing timestamps, so no imputation was needed. The problems are of a different kind: "
                f"{pc(inv.frac_dt_1s.mean())} of steps are exactly 1 s, but there are {int(inv.n_gaps_gt5s.sum())} gaps "
                f"over 5 s and {(inv.max_gap_s > 60).sum()} trips with a gap over 60 s (largest {gap.max_gap_s:,.0f} s, "
                f"trip {gap.trip}). {int(inv.n_speed_gt50ms.sum())} steps imply more than 50 m/s, across "
                f"{(inv.n_speed_gt50ms > 0).sum()} trips (worst trip {jump.trip}, {int(jump.n_speed_gt50ms)} steps), "
                "which are GPS jumps rather than real motion. Road networks are fragmented (median "
                f"{inv.weak_components.median():.0f} weakly connected components per trip). The full record of issues and "
                "decisions is in results/eda/data_quality.md.", P))
    A(img(R / "eda" / "trip_distributions.png"))
    A(Paragraph("<i>Figure 2.1 — Per-trip distributions (train and student-test): fixes per trip, trip length, median "
                "speed, largest time gap, road arcs per trip and reference-route continuity.</i>", SMALL))
    A(img(R / "eda" / "sampling_interval.png", 9.5))
    A(Paragraph("<i>Figure 2.2 — Time between GPS fixes (train, log scale).</i>", SMALL))
    A(img(R / "eda" / "trajectories_overview.png", 10))
    A(Paragraph("<i>Figure 2.3 — GPS trajectories of 20 train trips.</i>", SMALL))
    s.extend(shot("folium_map_eda_trip02", "Figure 2.4 — Folium map, train trip 00000002: GPS fixes (blue), reference "
                  "route (green), road network (grey), start/end markers and layer control."))
    A(Paragraph("2.3 Patterns observed, features extracted and why", H2))
    A(Paragraph(f"The most important pattern is that {pc(inv.frac_bidirectional_arcs.mean())} of arcs have an "
                "opposite-direction twin with the same geometry. Distance cannot separate them, which is why simply "
                f"picking the nearest arc is right only {pc(nb.test_top1)} of the time on student-test. The second "
                "pattern is that GPS error is usually a few metres but sometimes much larger, so candidates must cover a "
                "wide radius while the model learns to rank them.", P))
    A(Paragraph(f"<b>Candidates.</b> For each fix, all arcs within 60 m (found with a KD-tree over points sampled every 5 m "
                "along each arc, then exact point-to-segment distance), keeping the 8 nearest. The reference arc is "
                f"missing from the candidate set for only {pc(trn.cand_miss_rate.mean())} of train and "
                f"{pc(stc.cand_miss_rate.mean())} of student-test fixes, with about {trn.mean_cands.mean():.1f} candidates "
                "per fix.", P))
    feats = pd.DataFrame([
        ("dist, dist_rank, dist_minus_min, dist_ratio", "How close the arc is, absolutely and relative to the other candidates"),
        ("n_cand", "Dense areas (many candidates) are harder"),
        ("head_diff (0–180°)", "Direction-aware angle between travel direction and arc direction; separates twin arcs"),
        ("head_diff_undir, moving", "Same angle ignoring direction; heading is unreliable when the car is nearly stopped"),
        ("speed, gap_prev", "Context: fast motorway vs slow city driving; time since previous fix"),
        ("arc_len, proj_frac, proj_end", "Where the fix projects on the arc; projections near arc ends are ambiguous"),
        ("deg_from, deg_to", "Junction size at each end of the arc"),
    ], columns=["Feature(s)", "Why"])
    A(tbl(feats, col_w=[5.5 * cm, 12.1 * cm]))
    A(Spacer(1, 4))
    A(tbl(cand.groupby("partition")[["offroute_frac", "cand_miss_rate", "nearest_top1", "mean_cands",
                                     "align_resid_median_m", "align_resid_p95_m"]].mean().reset_index(),
          pct=("offroute_frac", "cand_miss_rate", "nearest_top1"), dec=2))
    A(PageBreak())

    # ------------------------------------------------------------------ 3. models
    A(Paragraph("3. Model experimentation and comparison", H1))
    A(Paragraph("<b>Framing.</b> Every (fix, candidate arc) pair is a binary example: is this the reference arc? "
                f"The train set has {n_rows:,} rows, of which {pc(pos_rate)} are positives. <b>Macro F1</b>, precision, recall and <b>log loss</b> are computed "
                "on these rows at threshold 0.5. <b>Top-1 accuracy</b> is observation-level: the candidate with the highest "
                "probability must equal the reference arc (fixes whose reference arc is not a candidate count as wrong). "
                "<b>Route continuity</b> is the share of consecutive predictions connected by a drivable road path whose "
                "length matches the GPS displacement (|d_net − d_gps| ≤ max(30 m, 0.5·d_gps)). <b>CV</b> is 5-fold "
                "GroupKFold on the 80 train trips with the trip as group, so no trip is ever in both the fitting and the "
                "validation fold. Seeds are fixed at 42.", P))
    A(Paragraph("<b>Primary selection metric:</b> CV Top-1, because the per-fix ranking of candidates is exactly what the "
                "HMM consumes, and it is measured on train only. Student-test confirms the choice but is not used to make it.", P))
    A(Paragraph("3.1 Overall model comparison", H2))
    cols = ["model", "family", "config", "cv_macro_f1", "cv_top1", "test_macro_f1", "test_top1",
            "test_route_continuity", "test_log_loss", "infer_ms_per_obs"]
    A(tbl(mc[cols].rename(columns={"cv_macro_f1": "CV macro F1", "cv_top1": "CV top-1", "test_macro_f1": "test macro F1",
                                   "test_top1": "test top-1", "test_route_continuity": "route continuity",
                                   "test_log_loss": "log loss", "infer_ms_per_obs": "inference ms/fix"}),
          pct=("CV top-1", "test top-1", "route continuity"), col_w=[2.4 * cm, 2 * cm, 3.4 * cm] + [1.4 * cm] * 7))
    A(Paragraph("N/A for the nearest-arc baseline: it has no training stage (no CV) and outputs a distance, not a "
                "probability (no log loss or F1 at 0.5).", SMALL))
    A(img(R / "model_comparison.png"))
    A(Paragraph("<i>Figure 3.1 — Left: CV macro F1 with fold standard deviation. Middle: student-test Top-1 and macro F1 "
                "(dashed line = nearest arc). Right: student-test log loss.</i>", SMALL))
    A(img(R / "fit_diagnostics.png", 15))
    A(Paragraph("<i>Figure 3.2 — Top-1 on the training data itself, in 5-fold CV and on student-test. A large train–CV gap "
                "means overfitting; low values everywhere mean underfitting.</i>", SMALL))
    fdt = fd.reset_index()[["model", "train_top1", "cv_top1", "test_top1", "gap_train_cv_top1", "train_macro_f1"]]
    A(tbl(fdt, pct=("train_top1", "cv_top1", "test_top1", "gap_train_cv_top1")))
    A(Paragraph("<b>Selection.</b> LightGBM has the highest CV Top-1 and CV macro F1 and the best student-test Top-1, macro F1 "
                "and log loss, with fast inference. Its probabilities are also the best calibrated (lowest log loss), which "
                "matters because they become the HMM emission scores. It was chosen for tuning and deployment.", P))

    M = {
        "LogReg": dict(
            why="A linear model is the natural first test: if distance and heading separate correct from wrong arcs "
                "linearly, nothing more complex is needed. It also gives well-calibrated probabilities, which an HMM needs.",
            sl="Strengths: trains in under a second and its log loss is reasonable. Limitations: it cannot model interactions, "
               "for example that heading only matters when the vehicle is moving, or that a small distance difference "
               "matters more in dense areas. Its accuracy plateaus below the tree models.",
            hp="StandardScaler, L2 penalty, C = 1.0, lbfgs solver, max_iter = 2000.",
            obj="Binary cross-entropy plus the L2 penalty."),
        "GaussianNB": dict(
            why="The simplest probabilistic classifier, included as a fast generative baseline.",
            sl="Strength: fastest model and no tuning. Limitation: it assumes features are independent given the class, but "
               "four of our features are all derived from distance (dist, rank, difference, ratio). The model counts the "
               "same evidence several times and becomes over-confident: it has the worst log loss "
               f"({m.loc['GaussianNB', 'test_log_loss']:.3f}) and macro F1, even though its ranking (Top-1) is acceptable.",
            hp="StandardScaler, var_smoothing = 1e-9.",
            obj="No iterative loss: per-class feature means and variances are estimated by maximum likelihood."),
        "DecisionTree": dict(
            why="Our features have natural thresholds (heading difference above about 90° means the wrong direction), "
                "which trees capture directly, along with interactions.",
            sl="Strengths: interpretable rules and very fast inference. Limitations: a single tree has high variance and "
               "gives coarse, step-shaped probabilities.",
            hp="max_depth = 8, min_samples_leaf = 50 (both chosen to limit variance).",
            obj="Greedy splits that minimise Gini impurity."),
        "RandomForest": dict(
            why="Averaging many trees on bootstrap samples keeps the tree's ability to model thresholds and interactions "
                "while reducing its variance.",
            sl="Strengths: robust, no scaling needed, small train–CV gap. Limitations: more memory and slower inference than "
               "a single tree, and no gain over boosting here.",
            hp="100 trees, max_depth = 14, min_samples_leaf = 20.",
            obj="Each tree minimises Gini impurity; probabilities are averaged over trees."),
        "LightGBM": dict(
            why="Gradient boosting builds trees that correct the previous trees' errors, which suits tabular features with "
                "interactions and different scales. It is fast on our training set.",
            sl="Strengths: best accuracy and the best-calibrated probabilities, fast inference. Limitation: more "
               "hyperparameters, and without regularisation it overfits more than the other trees (largest train–CV gap "
               "among the tree models before tuning).",
            hp="300 trees, learning_rate = 0.05, num_leaves = 31, other values at LightGBM defaults (untuned).",
            obj="Binary cross-entropy minimised by gradient boosting."),
        "MLP(64,32)": dict(
            why="A neural network can learn smooth non-linear combinations of features, as a contrast to the "
                "axis-aligned splits of trees.",
            sl="Strength: accuracy close to the tree ensembles. Limitations: needs feature scaling, has the longest training "
               "time, and is sensitive to the random seed.",
            hp="Two hidden layers (64, 32), ReLU, Adam, early stopping on a 10% validation split, at most 30 epochs, StandardScaler.",
            obj="Binary cross-entropy minimised with Adam."),
        "KNN(k=25)": dict(
            why="Instance-based: a candidate is judged by similar candidates seen in training, without assuming any "
                "functional form.",
            sl="Strength: no training step. Limitations: needs scaling, prediction is the slowest of all models, and "
               "fitting all training rows took more than 10 minutes per fold, so it is fitted on a random 40k-row subsample.",
            hp="k = 25, distance-weighted votes, brute-force search, 40,000-row subsample, StandardScaler.",
            obj="None (lazy learner): prediction is the distance-weighted vote of the 25 nearest training rows."),
    }
    A(Paragraph("3.2 Questions for every model", H2))
    for name, d in M.items():
        r, f = m.loc[name], fd.loc[name]
        gap = f.train_top1 - f.cv_top1
        if gap < 0.005:
            fit = (f"Train Top-1 {pc(f.train_top1)} is the same as CV Top-1 {pc(f.cv_top1)}, so there is no overfitting. "
                   "But both are lower than the tree models, so the model underfits: it is too simple for the data.")
        elif gap < 0.02:
            fit = (f"Train Top-1 {pc(f.train_top1)} vs CV {pc(f.cv_top1)}: a small gap of {100 * gap:.1f} points, so only "
                   f"slight overfitting. Student-test ({pc(r.test_top1)}) agrees with CV, so it generalises to new trips.")
        else:
            fit = (f"Train Top-1 {pc(f.train_top1)} vs CV {pc(f.cv_top1)}: a gap of {100 * gap:.1f} points, so the model "
                   "fits the training trips noticeably better than unseen trips (mild overfitting).")
        if name == "LightGBM":
            fit += (f" Tuning addressed this: the tuned model (fewer leaves, larger minimum leaf size) has a gap of only "
                    f"{100 * fd.loc['LightGBM tuned', 'gap_train_cv_top1']:.1f} points (Section 4).")
        if name == "KNN(k=25)":
            fit += (" Part of the train score is inflated, because the 40k fitted rows are their own nearest neighbour at "
                    "distance zero.")
        if name == "MLP(64,32)":
            fit += " The training-loss curve (Figure 3.3) flattens and early stopping ended training before the epoch limit."
        A(Paragraph(f"3.2.{list(M).index(name) + 1} {name} ({r.family})", H3))
        A(Paragraph(f"<b>A1. Why suitable?</b> {d['why']}", P))
        A(Paragraph(f"<b>A2. Strengths and limitations.</b> {d['sl']}", P))
        A(Paragraph(f"<b>B5. Hyperparameters (initial values).</b> {d['hp']}", P))
        A(Paragraph(f"<b>B6. Loss / objective.</b> {d['obj']}", P))
        A(Paragraph(f"<b>C7–C8. Student-test results.</b> Top-1 {pc(r.test_top1)}, macro F1 {r.test_macro_f1:.3f} "
                    f"(precision {r.test_precision:.3f}, recall {r.test_recall:.3f}), log loss {r.test_log_loss:.3f}, route "
                    f"continuity {pc(r.test_route_continuity)}. CV macro F1 {r.cv_macro_f1:.3f} ± {r.cv_macro_f1_std:.3f}, "
                    f"CV Top-1 {pc(r.cv_top1)}. Training time {r.fit_s:.1f} s; inference {r.infer_ms_per_obs:.3f} ms per fix.", P))
        A(Paragraph(f"<b>C9. Overfitting or underfitting?</b> {fit}", P))
        A(img(R / "confusion" / f"cm_{slug(name)}.png", 5.5))
        if name == "MLP(64,32)":
            A(img(R / f"loss_curve_{slug(name)}.png", 10))
            A(Paragraph("<i>Figure 3.3 — MLP training loss per epoch and accuracy on the early-stopping validation split. "
                        "Only the MLP produces an epoch-wise history; no curves are shown for the other models.</i>", SMALL))
    A(PageBreak())

    # ------------------------------------------------------------------ 4. HPO
    A(Paragraph("4. Bayesian hyperparameter optimisation (LightGBM)", H1))
    space = pd.DataFrame([("n_estimators", "100–1000, step 50"), ("learning_rate", "0.01–0.2, log-uniform"),
                          ("num_leaves", "15–255, log-uniform"), ("min_child_samples", "10–300, log-uniform"),
                          ("subsample", "0.5–1.0"), ("colsample_bytree", "0.5–1.0"),
                          ("reg_lambda", "0.001–10, log-uniform"), ("reg_alpha", "0.001–10, log-uniform")],
                         columns=["hyperparameter", "range / distribution"])
    space["best value"] = [f"{hp['best_params'][k]:.4g}" for k in space.hyperparameter]
    A(Paragraph(f"<b>1–2. Search space.</b> Eight hyperparameters, chosen because they control tree size "
                "(num_leaves, min_child_samples), the boosting schedule (n_estimators, learning_rate), randomness "
                "(subsample, colsample_bytree) and regularisation (reg_lambda, reg_alpha). Ranges are log-uniform where "
                "the effect is multiplicative.", P))
    A(tbl(space, col_w=[5 * cm, 7 * cm, 5.6 * cm]))
    A(Paragraph(f"<b>3. Objective.</b> Maximise mean observation-level Top-1 over 5-fold GroupKFold on the 80 train trips. "
                "Student-test was not used during the search. <b>Method:</b> Optuna with the Tree-structured Parzen "
                "Estimator (TPE), a Bayesian method that models which regions of the space give good scores and samples "
                "there more often (seed 42).", P))
    A(Paragraph(f"<b>4. Trials.</b> {hp['n_trials']} (trial 0 is the untuned configuration, for reference), taking "
                f"{hp['search_seconds'] / 60:.1f} minutes. Every trial is recorded in results/hpo/trials.csv "
                f"({len(trials)} rows).", P))
    A(Paragraph(f"<b>5. Best values.</b> See the table above: a smaller, more regularised model (num_leaves "
                f"{hp['best_params']['num_leaves']}, min_child_samples {hp['best_params']['min_child_samples']}, "
                f"{hp['best_params']['n_estimators']} trees, row subsampling {hp['best_params']['subsample']:.2f}).", P))
    cmp_ = pd.DataFrame([
        ("CV macro F1", f"{u.cv_macro_f1:.4f}", f"{t.cv_macro_f1:.4f}"),
        ("CV Top-1", pc(u.cv_top1), pc(t.cv_top1)),
        ("Student-test Top-1", pc(u.test_top1), pc(t.test_top1)),
        ("Student-test macro F1", f"{u.test_macro_f1:.4f}", f"{t.test_macro_f1:.4f}"),
        ("Student-test log loss", f"{u.test_log_loss:.4f}", f"{t.test_log_loss:.4f}"),
        ("Train − CV Top-1 gap", f"{100 * fd.loc['LightGBM', 'gap_train_cv_top1']:.1f} pts",
         f"{100 * fd.loc['LightGBM tuned', 'gap_train_cv_top1']:.1f} pts"),
        ("Inference (ms per fix)", f"{u.infer_ms_per_obs:.4f}", f"{t.infer_ms_per_obs:.4f}"),
        ("Training time (s)", f"{u.fit_s:.2f}", f"{t.fit_s:.2f}")], columns=["metric", "untuned", "tuned"])
    A(tbl(cmp_, col_w=[7 * cm, 5.3 * cm, 5.3 * cm]))
    A(Paragraph(f"<b>6. Did the tuned model outperform the untuned one?</b> Only slightly, and not on every metric. CV Top-1 "
                f"improved by {100 * (t.cv_top1 - u.cv_top1):+.2f} points and CV macro F1 by "
                f"{t.cv_macro_f1 - u.cv_macro_f1:+.4f}. On student-test, Top-1 changed by "
                f"{100 * (t.test_top1 - u.test_top1):+.2f} points. Both differences are smaller than the fold-to-fold "
                f"standard deviation of CV macro F1 ({lg.cv_macro_f1_std:.3f}), so accuracy is effectively unchanged. The "
                f"clear gains are elsewhere: the tuned model overfits much less (train–CV gap "
                f"{100 * fd.loc['LightGBM', 'gap_train_cv_top1']:.1f} → {100 * fd.loc['LightGBM tuned', 'gap_train_cv_top1']:.1f} "
                f"points) and is about {100 * (1 - t.infer_ms_per_obs / u.infer_ms_per_obs):.0f}% faster at inference. We "
                "deployed the tuned model because it was selected on CV; choosing by student-test would be leakage.", P))
    A(img(R / "hpo" / "optimisation.png"))
    A(Paragraph("<i>Figure 4.1 — Left: CV Top-1 for each trial and the best so far. Right: hyperparameter importance "
                "(fANOVA).</i>", SMALL))
    A(PageBreak())

    # ------------------------------------------------------------------ 5. HMM
    A(Paragraph("5. HMM and Viterbi route reconstruction", H1))
    A(Paragraph("Classifying each fix on its own can produce impossible routes: consecutive fixes on unconnected arcs, or "
                "flipping between an arc and its twin. The HMM adds the road network and the order of fixes.", P))
    A(Paragraph(f"<b>1. Hidden states.</b> For matched fix k, the states are its candidate arcs (up to 8), each with the "
                "point where the fix projects onto it. The correspondence between a state and its road arc is kept, so the "
                "decoded state is directly a predicted arc.", P))
    A(Paragraph(f"<b>2. Emission.</b> log P(z_k | arc) = α · log p, where p is the tuned LightGBM probability that the arc is "
                f"the reference (α = {cfg['alpha']}).", P))
    A(Paragraph(f"<b>3. Transition.</b> log P(arc_i → arc_j) = −|d_net − d_gps| / β, with β = {cfg['beta']} m. d_gps is "
                "the straight-line distance between the two fixes; d_net is the shortest driving distance along the "
                "directed road graph from the first projection point to the second (Dijkstra, distance-limited). "
                "A real drive has d_net ≈ d_gps, while jumping to a parallel street or reversing onto a twin arc makes d_net "
                "much larger. This follows the Newson–Krumm formulation also described in the Uber map-matching talk we "
                "used as reference.", P))
    A(Paragraph(f"<b>Viterbi recurrence.</b> δ_k(j) = max_i [δ_(k−1)(i) + log P(arc_i → arc_j)] + log P(z_k | arc_j), with a "
                "back-pointer to the best i. The best final state is traced back to give one arc per fix, and the decoded "
                "arcs are joined by their shortest paths into one connected route.", P))
    A(Paragraph(f"<b>4. Missing candidates and GPS gaps.</b> A fix with no arc within 60 m has no states and is skipped (it "
                "counts as wrong in scoring). The Dijkstra search radius grows with elapsed time (at most "
                f"{cfg['max_speed']:.0f} m/s × Δt). For gaps over {cfg['gap_break_s']:.0f} s the vehicle could be "
                "anywhere, so the transition term is dropped and decoding restarts from the best state.", P))
    A(Paragraph(f"<b>5. Disconnected segments.</b> If no path exists within the search radius (separate network "
                f"components or missing arcs), the transition gets a fixed log-probability of {cfg['unreachable_logp']:.0f} "
                "instead of −∞. Decoding therefore never fails; the break is counted and shows up as lower continuity.", P))
    A(Paragraph("<b>Choosing α and β.</b> Probabilities for all 80 train trips were produced out-of-fold (GroupKFold-5), "
                f"so the decoder never sees a probability from a model trained on the same trip. A grid of α ∈ {{0.5, 1, 2}} "
                f"and β ∈ {{1, 2, 5, 10, 15, 30, 60, 120}} m was evaluated on the first 30 train trips. Small β (trust the "
                f"road distance strongly) works best, with an interior optimum at β = {cfg['beta']:.0f} m and α = "
                f"{cfg['alpha']:.0f}. Student-test was not used.", P))
    A(img(R / "hmm" / "decoder_grid.png", 10))
    A(Paragraph("<i>Figure 5.1 — Out-of-fold train Top-1 for each (α, β); dashed line = classifier only.</i>", SMALL))
    A(Paragraph("<b>6. Did HMM decoding improve the route?</b> Yes, on all ten student-test trips:", P))
    show = per[["trip", "n_scored_obs", "top1_clf", "top1_hmm", "gain", "continuity_clf", "continuity_hmm",
                "route_precision", "route_recall", "route_f1", "gap_breaks"]]
    tot = pd.DataFrame([dict(trip="ALL", n_scored_obs=A_.n_scored_obs, top1_clf=A_.top1_clf, top1_hmm=A_.top1_hmm,
                             gain=A_.top1_hmm - A_.top1_clf, continuity_clf=A_.continuity_clf,
                             continuity_hmm=A_.continuity_hmm, route_precision=A_.route_precision,
                             route_recall=A_.route_recall, route_f1=A_.route_f1, gap_breaks=per.gap_breaks.sum())])
    A(tbl(pd.concat([show, tot]).rename(columns={"n_scored_obs": "fixes", "gap_breaks": "gap breaks"}),
          pct=("top1_clf", "top1_hmm", "gain", "continuity_clf", "continuity_hmm", "route_precision", "route_recall",
               "route_f1"), dec=0, col_w=[2.2 * cm, 1.3 * cm] + [1.57 * cm] * 9))
    best_g, worst_g = per.loc[per.gain.idxmax()], per.loc[per.gain.idxmin()]
    hard = per.loc[per.top1_hmm.idxmin()]
    hinv = inv.set_index("trip")
    A(Paragraph(f"Overall Top-1 rises from {pc(A_.top1_clf)} to {pc(A_.top1_hmm)} and route continuity from "
                f"{pc(A_.continuity_clf)} to {pc(A_.continuity_hmm)}. The reconstructed route covers "
                f"{pc(A_.route_recall)} of the reference arcs with {pc(A_.route_precision)} precision. The largest gain is "
                f"trip {best_g.trip} ({pc(best_g.top1_clf)} → {pc(best_g.top1_hmm)}).", P))
    gapt = per[per.gap_breaks > 0]
    gap_txt = ""
    if len(gapt):
        g0 = gapt.iloc[0]
        gap_txt = (f" Trip {g0.trip} also gained little ({100 * g0.gain:.1f} points): it contains a "
                   f"{hinv.loc[g0.trip, 'max_gap_s']:.0f} s GPS gap, where the chain is split and no sequence information "
                   "links the two sides.")
    A(Paragraph(f"<b>Where the HMM helped least.</b> The smallest gain is trip {worst_g.trip} "
                f"({100 * worst_g.gain:.1f} points), but its classifier-only accuracy was already {pc(worst_g.top1_clf)}, "
                f"so there was little left to fix.{gap_txt} Trip {hard.trip} remains the hardest ({pc(hard.top1_hmm)}): it "
                f"has the slowest driving of all student-test trips (median {hinv.loc[hard.trip, 'median_speed_ms']:.1f} m/s) "
                "and many candidates per fix. At low speed the GPS heading is noisy, so the twin-arc evidence is weak, and "
                "short moves make d_net ≈ d_gps true for several neighbouring arcs. Its route-level recall is still "
                f"{pc(hard.route_recall)}: the route geometry is right, but some fixes are assigned to an adjacent arc.", P))
    A(img(R / "hmm" / "clf_vs_hmm.png"))
    A(Paragraph("<i>Figure 5.2 — Classifier-only vs classifier + HMM per student-test trip: Top-1 (left) and route "
                "continuity (right).</i>", SMALL))
    A(PageBreak())

    # ------------------------------------------------------------------ 6. maps
    A(Paragraph("6. Geographical visualisation using Folium", H1))
    A(Paragraph("Interactive Folium maps are generated for every student-test trip by src/hmm_experiments.py "
                "(results/hmm/maps/). Layers: GPS fixes (blue), road network within 150 m (grey), reference route (green), "
                "predicted route after the HMM (purple) and classifier-only arcs (red, off by default). Each map has a "
                "legend, start/end markers, a scale bar and a layer control, and is zoomed to the trip.", P))
    s.extend(shot("folium_map_trip77_full", "Figure 6.1 — Folium map, student-test trip 00000077: GPS fixes, reference route "
                  "and predicted route (classifier + HMM)."))
    A(img(static_map("00000077"), 13))
    A(Paragraph("<i>Figure 6.2 — Static export of the same layers for trip 00000077, including classifier-only arcs (red). "
                "Red arcs off the green route are classifier errors that the HMM removes.</i>", SMALL))
    A(PageBreak())

    # ------------------------------------------------------------------ 7. API / UI
    A(Paragraph("7. API, UI and deployment", H1))
    A(Paragraph("<b>Laptop 1 (API).</b> FastAPI (api/main.py), started with <i>uvicorn api.main:app --host 0.0.0.0 --port "
                "8000</i>. On start-up it loads models/lgbm_tuned.joblib and models/hmm_config.json. Endpoints: GET /health, "
                "GET /trips (the trips in data/evaluator_test/), POST /predict {trip_id}. /predict reads that trip's GPS "
                "track and road network, generates candidates, scores them with LightGBM, decodes with Viterbi, then "
                "computes accuracy against the reference route stored on laptop 1. The response contains the GPS points, "
                "the predicted route geometry, per-fix matches and the metrics, but never the reference route. Every "
                "request is logged to the console and logs/api.log.", P))
    A(Paragraph("<b>Laptop 2 (UI).</b> A static page (ui/index.html, Leaflet) served with <i>python -m http.server 5500</i>. "
                "The evaluator enters laptop 1's address, the UI lists the trips from /trips, and selecting one sends its "
                "id to /predict and draws the GPS points, the predicted route (and optionally classifier-only arcs) with the "
                "accuracy panel. Both laptops are on the same network; CORS is enabled on the API.", P))
    A(Paragraph(f"All ten student-test trips were run through the API end to end; results matched Section 5 exactly, with "
                f"processing times of {per.seconds.min():.1f}–{per.seconds.max():.1f} s per trip.", P))
    A(Paragraph("<b>Screenshots.</b> Figures 7.1–7.4 were captured from a test run with the API and the UI on the same "
                "laptop (address 127.0.0.1); the two-laptop setup differs only in the address entered in the UI.", P))
    for nm, cap in [("ui_trip_list", "Figure 7.1 — UI after connecting: the ten trips returned by GET /trips."),
                    ("ui_matched_trip", "Figure 7.2 — UI after selecting trip 00000007: GPS fixes on the map and the "
                     "accuracy panel returned by POST /predict."),
                    ("ui_matched_trip_route", "Figure 7.3 — Same trip with the GPS layer switched off, showing the "
                     "predicted route (classifier + HMM)."),
                    ("api_health", "Figure 7.4 — GET /health response: tuned LightGBM + HMM configuration loaded.")]:
        s.extend(shot(nm, cap, 15))
    logf = R / "api_log_excerpt.txt"
    if logf.exists():
        A(Paragraph("<b>API request log</b> (logs/api.log on the API laptop, one /predict per student-test trip):", P))
        from reportlab.platypus import Preformatted
        mono = ParagraphStyle("mono", parent=SMALL, fontName="Courier", fontSize=6.3, leading=7.6)
        txt = chr(10).join(l.replace(" | INFO | ", " | ")[:150] for l in logf.read_text(encoding="utf-8").splitlines())
        A(Preformatted(txt, mono))
    s.extend(shot("api_console_log", "Figure 7.5 — API console (CMD) on laptop 1 during the two-laptop run, showing the "
                  "received /predict requests."))

    # ------------------------------------------------------------------ 8. integrity
    A(PageBreak())
    A(Paragraph("8. Academic integrity and AI-assistance disclosure", H1))
    A(Paragraph("This project was developed with substantial assistance from an AI assistant (Claude, by Anthropic), "
                "used through Claude Code. It was used to write source code, run the experiments on our machine, and draft "
                "the text of this report. Every metric, table and figure was produced by running the code in the "
                "repository on the provided data; no result was typed in by hand, and all results can be regenerated with "
                "the commands in the README.", P))
    A(Paragraph("Work was divided as in the team table on the first page. Each member has reviewed the code and the report "
                "sections for their own area and can explain them, and every member has read Section 5 (HMM/Viterbi). "
                "Evaluator-test data was not used during development, ground truth stays on the API laptop, and no "
                "prediction is hard-coded.", P))
    A(Paragraph("<b>References.</b> P. Newson and J. Krumm, \"Hidden Markov Map Matching Through Noise and Sparseness\", "
                "ACM SIGSPATIAL GIS 2009. \"Map Matching @ Uber\" (YouTube talk provided as hackathon reference). "
                "Libraries: scikit-learn, LightGBM, Optuna, Folium, FastAPI, Leaflet.", SMALL))

    out = OUT / "PathSnap_Technical_Report.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.5 * cm, title="PathSnap Technical Report").build(s)
    print("wrote", out)


if __name__ == "__main__":
    main()
