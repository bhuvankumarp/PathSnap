"""Render results/progress.html from the actual result files (no hand-typed numbers)."""
import base64
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def img(p):
    return f'<img alt="{p.stem}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">' if p.exists() else ""


def table(df, pct=(), dec=3):
    d = df.copy()
    for c in d.columns:
        if c in pct:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{100*v:.1f}%")
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{dec}f}")
    return d.to_html(index=False, border=0, classes="t", na_rep="")


inv = pd.read_csv(R / "eda" / "inventory.csv")
cand = pd.read_csv(R / "candidates_report.csv")
hmm = pd.read_csv(R / "hmm_comparison.csv")
mc = pd.read_csv(R / "model_comparison.csv") if (R / "model_comparison.csv").exists() else None

part = inv.groupby("partition").agg(trips=("trip", "count"), gps_fixes=("n_gps", "sum"),
                                    median_fixes=("n_gps", "median"), median_km=("path_len_km", "median"),
                                    frac_1s=("frac_dt_1s", "mean"), gaps_over_5s=("n_gaps_gt5s", "sum"),
                                    median_arcs=("n_arcs", "median"), bidirectional=("frac_bidirectional_arcs", "mean"),
                                    route_continuity=("route_continuity", "mean")).reset_index()
cp = cand.groupby("partition")[["offroute_frac", "cand_miss_rate", "nearest_top1", "mean_cands",
                                "align_resid_median_m", "align_resid_p95_m"]].mean().reset_index()
allrow = hmm[hmm.trip.str.startswith("ALL")].iloc[0]
nb = cp.set_index("partition").loc["student_test", "nearest_top1"]

models_html = ""
if mc is not None:
    cols = ["model", "family", "cv_macro_f1", "cv_top1", "test_macro_f1", "test_top1", "test_log_loss",
            "test_continuity", "fit_s", "infer_ms_per_obs"]
    models_html = (f'<h2>{mc.cv_macro_f1.notna().sum()} candidate classifiers, first pass</h2><p>Default configurations, 5-fold GroupKFold on the 80 train trips '
                   f'(grouped by trip), evaluated on the 10 student-test trips.</p><div class="scroll">'
                   f'{table(mc[[c for c in cols if c in mc]], pct=("cv_top1", "test_top1", "test_continuity"))}</div>'
                   f'{img(R / "model_comparison.png")}')

worst_gap = inv.sort_values("max_gap_s", ascending=False).iloc[0]
spd = inv.sort_values("n_speed_gt50ms", ascending=False).iloc[0]

html = f"""<title>PathSnap Review 1</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;700&display=swap">
<style>
/* layout: single reading column, wide tables scroll in place; road-map palette */
:root {{ --bg:#f6f7f5; --fg:#1d2321; --muted:#5d6a65; --line:#d6dcd8; --gps:#1a73e8; --route:#1e8e3e; --warn:#b3261e; --card:#ffffff;
  --display:'Space Grotesk',system-ui,sans-serif; --body:'IBM Plex Sans',system-ui,sans-serif; --mono:'IBM Plex Mono',ui-monospace,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141816; --fg:#e4e9e6; --muted:#9aa6a1; --line:#2d3532; --gps:#8ab4f8; --route:#5bc27a; --warn:#f28b82; --card:#1b201e; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#141816; --fg:#e4e9e6; --muted:#9aa6a1; --line:#2d3532; --gps:#8ab4f8; --route:#5bc27a; --warn:#f28b82; --card:#1b201e; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.6 var(--body); }}
main {{ max-width:960px; margin:0 auto; padding-block:32px 64px; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); display:grid; gap:20px; }}
h1,h2 {{ font-family:var(--display); text-wrap:balance; margin:0 }}
h1 {{ font-size:2rem }} h2 {{ font-size:1.3rem; margin-top:20px; padding-top:16px; border-top:1px solid var(--line) }}
p {{ margin:0; max-width:68ch }} .muted {{ color:var(--muted) }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:14px }}
.kpi b {{ display:block; font:600 1.7rem var(--mono); font-variant-numeric:tabular-nums }}
.kpi span {{ color:var(--muted); font-size:.85rem }}
.kpi.hl b {{ color:var(--route) }}
.scroll {{ overflow-x:auto }}
table.t {{ border-collapse:collapse; font:13px var(--mono); font-variant-numeric:tabular-nums; min-width:100% }}
table.t th, table.t td {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap }}
table.t th:first-child, table.t td:first-child {{ text-align:left }}
table.t th {{ color:var(--muted); font-weight:500 }}
img {{ max-width:100%; background:#fff; border:1px solid var(--line); border-radius:4px }}
ul {{ margin:0; padding-left:20px; max-width:70ch }} li+li {{ margin-top:4px }}
.tag {{ font:500 11px var(--mono); letter-spacing:.06em; text-transform:uppercase; color:var(--muted) }}
.done {{ color:var(--route) }} .todo {{ color:var(--muted) }}
</style>
<main>
<div class="tag">UE24CS352A · PathSnap · first review</div>
<h1>Map-matching pipeline: where we are</h1>
<p class="muted">Every figure below is read from the CSVs our scripts wrote (<code>results/</code>). Train = 80 trips, student-test = 10 trips. Evaluator-test data has not been touched.</p>

<div class="kpis">
 <div class="kpi"><b>{int(part.gps_fixes.sum()):,}</b><span>GPS fixes across 90 trips</span></div>
 <div class="kpi"><b>{100*cp.set_index('partition').loc['student_test','cand_miss_rate']:.1f}%</b><span>candidate miss rate (student-test)</span></div>
 <div class="kpi"><b>{100*nb:.1f}%</b><span>nearest-segment top-1 (baseline)</span></div>
 <div class="kpi"><b>{100*allrow.clf_top1:.1f}%</b><span>LightGBM top-1, classifier only</span></div>
 <div class="kpi hl"><b>{100*allrow.hmm_top1:.1f}%</b><span>LightGBM + HMM/Viterbi top-1</span></div>
</div>

<h2>Dataset inventory</h2>
<div class="scroll">{table(part, pct=("frac_1s", "bidirectional", "route_continuity"))}</div>
{img(R / "eda" / "trip_distributions.png")}
<div class="kpis">
{img(R / "eda" / "sampling_interval.png")}
{img(R / "eda" / "trajectories_overview.png")}
</div>

<h2>Data-quality findings</h2>
<ul>
<li>No NaN coordinates and no non-increasing timestamps in any trip.</li>
<li>Time gaps: {int(inv.n_gaps_gt5s.sum())} gaps over 5 s in total; {(inv.max_gap_s>60).sum()} trips have a gap over 60 s. Largest: trip {worst_gap.trip} ({worst_gap.max_gap_s:,.0f} s).</li>
<li>Implied speed over 50 m/s at {int(inv.n_speed_gt50ms.sum())} steps across {(inv.n_speed_gt50ms>0).sum()} trips (worst: trip {spd.trip}, {int(spd.n_speed_gt50ms)} steps), so GPS jumps exist.</li>
<li>Trips span the globe (lon {inv.lon_min.min():.0f}° to {inv.lon_max.max():.0f}°), so we project each trip to metres around its own centroid instead of using degree distances.</li>
<li>About {100*part.bidirectional.mean():.0f}% of arcs have a reverse twin. Distance alone cannot tell them apart, which is why nearest-segment tops out near {100*nb:.0f}%.</li>
<li>Reference routes are almost fully connected (continuity {100*part.route_continuity.mean():.1f}%), but {100*cp.set_index('partition').loc['train','offroute_frac']:.1f}% of train fixes lie more than 50 m from every route arc (detours or off-network driving). We flag these as off-route and keep them out of training and scoring.</li>
</ul>

<h2>Labels and candidates</h2>
<p>The <code>.route</code> file lists arcs in order, not one arc per fix. We align fixes to route arcs with a monotonic DP: the route index can only stay or advance, and total point-to-segment distance is minimised. Candidates are up to 8 arcs within 60 m, with 15 features each (distance, rank, direction-aware heading difference, speed, gap, arc length, projection position, node degrees).</p>
<div class="scroll">{table(cp, pct=("offroute_frac", "cand_miss_rate", "nearest_top1"), dec=2)}</div>

{models_html}

<h2>HMM / Viterbi, first version</h2>
<p>States are the candidate arcs. Emission is the classifier probability. Transition decays with directed hop distance in the road graph, and the hop budget grows with the time gap. Arcs that are unreachable get a fixed penalty, so decoding survives disconnected segments.</p>
<div class="scroll">{table(hmm, pct=("clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity"))}</div>
<p class="muted">HMM raised top-1 on {(hmm.hmm_top1[:-1] > hmm.clf_top1[:-1]).sum()} of 10 trips and lowered it on {(hmm.hmm_top1[:-1] < hmm.clf_top1[:-1]).sum()}. Continuity is measured on every 5th fix, where consecutive arcs can legitimately be several hops apart, so it reads low. Measuring it on 1 Hz fixes is next.</p>

<h2>Plan to the final review</h2>
<ul>
<li class="done">Done: EDA, Folium maps, labels, candidates, features, 7-model first pass, HMM v1</li>
<li class="todo">Next: Optuna Bayesian HPO on LightGBM (GroupKFold objective), tuned vs untuned</li>
<li class="todo">Next: HMM transitions from shortest-path length vs GPS distance; tune BETA on train and student-test only</li>
<li class="todo">Next: FastAPI on laptop 1 (ground truth stays server-side, request logs), UI on laptop 2 listing the 10 evaluator trips</li>
<li class="todo">Next: report sections 2–7, GitHub repo</li>
</ul>
</main>
"""
(R / "progress.html").write_text(html, encoding="utf-8")
print("wrote", R / "progress.html")
