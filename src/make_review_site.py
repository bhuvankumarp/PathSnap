"""Build review1.html (repo root): the first-review walkthrough, generated from result files.
Serve the repo root and open it:   python -m http.server 5600   ->  http://localhost:5600/review1.html
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

inv = pd.read_csv(R / "eda" / "inventory.csv")
cand = pd.read_csv(R / "candidates_report.csv")
mc = pd.read_csv(R / "model_comparison.csv")
hmm = pd.read_csv(R / "hmm" / "student_test_clf_vs_hmm.csv")
allr = hmm.iloc[-1]
tr_off = cand[cand.partition == "train"].offroute_frac.mean()
nb = mc[mc.model.str.startswith("Nearest")].iloc[0]
best = mc.dropna(subset=["cv_macro_f1"]).sort_values("cv_top1", ascending=False).iloc[0]

rows = ""
for _, r in mc.iterrows():
    f = lambda v, p=False: "–" if pd.isna(v) else (f"{100*v:.1f}%" if p else f"{v:.3f}")
    hl = ' class="best"' if r.model == best.model else ""
    rows += (f"<tr{hl}><td>{r.model}</td><td>{r.family}</td><td>{f(r.cv_macro_f1)}</td><td>{f(r.cv_top1, True)}</td>"
             f"<td>{f(r.test_macro_f1)}</td><td>{f(r.test_top1, True)}</td><td>{f(r.test_log_loss)}</td></tr>")

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathSnap Review 1</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;700&display=swap">
<style>
:root {{ --bg:#f5f7f6; --card:#fff; --fg:#1b2220; --muted:#5b6863; --line:#d5dcd8; --gps:#1a73e8; --route:#1e8e3e; --pred:#7b1fa2; --bad:#c5221f;
  --display:'Space Grotesk',system-ui,sans-serif; --body:'IBM Plex Sans',system-ui,sans-serif; --mono:'IBM Plex Mono',ui-monospace,monospace }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 var(--body) }}
nav {{ position:sticky; top:0; z-index:10; background:var(--fg); color:#fff; display:flex; gap:4px; flex-wrap:wrap; padding:8px 16px; align-items:center }}
nav b {{ font-family:var(--display); margin-right:12px }}
nav a {{ color:#cfe3d6; text-decoration:none; font-size:14px; padding:4px 10px; border-radius:4px }}
nav a:hover {{ background:#ffffff1f; color:#fff }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 80px; display:grid; gap:56px }}
section {{ display:grid; gap:14px; scroll-margin-top:60px }}
.step {{ font:500 12px var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--muted) }}
h1 {{ font:700 2.2rem/1.15 var(--display); margin:0; text-wrap:balance }}
h2 {{ font:700 1.6rem/1.2 var(--display); margin:0 }}
p {{ margin:0; max-width:75ch }}
.say {{ border-left:4px solid var(--pred); background:var(--card); padding:12px 16px; border-radius:0 6px 6px 0 }}
.say::before {{ content:"Say: "; font-weight:600; color:var(--pred) }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px 14px }}
.kpi b {{ display:block; font:600 1.6rem var(--mono); font-variant-numeric:tabular-nums }}
.kpi span {{ font-size:13px; color:var(--muted) }}
.kpi.bad b {{ color:var(--bad) }} .kpi.good b {{ color:var(--route) }}
iframe {{ width:100%; height:560px; border:1px solid var(--line); border-radius:6px; background:#fff }}
img {{ max-width:100%; border:1px solid var(--line); border-radius:6px; background:#fff }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:12px }}
@media (max-width:800px) {{ .two {{ grid-template-columns:1fr }} }}
.scroll {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; font:14px var(--mono); font-variant-numeric:tabular-nums; background:var(--card) }}
th,td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap }}
th:nth-child(-n+2), td:nth-child(-n+2) {{ text-align:left }}
th {{ color:var(--muted); font-weight:500 }}
tr.best td {{ background:#e7f4ea; font-weight:600 }}
.flow {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; font:500 14px var(--mono) }}
.flow span {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:8px 12px }}
.flow i {{ color:var(--muted); font-style:normal }}
ul {{ margin:0; padding-left:22px; max-width:75ch }} li+li {{ margin-top:4px }}
a.btn {{ display:inline-block; background:var(--pred); color:#fff; text-decoration:none; padding:8px 14px; border-radius:6px; font-weight:600 }}
.muted {{ color:var(--muted); font-size:14px }}
</style></head><body>
<nav><b>PathSnap · Review 1</b>
<a href="#problem">1 Problem</a><a href="#data">2 Data</a><a href="#approach">3 Approach</a><a href="#models">4 Models</a><a href="#next">5 Next</a><a href="#extra">Extra</a></nav>
<main>

<section id="problem">
<div class="step">Step 1 · The problem</div>
<h1>Snap noisy GPS to the road the car really drove</h1>
<p>Blue dots are raw GPS fixes; the green line is the reference route. Toggle layers top-right.</p>
<iframe src="results/eda/map_train_00000002.html" title="Folium map of train trip 00000002"></iframe>
<p class="say">GPS drifts onto wrong or parallel roads. Given the GPS track and the road network, we must find which road the car was on at each moment.</p>
</section>

<section id="data">
<div class="step">Step 2 · Data exploration</div>
<h2>What the data looks like</h2>
<div class="kpis">
 <div class="kpi"><b>{len(inv)}</b><span>trips: {(inv.partition=='train').sum()} train, {(inv.partition=='student_test').sum()} student-test</span></div>
 <div class="kpi"><b>{inv.n_gps.sum():,}</b><span>GPS fixes, {100*inv.frac_dt_1s.mean():.0f}% exactly 1 s apart</span></div>
 <div class="kpi"><b>{int(inv.nan_gps.sum())}</b><span>missing / NaN coordinates</span></div>
 <div class="kpi bad"><b>{inv.max_gap_s.max()/3600:.0f} h</b><span>largest time gap (trip {inv.loc[inv.max_gap_s.idxmax(),'trip']})</span></div>
 <div class="kpi bad"><b>{int(inv.n_speed_gt50ms.sum()):,}</b><span>GPS jumps faster than 50 m/s ({(inv.n_speed_gt50ms>0).sum()} trips)</span></div>
 <div class="kpi bad"><b>{100*tr_off:.1f}%</b><span>train fixes &gt;50 m from the route (excluded)</span></div>
</div>
<img src="results/eda/trip_distributions.png" alt="Per-trip distributions">
<div class="two"><img src="results/eda/sampling_interval.png" alt="Sampling interval"><img src="results/eda/trajectories_overview.png" alt="Trajectories"></div>
<p class="say">No missing values, but large time gaps, GPS jumps and some off-route driving. Trips are spread over several continents (longitude {inv.lon_min.min():.0f}° to {inv.lon_max.max():.0f}°), so we convert to metres per trip instead of using degrees.</p>
</section>

<section id="approach">
<div class="step">Step 3 · Approach</div>
<h2>Candidates → features → classifier → HMM</h2>
<div class="flow"><span>GPS fix</span><i>→</i><span>≤8 roads within 60 m</span><i>→</i><span>15 features each</span><i>→</i><span>classifier: P(right road)</span><i>→</i><span>HMM/Viterbi: connected route</span></div>
<ul>
<li><b>Labels:</b> the route file is an ordered road list. Each GPS fix is matched to it in order (never going backwards).</li>
<li><b>Features:</b> distance, direction difference, speed, time gap, road length, position along road, junction degree…</li>
<li><b>Why direction matters:</b> {100*inv.frac_bidirectional_arcs.mean():.0f}% of roads exist in both directions at the same distance.</li>
</ul>
<div class="kpis">
 <div class="kpi"><b>{100*cand[cand.partition=='student_test'].cand_miss_rate.mean():.1f}%</b><span>right road missing from candidates (student-test)</span></div>
 <div class="kpi bad"><b>{100*nb.test_top1:.1f}%</b><span>accuracy of "pick the nearest road"</span></div>
</div>
<p class="say">Nearest road gets only {100*nb.test_top1:.0f}% because it can't tell a road from its opposite direction. Our heading feature fixes that.</p>
</section>

<section id="models">
<div class="step">Step 4 · Model comparison</div>
<h2>Seven models, grouped cross-validation by trip</h2>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Family</th><th>CV macro F1</th><th>CV top-1</th><th>Test macro F1</th><th>Test top-1</th><th>Test log loss</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<img src="results/model_comparison.png" alt="Model comparison">
<p class="muted">CV = 5-fold GroupKFold on 80 train trips (a trip never appears in both train and validation). Test = 10 student-test trips. Top-1 = share of GPS fixes whose highest-scoring road is correct.</p>
<p class="say">{best.model} is best: {100*best.test_top1:.1f}% top-1 vs {100*nb.test_top1:.1f}% for nearest road, with the best log loss, and it's fast.</p>
</section>

<section id="next">
<div class="step">Step 5 · Next steps</div>
<h2>Plan to the final review</h2>
<ul>
<li>Bayesian hyperparameter tuning of {best.model} (Optuna)</li>
<li>HMM / Viterbi decoding for a connected route</li>
<li>FastAPI backend on laptop 1, evaluator UI on laptop 2</li>
<li>Technical report and GitHub repository</li>
</ul>
</section>

<section id="extra">
<div class="step">Extra · only if they ask for more</div>
<h2>HMM result and live demo</h2>
<div class="kpis">
 <div class="kpi"><b>{100*allr.top1_clf:.1f}%</b><span>top-1, classifier only</span></div>
 <div class="kpi good"><b>{100*allr.top1_hmm:.1f}%</b><span>top-1, classifier + HMM</span></div>
 <div class="kpi good"><b>{100*allr.continuity_hmm:.1f}%</b><span>route continuity with HMM (was {100*allr.continuity_clf:.1f}%)</span></div>
 <div class="kpi good"><b>{100*allr.route_f1:.1f}%</b><span>route F1</span></div>
</div>
<iframe src="results/hmm/maps/student_test_00000077.html" title="HMM vs classifier map, trip 00000077"></iframe>
<p class="muted">Green = reference, purple = predicted route (HMM). Toggle "Classifier-only" in the layer menu to see the jumpy route without the HMM.</p>
<p><a class="btn" href="http://localhost:5500" target="_blank" rel="noopener">Open live demo UI</a> <span class="muted">(needs the API and UI running: run_api.bat + run_ui.bat)</span></p>
</section>
</main></body></html>
"""
(ROOT / "review1.html").write_text(html, encoding="utf-8")
print("wrote", ROOT / "review1.html")
