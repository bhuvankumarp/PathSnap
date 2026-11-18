"""Over/underfitting evidence for every model: in-sample (train) vs 5-fold GroupKFold CV vs
student-test, on the same metrics (macro F1 at 0.5, observation-level Top-1, log loss).
Writes results/fit_diagnostics.csv and results/fit_diagnostics.png."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, log_loss

sys.path.insert(0, str(Path(__file__).parent))
from candidates import FEATURES
from train_models import models, top1

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def main():
    tr = pd.read_pickle(ROOT / "features" / "train.pkl")
    te = pd.read_pickle(ROOT / "features" / "student_test.pkl")
    mc = pd.read_csv(R / "model_comparison.csv").set_index("model")
    X, y = tr[FEATURES].to_numpy(), tr.y.to_numpy()
    rows = []
    for name, (fam, est) in models().items():
        est.fit(X, y)
        p = est.predict_proba(X)[:, 1]
        rows.append(dict(model=name, train_macro_f1=f1_score(y, p > 0.5, average="macro"),
                         train_top1=top1(tr, p)[0], train_log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                         cv_macro_f1=mc.loc[name, "cv_macro_f1"], cv_macro_f1_std=mc.loc[name, "cv_macro_f1_std"],
                         cv_top1=mc.loc[name, "cv_top1"],
                         test_macro_f1=mc.loc[name, "test_macro_f1"], test_top1=mc.loc[name, "test_top1"],
                         test_log_loss=mc.loc[name, "test_log_loss"]))
        r = rows[-1]
        print(f"{name:14s} train_top1={r['train_top1']:.4f} cv_top1={r['cv_top1']:.4f} test_top1={r['test_top1']:.4f} "
              f"train_f1={r['train_macro_f1']:.4f} cv_f1={r['cv_macro_f1']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df["gap_train_cv_top1"] = df.train_top1 - df.cv_top1
    df.to_csv(R / "fit_diagnostics.csv", index=False)
    x = np.arange(len(df)); w = 0.27
    fig, a = plt.subplots(figsize=(11, 4.2))
    a.bar(x - w, df.train_top1, w, label="train (in-sample)")
    a.bar(x, df.cv_top1, w, label="5-fold GroupKFold CV")
    a.bar(x + w, df.test_top1, w, label="student-test")
    a.set_xticks(x, df.model, rotation=25); a.set_ylim(0.75, 1.0); a.set_ylabel("observation-level Top-1")
    a.set_title("Train vs CV vs student-test Top-1 (gap = overfitting)"); a.legend()
    fig.tight_layout(); fig.savefig(R / "fit_diagnostics.png", dpi=110)


if __name__ == "__main__":
    main()
