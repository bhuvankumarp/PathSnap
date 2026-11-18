"""Train and compare candidate classifiers (first pass, default-ish configs).

CV: 5-fold GroupKFold on the 80 train trips (group = trip, no leakage).
Dev evaluation: the 10 student_test trips.
Metrics:
  macro_f1      candidate-level binary (is reference arc), macro over {0,1}
  log_loss      candidate-level
  top1_acc      observation-level: argmax-probability candidate == reference arc
                (observations whose reference arc is not a candidate count as wrong)
  continuity    fraction of consecutive predicted arcs that are equal or share a node
Writes results/model_comparison.csv and plots.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent))
from candidates import FEATURES
from data_io import load_trip

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 42


PRED = RES / "predictions"
(RES / "confusion").mkdir(parents=True, exist_ok=True)


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


class SubsampledKNN(BaseEstimator, ClassifierMixin):
    """Distance-weighted KNN fitted on a random subsample (full 225k-row KNN takes >10 min per fold)."""

    def __init__(self, k=25, n_fit=40000, random_state=SEED):
        self.k, self.n_fit, self.random_state = k, n_fit, random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        i = rng.choice(len(X), min(self.n_fit, len(X)), replace=False)
        self.m_ = make_pipeline(StandardScaler(), KNeighborsClassifier(self.k, weights="distance", algorithm="brute", n_jobs=4)).fit(X[i], y[i])
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)


CONFIGS = {
    "LogReg": "StandardScaler + L2, C=1.0, lbfgs",
    "GaussianNB": "StandardScaler, var_smoothing=1e-9",
    "DecisionTree": "max_depth=8, min_samples_leaf=50",
    "RandomForest": "100 trees, max_depth=14, min_samples_leaf=20",
    "LightGBM": "300 trees, lr=0.05, num_leaves=31",
    "MLP(64,32)": "ReLU, Adam, early stopping, <=30 epochs",
    "KNN(k=25)": "k=25, distance-weighted, fit on 40k-row subsample",
}


def models():
    return {
        "LogReg": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))),
        "GaussianNB": ("Probabilistic", make_pipeline(StandardScaler(), GaussianNB())),
        "DecisionTree": ("Tree", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=SEED)),
        "RandomForest": ("Ensemble/bagging", RandomForestClassifier(100, min_samples_leaf=20, max_depth=14, n_jobs=-1, random_state=SEED)),
        "LightGBM": ("Boosting", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                                random_state=SEED, verbose=-1)),
        "MLP(64,32)": ("Neural net", make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=30,
                                                   early_stopping=True, random_state=SEED))),
        "KNN(k=25)": ("Instance-based", SubsampledKNN()),
    }


def top1(df, p):
    d = df[["trip", "obs", "arc", "y"]].copy(); d["p"] = p
    best = d.loc[d.groupby(["trip", "obs"]).p.idxmax()]
    return best.y.mean(), best


def continuity(best, arcs_by_trip):
    ok = tot = 0
    for tid, g in best.sort_values(["trip", "obs"]).groupby("trip"):
        arcs = arcs_by_trip[tid]
        a = g.arc.to_numpy()
        f, t = arcs[a, 0], arcs[a, 1]
        same = a[1:] == a[:-1]
        nodes_touch = (t[:-1] == f[1:]) | (f[:-1] == f[1:]) | (t[:-1] == t[1:]) | (f[:-1] == t[1:])
        ok += np.sum(same | nodes_touch); tot += len(a) - 1
    return ok / tot


def evaluate(name, fam, est, tr, te, arcs_by_trip):
    X, y, g = tr[FEATURES].to_numpy(), tr.y.to_numpy(), tr.trip.to_numpy()
    cv_f1, cv_top1 = [], []
    for k, (a, b) in enumerate(GroupKFold(5).split(X, y, g)):
        est.fit(X[a], y[a])
        p = est.predict_proba(X[b])[:, 1]
        cv_f1.append(f1_score(y[b], p > 0.5, average="macro"))
        cv_top1.append(top1(tr.iloc[b], p)[0])
    t0 = time.time(); est.fit(X, y); fit_s = time.time() - t0
    t0 = time.time(); p = est.predict_proba(te[FEATURES].to_numpy())[:, 1]; inf_s = time.time() - t0
    acc, best = top1(te, p)
    PRED.mkdir(exist_ok=True)
    te[["trip", "obs", "arc", "y"]].assign(p=p).to_pickle(PRED / f"{slug(name)}.pkl")
    # confusion matrix (candidate level, threshold 0.5)
    cm = confusion_matrix(te.y, p > 0.5)
    fig, a = plt.subplots(figsize=(3.6, 3.2))
    ConfusionMatrixDisplay(cm, display_labels=["not ref", "ref"]).plot(ax=a, colorbar=False, values_format="d")
    a.set_title(name); fig.tight_layout(); fig.savefig(RES / "confusion" / f"cm_{slug(name)}.png", dpi=110); plt.close(fig)
    # epoch-wise curves exist only for the MLP
    mlp = est[-1] if hasattr(est, "steps") else est
    if hasattr(mlp, "loss_curve_"):
        fig, a = plt.subplots(figsize=(6, 3.5))
        a.plot(mlp.loss_curve_, label="training log-loss")
        if getattr(mlp, "validation_scores_", None) is not None:
            b = a.twinx(); b.plot(mlp.validation_scores_, c="tab:orange", label="validation accuracy (10% hold-out)")
            b.set_ylabel("val accuracy"); b.legend(loc="center right")
        a.set_xlabel("epoch"); a.set_ylabel("loss"); a.legend(loc="upper right"); a.set_title(f"{name} training history")
        fig.tight_layout(); fig.savefig(RES / f"loss_curve_{slug(name)}.png", dpi=110); plt.close(fig)
    pr, rc, _, _ = precision_recall_fscore_support(te.y, p > 0.5, average="macro")
    row = dict(model=name, family=fam, config=CONFIGS.get(name, ""), cv_macro_f1=np.mean(cv_f1), cv_macro_f1_std=np.std(cv_f1),
               cv_top1=np.mean(cv_top1), test_precision=pr, test_recall=rc,
               test_macro_f1=f1_score(te.y, p > 0.5, average="macro"),
               test_log_loss=log_loss(te.y, np.clip(p, 1e-6, 1 - 1e-6)),
               test_top1=acc, test_continuity=continuity(best, arcs_by_trip),
               fit_s=fit_s, infer_ms_per_obs=1000 * inf_s / te.groupby(["trip", "obs"]).ngroups)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return row


def main():
    tr = pd.read_pickle(ROOT / "features" / "train.pkl")
    te = pd.read_pickle(ROOT / "features" / "student_test.pkl")
    arcs_by_trip = {t: load_trip("student_test", t).arcs for t in te.trip.unique()}
    print("train rows", len(tr), "pos rate", tr.y.mean().round(4), "| test rows", len(te))
    # nearest-segment heuristic (no learning) as reference point
    nb_acc, nb_best = top1(te, -te.dist.to_numpy())
    rows = [dict(model="Nearest-segment (baseline)", family="Heuristic", test_top1=nb_acc,
                 test_continuity=continuity(nb_best, arcs_by_trip))]
    for name, (fam, est) in models().items():
        rows.append(evaluate(name, fam, est, tr, te, arcs_by_trip))
        pd.DataFrame(rows).to_csv(RES / "model_comparison.csv", index=False)  # save as we go
    df = pd.DataFrame(rows)
    df.to_csv(RES / "model_comparison.csv", index=False)
    print(df.round(4).to_string())

    m = df.dropna(subset=["cv_macro_f1"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].bar(m.model, m.cv_macro_f1, yerr=m.cv_macro_f1_std, capsize=4)
    ax[0].set_title("5-fold GroupKFold macro F1 (train)"); ax[0].set_ylim(0.5, 1)
    w = 0.4; x = np.arange(len(m))
    ax[1].bar(x - w / 2, m.test_top1, w, label="Top-1 acc"); ax[1].bar(x + w / 2, m.test_macro_f1, w, label="Macro F1")
    ax[1].axhline(nb_acc, ls="--", c="k", lw=1, label="nearest-seg top-1")
    ax[1].set_xticks(x, m.model); ax[1].set_ylim(0.5, 1); ax[1].legend(); ax[1].set_title("Student-test")
    ax[2].bar(m.model, m.test_log_loss); ax[2].set_title("Student-test log loss (lower better)")
    for a in ax: a.tick_params(axis="x", rotation=40)
    fig.tight_layout(); fig.savefig(RES / "model_comparison.png", dpi=110)


if __name__ == "__main__":
    main()
