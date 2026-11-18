"""Bayesian hyperparameter optimisation of LightGBM with Optuna (TPE sampler).

Objective : mean observation-level Top-1 accuracy over 5-fold GroupKFold on the 80 train
            trips (group = trip). Top-1 is the quantity the route pipeline consumes, so it is
            the primary selection metric; macro F1 and log loss are recorded per trial too.
Student-test is used only AFTER the search, to compare untuned vs tuned.
Outputs: results/hpo/trials.csv, best_params.json, tuned_vs_untuned.csv, plots;
         models/lgbm_tuned.joblib
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

sys.path.insert(0, str(Path(__file__).parent))
from candidates import FEATURES
from train_models import top1, continuity
from data_io import load_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hpo"; OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42
N_TRIALS = 40
UNTUNED = dict(n_estimators=300, learning_rate=0.05, num_leaves=31)


def space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )


def make(params):
    return LGBMClassifier(**params, subsample_freq=1 if params.get("subsample", 1) < 1 else 0,
                          random_state=SEED, verbose=-1, n_jobs=-1)


def cv_eval(params, tr, folds):
    X, y = tr[FEATURES].to_numpy(), tr.y.to_numpy()
    t1, f1, ll = [], [], []
    for a, b in folds:
        m = make(params).fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        t1.append(top1(tr.iloc[b], p)[0]); f1.append(f1_score(y[b], p > 0.5, average="macro"))
        ll.append(log_loss(y[b], np.clip(p, 1e-6, 1 - 1e-6)))
    return np.mean(t1), np.mean(f1), np.mean(ll)


def test_eval(params, tr, te, arcs_by):
    t0 = time.time(); m = make(params).fit(tr[FEATURES], tr.y); fit_s = time.time() - t0
    t0 = time.time(); p = m.predict_proba(te[FEATURES])[:, 1]; inf = time.time() - t0
    acc, best = top1(te, p)
    return m, dict(test_top1=acc, test_macro_f1=f1_score(te.y, p > 0.5, average="macro"),
                   test_log_loss=log_loss(te.y, np.clip(p, 1e-6, 1 - 1e-6)),
                   test_continuity=continuity(best, arcs_by), fit_s=fit_s,
                   infer_ms_per_obs=1000 * inf / te.groupby(["trip", "obs"]).ngroups)


def main():
    tr = pd.read_pickle(ROOT / "features" / "train.pkl")
    te = pd.read_pickle(ROOT / "features" / "student_test.pkl")
    arcs_by = {t: load_trip("student_test", t).arcs for t in te.trip.unique()}
    folds = list(GroupKFold(5).split(tr, tr.y, tr.trip))

    def objective(trial):
        t1, f1, ll = cv_eval(space(trial), tr, folds)
        trial.set_user_attr("cv_macro_f1", f1); trial.set_user_attr("cv_log_loss", ll)
        return t1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="lgbm_pathsnap")
    study.enqueue_trial({**UNTUNED, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                         "reg_lambda": 1e-3, "reg_alpha": 1e-3})  # trial 0 = untuned reference
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number:2d} top1={t.value:.4f} best={s.best_value:.4f}", flush=True)])
    hpo_s = time.time() - t0
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    df.to_csv(OUT / "trials.csv", index=False)
    best = study.best_params
    json.dump({"best_params": best, "best_cv_top1": study.best_value, "n_trials": N_TRIALS,
               "search_seconds": hpo_s, "sampler": "TPE(seed=42)", "cv": "GroupKFold(5) by trip"},
              open(OUT / "best_params.json", "w"), indent=2)

    rows = []
    for name, params in [("LightGBM untuned", UNTUNED), ("LightGBM tuned", best)]:
        cv_t1, cv_f1, cv_ll = cv_eval(params, tr, folds)
        m, r = test_eval(params, tr, te, arcs_by)
        rows.append(dict(model=name, cv_top1=cv_t1, cv_macro_f1=cv_f1, cv_log_loss=cv_ll, **r))
        print(rows[-1], flush=True)
        if name.endswith("tuned") and not name.endswith("untuned"):
            joblib.dump({"model": m, "features": FEATURES, "params": best}, ROOT / "models" / "lgbm_tuned.joblib")
    pd.DataFrame(rows).to_csv(OUT / "tuned_vs_untuned.csv", index=False)

    v = df["value"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(v, "o", ms=4, alpha=0.6, label="trial"); ax[0].plot(np.maximum.accumulate(v), label="best so far")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV top-1"); ax[0].set_title("Optuna TPE optimisation history"); ax[0].legend()
    imp = optuna.importance.get_param_importances(study)
    ax[1].barh(list(imp)[::-1], list(imp.values())[::-1]); ax[1].set_title("Hyperparameter importance (fANOVA)")
    fig.tight_layout(); fig.savefig(OUT / "optimisation.png", dpi=110)
    print(json.dumps(best, indent=1), "\nsearch time", round(hpo_s), "s")


if __name__ == "__main__":
    main()
