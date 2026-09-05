"""Training the cart-abandonment model.

    python -m aether.ml.train data/raw/2019-Oct.csv --limit 2000000

## The model

`HistGradientBoostingClassifier`: histogram-based gradient boosted trees,
scikit-learn's equivalent of LightGBM. Chosen for three concrete reasons
rather than by habit.

Tabular data of this shape is what boosted trees are best at, and a neural
network here would be theatre. The features interact in ways a linear model
cannot express: a large cart is a mild signal on its own, a long silence is a
mild signal on its own, and a large cart *plus* a long silence is a strong
one. Trees find that; logistic regression cannot without being told.

It handles missing values natively. A third of REES46 rows have no
`category_code` and one in seven has no `brand`, and every alternative needs
imputation, which invents data and pulls the mean around.

And it is fast enough that training is free: a couple of million events fit in
seconds, so this can run in a GitHub Actions job.

## Why a baseline is trained too

Logistic regression is fitted alongside, on identical data, and reported next
to the real model. If the boosted trees cannot clearly beat a linear model,
they have not earned their complexity, and the honest thing is to know that
rather than to present a single impressive-looking number with nothing to
compare it against.

## Why PR-AUC rather than accuracy

Most sessions with a cart abandon it, so a model that predicts "abandoned" for
everything scores well on accuracy and is worthless. Precision-recall AUC
measures what is actually wanted: finding the abandoners without drowning in
false alarms. ROC-AUC is reported too, and is the more flattering of the two,
which is why it is not the headline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from aether.data.rees46 import iter_events
from aether.env import load_dotenv
from aether.ml.dataset import BuildStats, Dataset, build_dataset, time_split
from aether.ml.features import FEATURE_NAMES
from aether.ml.model import ModelArtifact

DEFAULT_MODEL_PATH = Path("data/model.pkl")


def fit_model(train: Dataset, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        # Enough examples per leaf that a rule has to hold for a meaningful
        # number of sessions before the model will learn it.
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=seed,
    )
    model.fit(train.X, train.y)
    return model


def fit_baseline(train: Dataset, seed: int = 0):
    """A linear model on the same data, to keep the real one honest."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    ).fit(train.X, train.y)


def evaluate(model, data: Dataset) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    if len(data) == 0 or len(set(data.y.tolist())) < 2:
        return {"note": "not enough label variety in this split to evaluate"}

    scores = model.predict_proba(data.X)[:, 1]
    return {
        # The headline. Positives dominate, so this is the number that
        # reflects whether abandoners are actually being found.
        "pr_auc": float(average_precision_score(data.y, scores)),
        "roc_auc": float(roc_auc_score(data.y, scores)),
        "log_loss": float(log_loss(data.y, scores)),
        # How well the probabilities mean what they say. A 0.8 should be
        # right about 80% of the time, and the dashboard shows a probability
        # rather than a class, so calibration is not cosmetic.
        "brier": float(brier_score_loss(data.y, scores)),
        "base_rate": float(data.y.mean()),
        "examples": int(len(data)),
    }


def calibration_curve(model, data: Dataset, bins: int = 10) -> list[dict]:
    """Predicted probability against observed rate, bucketed.

    A well-calibrated model puts its 0.7 bucket at roughly 70% observed.
    """
    if len(data) == 0:
        return []
    scores = model.predict_proba(data.X)[:, 1]
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for low, high in zip(edges, edges[1:]):
        in_bin = (scores >= low) & (scores < high if high < 1.0 else scores <= 1.0)
        if not in_bin.any():
            continue
        out.append(
            {
                "predicted": float(scores[in_bin].mean()),
                "observed": float(data.y[in_bin].mean()),
                "count": int(in_bin.sum()),
            }
        )
    return out


def feature_importance(model, train: Dataset, test: Dataset, seed: int = 0) -> list[dict]:
    """Permutation importance: how much worse the model gets when one feature
    is shuffled.

    Preferred over a tree's own split counts, which reward high-cardinality
    features regardless of whether they help.
    """
    from sklearn.inspection import permutation_importance

    if len(test) < 50:
        return []
    sample = min(len(test), 20_000)
    result = permutation_importance(
        model,
        test.X[:sample],
        test.y[:sample],
        n_repeats=3,
        random_state=seed,
        scoring="average_precision",
    )
    ranked = sorted(
        zip(FEATURE_NAMES, result.importances_mean), key=lambda pair: -pair[1]
    )
    return [{"feature": name, "importance": float(value)} for name, value in ranked]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.ml.train",
        description="Train the cart-abandonment model.",
    )
    parser.add_argument("input", type=Path, help="REES46 .csv or .csv.gz")
    parser.add_argument("--limit", type=int, default=2_000_000, help="events to read")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    load_dotenv()

    if not args.input.exists():
        parser.error(f"{args.input} not found. See docs/DATA.md.")

    print(f"reading {args.input}")
    began = time.perf_counter()
    stats = BuildStats()
    data = build_dataset(iter_events(args.input, limit=args.limit), stats)
    print(f"  {stats}")
    print(f"  built in {time.perf_counter() - began:,.1f}s")

    if len(data) == 0:
        print("  no sessions with a cart; nothing to train on")
        return 1

    train, test = time_split(data, args.train_fraction)
    print()
    print("SPLIT   by session start time, never at random: a random split lets")
    print("        the model see the period it is tested on.")
    print(f"  train          {len(train):,} examples, {train.sessions:,} sessions, "
          f"{100 * train.abandonment_rate:.1f}% abandoned")
    print(f"  test           {len(test):,} examples, {test.sessions:,} sessions, "
          f"{100 * test.abandonment_rate:.1f}% abandoned")

    overlap = set(train.session_ids.tolist()) & set(test.session_ids.tolist())
    print(f"  sessions in both: {len(overlap)}   (must be 0)")

    print()
    began = time.perf_counter()
    model = fit_model(train, args.seed)
    fit_seconds = time.perf_counter() - began
    baseline = fit_baseline(train, args.seed)

    model_metrics = evaluate(model, test)
    baseline_metrics = evaluate(baseline, test)

    print(f"RESULTS on held-out future, fitted in {fit_seconds:,.1f}s")
    print(f"  {'':<22}{'PR-AUC':>9}{'ROC-AUC':>9}{'log loss':>10}{'Brier':>8}")
    for label, metrics in (
        ("gradient boosting", model_metrics),
        ("logistic baseline", baseline_metrics),
    ):
        if "pr_auc" not in metrics:
            print(f"  {label:<22} {metrics['note']}")
            continue
        print(
            f"  {label:<22}{metrics['pr_auc']:>9.4f}{metrics['roc_auc']:>9.4f}"
            f"{metrics['log_loss']:>10.4f}{metrics['brier']:>8.4f}"
        )
    if "pr_auc" in model_metrics:
        print(f"  {'always-abandon':<22}{model_metrics['base_rate']:>9.4f}"
              f"{'0.5000':>9}   (predicting the majority class)")

    curve = calibration_curve(model, test)
    if curve:
        print()
        print("CALIBRATION   does a 0.8 mean 80%?")
        print(f"  {'predicted':>10}{'observed':>10}{'n':>9}")
        for point in curve:
            print(f"  {point['predicted']:>10.2f}{point['observed']:>10.2f}{point['count']:>9,}")

    importances = feature_importance(model, train, test, args.seed)
    if importances:
        print()
        print("WHAT THE MODEL USES   permutation importance, top 8")
        for entry in importances[:8]:
            print(f"  {entry['feature']:<24}{entry['importance']:>9.4f}")

    artifact = ModelArtifact(
        estimator=model,
        feature_names=FEATURE_NAMES,
        metrics={
            "model": model_metrics,
            "baseline": baseline_metrics,
            "calibration": curve,
            "importance": importances[:12],
            "dataset": {
                "events": stats.events,
                "sessions": stats.sessions,
                "sessions_with_cart": stats.sessions_with_cart,
                "examples": stats.examples,
            },
        },
        trained_at=time.time(),
        trained_on_events=stats.events,
    )
    written = artifact.save(args.out)
    print()
    print(f"wrote {args.out}  ({written / 1024:,.0f} KB)   scikit-learn, for retraining")
    print(f"      {args.out.with_suffix('.json')}  (metrics, readable without unpickling)")

    # And the serving format. Two files rather than one because they have
    # different jobs: the pickle can still fit and explain itself, and the
    # export is what a container should load, since it needs no scikit-learn
    # and cannot execute anything on the way in.
    from aether.ml.export import export

    numpy_path = args.out.with_suffix(".npz")
    numpy_written = export(artifact).save(numpy_path)
    print(f"      {numpy_path}  ({numpy_written / 1024:,.0f} KB)   numpy only, for serving")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
