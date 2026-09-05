"""Exporting the trained ensemble to plain arrays.

A gradient boosted tree ensemble is, once fitted, a pile of thresholds. The
scikit-learn object that produced it is a training apparatus, and carrying it
into the serving path costs three things that have nothing to do with
prediction.

**Start-up time.** Importing scikit-learn measured 783 ms of a 2,718 ms cold
start in the container. On something that scales to zero, that is paid by
whoever visits first, every time the service has been idle.

**Image size.** scikit-learn and its dependencies are most of a 693 MB image,
for code that runs once during training and never again.

**Unpickling a model is arbitrary code execution.** `ModelArtifact` is a
pickle, so loading it from a bucket means whoever can write to that bucket can
run code in the serving container. The format here is `.npz` loaded with
`allow_pickle=False`: arrays and a JSON header, and nothing that can execute.
That is the part that matters most, and it is the reason this is worth doing
even where the milliseconds are not.

## What is exported

`HistGradientBoostingClassifier` keeps each iteration's tree as a structured
array of nodes. For binary classification there is one tree per iteration, and
the raw prediction is the baseline plus every tree's leaf value, squashed
through a logistic.

    raw   = baseline + sum(leaf_value(tree, x) for tree in trees)
    P     = 1 / (1 + exp(-raw))

Each node carries a feature, a threshold, and two children, so the traversal
is six columns wide. The rest of the structured array, the gain, the counts,
the bin thresholds, are training bookkeeping and are dropped.

Trees are concatenated into flat arrays with an offsets index, rather than
kept as a list, because an `.npz` of 200 small arrays is 200 members to read
and one array plus offsets is one.

## What is deliberately not supported

**Categorical splits.** `HistGradientBoostingClassifier` can split on a bitset
of categories rather than a threshold, and every feature this project computes
is numeric, so such a split can only mean the exporter is being used on
something it was not written for. It raises rather than guessing.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aether.ml.features import FEATURE_NAMES, compute_features
from aether.ml.session import SessionState

FORMAT_VERSION = 1


@dataclass
class TreeEnsemble:
    """A fitted ensemble as flat arrays, and the scorer over them.

    Interchangeable with `ModelArtifact` wherever a model is consumed: the
    predictor and the service call `probability` and `probabilities` and do
    not care which they were handed.
    """

    baseline: float
    # One entry per node, all trees concatenated. `offsets[t]:offsets[t+1]`
    # is tree t, and child indices are relative to the start of their tree.
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    is_leaf: np.ndarray
    value: np.ndarray
    missing_left: np.ndarray
    offsets: np.ndarray

    feature_names: tuple[str, ...] = FEATURE_NAMES
    metrics: dict | None = None
    trained_at: float = 0.0
    trained_on_events: int = 0
    version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != FEATURE_NAMES:
            # Same guard `ModelArtifact` carries, for the same reason: a model
            # fed a differently ordered vector scores garbage confidently.
            raise ValueError(
                "model was trained on different features than this build computes.\n"
                f"  trained: {list(self.feature_names)}\n"
                f"  current: {list(FEATURE_NAMES)}"
            )

    @property
    def trees(self) -> int:
        return len(self.offsets) - 1

    @property
    def nodes(self) -> int:
        return len(self.feature)

    # -- scoring -----------------------------------------------------------

    def probability(self, state: SessionState) -> float:
        return float(self.probabilities(compute_features(state).reshape(1, -1))[0])

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        """Abandonment probability for a batch of feature vectors."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected (n, {len(FEATURE_NAMES)}) features, got {X.shape}"
            )
        raw = self._raw(X)
        # The stable logistic: exp of a large positive raw overflows, so the
        # sign is folded out first.
        out = np.empty_like(raw)
        positive = raw >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
        exp_raw = np.exp(raw[~positive])
        out[~positive] = exp_raw / (1.0 + exp_raw)
        return out

    def _raw(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 1:
            # Serving scores one session at a time, and the vectorised walk
            # below spends its time in numpy call overhead on one-element
            # arrays: 200 trees times a handful of operations each. Walking
            # the tree in Python over lists measured about six times faster
            # for a single row, and identically for a thousand.
            return np.array([self._raw_one(X[0])])

        rows = np.arange(len(X))
        raw = np.full(len(X), self.baseline, dtype=np.float64)

        for tree in range(self.trees):
            lo, hi = int(self.offsets[tree]), int(self.offsets[tree + 1])
            feature = self.feature[lo:hi]
            threshold = self.threshold[lo:hi]
            left = self.left[lo:hi]
            right = self.right[lo:hi]
            is_leaf = self.is_leaf[lo:hi]
            missing_left = self.missing_left[lo:hi]

            # Every row walks the tree together, one level per iteration.
            # Rows that reach a leaf early simply stop moving, so the loop
            # runs for the depth of the tree rather than the depth per row.
            node = np.zeros(len(X), dtype=np.int64)
            while True:
                at_leaf = is_leaf[node]
                if at_leaf.all():
                    break
                x = X[rows, feature[node]]
                go_left = np.where(
                    np.isnan(x), missing_left[node], x <= threshold[node]
                )
                node = np.where(at_leaf, node, np.where(go_left, left[node], right[node]))
            raw += self.value[lo:hi][node]
        return raw

    def _raw_one(self, x: np.ndarray) -> float:
        """One row, walked in Python. Same arithmetic, no array overhead."""
        feature, threshold, left, right, is_leaf, value, missing_left = self._lists()
        offsets = self._offsets_list()
        row = x.tolist()

        total = self.baseline
        for tree in range(len(offsets) - 1):
            base = offsets[tree]
            node = base
            while not is_leaf[node]:
                seen = row[feature[node]]
                # NaN compares false against everything, including itself.
                if seen != seen:
                    go_left = missing_left[node]
                else:
                    go_left = seen <= threshold[node]
                node = base + (left[node] if go_left else right[node])
            total += value[node]
        return total

    def _lists(self):
        """Node columns as Python lists, built once.

        Indexing a numpy array with a Python integer builds a scalar object
        every time, which is most of the cost of a scalar tree walk.
        """
        cached = getattr(self, "_columns", None)
        if cached is None:
            cached = (
                self.feature.tolist(),
                self.threshold.tolist(),
                self.left.tolist(),
                self.right.tolist(),
                self.is_leaf.tolist(),
                self.value.tolist(),
                self.missing_left.tolist(),
            )
            object.__setattr__(self, "_columns", cached)
        return cached

    def _offsets_list(self) -> list[int]:
        cached = getattr(self, "_offsets_cache", None)
        if cached is None:
            cached = self.offsets.tolist()
            object.__setattr__(self, "_offsets_cache", cached)
        return cached

    # -- persistence -------------------------------------------------------

    def to_bytes(self) -> bytes:
        header = json.dumps(
            {
                "version": self.version,
                "baseline": self.baseline,
                "feature_names": list(self.feature_names),
                "metrics": self.metrics or {},
                "trained_at": self.trained_at,
                "trained_on_events": self.trained_on_events,
            }
        )
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            header=np.frombuffer(header.encode(), dtype=np.uint8),
            feature=self.feature,
            threshold=self.threshold,
            left=self.left,
            right=self.right,
            is_leaf=self.is_leaf,
            value=self.value,
            missing_left=self.missing_left,
            offsets=self.offsets,
        )
        return buffer.getvalue()

    def save(self, path: Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_bytes()
        path.write_bytes(payload)
        return len(payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> TreeEnsemble:
        """Load, without ever unpickling.

        `allow_pickle=False` is the entire security argument for this format:
        a `.npz` read this way cannot execute anything, so an artifact fetched
        from object storage is data rather than code.
        """
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            header = json.loads(bytes(loaded["header"]).decode())
            if header["version"] != FORMAT_VERSION:
                raise ValueError(
                    f"model format v{header['version']}, expected v{FORMAT_VERSION}"
                )
            return cls(
                baseline=header["baseline"],
                feature=loaded["feature"],
                threshold=loaded["threshold"],
                left=loaded["left"],
                right=loaded["right"],
                is_leaf=loaded["is_leaf"],
                value=loaded["value"],
                missing_left=loaded["missing_left"],
                offsets=loaded["offsets"],
                feature_names=tuple(header["feature_names"]),
                metrics=header["metrics"],
                trained_at=header["trained_at"],
                trained_on_events=header["trained_on_events"],
            )

    @classmethod
    def load(cls, path: Path) -> TreeEnsemble:
        return cls.from_bytes(Path(path).read_bytes())


def export_estimator(estimator) -> tuple[float, dict[str, np.ndarray]]:
    """Flatten a fitted HistGradientBoostingClassifier into arrays."""
    predictors = getattr(estimator, "_predictors", None)
    if predictors is None:
        raise TypeError(
            f"{type(estimator).__name__} is not a fitted "
            "HistGradientBoostingClassifier"
        )
    if any(len(iteration) != 1 for iteration in predictors):
        raise ValueError(
            "expected one tree per iteration; multiclass models are not supported"
        )

    trees = [iteration[0].nodes for iteration in predictors]
    if any(tree["is_categorical"].any() for tree in trees):
        raise ValueError(
            "categorical splits are not supported: every feature this project "
            "computes is numeric, so this model was not trained by this code"
        )

    offsets = np.zeros(len(trees) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(tree) for tree in trees])

    def column(name: str, dtype) -> np.ndarray:
        return np.concatenate([tree[name] for tree in trees]).astype(dtype)

    arrays = {
        "feature": column("feature_idx", np.int64),
        "threshold": column("num_threshold", np.float64),
        "left": column("left", np.int64),
        "right": column("right", np.int64),
        "is_leaf": column("is_leaf", bool),
        "value": column("value", np.float64),
        "missing_left": column("missing_go_to_left", bool),
        "offsets": offsets,
    }
    baseline = float(np.asarray(estimator._baseline_prediction).ravel()[0])
    return baseline, arrays


def export(artifact) -> TreeEnsemble:
    """Turn a `ModelArtifact` into a numpy-only one."""
    baseline, arrays = export_estimator(artifact.estimator)
    return TreeEnsemble(
        baseline=baseline,
        **arrays,
        feature_names=tuple(artifact.feature_names),
        metrics=artifact.metrics,
        trained_at=artifact.trained_at,
        trained_on_events=artifact.trained_on_events,
    )
