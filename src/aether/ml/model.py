"""The trained model, and the contract around it.

An artifact is not just a fitted estimator. It is the estimator plus the exact
feature contract it was fitted against, because a model handed features in a
different order does not fail: it returns confident, wrong numbers. So the
feature names travel with the weights and are checked on load, and a mismatch
raises rather than scores.

The artifact is a single file that goes to R2, is loaded by the predictor
replicas, and is loaded by the serving endpoint. One file, three consumers,
one version.

## On pickle, and the trust boundary it creates

Unpickling executes code, so this is only safe because the artifact is written
by our own training job into our own private bucket and read back by our own
services. Nothing user-supplied is ever unpickled, and no path here accepts an
artifact from a request.

The boundary is real rather than theoretical: a leaked R2 write key would let
someone replace the model with a payload that runs inside the serving process
the next time it cold-starts. That is worth knowing rather than discovering.
Two things follow. The R2 token used by serving should be read-only, separate
from the one the training job writes with. And if this ever loads a model it
did not produce, the format should change to `skops`, scikit-learn's own
schema-validated serializer, which reconstructs only declared estimator types
instead of running arbitrary opcodes.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aether.ml.features import FEATURE_NAMES, compute_features
from aether.ml.session import SessionState

ARTIFACT_VERSION = 1


@dataclass
class ModelArtifact:
    """A fitted classifier plus everything needed to use it correctly."""

    estimator: Any
    feature_names: tuple[str, ...]
    metrics: dict = field(default_factory=dict)
    trained_at: float = 0.0
    trained_on_events: int = 0
    version: int = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != FEATURE_NAMES:
            # Not a warning. A model fitted on a different feature order will
            # happily score garbage with high confidence.
            raise ValueError(
                "model was trained on different features than this build computes.\n"
                f"  trained: {list(self.feature_names)}\n"
                f"  current: {list(FEATURE_NAMES)}"
            )

    # -- predicting --------------------------------------------------------

    def probability(self, state: SessionState) -> float:
        """P(this session abandons its cart), given what is known so far."""
        return float(self.probabilities(compute_features(state).reshape(1, -1))[0])

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        """Abandonment probability for a batch of feature vectors."""
        return self.estimator.predict_proba(X)[:, 1]

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        path.write_bytes(payload)

        # A sidecar of the metrics, readable without unpickling anything. The
        # dashboard's model card reads this, and so can a human.
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "version": self.version,
                    "trained_at": self.trained_at,
                    "trained_on_events": self.trained_on_events,
                    "features": list(self.feature_names),
                    "metrics": self.metrics,
                },
                indent=2,
            )
        )
        return len(payload)

    @classmethod
    def load(cls, path: Path) -> ModelArtifact:
        artifact = pickle.loads(Path(path).read_bytes())
        if not isinstance(artifact, cls):
            raise ValueError(f"{path} does not contain a model artifact")
        return artifact

    @classmethod
    def from_bytes(cls, data: bytes) -> ModelArtifact:
        """Load from object storage, without touching the filesystem.

        How the serving endpoint gets the model: read the object from R2 at
        cold start and unpickle it in memory.
        """
        artifact = pickle.loads(data)
        if not isinstance(artifact, cls):
            raise ValueError("bytes do not contain a model artifact")
        return artifact
