"""Loading a model, in whichever of the two formats it was written.

There are two on purpose, and the suffix chooses.

`.npz` is what serving reads: arrays and a JSON header, no scikit-learn, no
pickle. `.pkl` is the scikit-learn object itself, which is what retraining and
analysis want because it still knows how to fit and to explain itself.

Serving should always be pointed at the `.npz`. Reading a `.pkl` works and is
useful on a laptop, but it drags scikit-learn into the process and unpickles a
file that may have come from a bucket, and neither belongs in a container that
answers requests.
"""

from __future__ import annotations

from pathlib import Path

NUMPY_SUFFIX = ".npz"
PICKLE_SUFFIX = ".pkl"


def model_from_bytes(data: bytes, name: str):
    """Load a model from bytes, choosing the format by the object's name."""
    if name.endswith(NUMPY_SUFFIX):
        from aether.ml.export import TreeEnsemble

        return TreeEnsemble.from_bytes(data)
    if name.endswith(PICKLE_SUFFIX):
        from aether.ml.model import ModelArtifact

        return ModelArtifact.from_bytes(data)
    raise ValueError(
        f"cannot tell the format of {name!r}: expected "
        f"{NUMPY_SUFFIX} (serving) or {PICKLE_SUFFIX} (training)"
    )


def load_model(path: Path):
    path = Path(path)
    return model_from_bytes(path.read_bytes(), path.name)
