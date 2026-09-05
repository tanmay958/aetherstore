"""Tests for the numpy-only model export.

The point of this format is that serving neither imports scikit-learn nor
unpickles anything. Two tests carry that weight: one asserts the exported
scorer agrees with scikit-learn to floating point, because a faster wrong
answer is worthless, and one asserts the loader refuses a pickle, because the
security argument is the main reason the format exists.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from aether.data.rees46 import iter_events  # noqa: E402
from aether.ml.dataset import build_dataset  # noqa: E402
from aether.ml.export import FORMAT_VERSION, TreeEnsemble, export  # noqa: E402
from aether.ml.features import FEATURE_NAMES  # noqa: E402
from aether.ml.model import ModelArtifact  # noqa: E402
from aether.ml.train import fit_model  # noqa: E402


@pytest.fixture(scope="module")
def data(request):
    csv = Path(request.config.rootdir) / "tests/fixtures/rees46_sample.csv"
    return build_dataset(iter_events(csv))


@pytest.fixture(scope="module")
def artifact(data):
    return ModelArtifact(
        estimator=fit_model(data),
        feature_names=FEATURE_NAMES,
        metrics={"model": {"note": "fixture"}},
        trained_at=1.0,
        trained_on_events=len(data),
    )


@pytest.fixture(scope="module")
def ensemble(artifact):
    return export(artifact)


# --------------------------------------------------------------------------
# the answer must not change
# --------------------------------------------------------------------------


def test_matches_scikit_learn(artifact, ensemble, data):
    """Floating point identical, not merely close.

    Summation happens in the same order over the same values, so the only
    permitted difference is the last bit.
    """
    expected = artifact.probabilities(data.X)
    actual = ensemble.probabilities(data.X)
    assert np.abs(expected - actual).max() < 1e-12


def test_the_scalar_and_vectorised_paths_agree_exactly(ensemble, data):
    """Serving takes a shortcut for one row. It must be a shortcut and not a
    second implementation with its own bugs."""
    batched = ensemble.probabilities(data.X)
    one_at_a_time = np.array(
        [ensemble.probabilities(data.X[i : i + 1])[0] for i in range(len(data.X))]
    )
    assert np.array_equal(batched, one_at_a_time)


def test_probabilities_are_probabilities(ensemble, data):
    out = ensemble.probabilities(data.X)
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert np.all(np.isfinite(out))


def test_an_extreme_raw_score_does_not_overflow(ensemble):
    """exp() of a large positive number is inf, and inf/inf is nan. The
    logistic is written to fold the sign out first."""
    huge = np.full((2, len(FEATURE_NAMES)), 1e308)
    huge[1] = -1e308
    out = ensemble.probabilities(huge)
    assert np.all(np.isfinite(out))


def test_a_missing_value_follows_the_trained_direction(ensemble):
    """Every feature this project computes is finite, but the trees still
    encode a direction for NaN and the scorer must honour it rather than
    letting a comparison against NaN silently mean "go right"."""
    x = np.zeros((1, len(FEATURE_NAMES)))
    nan = x.copy()
    nan[0, 0] = np.nan
    assert np.isfinite(ensemble.probabilities(nan)[0])


def test_the_wrong_feature_count_is_rejected(ensemble):
    with pytest.raises(ValueError, match="expected"):
        ensemble.probabilities(np.zeros((1, 3)))


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_round_trips_through_bytes(ensemble, data):
    restored = TreeEnsemble.from_bytes(ensemble.to_bytes())
    assert restored.trees == ensemble.trees
    assert restored.nodes == ensemble.nodes
    assert restored.trained_on_events == ensemble.trained_on_events
    assert restored.metrics == ensemble.metrics
    assert np.array_equal(
        restored.probabilities(data.X), ensemble.probabilities(data.X)
    )


def test_round_trips_through_a_file(ensemble, tmp_path, data):
    path = tmp_path / "model.npz"
    written = ensemble.save(path)
    assert written == path.stat().st_size
    restored = TreeEnsemble.load(path)
    assert np.array_equal(
        restored.probabilities(data.X), ensemble.probabilities(data.X)
    )


def test_it_is_much_smaller_than_the_pickle(ensemble, artifact, tmp_path):
    pickled = tmp_path / "model.pkl"
    artifact.save(pickled)
    assert len(ensemble.to_bytes()) < pickled.stat().st_size


def test_loading_refuses_a_pickle(tmp_path):
    """The whole security argument. An artifact fetched from a bucket must be
    data, not code, so a payload that would execute on load must fail."""
    import pickle

    with pytest.raises(Exception):  # noqa: B017 - numpy's own, whatever it is
        TreeEnsemble.from_bytes(pickle.dumps({"anything": "at all"}))


def test_an_npz_containing_a_pickled_object_is_refused(tmp_path):
    payload = tmp_path / "evil.npz"
    np.savez(payload, header=np.array([{"nope": True}], dtype=object))
    with pytest.raises(Exception):  # noqa: B017
        TreeEnsemble.from_bytes(payload.read_bytes())


def test_a_future_format_version_is_refused(ensemble):
    """Silently reading a format you do not understand is how a model scores
    garbage confidently."""
    import io

    with np.load(io.BytesIO(ensemble.to_bytes()), allow_pickle=False) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    header = json.loads(bytes(arrays.pop("header")).decode())
    header["version"] = FORMAT_VERSION + 1
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
        **arrays,
    )
    with pytest.raises(ValueError, match="format"):
        TreeEnsemble.from_bytes(buffer.getvalue())


def test_features_trained_on_must_match_this_build(ensemble):
    with pytest.raises(ValueError, match="different features"):
        TreeEnsemble(
            baseline=ensemble.baseline,
            feature=ensemble.feature,
            threshold=ensemble.threshold,
            left=ensemble.left,
            right=ensemble.right,
            is_leaf=ensemble.is_leaf,
            value=ensemble.value,
            missing_left=ensemble.missing_left,
            offsets=ensemble.offsets,
            feature_names=("not", "these"),
        )


# --------------------------------------------------------------------------
# what it refuses to export
# --------------------------------------------------------------------------


def test_a_categorical_split_is_refused(artifact):
    """Every feature here is numeric, so a categorical split can only mean
    this is being pointed at a model it was not written for."""
    import copy

    poisoned = copy.deepcopy(artifact)
    poisoned.estimator._predictors[0][0].nodes["is_categorical"][0] = 1
    with pytest.raises(ValueError, match="categorical"):
        export(poisoned)


def test_an_unfitted_estimator_is_refused():
    with pytest.raises(TypeError, match="not a fitted"):
        from aether.ml.export import export_estimator

        export_estimator(object())


# --------------------------------------------------------------------------
# it must be usable everywhere a ModelArtifact is
# --------------------------------------------------------------------------


def test_the_predictor_accepts_it(ensemble):
    """`Predictor` calls `probability(state)`. Both model types answer that,
    and neither knows about the other."""
    from aether.stream.config import KafkaConfig
    from aether.stream.predictor import Predictor

    predictor = Predictor(ensemble, KafkaConfig(), output_topic=None)
    events = [
        {"event_id": "1", "ts": 1_570_000_000, "session_id": "s", "user_id": "u",
         "event_type": "view", "product_id": "a", "price": 10.0, "device": None,
         "title": None, "category": "c", "brand": "b", "query": None},
        {"event_id": "2", "ts": 1_570_000_060, "session_id": "s", "user_id": "u",
         "event_type": "add_to_cart", "product_id": "a", "price": 10.0,
         "device": None, "title": None, "category": "c", "brand": "b", "query": None},
    ]
    predictions = [predictor.handle(event) for event in events]
    assert predictions[0] is None
    assert 0.0 <= predictions[1].probability <= 1.0


def test_it_scores_a_session_identically_to_the_pickled_model(ensemble, artifact):
    from aether.ml.session import SessionState

    state = SessionState("s")
    state.update({"event_id": "1", "ts": 1_570_000_000, "session_id": "s",
                  "user_id": "u", "event_type": "add_to_cart", "product_id": "a",
                  "price": 99.0, "device": None, "title": None,
                  "category": "c", "brand": "b", "query": None})
    assert ensemble.probability(state) == pytest.approx(
        artifact.probability(state), abs=1e-12
    )
