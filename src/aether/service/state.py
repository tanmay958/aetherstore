"""What the service loads once and shares across every request.

The service holds no data of its own. An index lives in object storage and a
model is a file in the same bucket, so a fresh instance can answer a query
about a million documents having downloaded a few kilobytes. That is the
property that makes it deployable on something that scales to zero, and the
reason several instances behind a load balancer need no coordination at all.

What is held is a cache, and it is safe to hold because segments are
immutable. A segment's footer and hotcache are read once and kept for the
life of the process; they cannot go stale, because the object they describe
can never change. Publishing new data writes a *new* segment and a new
manifest, which `refresh()` picks up.

## Measuring what a query costs

The whole argument of this engine is about requests and bytes, so the service
can report exactly what a query spent. That is harder than it looks: reads
pass through one shared `CountingStore`, and concurrent requests would land
in each other's totals.

Rather than thread a request-scoped counter through the coordinator's own
fan-out threads, cost reporting is opt-in. `?explain=1` takes a lock and
reads the counters before and after, so the number is exactly this query's.
Plain queries take no lock and run fully concurrently. Measurement is a demo
and debugging affordance, and it should not be allowed to shape the hot path.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from aether.index.coordinator import Coordinator
from aether.index.manifest import DEFAULT_MANIFEST_KEY
from aether.storage.base import ObjectStore
from aether.storage.counting import CountingStore, ReadStats

# Where the index and the model live, overridable so the same image serves a
# local directory in a test and an R2 bucket in production.
INDEX_URI_ENV = "AETHER_INDEX"
MODEL_URI_ENV = "AETHER_MODEL"
DEFAULT_INDEX_URI = "data/idx1m"
DEFAULT_MODEL_URI = "data/model.pkl"


@dataclass(frozen=True)
class Cost:
    """What one query spent in object storage."""

    requests: int
    bytes_read: int

    @classmethod
    def of(cls, stats: ReadStats) -> Cost:
        return cls(stats.requests, stats.bytes_read)


class ServiceState:
    """Index and model, loaded once per process."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        manifest_key: str = DEFAULT_MANIFEST_KEY,
        model_uri: str | None = DEFAULT_MODEL_URI,
        max_workers: int = 16,
    ) -> None:
        # Counting wraps the store rather than the coordinator so that every
        # read is seen, including the ones the model load makes.
        self.store = CountingStore(store)
        self.manifest_key = manifest_key
        self.model_uri = model_uri
        self.coordinator = Coordinator(
            self.store, manifest_key=manifest_key, max_workers=max_workers
        )
        self._measuring = threading.Lock()
        self.started_at = time.time()

        self.model = None
        self.model_error: str | None = None
        self.index_error: str | None = None

    # -- loading -----------------------------------------------------------

    def load(self) -> None:
        """Open the index and the model, recording rather than raising.

        A failure here must not stop the process from starting. A service that
        refuses to boot without a model cannot serve search either, and an
        instance that exits on startup is far harder to diagnose than one
        that answers `/health` with the reason it is degraded.
        """
        try:
            self.coordinator.refresh()
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            self.index_error = f"{type(error).__name__}: {error}"

        if not self.model_uri:
            self.model_error = "no model configured"
            return
        try:
            from aether.ml.model import ModelArtifact
            from aether.storage.factory import open_object

            # Read straight into memory: the artifact never touches a
            # filesystem, because a scale-to-zero container may not have a
            # writable one and copying it through disk buys nothing.
            store, key = open_object(self.model_uri)
            self.model = ModelArtifact.from_bytes(
                store.get_range(key, 0, store.size(key))
            )
        except Exception as error:  # noqa: BLE001
            self.model_error = f"{type(error).__name__}: {error}"

    # -- measurement -------------------------------------------------------

    @contextmanager
    def measured(self) -> Iterator[list[Cost]]:
        """Exact storage cost for the work done inside.

        Serialised, because the counters are shared. Only `?explain=1` asks
        for this, so the ordinary query path never waits on it.
        """
        with self._measuring:
            box: list[Cost] = []
            with self.store.measure() as delta:
                yield box
            box.append(Cost.of(delta))

    # -- reporting ---------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self.index_error is None

    @property
    def status(self) -> str:
        """`empty` is deliberately distinct from `ok`.

        An absent manifest is a valid empty index rather than an error, which
        is right for a service that may start before its first ingest. But a
        service pointed at the wrong bucket looks exactly the same, and
        reporting both as healthy turns a misconfiguration into a silent one.
        """
        if self.index_error:
            return "degraded"
        return "ok" if self.coordinator.manifest.docs else "empty"

    def index_summary(self) -> dict:
        if self.index_error:
            return {"error": self.index_error}
        manifest = self.coordinator.manifest
        return {
            "documents": manifest.docs,
            "segments": len(manifest.segments),
            "bytes": manifest.bytes,
            "manifest_key": self.manifest_key,
        }

    def model_summary(self) -> dict:
        if self.model is None:
            return {"loaded": False, "error": self.model_error}
        return {
            "loaded": True,
            "version": self.model.version,
            "trained_at": self.model.trained_at,
            "trained_on_events": self.model.trained_on_events,
            "features": list(self.model.feature_names),
            "metrics": self.model.metrics,
        }


def state_from_env() -> ServiceState:
    """Build the service state from environment, the only configuration."""
    from aether.env import load_dotenv
    from aether.storage.factory import open_store


    load_dotenv()
    uri = os.environ.get(INDEX_URI_ENV, DEFAULT_INDEX_URI)
    model_uri = os.environ.get(MODEL_URI_ENV, DEFAULT_MODEL_URI)
    state = ServiceState(open_store(uri), model_uri=model_uri or None)
    state.load()
    return state
