.PHONY: help test up down logs bench clean

help:
	@echo "make test    run the suite (MinIO tests skip if it is not running)"
	@echo "make up      start MinIO and create the bucket"
	@echo "make down    stop MinIO"
	@echo "make bench   build a segment from the fixture and search it"

test:
	uv run pytest

# Two steps on purpose. `--wait` considers a container that exited a failure,
# and createtopic/createbucket are one-shot jobs that exit 0 by design, so
# waiting on them reports a crash that did not happen. Wait on the long-lived
# services, then let the init jobs run.
up:
	docker compose up -d --wait redpanda minio
	docker compose up -d
	@echo "MinIO      http://localhost:9000   console http://localhost:9001"
	@echo "Redpanda   localhost:19092         console http://localhost:8080"

down:
	docker compose down

logs:
	docker compose logs -f

topic:
	docker compose exec redpanda rpk topic describe clickstream --brokers redpanda:9092

lag:
	docker compose exec redpanda rpk group describe aether-indexers --brokers redpanda:9092

bench:
	uv run python -m aether.index.build tests/fixtures/rees46_sample.csv data/fixture.seg
	@echo
	uv run python -m aether.index.search data/fixture.seg "samsung smartphone"

clean:
	rm -rf .pytest_cache data/*.seg
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
