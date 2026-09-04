.PHONY: help test up down logs bench clean

help:
	@echo "make test    run the suite (MinIO tests skip if it is not running)"
	@echo "make up      start MinIO and create the bucket"
	@echo "make down    stop MinIO"
	@echo "make bench   build a segment from the fixture and search it"

test:
	uv run pytest

up:
	docker compose up -d --wait
	@echo "MinIO on http://localhost:9000, console on http://localhost:9001"

down:
	docker compose down

logs:
	docker compose logs -f minio

bench:
	uv run python -m aether.index.build tests/fixtures/rees46_sample.csv data/fixture.seg
	@echo
	uv run python -m aether.index.search data/fixture.seg "samsung smartphone"

clean:
	rm -rf .pytest_cache data/*.seg
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
