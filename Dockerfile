# The query service, and nothing else.
#
# What is deliberately absent matters as much as what is here. There is no
# Kafka client, because a scale-to-zero container cannot be a consumer, and
# librdkafka would add tens of megabytes to an image whose job is to start
# fast. There is no raw data, because the CSV is input to a build step and is
# never deployed. The index stays in object storage: this image ships code.
#
#   docker build -t aether .
#   docker run -p 8000:8000 --env-file .env \
#     -e AETHER_INDEX=r2://aether/idx100k \
#     -e AETHER_MODEL=r2://aether/models/model.pkl aether

FROM python:3.13-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Dependencies first, so a code change does not reinstall scikit-learn.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --prefix=/install ".[s3,ml,service]"


FROM python:3.13-slim

# Unbuffered so Cloud Run's log tail is not a surprise, and no .pyc writes
# because the filesystem is throwaway.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

COPY --from=build /install /usr/local

# Nothing here needs to write anything, so it should not be able to.
RUN useradd --create-home --uid 10001 aether
USER aether
WORKDIR /home/aether

EXPOSE 8000
# Shell form, because Cloud Run supplies PORT at runtime and it has to expand.
CMD exec python -m aether.service --host 0.0.0.0 --port ${PORT}
