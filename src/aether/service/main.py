"""Run the query service.

    uv run python -m aether.service --index r2://aether/idx100k --model data/model.pkl

Configuration is environment first, flags second, because a container gets
environment and a laptop gets flags, and the same image has to serve both.
Cloud Run supplies `PORT`, so that is the default when it is set.
"""

from __future__ import annotations

import argparse
import os

from aether.env import load_dotenv
from aether.service.state import (
    DEFAULT_INDEX_URI,
    DEFAULT_MODEL_URI,
    INDEX_URI_ENV,
    MODEL_URI_ENV,
    ServiceState,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.service", description="Serve search and prediction over HTTP."
    )
    parser.add_argument("--index", default=None, help="index URI, e.g. r2://aether/idx")
    parser.add_argument("--model", default=None, help="model object URI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv()

    if args.index:
        os.environ[INDEX_URI_ENV] = args.index
    if args.model:
        os.environ[MODEL_URI_ENV] = args.model

    index_uri = os.environ.get(INDEX_URI_ENV, DEFAULT_INDEX_URI)
    model_uri = os.environ.get(MODEL_URI_ENV, DEFAULT_MODEL_URI)

    import uvicorn

    from aether.service.state import state_from_env

    # Loaded here rather than on the first request so a misconfiguration
    # shows up in the startup log instead of in a user's first query.
    state = state_from_env()
    print(f"aether.service  index={index_uri}  model={model_uri}")
    print(f"  index   {state.index_summary()}")
    print(f"  model   {'loaded' if state.model else state.model_error}")
    print(f"  serving http://{args.host}:{args.port}")

    if args.reload:
        uvicorn.run("aether.service.app:app", host=args.host, port=args.port, reload=True)
    else:
        from aether.service.app import create_app

        uvicorn.run(create_app(state), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
