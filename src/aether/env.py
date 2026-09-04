"""Load a .env file into the environment.

Credentials belong in the environment, not in code or on a command line where
they land in shell history. A .env file is the usual way to get them there
during development, and this reads one without adding a dependency.

Two rules make it safe to call unconditionally. A variable already present in
the environment is never overwritten, so an explicit `export` or a value
injected by CI always wins over the file. And a missing file is simply nothing
to do, so deployments that inject real secrets some other way are unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATH = Path(".env")


def load_dotenv(path: Path | str = DEFAULT_PATH, *, override: bool = False) -> int:
    """Read KEY=value lines into os.environ. Returns how many were set.

    Understands blank lines, `#` comments, an optional leading `export`, and
    values wrapped in single or double quotes. Anything else is left alone
    rather than guessed at.
    """
    path = Path(path)
    if not path.is_file():
        return 0

    loaded = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, separator, value = line.partition("=")
        if not separator:
            continue

        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded
