"""API keys for remote judges, read without printing or exporting them.

    from natsteer.apikeys import get_key
    key = get_key("OPENROUTER_API_KEY")      # Secret; str(key) and repr(key) are masked
    headers = key.auth_header()              # {"Authorization": "Bearer ..."}; send it, never log it

Lookup order: the process environment (if the caller set the name there), then a `NAME=value` or
`export NAME=value` line in the key files (default `~/.bashrc`; override with NATSTEER_KEY_FILES, a
colon-separated list). The file is parsed as text and never run as shell code, so the key is not placed in the
environment of this process or its children. Values that need shell expansion (`$`, backticks) are refused.
"""

import os
import re
import stat
import warnings
from pathlib import Path


class Secret:
    """A key that does not show itself in prints, logs, tracebacks or saved json."""

    __slots__ = ("_value", "name")

    def __init__(self, name: str, value: str):
        self.name, self._value = name, value

    def reveal(self) -> str:
        return self._value

    def auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self._value}"}

    def __repr__(self) -> str:
        return f"Secret({self.name}, {len(self._value)} chars)"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("Secret cannot be pickled")


def _key_files() -> list[Path]:
    env = os.environ.get("NATSTEER_KEY_FILES")
    return [Path(p).expanduser() for p in env.split(":")] if env else [Path.home() / ".bashrc"]


def _read_assignment(path: Path, name: str) -> str | None:
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}=(.*?)\s*(?:#.*)?$")
    value = None
    for line in path.read_text(errors="replace").splitlines():
        m = pat.match(line)
        if m:
            value = m.group(1)  # the last assignment wins, as in the shell
    if value is None:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if any(c in value for c in "$`") or not value:
        raise ValueError(f"{name} in {path} needs shell expansion or is empty; put the literal key there")
    return value


def get_key(name: str) -> Secret:
    if os.environ.get(name):
        return Secret(name, os.environ[name])
    for path in _key_files():
        if not path.is_file():
            continue
        if path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH):
            warnings.warn(f"{path} is readable by other users (chmod 600 {path} to fix)", stacklevel=2)
        value = _read_assignment(path, name)
        if value:
            return Secret(name, value)
    raise KeyError(f"{name} not found in the environment or in {', '.join(map(str, _key_files()))}")
