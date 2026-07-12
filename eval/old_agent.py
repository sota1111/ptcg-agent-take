"""Reference an *old version* of the rule-based agent for new-vs-old regression (SOT-1636).

The regression harness (``eval/regression.py``) needs to play the current
(working-tree) agent head to head against a **previous version** of itself so a
rule change can be shown not to have regressed against the prior agent (guarding
against over-fitting to Random). Both agents must run in the *same* match — and
therefore in the same interpreter — but the ``agents`` package uses absolute
imports (``from agents.base import ...``), so a second copy cannot simply be
imported under its own name without clashing with the live ``agents`` package.

This module establishes the "旧版エージェント参照" mechanism the issue asks for
(git tag / module copy): it materialises the ``agents/`` tree from any git ref
via ``git archive`` into a temp dir, renames the top-level package to a unique
``agents_<sha8>`` and rewrites the package's own ``import agents…`` statements to
match, so the historical agent imports cleanly **side by side** with the current
one. The old agent runs against the *current* engine (``cg``) — the engine is the
fixed evaluation substrate; only the agent policy is under regression.

Design notes / constraints:
* The rewrite only touches ``from|import agents`` at the start of a logical line
  (the only shape the package uses); string/comment occurrences of the word are
  left alone and are harmless for execution.
* Materialisation happens in the *parent* process; the temp ``import_root`` is
  passed to arena workers (which prepend it to ``sys.path`` and import lazily), so
  it works under both fork and spawn start methods.
* The caller owns cleanup — use :func:`materialize_agents_ref` inside a
  ``try/finally`` (or the :func:`agents_ref` context manager) and remove the
  returned ``import_root``.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Matches an import of the ``agents`` package at the start of a logical line:
#   from agents.base import X   /   from agents import damage   /   import agents
# Captures the leading ``from ``/``import `` keyword so it can be preserved.
_IMPORT_RE = re.compile(r"(?m)^(\s*(?:from|import)\s+)agents(\b)")


def _run_git(args: list[str]) -> str:
    """Run ``git`` in the repo root and return stdout (raises on failure)."""
    out = subprocess.run(
        ["git", "-C", REPO, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def resolve_ref(git_ref: str) -> str:
    """Resolve ``git_ref`` (tag / branch / sha / ``HEAD``) to a full commit sha."""
    return _run_git(["rev-parse", git_ref]).strip()


def _rewrite_package(pkg_dir: str, pkg_name: str) -> None:
    """Rewrite ``import agents…`` → ``import <pkg_name>…`` in every ``.py`` under ``pkg_dir``."""
    for root, _dirs, files in os.walk(pkg_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            new = _IMPORT_RE.sub(rf"\1{pkg_name}\2", src)
            if new != src:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new)


def materialize_agents_ref(git_ref: str, workdir: Optional[str] = None) -> tuple[str, str]:
    """Materialise the ``agents/`` package from ``git_ref`` as an importable copy.

    ``git archive <ref> agents`` is extracted into a fresh temp dir, the package
    is renamed to a unique ``agents_<sha8>`` and its internal ``agents`` imports
    are rewritten to match. Returns ``(import_root, package_name)``: prepend
    ``import_root`` to ``sys.path`` and ``import <package_name>.rule_based``.

    Raises ``subprocess.CalledProcessError`` if the ref has no ``agents/`` tree
    (e.g. a commit predating the package). The caller owns ``import_root`` and
    must remove it when done.
    """
    sha = resolve_ref(git_ref)
    pkg_name = f"agents_{sha[:8]}"
    import_root = tempfile.mkdtemp(prefix="oldagent_", dir=workdir)
    try:
        # `git archive <ref> agents` -> tar stream -> extract into import_root/agents
        archive = _run_git_bytes(["archive", sha, "agents"])
        _extract_tar(archive, import_root)
        src_dir = os.path.join(import_root, "agents")
        dst_dir = os.path.join(import_root, pkg_name)
        os.rename(src_dir, dst_dir)
        # Drop the historical test modules — they are not needed to run the agent
        # and may import fixtures absent from the current tree.
        for fn in os.listdir(dst_dir):
            if fn.startswith("test_") and fn.endswith(".py"):
                os.remove(os.path.join(dst_dir, fn))
        _rewrite_package(dst_dir, pkg_name)
    except Exception:
        shutil.rmtree(import_root, ignore_errors=True)
        raise
    return import_root, pkg_name


def _run_git_bytes(args: list[str]) -> bytes:
    """Like :func:`_run_git` but returns raw bytes (for the tar archive stream)."""
    out = subprocess.run(
        ["git", "-C", REPO, *args],
        check=True,
        capture_output=True,
    )
    return out.stdout


def _extract_tar(blob: bytes, dest: str) -> None:
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(dest)


@contextlib.contextmanager
def agents_ref(git_ref: str) -> Iterator[tuple[str, str]]:
    """Context manager wrapping :func:`materialize_agents_ref` with cleanup."""
    import_root, pkg_name = materialize_agents_ref(git_ref)
    try:
        yield import_root, pkg_name
    finally:
        shutil.rmtree(import_root, ignore_errors=True)


def load_rule_based_class(import_root: str, pkg_name: str):
    """Import and return the historical ``RuleBasedAgent`` class (arena-worker side)."""
    import importlib

    if import_root not in sys.path:
        sys.path.insert(0, import_root)
    mod = importlib.import_module(f"{pkg_name}.rule_based")
    return mod.RuleBasedAgent
