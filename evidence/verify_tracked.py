#!/usr/bin/env python3
"""Verify evidence that is expected to work from a fresh Git clone."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [ROOT / entry.decode() for entry in result.stdout.split(b"\0") if entry]


def main():
    failures = []
    tracked = tracked_files()

    for path in tracked:
        if not path.is_file():
            failures.append("missing tracked file: %s" % path.relative_to(ROOT))
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                failures.append("invalid JSON %s: %s" % (path.relative_to(ROOT), exc))

    manifest = ROOT / "evidence" / "MANIFEST.sha256"
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append("invalid manifest line %d" % number)
            continue
        target = manifest.parent / relative
        if not target.is_file():
            failures.append("manifest target missing: evidence/%s" % relative)
        elif sha256(target) != expected:
            failures.append("manifest hash mismatch: evidence/%s" % relative)

    if failures:
        for failure in failures:
            print("FAIL: %s" % failure)
        return 1

    json_count = sum(path.suffix == ".json" for path in tracked)
    print("PASS: %d tracked files present; %d JSON files parse; evidence manifest matches"
          % (len(tracked), json_count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
