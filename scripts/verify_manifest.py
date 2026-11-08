"""Verify the released source/input/reference files against their hashes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "MANIFEST.json").read_text())
failures = []
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        failures.append(item["path"])
if failures:
    raise SystemExit("Changed or missing files: " + ", ".join(failures))
print("Verified", len(manifest["files"]), "release-candidate files.")
