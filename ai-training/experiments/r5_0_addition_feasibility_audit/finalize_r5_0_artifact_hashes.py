from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


files = sorted(path for path in HERE.iterdir() if path.is_file() and path.name != "artifact_hashes.json")
entries = [
    {"relative_path": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}
    for path in files
]
manifest = {
    "schema_version": 2,
    "artifact_count": len(entries),
    "artifacts": entries,
    "self_excluded": "artifact_hashes.json cannot contain a stable hash of itself",
    "verification": "PENDING_REOPEN",
    "failures": [],
}
target = HERE / "artifact_hashes.json"
target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
reopened = json.loads(target.read_text(encoding="utf-8"))
failures = []
for entry in reopened["artifacts"]:
    path = HERE / entry["relative_path"]
    if not path.is_file() or path.stat().st_size != entry["byte_size"] or sha256(path) != entry["sha256"]:
        failures.append(entry["relative_path"])
manifest["verification"] = "HASH_AUDIT_PASS" if not failures else "HASH_AUDIT_FAIL"
manifest["failures"] = failures
target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"artifact_count": len(entries), "verification": manifest["verification"], "failures": failures}))
