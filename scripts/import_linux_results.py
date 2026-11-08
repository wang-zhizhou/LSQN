"""Archive a completed Linux run without altering numerical JSON tokens.

Usage from the project root:
    python scripts/import_linux_results.py /path/to/linux-paper-results
    python scripts/import_linux_results.py --verify

Only JSON inputs are imported. Raw run files are individually gzipped; metadata
stays readable. An existing archive is verified, never replaced by another batch.
This script uses only the Python standard library and never runs a solver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile


SUITES = ("main", "public", "scaling")
METADATA = ("analysis.json", "environment.json", "protocol.json", "summary.json")
EXPECTED_COUNTS = {"main": 960, "public": 24, "scaling": 36}
JSON_STRING = re.compile(rb'"(?:[^"\\]|\\.)*"')
HOME_PREFIX = re.compile(r"/(?:home|Users)/([^/\s\"'<>\\]+)")
HOST_KEYS = {"hostname", "host_name", "nodename", "computer_name", "computername", "platform_node"}
SCHEMA = "lsqn-published-linux-archive-v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "benchmarks" / "linux-intel-2026-10-25"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def numeric_structure_digest(data: bytes) -> str:
    """Hash every byte outside JSON strings, including exact number spellings."""
    return digest(JSON_STRING.sub(b'""', data))


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def find_hosts(value, parent=""):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            is_host = normalized in HOST_KEYS or (normalized == "node" and parent in {"uname", "platform", "system"})
            if is_host and isinstance(item, str) and item:
                yield item
            yield from find_hosts(item, normalized)
    elif isinstance(value, list):
        for item in value:
            yield from find_hosts(item, parent)


class Redactor:
    def __init__(self, metadata_objects, extra_hosts=()):
        self.hosts = set(extra_hosts)
        self.users = set()
        for obj in metadata_objects:
            self.hosts.update(find_hosts(obj))
            for text in strings(obj):
                self.users.update(HOME_PREFIX.findall(text))
        self.host_patterns = [re.compile(r"(?<![\w.-])" + re.escape(host) + r"(?![\w.-])") for host in sorted(self.hosts)]
        self.user_patterns = [re.compile(r"(?<![\w.-])" + re.escape(user) + r"(?![\w.-])") for user in sorted(self.users)]

    def clean_string(self, text):
        text = HOME_PREFIX.sub("<HOME>", text)
        for pattern in self.host_patterns:
            text = pattern.sub("<HOSTNAME>", text)
        return text

    def transform(self, data):
        counts = Counter()

        def replace(match):
            token = match.group()
            original = json.loads(token)
            cleaned = self.clean_string(original)
            if original == cleaned:
                return token
            counts["changed_string_tokens"] += 1
            counts["home_prefix_replacements"] += len(HOME_PREFIX.findall(original))
            counts["hostname_replacements"] += sum(len(p.findall(original)) for p in self.host_patterns)
            return json.dumps(cleaned, ensure_ascii=True).encode()

        # Most raw records contain no machine-specific strings. Leave their
        # original bytes untouched, including whitespace and floating literals.
        if b"/home/" not in data and b"/Users/" not in data and b"\\/" not in data and b"\\u" not in data and not self.hosts:
            return data, counts
        return JSON_STRING.sub(replace, data), counts

    def assert_private_values_removed(self, value):
        for text in strings(value):
            if HOME_PREFIX.search(text) or any(p.search(text) for p in self.host_patterns):
                raise ValueError("A personal home prefix or detected hostname remains in a JSON string")
            if any(p.search(text) for p in self.user_patterns):
                raise ValueError("A detected home-directory username remains outside its redacted prefix")


def compress(data):
    stream = io.BytesIO()
    # GzipFile sets OS=255, omits the filename, and fixes MTIME=0. There is no
    # local filename or current clock embedded in the compressed file.
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return stream.getvalue()


def relative_path(text):
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ValueError("Manifest paths must be relative and must not traverse parents")
    return path


def inventory(source):
    selected = {"collection.json": False}
    counts = {}
    for suite in SUITES:
        for name in METADATA:
            selected[f"{suite}/{name}"] = False
        raw = sorted((source / suite).glob(f"{suite}_*.json"))
        counts[suite] = len(raw)
        for path in raw:
            selected[path.relative_to(source).as_posix()] = True
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Expected the complete 1020-run batch; found {counts}")
    for relative in selected:
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked input: {relative}")
    extra_json = {p.relative_to(source).as_posix() for p in source.rglob("*.json")} - selected.keys()
    if extra_json:
        raise ValueError(f"Unexpected JSON inputs require an explicit import decision: {sorted(extra_json)}")
    return sorted(selected.items()), counts


def file_key(records):
    return digest("".join(f"{r['source_path']}\0{r['source_sha256']}\n" for r in records).encode())


def verify_archive(destination):
    manifest = json.loads((destination / "provenance.json").read_bytes())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unrecognized archive manifest")
    expected = {"provenance.json", "README.md"}
    counts = Counter()
    byte_total = 0
    for record in manifest["files"]:
        relative_path(record["source_path"])
        relative = str(relative_path(record["published_path"]))
        path = destination / relative
        if path.is_symlink():
            raise ValueError(f"Symlinked archive input: {relative}")
        expected.add(relative)
        data = path.read_bytes()
        if digest(data) != record["published_sha256"]:
            raise ValueError(f"Published hash mismatch: {relative}")
        payload = gzip.decompress(data) if record["compression"] == "gzip" else data
        if record["compression"] == "gzip" and (data[3] != 0 or data[4:8] != b"\0\0\0\0" or data[9] != 255):
            raise ValueError(f"Non-deterministic gzip header: {relative}")
        if digest(payload) != record["sanitized_json_sha256"] or numeric_structure_digest(payload) != record["non_string_content_sha256"]:
            raise ValueError(f"JSON payload hash mismatch: {relative}")
        parsed = json.loads(payload)
        if any(HOME_PREFIX.search(text) for text in strings(parsed)):
            raise ValueError(f"Unredacted personal home prefix: {relative}")
        if record["kind"] == "raw_run":
            counts[record["suite"]] += 1
        byte_total += len(data)
    present = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    if present != expected:
        raise ValueError(f"Unexpected or missing archive files: {sorted(present ^ expected)}")
    if dict(counts) != EXPECTED_COUNTS or file_key(manifest["files"]) != manifest["source_batch_sha256"]:
        raise ValueError("Archive inventory does not match the manifest")
    readme = (destination / "README.md").read_bytes()
    if digest(readme) != manifest["archive_readme_sha256"]:
        raise ValueError("Archive README hash mismatch")
    return {"verified": True, "raw_runs": dict(counts), "imported_json_files": len(manifest["files"]),
            "published_payload_bytes": byte_total, "archive_bytes": byte_total + len(readme) + (destination / "provenance.json").stat().st_size,
            "source_batch_sha256": manifest["source_batch_sha256"], "privacy_check": manifest["privacy_check"]}


def archive(source, destination, extra_hosts=()):
    selected, counts = inventory(source)
    metadata_bytes = {rel: (source / rel).read_bytes() for rel, raw in selected if not raw}
    metadata = {rel: json.loads(data) for rel, data in metadata_bytes.items()}
    if metadata["collection.json"]["planned_solves"] != sum(counts.values()):
        raise ValueError("Collection planned_solves disagrees with the complete inventory")
    redactor = Redactor(metadata.values(), extra_hosts)
    if destination.exists() and any(destination.iterdir()):
        if not (destination / "provenance.json").is_file():
            raise ValueError("Destination is nonempty and lacks a completed manifest; refusing to overwrite")
        old = json.loads((destination / "provenance.json").read_bytes())
        expected = {row["source_path"]: row["source_sha256"] for row in old["files"]}
        if set(expected) != {rel for rel, _ in selected}:
            raise ValueError("Destination belongs to a different input inventory; refusing to overwrite")
        for rel, _ in selected:
            data = metadata_bytes.get(rel)
            if data is None:
                data = (source / rel).read_bytes()
            if digest(data) != expected[rel]:
                raise ValueError(f"Destination belongs to a different source batch: {rel}; refusing to overwrite")
        report = verify_archive(destination)
        report["unchanged_existing_archive"] = True
        return report
    destination.mkdir(parents=True, exist_ok=True)
    records, run_counts, changes = [], {}, Counter()
    source_bytes = published_bytes = 0
    with tempfile.TemporaryDirectory(prefix=".import-", dir=destination) as temporary:
        staging = Path(temporary)
        for index, (relative, is_raw) in enumerate(selected, 1):
            data = metadata_bytes.get(relative)
            if data is None:
                data = (source / relative).read_bytes()
            sanitized, changed = redactor.transform(data)
            parsed = json.loads(sanitized)
            redactor.assert_private_values_removed(parsed)
            non_string_hash = numeric_structure_digest(data)
            if sanitized != data and numeric_structure_digest(sanitized) != non_string_hash:
                raise ValueError(f"A non-string token changed: {relative}")
            published = compress(sanitized) if is_raw else sanitized
            if is_raw and gzip.decompress(published) != sanitized:
                raise ValueError(f"Gzip round-trip mismatch: {relative}")
            published_relative = relative + (".gz" if is_raw else "")
            target = staging / published_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(published)
            suite = relative.split("/")[0] if "/" in relative else None
            records.append({"source_path": relative, "published_path": published_relative,
                "kind": "raw_run" if is_raw else "metadata", "suite": suite,
                "source_sha256": digest(data), "source_bytes": len(data),
                "sanitized_json_sha256": digest(sanitized), "sanitized_json_bytes": len(sanitized),
                "published_sha256": digest(published), "published_bytes": len(published),
                "non_string_content_sha256": non_string_hash, "compression": "gzip" if is_raw else "none",
                "redactions": dict(changed)})
            changes.update(changed)
            source_bytes += len(data)
            published_bytes += len(published)
            if is_raw:
                stats = run_counts.setdefault(suite, {"runs": 0, "success": 0, "timeout": 0, "other_failure": 0, "failure_messages": {}})
                stats["runs"] += 1
                stats["success"] += bool(parsed["success"])
                stats["timeout"] += bool(parsed["timeout"])
                stats["other_failure"] += bool(not parsed["success"] and not parsed["timeout"])
                if not parsed["success"]:
                    message = parsed.get("message", "")
                    stats["failure_messages"][message] = stats["failure_messages"].get(message, 0) + 1
            if index % 100 == 0:
                print(f"Imported {index}/{len(selected)} JSON files", flush=True)
        for suite in SUITES:
            rows = metadata[f"{suite}/summary.json"]["runs"]
            if len(rows) != counts[suite] or sum(bool(row["success"]) for row in rows) != run_counts[suite]["success"]:
                raise ValueError(f"Raw records disagree with the original summary: {suite}")
        readme = ("# Linux Intel result archive\n\n"
            "This directory preserves all 1,020 runs from the completed Linux paper batch: "
            "960 main comparisons, 24 public-environment runs, and 36 scaling runs. "
            "Failures and timeouts are retained. No experiment was rerun during import.\n\n"
            "Raw records are individual `.json.gz` files. The original collection, summary, "
            "analysis, protocol, and environment JSON files are included without compression. "
            "Summary `file` entries retain their original names: append `.gz` when opening the archived raw record.\n\n"
            "Only personal machine identifiers in JSON strings are redacted: home-directory prefixes "
            "become `<HOME>` and detected hostname values become `<HOSTNAME>`. "
            "CPU, BLAS, compiler, source hashes, all numerical values, numeric spellings, "
            "iteration histories, policies, queries, timings, and failure statuses are preserved. "
            "No console logs, executable files, or native libraries are included.\n\n"
            "`provenance.json` maps original relative paths and SHA-256 hashes to published "
            "relative paths and SHA-256 hashes, with a separate hash of each uncompressed, sanitized "
            "JSON payload. Its non-string-content hash replaces each complete JSON string token "
            "with an empty string before hashing; all other bytes, including numeric tokens and "
            "structure, remain unchanged. Gzip uses level 9, MTIME 0, no filename, and OS 255. "
            "Deflate bytes may depend on the zlib version; the uncompressed payload hash is independent of that.\n\n"
            "From the repository root, verify this archive with:\n\n"
            "```sh\npython scripts/import_linux_results.py --verify\n```\n\n"
            "To reproduce the import from the original directory, pass its path as the positional "
            "argument. An existing archive is checked and left untouched; a different batch is refused. "
            "Original source hashes require access to the original private files; published hashes "
            "and all archived records can be verified without those files.\n").encode()
        manifest = {"schema": SCHEMA, "batch_id": "linux-intel-2026-10-25",
            "source_batch_sha256": file_key(records), "importer_path": "scripts/import_linux_results.py",
            "importer_sha256": digest(Path(__file__).read_bytes()), "source_raw_counts": counts,
            "run_status_counts": run_counts, "imported_json_files": len(records),
            "source_json_bytes": source_bytes, "published_payload_bytes": published_bytes,
            "transformation": {"scope": "Only JSON string tokens containing personal home prefixes or hostname values are changed; all non-string bytes are preserved.",
                "home_placeholder": "<HOME>", "hostname_placeholder": "<HOSTNAME>",
                "gzip": {"compression_level": 9, "mtime": 0, "filename": "", "os_byte": 255},
                "summary_file_references": "Original .json names are preserved; published raw files add .gz.",
                "excluded": ["console.log", "executables", "native libraries", ".DS_Store"]},
            "privacy_check": {"all_imported_json_strings_checked": True, "residual_home_prefixes": 0,
                "residual_detected_hostnames": 0, "residual_detected_usernames": 0,
                "detected_home_username_count": len(redactor.users), "detected_hostname_count": len(redactor.hosts),
                "redactions": dict(changes), "note": "Hostname discovery uses structured hostname fields and optional --hostname values; this is not a claim to identify arbitrary unlabelled personal names."},
            "archive_readme_sha256": digest(readme), "files": records}
        (staging / "README.md").write_bytes(readme)
        (staging / "provenance.json").write_bytes(json_bytes(manifest))
        verify_archive(staging)
        # The manifest is the final completion marker. No pre-existing file is
        # overwritten, including on interrupted or repeated imports.
        children = sorted(staging.iterdir(), key=lambda p: p.name == "provenance.json")
        for child in children:
            target = destination / child.name
            if target.exists():
                raise ValueError(f"Refusing to overwrite archive entry: {child.name}")
            child.rename(target)
    return verify_archive(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="Original completed Linux results directory (read only).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hostname", action="append", default=[], help="Additional known hostname to redact from JSON strings.")
    parser.add_argument("--verify", action="store_true", help="Verify the published archive without reading source files.")
    args = parser.parse_args()
    if args.verify:
        if args.source is not None:
            parser.error("--verify accepts no source; omit --verify to check/reimport the original batch")
        report = verify_archive(args.output)
    else:
        if args.source is None:
            parser.error("Pass the source results directory, or use --verify")
        report = archive(args.source, args.output, args.hostname)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
