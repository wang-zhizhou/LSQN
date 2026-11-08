# Linux Intel result archive

This directory preserves all 1,020 runs from the completed Linux paper batch: 960 main comparisons, 24 public-environment runs, and 36 scaling runs. Failures and timeouts are retained. No experiment was rerun during import.

Raw records are individual `.json.gz` files. The original collection, summary, analysis, protocol, and environment JSON files are included without compression. Summary `file` entries retain their original names: append `.gz` when opening the archived raw record.

Only personal machine identifiers in JSON strings are redacted: home-directory prefixes become `<HOME>` and detected hostname values become `<HOSTNAME>`. CPU, BLAS, compiler, source hashes, all numerical values, numeric spellings, iteration histories, policies, queries, timings, and failure statuses are preserved. No console logs, executable files, or native libraries are included.

`provenance.json` maps original relative paths and SHA-256 hashes to published relative paths and SHA-256 hashes, with a separate hash of each uncompressed, sanitized JSON payload. Its non-string-content hash replaces each complete JSON string token with an empty string before hashing; all other bytes, including numeric tokens and structure, remain unchanged. Gzip uses level 9, MTIME 0, no filename, and OS 255. Deflate bytes may depend on the zlib version; the uncompressed payload hash is independent of that.

From the repository root, verify this archive with:

```sh
python scripts/import_linux_results.py --verify
```

To reproduce the import from the original directory, pass its path as the positional argument. An existing archive is checked and left untouched; a different batch is refused. Original source hashes require access to the original private files; published hashes and all archived records can be verified without those files.
