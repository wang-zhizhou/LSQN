# Verification records

## Historical development checks

The existing `mac-*` directories and development check reports were generated on the development Mac, using Python 3.9.13, NumPy 1.26.4, SciPy 1.13.0, and one BLAS thread. They describe the historical full package before the two baseline implementations were removed from the public distribution. Their numerical records and original source hashes are retained unchanged.

- `mac-checks.json`: final installed-package checks at that historical stage.
- `mac-installed-smoke/`: all ten methods on six small instances; this is an archived batch, not a claim that the public package can rerun all ten methods.
- `mac-installed-large/`: one LSQN run per family at n=4096 and gamma=0.999.
- `mac-optimized-smoke/`, `mac-curves-smoke/`, `mac-reference-smoke/`, `mac-optimized-main-small/`: earlier profile verification (134 runs), identified by `implementation-checks.json`.
- `reference-checks.json`: historical rc1 checks before optimized-backend integration.

Each solve was independently checked using the Bellman residual. LSQN runs also checked forcing, descent, and Armijo certificates. Small smoke instances were compared with an exact PI solve outside method timing. The complete 960-run main comparison was not rerun as part of those development checks.

The separate completed Linux experiment archive is under [`benchmarks/`](../benchmarks/README.md). Historical development timings are verification records, not a controlled hardware comparison.

## Public-package checks

The [public review check record](public-release-checks.json) reports the 8 November 2026 checks: 33 unit tests, 15 native-backend self-checks, 48 successful smoke solves, and regeneration of the archived macOS and Linux figures and runtime tables. These new execution checks were run on macOS; the Linux measurements remain the separately identified historical batch.

Current public-package checks are recorded separately from the historical batches. The runnable smoke scope is eight methods on six small models (48 solves). Source and method-set changes mean that this batch should not be required to match the old ten-method package's source hashes or method set.

The original LSQN implementation and the seven retained baseline implementations are preserved; the public-package checks address packaging, available methods, independent numerical correctness, and regeneration of figures and tables from archived records. They do not substitute new measurements for the paper's results or claim a new ten-method timing comparison.

Exported records redact personal absolute paths while retaining input hashes, source hashes, and numerical results. The original full source archive remains private.
