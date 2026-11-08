# Cross-platform results

These archives compare the paper's Apple M2/macOS measurements with a separate
Intel Core i9-14900K/Linux run of the release candidate on 25 October 2026.
They support software portability and broadly consistent experimental findings;
they are not a controlled comparison of CPU performance.

## Results at a glance

- Linux completed all 1,020 planned executions: 960 main, 24 public, and 36 scaling.
- LSQN succeeded in all 156 executions. Among the baselines, 51 runs timed out
  and three AA-I-S-5 runs reached the 30,000-iteration limit; these remain in the archive.
- LSQN has the second-lowest mean runtime rank on both platforms. Its advantage
  is strongest on Cluster; PI is fastest on Queue, and QPI-U/BT are faster on
  Garnet and Latent. An overall rank does not imply superiority on most cases.
- LSQN outer-iteration and total-query counts match the historical timing records
  in 30/32 main configurations. Exact trajectories need not match across platforms.

## Runtime ranks

For each configuration, take the median of three runs only if all three succeed.
Rank successful methods by this median; failed methods share the mean of the
remaining ranks. Average these ranks equally over the 32 configurations.
Lower is better. This is a mean rank, not a mean wall time or a significance test.

| Method | macOS mean rank | Linux mean rank | Linux successful cases |
| --- | --- | --- | --- |
| AA-I-S-5 | 3.453 | 3.297 | 31/32 |
| LSQN | 3.531 | 3.500 | 32/32 |
| QPI-BT | 3.562 | 3.688 | 32/32 |
| QPI-U | 4.062 | 4.125 | 32/32 |
| iGMRES-PI | 5.031 | 4.969 | 28/32 |
| PI | 4.875 | 5.094 | 32/32 |
| MPI-20 | 6.438 | 6.438 | 30/32 |
| QPI-S | 7.016 | 6.953 | 28/32 |
| VI | 8.188 | 8.109 | 28/32 |
| MPI-4 | 8.844 | 8.828 | 29/32 |

| Family | macOS LSQN mean rank | Linux LSQN mean rank |
| --- | --- | --- |
| Queue | 6.125 | 6.000 |
| Cluster | 1.000 | 1.000 |
| Garnet | 3.000 | 3.000 |
| Latent | 4.000 | 4.000 |

## Inner learning

Public results compare per-outer-iteration query counts, including zero-query steps.

| Environment | n | Discount | Mean queries: macOS / Linux | Max queries: macOS / Linux |
| --- | --- | --- | --- | --- |
| frozenlake4x4 | 17 | 0.99 | 2.71 / 2.71 | 5 / 5 |
| frozenlake4x4 | 17 | 0.999 | 2.71 / 2.71 | 5 / 5 |
| cliffwalking | 49 | 0.99 | 1.76 / 1.45 | 4 / 3 |
| cliffwalking | 49 | 0.999 | 2.96 / 4.16 | 6 / 14 |
| frozenlake8x8 | 65 | 0.99 | 5.25 / 5.25 | 9 / 9 |
| frozenlake8x8 | 65 | 0.999 | 5.43 / 5.43 | 10 / 10 |
| taxi | 501 | 0.99 | 24.17 / 23.37 | 100 / 76 |
| taxi | 501 | 0.999 | 75.94 / 70.90 | 353 / 275 |

For scaling, first average queries over the iterations of each instance, then
average the three instance means equally at each size. Fit log(mean queries)
against log(n) over the six sizes. These empirical exponents are not asymptotic
complexity bounds and do not establish that every run entered a superlinear regime.

| Discount | macOS query-growth exponent | Linux query-growth exponent |
| --- | --- | --- |
| 0.99 | 0.1643 | 0.1585 |
| 0.999 | 0.3342 | 0.3271 |

## Environment and comparison limits

- Both runs use CPUs; the Linux machine's GPU was not used. The Linux logs record
  one thread in each BLAS library and no fixed CPU affinity.
- Linux used Python 3.11.17, NumPy 1.26.4, and SciPy 1.13.0. Its NumPy OpenBLAS
  0.3.23.dev selected **Prescott**, while SciPy OpenBLAS 0.3.26.dev selected
  **Haswell**. This is the configuration actually measured, not a retuned run.
  `OPENBLAS_CORETYPE` was not recorded. The difference warrants checking before
  interpreting these timings as fully tuned hardware performance.
- macOS values are the paper's final table assembled from several recorded
  batches, rather than one fresh run of this package. Per-cell source identifiers
  and hashes are retained. Missing historical environment fields stay unknown.
- Comparing the Linux main-model hashes with the Mac reference-stage generator
  records gave matching Queue/Garnet hashes and differing Cluster/Latent hashes;
  this does not verify the historical paper timing inputs byte for byte. Linux
  did not retain those full main arrays, so the size or cause of the differences is unresolved.
  Public/scaling inputs are frozen in the package; some trajectories differ even
  for identical inputs. This comparison is not a bitwise reproduction claim.
- Repeated runs measure timing variation on fixed instances; they are not
  independent random model samples. Do not mix selected timings from the two hosts
  into a single performance table or attribute a speed difference solely to CPU.

## Inspect and reproduce the comparison

- [Linux archive](linux-intel-2026-10-25/): all 1,020 run records, including failures,
  plus the recorded protocols, summaries, and environment metadata. Run JSON files
  are gzip-compressed; machine-specific paths are redacted. The import manifest
  records original and released hashes and the redaction policy.
- [macOS paper reference](macos-m2-paper/): per-configuration and per-run source
  extracts, historical environment records, and provenance. It is a reference
  export, not a new macOS execution.
- [Machine-readable comparison](comparison.json) and [audit scope](audit.json).
- [Suggested Section 6 text](../docs/section6_cross_platform.tex).

From the repository root, with Python 3.9 or newer:

```bash
python scripts/summarize_benchmarks.py --check
python scripts/verify_manifest.py
```

To regenerate this page and `comparison.json`, run
`python scripts/summarize_benchmarks.py`. This reads archived measurements only;
it does not run an experiment. See the [main README](../README.md) to run new experiments.

## LSQN per-configuration comparison

Times below are seconds (median of three runs). The machine-readable files retain
the measurements for all ten methods. These host-to-host times are descriptive.

| Family | n | Discount | macOS seconds | Linux seconds | Outer iterations: macOS / Linux | Total queries: macOS / Linux |
| --- | --- | --- | --- | --- | --- | --- |
| queue | 512 | 0.99 | 0.0685559 | 0.0412949 | 9 / 9 | 501 / 501 |
| queue | 512 | 0.999 | 0.0806642 | 0.0479402 | 13 / 13 | 510 / 510 |
| queue | 1024 | 0.99 | 0.273043 | 0.158996 | 10 / 10 | 755 / 755 |
| queue | 1024 | 0.999 | 0.471026 | 0.269571 | 15 / 15 | 1023 / 1023 |
| queue | 2048 | 0.99 | 0.521944 | 0.37353 | 10 / 10 | 774 / 774 |
| queue | 2048 | 0.999 | 6.88003 | 6.51132 | 343 / 341 | 5140 / 5204 |
| queue | 4096 | 0.99 | 0.809858 | 0.670033 | 10 / 10 | 668 / 668 |
| queue | 4096 | 0.999 | 26.9971 | 27.8876 | 203 / 205 | 5985 / 5987 |
| cluster | 512 | 0.99 | 0.00844608 | 0.00560396 | 7 / 7 | 26 / 26 |
| cluster | 512 | 0.999 | 0.00901075 | 0.00599522 | 7 / 7 | 29 / 29 |
| cluster | 1024 | 0.99 | 0.0255129 | 0.0214533 | 6 / 6 | 23 / 23 |
| cluster | 1024 | 0.999 | 0.0293793 | 0.0225141 | 6 / 6 | 26 / 26 |
| cluster | 2048 | 0.99 | 0.0903037 | 0.0938426 | 6 / 6 | 17 / 17 |
| cluster | 2048 | 0.999 | 0.105923 | 0.110019 | 7 / 7 | 24 / 24 |
| cluster | 4096 | 0.99 | 0.322458 | 0.383689 | 6 / 6 | 17 / 17 |
| cluster | 4096 | 0.999 | 0.412574 | 0.474197 | 7 / 7 | 24 / 24 |
| garnet | 512 | 0.99 | 0.00298696 | 0.00249807 | 11 / 11 | 22 / 22 |
| garnet | 512 | 0.999 | 0.00317079 | 0.00259911 | 11 / 11 | 26 / 26 |
| garnet | 1024 | 0.99 | 0.00402833 | 0.00324546 | 11 / 11 | 24 / 24 |
| garnet | 1024 | 0.999 | 0.0042655 | 0.00333177 | 11 / 11 | 27 / 27 |
| garnet | 2048 | 0.99 | 0.00602721 | 0.00510117 | 12 / 12 | 24 / 24 |
| garnet | 2048 | 0.999 | 0.00660271 | 0.00523731 | 12 / 12 | 27 / 27 |
| garnet | 4096 | 0.99 | 0.0100603 | 0.00815915 | 12 / 12 | 24 / 24 |
| garnet | 4096 | 0.999 | 0.0121299 | 0.00847347 | 12 / 12 | 27 / 27 |
| latent | 512 | 0.99 | 0.00309729 | 0.00259528 | 5 / 5 | 4 / 4 |
| latent | 512 | 0.999 | 0.00282329 | 0.00264852 | 4 / 4 | 6 / 6 |
| latent | 1024 | 0.99 | 0.00788317 | 0.00927009 | 4 / 4 | 3 / 3 |
| latent | 1024 | 0.999 | 0.00878754 | 0.00924194 | 4 / 4 | 4 / 4 |
| latent | 2048 | 0.99 | 0.0313666 | 0.0344461 | 4 / 4 | 3 / 3 |
| latent | 2048 | 0.999 | 0.028802 | 0.0311795 | 3 / 3 | 4 / 4 |
| latent | 4096 | 0.99 | 0.140383 | 0.140298 | 4 / 4 | 3 / 3 |
| latent | 4096 | 0.999 | 0.110505 | 0.141125 | 4 / 4 | 3 / 3 |
