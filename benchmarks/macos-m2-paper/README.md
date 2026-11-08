# Historical Mac paper reference

This directory exports the Mac results used in the paper. It does not rerun any
solver. `main.json` contains 32 configurations × 10 methods; `main_runs.jsonl`
contains the 960 original run measurements selected for those cells. The old
method label SCB is exposed as LSQN, with the original label retained.

`main_curves.jsonl.gz` retains the complete recorded residual-2 trace for the
middle repetition ordered by (whole-solver elapsed time, repeat index) in each
of those same 320 final-runtime cells. Failed triples are retained and labelled.
No curve from the separate block32 paper-figure profile is substituted. Initial
and terminal points are retained as recorded; explicit flags identify whether
the original trace includes k=0 and the recorded final iteration. No missing
trace point is reconstructed or invented.

The main table combines historical batches. Of the 288 baseline cells, 192 come
from the October 11 table (including the authors' iGMRES supplemental batch)
and 96 come from the October 14 targeted leader recheck. The final LSQN choices
are Queue `native_query`, Cluster `cache_reuse`, Garnet `fused_sparse`, and Latent
`native`, from their October 14 confirmation batches. No fastest repetition or
per-case variant is selected. This is a descriptive mixed-batch ranking, not a
fresh simultaneous comparison. Recomputing all 320 ranks gives LSQN 3.53125.

Every cell preserves all three timings, successes, failure messages, iterations,
and available LSQN query counts. Failed cells have null `median_wall_seconds`;
their observed elapsed times remain in `wall_seconds_all`. Missing timeout flags
remain null; the original failure message is retained. `outer_iterations` and
`total_queries` are medians, with all individual values in the corresponding
`*_all` fields. Query counts for other methods are null because they are not the
LSQN protected-learning query metric.

`public.json` contains the eight official Gymnasium configurations, each with
three runs. `scaling.json` contains 36 generated FrozenLake instances (six sizes,
two discounts, three seeds, one run each). `learning_runs.jsonl` retains their
query sequences and measurements. Query sequences are checked against the
existing `experiments/data/archive_reference.json`. Scaling is synthetic, and
its fitted growth exponents describe these six sizes only; they are not
asymptotic complexity bounds. Time means whole-solver elapsed time, not only
inner learning time.

`source_manifest.json` maps relative research-archive IDs to original SHA-256
hashes. `sources/` includes normalized table/protocol evidence. The per-run
summary files omit large value vectors and histories; the selected complete
residual traces are stored separately in `main_curves.jsonl.gz`. Original gzip
hashes can be checked against the separate research archive. No private home
directory is embedded. Historical system library paths are reduced to filenames.

`environment.json` uses only records made during the historical batches. Missing
fields are null; no current-machine information is substituted. In particular,
the exact CPU model and native compiler/flags are absent from the inspected
protocols. The directory name is an archive label, not a new hardware probe.

Maintainers with the separate original research archive can regenerate this
export using the standard library only:

    python scripts/export_macos_reference.py --archive-root /path/to/research-archive

This command reads the archive and writes only this benchmark directory. It
requires the original archive; a public checkout alone does not contain all
original raw files. The checked-in JSON files are directly usable without it.
