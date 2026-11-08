# Sources and third-party notices

The authors' original code in this public package is licensed under [MIT](LICENSE). Installed dependencies retain their own licenses.

## Historical baselines not distributed

The original AA-I-S-5 Python port, the adapted iGMRES-PI implementation, and their upstream source code are not included in this public package. The complete author archive remains separate. Neither implementation has been replaced, and this project grants no license to the omitted code.

Archived numerical results, experimental settings, and provenance for both methods are retained. Their inclusion does not mean that the corresponding implementations are available for execution from this package. See [baseline records](docs/BASELINE_RECORDS.md) for parameters, citations, source versions, and reproduction limits.

- **AA-I-S-5:** the historical Python port used the `aa1-safe` option from [nonexp_global_aa1](https://github.com/cvxgrp/nonexp_global_aa1), commit `457996cc6cebcaad606d24531214aa40a636125e`. The original source attribution is Copyright (c) 2018 Junzi Zhang.
- **iGMRES-PI:** the historical adapter used the `GMRES_PI` routine from [SIS_iPI](https://gitlab.ethz.ch/gmatilde/SIS_iPI), commit `1eddae2cd01d961469c349e3931e660796cce7ae`.

No explicit redistribution permission was established for these historical source versions during the review. This is the reason for omitting their implementations; it is not a claim that the authors refused permission or that the recorded algorithms were incorrect.

## Included implementations and model data

- **LSQN:** the authors' protected-learning implementation, including the original reference and optimized profiles. Source provenance is recorded in `core-port-notes.txt`, `optimized-port-notes.txt`, and `native-port-notes.txt`.
- **VI, PI, MPI, and QPI:** the local implementations used by the recorded comparison. The QPI implementations are not presented as the QPI authors' own code.
- **Official benchmark arrays:** extracted from Gymnasium 1.1.1 transition tables, with terminal transitions redirected to the documented sink. Gymnasium source is not bundled. Its [MIT copyright and license notice](LICENSES/Gymnasium.txt) is retained for the supplied derived model data.
- **Synthetic scaling arrays:** the separately labelled local reconstruction described in `experiment-notes.txt`; these are not represented as the official Gymnasium environments.

Historical source hashes and adaptations are retained in `baselines-port-notes.txt` and `experiment-notes.txt`. Their introductory scope notes distinguish the original full experiment package from this public distribution.
