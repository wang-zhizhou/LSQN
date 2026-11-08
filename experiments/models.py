"""Frozen model loading and platform-independent array fingerprints.

NPZ file hashes identify the archive bytes. Model hashes identify canonical
little-endian float64/int64 arrays, their shapes, sparse/dense storage and gamma.
Neither hash includes the name, machine paths, timing, or compression metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from lsqn import MDP
from .benchmarks import FAMILIES

DATA = Path(__file__).resolve().parent / "data"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(digest, label, array, dtype):
    array = np.asarray(array, dtype=dtype, order="C")
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(dtype.encode("ascii") + b"\0")
    # A memoryview avoids an additional n-by-n bytes copy for dense inputs.
    digest.update(memoryview(array).cast("B"))


def model_hash(mdp):
    digest = hashlib.sha256(b"lsqn-model-arrays-v1\0")
    _hash_array(digest, "gamma", np.array([mdp.gamma]), "<f8")
    _hash_array(digest, "rewards", mdp.rewards, "<f8")
    for action, matrix in enumerate(mdp.transitions):
        label = "p%d" % action
        if sp.issparse(matrix):
            matrix = matrix.tocsr()
            digest.update(b"csr\0")
            _hash_array(digest, label + "_shape", matrix.shape, "<i8")
            _hash_array(digest, label + "_indptr", matrix.indptr, "<i8")
            _hash_array(digest, label + "_indices", matrix.indices, "<i8")
            _hash_array(digest, label + "_data", matrix.data, "<f8")
        else:
            digest.update(b"dense\0")
            _hash_array(digest, label, matrix, "<f8")
    return digest.hexdigest()


def model_info(mdp):
    return dict(n_states=mdp.n_states, n_actions=mdp.n_actions,
                gamma=float(mdp.gamma), name=mdp.name,
                storage="csr" if mdp.sparse else "dense",
                transition_nnz=sum(p.nnz if sp.issparse(p) else int(np.count_nonzero(p))
                                   for p in mdp.transitions),
                model_sha256=model_hash(mdp), hash_schema="lsqn-model-arrays-v1")


def save_model(mdp, path):
    """Write a new NPZ; never overwrite an existing input."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(rewards=mdp.rewards, gamma=np.array(mdp.gamma),
                  name=np.array(mdp.name), schema=np.array("lsqn-model-npz-v1"))
    for action, matrix in enumerate(mdp.transitions):
        prefix = "p%d" % action
        if sp.issparse(matrix):
            matrix = matrix.tocsr()
            arrays.update({prefix + "_data": matrix.data,
                           prefix + "_indices": matrix.indices,
                           prefix + "_indptr": matrix.indptr,
                           prefix + "_shape": np.array(matrix.shape)})
        else:
            arrays[prefix + "_dense"] = np.asarray(matrix)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return file_hash(path)


def load_model(path, name="frozen_mdp"):
    """Accept release archives and the original September scaling archives."""
    with np.load(path, allow_pickle=False) as arrays:
        rewards = arrays["rewards"].copy()
        matrices = []
        for action in range(rewards.shape[0]):
            prefix = "p%d" % action
            if prefix + "_dense" in arrays:
                matrices.append(arrays[prefix + "_dense"].copy())
            else:
                shape = tuple(int(x) for x in arrays[prefix + "_shape"])
                matrices.append(sp.csr_matrix((arrays[prefix + "_data"].copy(),
                                               arrays[prefix + "_indices"].copy(),
                                               arrays[prefix + "_indptr"].copy()), shape=shape))
        if "name" in arrays:
            name = str(arrays["name"].item())
        return MDP(matrices, rewards, float(arrays["gamma"].item()), name)


def gamma_tag(gamma):
    return format(gamma, ".12g").replace(".", "p")


def case_id(spec):
    return "%s_%s_n%d_g%s_seed%d" % (
        spec["suite"], spec["family"], spec["n"], gamma_tag(spec["gamma"]), spec.get("seed", 0))


def load_manifest():
    return json.loads((DATA / "manifest.json").read_text())


def build_model(spec, input_models=None):
    """Frozen inputs take precedence when an external input directory is given.

    An external directory must contain every requested case and a manifest.json.
    It is deliberately not an opportunistic cache: missing inputs are errors.
    """
    identity = case_id(spec)
    if input_models is not None:
        base = Path(input_models)
        manifest = json.loads((base / "manifest.json").read_text())
        entry = manifest["models"][identity]
        path = base / entry["file"]
        if file_hash(path) != entry["file_sha256"]:
            raise ValueError("Input file hash mismatch: " + identity)
        mdp = load_model(path, spec["family"])
        if model_hash(mdp) != entry["model_sha256"]:
            raise ValueError("Input array hash mismatch: " + identity)
        return mdp, dict(source="external_frozen", source_id=entry.get("source_id", identity),
                         file_sha256=entry["file_sha256"], frozen_model_match=True)
    bundled = load_manifest()["models"]
    if spec["suite"] in ("public", "scaling") or identity in bundled:
        entry = bundled[identity]
        path = DATA / entry["file"]
        if file_hash(path) != entry["file_sha256"]:
            raise ValueError("Bundled input file hash mismatch: " + identity)
        mdp = load_model(path, spec["family"])
        if model_hash(mdp) != entry["model_sha256"]:
            raise ValueError("Bundled input array hash mismatch: " + identity)
        return mdp, dict(source="bundled_frozen", source_id=entry["source_id"],
                         file_sha256=entry["file_sha256"], frozen_model_match=True,
                         construction=entry["construction"])
    mdp = FAMILIES[spec["family"]](n=spec["n"], actions=4,
                                   gamma=spec["gamma"], seed=spec["seed"])
    return mdp, dict(source="original_generator", source_id="paper-main-seed130" if spec["seed"] == 130 else "smoke",
                     generator="experiments/benchmarks.py", generator_sha256=file_hash(Path(__file__).with_name("benchmarks.py")),
                     frozen_model_match=None,
                     note="Dense Cluster/Latent and sparse Queue/Garnet are retained. Save and share NPZ inputs for strict cross-machine input identity.")


def plans(suite):
    main = [dict(suite="main", family=family, n=n, gamma=gamma, seed=130,
                 initialization="reward_lower_bound_all_states")
            for n in (512, 1024, 2048, 4096) for family in FAMILIES for gamma in (.99, .999)]
    public = [dict(suite="public", family=family, n=n, gamma=gamma, seed=0,
                   initialization="reward_lower_bound_all_states")
              for family, n in (("frozenlake4x4", 17), ("frozenlake8x8", 65), ("taxi", 501), ("cliffwalking", 49))
              for gamma in (.99, .999)]
    scaling = [dict(suite="scaling", family="frozenlake", n=side * side,
                    side=side, gamma=gamma, seed=seed, initialization="zero")
               for gamma in (.99, .999) for side in (8, 16, 24, 32, 48, 64) for seed in (130, 131, 132)]
    if suite == "smoke":
        return ([dict(suite="main", family=family, n=32, gamma=.99, seed=130,
                      initialization="reward_lower_bound_all_states") for family in FAMILIES]
                + [public[0], scaling[0]])
    return {"main": main, "public": public, "scaling": scaling, "all": main + public + scaling}[suite]
