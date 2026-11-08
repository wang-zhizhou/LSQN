"""Explicit historical LSQN implementation profiles; never infer a fallback."""
PAPER_BACKENDS = {
    "queue": "native_query",
    "cluster": "cache_reuse",
    "garnet": "fused_sparse",
    "latent": "native",
}
NATIVE_BACKENDS = {"native", "fused_sparse", "native_query"}


def backend_for(spec, implementation):
    """The two query-count suites retain their original uncached arithmetic."""
    if implementation not in ("paper", "curves", "reference"):
        raise ValueError("Unknown implementation profile: " + str(implementation))
    if spec["suite"] in ("public", "scaling"):
        return "reference"
    if spec["suite"] != "main":
        raise ValueError("Unknown model suite: " + str(spec["suite"]))
    if implementation == "paper":
        return PAPER_BACKENDS[spec["family"]]
    return "block32_cache" if implementation == "curves" else "reference"
