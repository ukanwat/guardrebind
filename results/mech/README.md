# Mechanism experiments (src/mech/)

Outputs of `src/mech/{patching,probe,similarity,dose_response}.py`, one JSON + one PDF per run,
named `<experiment>_<split>_n<N>....`. Run with `uv run --frozen --with matplotlib python -m src.mech.<script>`.

- `patching_train_n3_uncached.*`: smoke test only (3 train items), made before the prefix KV-cache was
  added to `Model.patched_margins` (the uncached path: every patched run was a full forward). It has no
  `cache_drift` field. The cached path was checked against full forwards on a tiny random Ministral3
  decoder on CPU (max margin difference 3.6e-7 in fp32), not yet on Shieldstral on the GPU.
