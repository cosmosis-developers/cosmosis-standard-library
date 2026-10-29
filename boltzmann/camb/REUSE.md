# CAMB result reuse

Enable exact result reuse with `reuse_computation = T`. The default is false.
Results are retained in a least-recently-used cache. A hit refreshes recency;
inserting a new result evicts the oldest entry when capacity is exceeded.

| Option | Default | Constraint |
| --- | --- | --- |
| reuse_cache_size | 4 | Positive integer |
| reuse_cache_adaptive | false | Opt-in capacity growth |
| reuse_cache_max_size | max(8, initial size) | At least initial size |
| reuse_cache_window | 8 | Positive integer |
| reuse_cache_threshold | 0.25 | Finite, between 0 and 1 inclusive |

Invalid settings raise ValueError during setup. Adaptive mode computes the hit
fraction over each decision window: H = hits / (hits + misses). If H is strictly
below the threshold, capacity doubles, capped at reuse_cache_max_size. Window
counters then reset. This heuristic can grow on unique inputs without improving
reuse; fixed capacity is recommended until representative workloads demonstrate
a benefit. Capacity bounds entry count, not native CAMB memory in bytes.

Cache signatures cover CAMB parameters and output configuration. Changing neutrino
mass, primordial amplitude or target sigma_8 requires recomputation. A sigma_8
cache hit also restores the normalized A_s output. The normalization is
A_s_new = A_s_old * (sigma_8_target / sigma_8_computed)**2.

## Validation

Run from the repository root:

```sh
python -m pytest -q tests/unit_tests/test_camb.py
python tests/benchmark_camb_reuse.py
```

Latest local validation: 27 tests passed in 33.49 seconds (one expected warning
that sigma_8 takes precedence over A_s). Coverage includes capacity and recency,
adaptive growth, invalid settings, threshold boundaries, nonfinite numeric inputs,
mass/amplitude changes, and sigma_8 output restoration.

Independent fresh comparisons require
epsilon_max = max(abs(P_cached - P_fresh) / maximum(abs(P_fresh), 1e-30)) <= 1e-12.
The tested linear and nonlinear spectra satisfy this bound. This is a tested
tolerance, not a universal zero-error guarantee.

Five earlier local warmed-cache timings had fresh median 0.597485 seconds and
reuse median 0.062744 seconds, giving a ratio of approximately 9.523. Warm-up is
excluded. These historical timings predate subsequent validation changes and
are not end-to-end inference speedups. Actual benefit depends on repeated inputs,
cache hit rate, native memory costs and existing CosmoSIS fast/slow reuse.
