# CAMB result reuse

Experimental nonlinear sharing: `share_nonlinear_ratio = T` shares native-grid
ratios for delta_tot, delta_nonu and delta_cdm auto spectra. Default false.
Unsupported components, cross spectra, grid/unit mismatches and invalid arrays
fall back to native extraction; six dedicated fallback tests passed.
`python tests/benchmark_camb_ratio_metropolis.py` runs three alternating pairs
of 100-step block Metropolis tests. Local disabled median: 8.584172 seconds
(sample standard deviation 0.819588); enabled median: 10.373043 seconds
(standard deviation 2.339120). Ratio of medians: 0.827546. All six spectral
and posterior differences were zero. This did not reproduce the earlier
single-pair speedup. Keep sharing experimental and disabled by default.

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

## Alternating fast/slow pipeline benchmark

Run `python tests/benchmark_camb_fast_slow.py` from the repository root.
This runs real CAMB with a synthetic likelihood and automatic fast/slow splitting.
Four cosmologies are visited in sequence three times (12 requests per condition).
The existing slow-subspace cache in the tested CosmoSIS checkout holds three
entries; this workload intentionally exceeds that capacity. The CAMB LRU holds
four entries. Verify the slow-subspace capacity when using other CosmoSIS versions.

Each condition is measured three times in alternating order: off/on, on/off,
off/on. Setup and fast/slow analysis are excluded from total evaluation time.
Pipeline cache state after analysis is retained; the CAMB LRU and cumulative
counters are cleared before timing. This is a controlled revisit experiment,
not an observational likelihood or sampler convergence benchmark.

Local recorded times in seconds:

| CAMB LRU | Run 1 | Run 2 | Run 3 | Median | Sample standard deviation |
| --- | --- | --- | --- | --- | --- |
| Disabled | 3.956843 | 4.184688 | 4.312583 | 4.184688 | 0.180195 |
| Enabled | 1.518363 | 2.068216 | 1.663524 | 1.663524 | 0.284951 |

The ratio of medians is 4.184688 / 1.663524 = approximately 2.516;
time reduction is 1 - 1.663524 / 4.184688 = approximately 60.25 percent.
Each enabled run recorded eight hits and four misses: H = 8/12.
Maximum relative spectral difference against the first baseline run was zero
in all six runs, with acceptance threshold epsilon_max <= 1e-12.
Three repeats describe local variability and do not establish a confidence interval.

## Recommended use and sampler check

Keep reuse disabled for routine continuous Metropolis sampling unless measured
CAMB cache hits justify enabling it. The demonstrated use is exact cosmology
revisits exceeding the existing slow-subspace cache capacity, such as controlled
repeated-grid or replay workloads. No generic MCMC speedup is established.

Run `python tests/benchmark_camb_metropolis.py` for a seeded 100-step run of the
CosmoSIS Metropolis engine with fast/slow block proposals and oversampling 5.
It uses 6dFGS BAO plus a synthetic fast nuisance likelihood. This is longer than
the initial 20-step check, but remains a performance smoke test, not convergence
validation or an observational posterior result.

Local result: 101 posterior requests per condition; full-vector exact revisits
zero. With reuse enabled CAMB recorded 0 hits and 18 misses. Total evaluation
times were 6.384891 seconds disabled and 6.393932 seconds enabled. Their ratio
is 0.998586, approximately 0.14 percent more time with reuse; one pair cannot
resolve a difference this small. Spectral relative error and posterior log
probability absolute difference were both zero (limits 1e-12 and 1e-10).
Other requests were handled by pipeline reuse or prior rejection. Rejected
chain states should not be confused with reevaluated likelihood requests.
