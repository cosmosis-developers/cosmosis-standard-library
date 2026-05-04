# SP(k) CosmoSIS Module

This module applies baryonic suppression from py-SP(k) to a matter power
spectrum grid in a CosmoSIS pipeline.

## Dependency

Install pyspk in the active environment:

```bash
pip install pyspk
```

If `pyspk` is not installed, module setup raises an informative import error.

## What It Does

1. Reads an input power grid (`k_h`, `z`, `P_k`) from `input_section`.
2. Builds SP(k) suppression for each redshift.
3. Writes the modified power grid to `output_section`.
4. Optionally writes suppression values `S_k` to `suppression_section`.

Default behavior is in-place modification of `matter_power_nl`.

## Module Options

- `SO` (int, default `500`): spherical overdensity (`200` or `500`).
- `input_section` (str, default `matter_power_nl`): input power section.
- `output_section` (str, default `matter_power_nl`): output power section.
- `suppression_section` (str, default empty): optional section for `S_k`.
- `verbose` (bool, default `False`): pyspk verbose mode.
- `fb_table` (str, default empty): CSV table for binned mode.
- `extrapolate` (bool, default `False`): binned-mode extrapolation behavior.

## Relation Modes

Exactly one mode must be provided per sample:

1. Power law: `fb_a`, `fb_pow`, optional `fb_pivot`
2. Cosmology power law: `alpha`, `beta`, `gamma`
3. Double power law: `epsilon`, `alpha`, `beta`, `gamma`, `m_pivot`
4. Binned mode: set `fb_table` and do not set SP(k) relation parameters

## Minimal Configuration Snippet

```ini
[spk]
file = structure/spk/spk_interface.py
SO = 500
input_section = matter_power_nl
output_section = matter_power_nl
suppression_section =
verbose = F
extrapolate = F
```

In `values.ini`, set exactly one supported SP(k) relation mode in `[spk]`.

For a complete runnable example, see `examples/spk.ini` and
`examples/spk_values.ini`.
