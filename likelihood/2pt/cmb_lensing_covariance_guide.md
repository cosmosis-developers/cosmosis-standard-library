# CMB Lensing Gaussian Covariance in `save_2pt`

This guide covers how to include CMB lensing (κ_CMB) cross-correlations in
the Gaussian covariance computed by `save_2pt`.

---

## Supported observables

The following two-point functions can be combined in a joint covariance
involving κ_CMB:

| Observable | Real-space section | Fourier section |
|---|---|---|
| Galaxy clustering w(θ) | `galaxy_xi` | `galaxy_cl` |
| Galaxy–galaxy lensing γ_t | `galaxy_shear_xi` | `galaxy_shear_cl` |
| Galaxy × κ_CMB | `galaxy_cmbkappa_xi` | `galaxy_cmbkappa_cl` |

Full cross-covariance blocks between these observables are included.

---

## Required pipeline sections

For real-space covariances the following C_ell sections must be computed by
the pipeline (they are read automatically from the datablock):

`galaxy_cl`, `galaxy_shear_cl`, `shear_cl`, `galaxy_cmbkappa_cl`, `cmbkappa_cl`

---

## New parameters

### `cmb_lensing_noise_file`

Path to the CMB lensing reconstruction noise N_ell^κκ.  **Required** when
`make_covariance = T` and any CMB-kappa spectrum is included.

Three formats are accepted:

**1. Plain text, 1-D** — one N_ell value per row; row index = ell.

**2. Plain text, 2-D** — first column is ell, noise in another column:
```ini
cmb_lensing_noise_file = /path/to/noise.dat
cmb_lensing_noise_col  = 7   ; required; 0-indexed (col 0 is ell); for SO nlkk files col 7 = N_MV
```

**3. NumPy `.npy` dict** (e.g. SO / Simons Observatory format):
```ini
cmb_lensing_noise_file = /path/to/noise.npy
cmb_lensing_ells_key   = els    ; default
cmb_lensing_noise_key  = Nl_MV  ; default
```

> **Note:** the noise file must cover at least up to `ell_max`.  An error is
> raised at setup time if the file falls short.  Below the file's minimum ell
> the noise is held constant at the first valid value.

---

## Example: real-space joint analysis (w(θ), γ_t, galaxy×κ_CMB)

```ini
[save_2pt]
filename  = output/data_vector.fits
overwrite = T

; angular binning
theta_min   = 2.5
theta_max   = 250.0
n_theta     = 20
angle_units = arcmin

; real-space spectra to save
spectrum_sections = galaxy_xi  galaxy_shear_xi  galaxy_cmbkappa_xi
output_extensions = wtheta     gammat           galaxy_kappa

; corresponding C_ell sections and transform types
cl_sections    = galaxy_cl  galaxy_shear_cl  galaxy_cmbkappa_cl
cl_to_xi_types = 00         02+              00

; covariance
make_covariance              = T
fsky                         = 0.4
ell_max                      = 5000
number_density_shear_arcmin2 = 5.0 5.0 5.0 5.0
number_density_lss_arcmin2   = 3.0 3.0 3.0 3.0
sigma_e_total                = 0.26 0.26 0.26 0.26

; CMB lensing noise
cmb_lensing_noise_file = /path/to/nlkk.dat
; cmb_lensing_noise_col = 1          ; if multi-column text
; cmb_lensing_ells_key  = els        ; if .npy dict
; cmb_lensing_noise_key = Nl_MV      ; if .npy dict
```

### `cl_to_xi_types` quick reference

| Pairing | Type |
|---|---|
| Position × position, position × κ_CMB | `00` |
| Position × shear, shear × κ_CMB | `02+` |
| Shear × shear (ξ+/ξ−) | `22+` / `22-` |

---

## Example: Fourier-space analysis

```ini
[save_2pt]
filename  = output/cls.fits
overwrite = T

ell_min = 30
ell_max = 3000
n_ell   = 20

spectrum_sections = galaxy_cl  galaxy_shear_cl  galaxy_cmbkappa_cl
output_extensions = wtheta     gammat           galaxy_kappa

make_covariance              = T
fsky                         = 0.4
number_density_shear_arcmin2 = 5.0 5.0 5.0 5.0
number_density_lss_arcmin2   = 3.0 3.0 3.0 3.0
sigma_e_total                = 0.26 0.26 0.26 0.26

cmb_lensing_noise_file = /path/to/nlkk.npy
cmb_lensing_ells_key   = els
cmb_lensing_noise_key  = Nl_MV
```

> For Fourier-space runs `cl_sections` and `cl_to_xi_types` are not needed.

---

## Common errors

| Error | Fix |
|---|---|
| `Set cmb_lensing_noise_file...` | Add `cmb_lensing_noise_file` to the ini |
| `only provides noise up to ell=X, but ell_max=Y` | Use a file with higher ell coverage or lower `ell_max` |
| `KeyError` loading `.npy` | Check keys with `np.load(f, allow_pickle=True).item().keys()` and set `cmb_lensing_ells_key` / `cmb_lensing_noise_key` |
