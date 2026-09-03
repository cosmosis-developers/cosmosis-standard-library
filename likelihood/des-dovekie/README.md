# DES Y5 Dovekie Supernova Likelihood

This module computes the likelihood for the Dark Energy Survey Year 5 (DES Y5) Supernova sample (Dovekie).

### Data Provenance
* **Original Repository:** [DES-SN5YR](https://github.com/des-science/DES-SN5YR)
* **Data File:** `DES-Dovekie_HD_clean.csv`. Note: The original `DES-Dovekie_HD.csv` provided by DES was space-delimited with non-standard prefix headers (`VARNAMES:`, `SN:`). This has been cleaned and converted to a standard, comma-separated `.csv` so `astropy.table` can parse it natively without raising KeyErrors.
* **Covariance:** `STAT+SYS.npz` (Inverse precision matrix, analytically marginalized over $M$ and $H_0$).

### Usage
Include `des_dovekie` in your `modules` list and provide the paths to the data and covariance files in the configuration block. Nuisance parameters for absolute magnitude are not required as they are analytically marginalized.
