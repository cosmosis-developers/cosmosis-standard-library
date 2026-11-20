import sacc
import sys
sys.path.append("likelihood/2pt")
import twopoint
import numpy as np
from cosmosis import Inifile

# Load the DES Y3 2pt data
data1 = twopoint.TwoPointFile.from_fits("likelihood/des-y3/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits") 

# Extract the n(z) data
nz_source = data1.get_kernel("nz_source")
z = nz_source.z
nzs = nz_source.nzs

# Extract the xi+ and xi- data
xip = data1.get_spectrum("xip")
xim = data1.get_spectrum("xim")
# Check that ordering is as expected
assert data1.covmat_info.names.index("xip") < data1.covmat_info.names.index("xim")

# Make a new SACC file with just the shear-shear data and the n(z)s
s = sacc.Sacc()

# Add the tracer for the n(z) for the sources
for i in range(len(nzs)):
    s.add_tracer(
        "NZ",
        f"source_{i}",
        z = z,
        nz = nzs[i]
    )

# Add the xi+ data points
for i in range(len(xip.bin1)):
    window = sacc.TopHatWindow(xip.angle_min[i], xip.angle_max[i])
    bin1 = xip.bin1[i] - 1
    bin2 = xip.bin2[i] - 1
    s.add_data_point(
        "galaxy_shear_xi_plus",
        (f"source_{bin1}", f"source_{bin2}"),
        xip.value[i],
        theta = xip.angle[i],
        window = window
    )

# Add the xi- data points
for i in range(len(xim.bin1)):
    window = sacc.TopHatWindow(xim.angle_min[i], xim.angle_max[i])
    bin1 = xim.bin1[i] - 1
    bin2 = xim.bin2[i] - 1
    s.add_data_point(
        "galaxy_shear_xi_minus",
        (f"source_{bin1}", f"source_{bin2}"),
        xim.value[i],
        theta = xim.angle[i],
        window = window
    )


# Extract the subset of the covariance matrix corresponding to xi+ and xi-
# and their cross-covariance.
xip_block_index = data1.covmat_info.names.index("xip")
xim_block_index = data1.covmat_info.names.index("xim")
xip_start = data1.covmat_info.starts[xip_block_index]
xim_start = data1.covmat_info.starts[xim_block_index]
xip_end = xip_start + data1.covmat_info.lengths[xip_block_index]
xim_end = xim_start + data1.covmat_info.lengths[xim_block_index]
n = len(data1.covmat)
mask = np.zeros((n,), dtype=bool)
mask[xip_start:xip_end] = True
mask[xim_start:xim_end] = True
covmat = data1.covmat[mask][:, mask]

# Add the covariance matrix to the new SACC file
s.add_covariance(covmat)


# Extract the photo-z shift bias prior values from the ini file
nbin_source = len(nzs)
values = Inifile("examples/des-y3-priors.ini")
mean = np.zeros(nbin_source)
sigma = np.zeros(nbin_source)
for i in range(nbin_source):
    v = values.get("wl_photoz_errors", f"bias_{i+1}")
    mean[i] = float(v.split()[1])
    sigma[i] = float(v.split()[2])

# Save the photo-z bias prior as a tracer uncertainty in the SACC file
tracer_uncertainty = sacc.tracer_uncertainty.NZShiftUncertainty(
    name="wl_photoz_errors",
    tracer_names=[f"source_{i}" for i in range(nbin_source)],
    mean=mean,
    cholesky_or_sigma=sigma,
)
s.add_tracer_uncertainty_object(tracer_uncertainty)

# Save the whole file to the output
s.save_fits("examples/des-y3-shear.sacc")