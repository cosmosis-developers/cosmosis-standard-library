import numpy as np
from cosmosis.datablock import BlockError
import pathlib
import sys

# Get the SpectrumInterp class from the spec_tools module.
# Should really put this somewhere else!
twopoint_dir = pathlib.Path(__file__).parent.parent.parent.resolve() / "2pt"
sys.path.append(str(twopoint_dir))
from spec_tools import SpectrumInterp

def load_psf_template(options):
    filename = options.get_string("psf_file", "")
    psf_ell_bins = np.load(filename)['arr_0']
    psf_cl_arr = np.load(filename)['arr_1']
    filename2 = options.get_string("psf_transformation_file", "")
    psf_transform_matrix = np.load(filename2)['arr_0']
    psf_means = np.load(filename2)['arr_1']
    return psf_ell_bins, psf_cl_arr, psf_transform_matrix, psf_means

def compute_psf_template(options, block):
    alpha2 = block["psf_parameters", "psf_alpha2"]
    beta2 = block["psf_parameters", "psf_beta2"]
    alpha4 = block["psf_parameters", "psf_alpha4"]
    beta4 = block["psf_parameters", "psf_beta4"]

    psf_ell_bins, psf_template, psf_transform_matrix, psf_means = load_psf_template(options)
        
    uncorrelated_p_arr = np.array([alpha2, beta2, alpha4, beta4])
    p_arr = (np.linalg.inv(psf_transform_matrix)@uncorrelated_p_arr.T).T + psf_means
    delta_cl = np.zeros(len(psf_template[0][0]))
    for i in range(4):
        for j in range(4):
            delta_cl += p_arr[i]*p_arr[j]*psf_template[i][j]

    return delta_cl


def extract_hsc_prediction(sacc_data, block, data_type, section, **kwargs):
    """
    This is based on the original SACC likelihood, but adds the PSF subtraction
    """
    options = kwargs.get("options", None)
    if options is None:
        raise ValueError("HSC likelihood requires options to be passed")
    
    x_theory = block[section, "ell"]
    is_auto = block[section, "is_auto"]

    # We build up these vectors from all the data points.
    # Only the theory vector is needed for the likelihood - the others
    # are for convenience, debugging, etc.
    theory_vector = []
    angle_vector = []
    bin1_vector = []
    bin2_vector = []

    psf = compute_psf_template(options, block)

    # Because we called to_canonical_order when we loaded the data,
    # we know that the data is grouped by data type, and then by tracers (tomo bins).
    # So that means we can do a data type at a time and then concatenate them, and
    # within this do a bin pair at a time, and concatenate them too.
    for b1, b2 in sacc_data.get_tracer_combinations(data_type):
        # Here we assume that the bin names are formatted such that
        # they always end with _1, _2, etc. That isn't always true in
        # sacc, but is somewhat baked into cosmosis in other modules.
        # It would be nice to update that throughout, but that will
        # have to wait. Also, cosmosis bins start from 1 not 0.
        # We need to make sure that's fixed in the 
        i = int(b1.split("_")[-1]) + 1
        j = int(b2.split("_")[-1]) + 1

        if data_type in kwargs.get("flip", False):
            i, j = j, i

        try:
            theory = block[section, f"bin_{i}_{j}"]
        except BlockError:
            if is_auto:
                theory = block[section, f"bin_{j}_{i}"]
            else:
                raise

        # check that all the data points share the same window
        # object (window objects contain weights for a set of ell / theta values,
        # as a matrix).
        window = None
        for d in sacc_data.get_data_points(data_type, (b1, b2)):
            w = d['window']
            if (window is not None) and (w is not window):
                raise ValueError("Sacc likelihood currently assumes data types share a window object")
            window = w

        if window is None:
            raise ValueError("HSC likelihood assumes a window")
        # We need to interpolate between the sample ell / theta values
        # onto all the ell / theta values required by the weight function
        # This will give zero outside the range where we have
        # calculated the theory
        theory_spline = SpectrumInterp(x_theory, theory)
        x_window = window.values
        theory_interpolated = theory_spline(x_window)

        for d in sacc_data.get_data_points(data_type, (b1, b2)):
            index = d['window_ind']
            x_nominal = d['ell']
            weight = window.weight[:, index]

            # The weight away should hopefully sum to 1 anyway but we should
            # probably not rely on that always being true.
            binned_theory = (weight @ theory_interpolated) / weight.sum() + psf[index]

            theory_vector.append(binned_theory)
            angle_vector.append(x_nominal)
            bin1_vector.append(i - 1)
            bin2_vector.append(j - 1)

    # Return the whole collection as a single array
    theory_vector = np.array(theory_vector)

    # For convenience we also save the angle vector (ell or theta)
    # and bin indices
    angle_vector = np.array(angle_vector)
    bin1_vector = np.array(bin1_vector, dtype=int)
    bin2_vector = np.array(bin2_vector, dtype=int)

    metadata = {
        "angle": angle_vector,
        "bin1": bin1_vector,
        "bin2": bin2_vector
    }

    return theory_vector, metadata



