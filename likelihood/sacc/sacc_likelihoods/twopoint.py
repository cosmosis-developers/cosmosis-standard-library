import numpy as np
from cosmosis.datablock import BlockError
import pathlib
import sys

# Get the SpectrumInterp class from the spec_tools module.
# Should really put this somewhere else!
twopoint_dir = pathlib.Path(__file__).parent.parent.parent.resolve() / "2pt"
sys.path.append(str(twopoint_dir))
from spec_tools import SpectrumInterp


def extract_spectrum_prediction(sacc_data, block, data_type, section, **kwargs):

    category = kwargs.get("category")
    if category == "spectrum":
        x_theory = block[section, "ell"]
    elif category == "real":
        x_theory = block[section, "theta"]
    #TO-DO: Decide on final nomenclature for cosebis and psi-stats!
    # Given current cosebis module in standard library, the x_nominal should be simply n
    elif category == "cosebis":
        x_theory = block[section, "n"]
    is_auto = block[section, "is_auto"]

    # We build up these vectors from all the data points.
    # Only the theory vector is needed for the likelihood - the others
    # are for convenience, debugging, etc.
    theory_vector = []
    angle_vector = []
    bin1_vector = []
    bin2_vector = []


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
        # as a matrix), or that none have windows.
        window = None
        for d in sacc_data.get_data_points(data_type, (b1, b2)):
            w = d.get_tag('window')
            if (window is not None) and (w is not window):
                raise ValueError("Sacc likelihood currently assumes data types share a window object")
            window = w

        # We need to interpolate between the sample ell / theta values
        # onto all the ell / theta values required by the weight function
        # This will give zero outside the range where we have
        # calculated the theory
        theory_spline = SpectrumInterp(x_theory, theory)
        if window is not None:
            x_window = window.values
            theory_interpolated = theory_spline(x_window)

        for d in sacc_data.get_data_points(data_type, (b1, b2)):
            if category == "spectrum":
                x_nominal = d['ell']
            elif category == "real":
                x_nominal = d['theta']
            #TO-DO: Decide on final nomenclature for cosebis and psi-stats!
            # Given current cosebis module in standard library, the x_nominal should be simply n
            elif category == "cosebis":
                x_nominal = d['n']

            if window is None:
                binned_theory = theory_spline(x_nominal)
            else:
                index = d['window_ind']
                weight = window.weight[:, index]

                # TO-DO: Check this for real statistics, but should be ok.
                # We don't automatically renormalize the weights.
                # Some contexts, like the output from NaMaster,
                # use non-unit-sum weights
                binned_theory = (weight @ theory_interpolated)

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
