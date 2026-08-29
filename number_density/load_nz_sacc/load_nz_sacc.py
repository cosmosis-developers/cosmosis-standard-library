import numpy as np
from cosmosis.datablock import names, option_section
import sacc
import scipy.interpolate

def setup(options):
    nz_file = options.get_string(option_section, "nz_file")
    prefix_section = options.get_bool(option_section, "prefix_section", True)
    data_sets = options.get_string(option_section, "data_sets").split()
    uncertainty_names = options.get_string(option_section, "uncertainty_names", default="").split()
    uncertainty_sections = options.get_string(option_section, "uncertainty_sections", default="nz_uncertainty").split()

    if not data_sets:
        raise RuntimeError(
            "Option data_sets empty; please set the option data_sets=name1 name2 etc and I will search the sacc file for those tracers")

    print("Loading number density data from {0}:".format(nz_file))

    s = sacc.Sacc.load_fits(nz_file)

    # Check if the requested uncertainties are present
    for uncertainty_name in uncertainty_names:
        if uncertainty_name and (uncertainty_name not in s.tracer_uncertainties):
            raise ValueError(f"Requested tracer uncertainty {uncertainty_name} not found in SACC file {nz_file}, available uncertainties: {s.tracer_uncertainties}")


    tracer_uncertainties = {
        name: s.tracer_uncertainties[name] 
        for name in uncertainty_names 
        if name in s.tracer_uncertainties
    }


    data = {}
    tracers_found = {name: False for name in data_sets}
    for tracer in s.tracers.values():
        # Check if we want to use this tracer
        for d in data_sets:
            if tracer.name.startswith(d):
                data_set = d
                tracers_found[d] = True
                break
        else:
            continue

        index = int(tracer.name.split("_")[-1])
        z = tracer.z
        nz = tracer.nz
        data[data_set, index] = (z, nz)

    # Check that the number of n(z) values matches the size of the linear transformation matrix
    for tracer_uncertainty in tracer_uncertainties.values():
        if not isinstance(tracer_uncertainty, sacc.tracer_uncertainty.NZLinearUncertainty):
            continue
        _, nzs_for_tu = get_nzs_for_uncertainty(data, tracer_uncertainty)
        nzs_for_tu_cat = np.concatenate(nzs_for_tu)
        n1 = len(nzs_for_tu_cat)
        n2 =  tracer_uncertainty.linear_transformation.shape[0]
        if not n1 == n2:
            raise ValueError(f"Number of n(z) values for tracer uncertainty {tracer_uncertainty.name} ({n1}) does not match the size of its linear transformation matrix ({tracer_uncertainty.linear_transformation.shape})"  )

    output = {
        "tracer_uncertainties": tracer_uncertainties,
        "uncertainty_sections": uncertainty_sections,
        "output_nz": data,
        "prefix_section": prefix_section,
    }

    # CosmoSIS is currently expecting all the bins to have the
    # same z grid. Changing this is possible but would require
    # some changes later in the pipeline, so we check here.
    # We also check for a few other possible issues, and slightly
    # reorder the data. This isn't critical, we could do this better.
    for data_set in data_sets:
        if not tracers_found[data_set]:
            raise ValueError(f"Requested data set {data_set} not found in SACC file {nz_file}, available tracers: {[tracer.name for tracer in s.tracers.values()]}")
        n = len([key for key in data.keys() if key[0] == data_set])
        z = None
        for i in range(n):
            if (data_set, i) not in data:
                raise ValueError(f"n(z) data in {data_set} not contiguous bins in file {nz_file}")
            zi, nz = data[data_set, i]
            if (z is not None) and (not np.allclose(zi, z)):
                raise ValueError(f"z values different for different bins in {nz_file}")



    return output


def execute(block, config):
    # Extract the various items we set during setup
    # nz_info is a dict like nz_info["source", 0] = (z, nz)
    nz_info = config["output_nz"]
    tracer_uncertainties = config["tracer_uncertainties"]
    uncertainty_sections = config["uncertainty_sections"]
    prefix_section = config["prefix_section"]

    # Generate the morphed n(z)s if we have any uncertainties to apply
    for tracer_uncertainty, uncertainty_section in zip(tracer_uncertainties.values(), uncertainty_sections):
        nz_info = update_nzs(block, nz_info, tracer_uncertainty, uncertainty_section)
        
    # Copy the nz_info (which may have been updated)
    # into the datablock
    counts = {}
    done_z = {}
    for (name, index), (z, nz) in nz_info.items():
        if prefix_section:
            name = "nz_" + name
        ns = len(z)
        if not done_z.get(name, False):
            block[name, "z"] = z
            block[name, "nz"] = ns
            done_z[name] = True
        counts[name] = counts.get(name, 0) + 1
        block[name, "bin_{0}".format(index + 1)] = nz

    # Record the number of bins in each section in the block
    for name, n in counts.items():
        block[name, "nbin"] = n
    
    return 0


def get_nzs_for_uncertainty(nz_info, tracer_uncertainty):
    nzs = []
    zs = []
    for name in tracer_uncertainty.tracer_names:
        name, index = name.split("_")
        index = int(index)
        z, nz = nz_info[name, index]
        nzs.append(nz)
        zs.append(z)
    return zs, nzs

def update_nzs(block, nz_info, tracer_uncertainty, uncertainty_section):
    """
    
    Parameters
    ----------
    nz_info : dict[str, tuple[np.ndarray, dict[int, np.ndarray]]]
    """
    ntracer = len(tracer_uncertainty.tracer_names)
    M = tracer_uncertainty.linear_transformation
    mu = tracer_uncertainty.mean

    # Get the n(z)s as a list in the order expected in the datablock
    zs, nzs = get_nzs_for_uncertainty(nz_info, tracer_uncertainty)

    # The alpha parameters are the normally distributed variables that
    # are converted into either new n(z) additions directly or into
    # shift/stretch parameters.
    alpha = read_alpha_parameters(block, tracer_uncertainty, uncertainty_section)
    mode = determine_tracer_uncertainty_type(tracer_uncertainty)

    # In the linear model case the values that are stored in the uncertainty
    # object are the vectors that apply to the n(z) itself, so we just need to do
    # a linear combination of these.
    # In this case there is a mean object attached to the uncertainty,
    # but it should always be zero.
    nzs_cat = np.concatenate(nzs)
    if mode == "linear_combination":
        new_nzs = nzs_cat + M @ alpha
        new_nzs = np.split(new_nzs, ntracer)
    else:
        # Otherwise, the values stored are shift (and perhaps stretch) values.
        # We need to generate them and then apply them to each n(z).
        if mode == "shift":
            shifts = mu + M @ alpha
            width = 1 + np.zeros_like(shifts)

            # write the shift parameters to the block for record-keeping
            for i, shift in enumerate(shifts):
                nm = tracer_uncertainty.tracer_names[i]
                block[uncertainty_section, f"bias_{nm}"] = shift

        elif mode == "shift_stretch":
            # In the shift-stretch case, the parameters are interleaved shift, stretch
            # values.
            params = mu + M @ alpha
            shifts = params[0::2]
            width = params[1::2]
        new_nzs = []

        # In either of these cases we apply the shift/width model to each n(z)
        for i, (z, nz) in enumerate(zip(zs, nzs)):
            print("Applying shift =", shifts[i], "width =", width[i], " to bin ", i)
            new_nz = shift_and_width_model(z, nz, shifts[i], width[i])
            new_nzs.append(new_nz)

    output = {}
    for (name, z, new_nz) in zip(tracer_uncertainty.tracer_names, zs, new_nzs):
        name, index = name.split("_")
        index = int(index)
        output[name, index] = (z, new_nz)

    # return the original nz_info updated with the new n(z)s
    return nz_info | output


def read_alpha_parameters(block, tracer_uncertainty, values_section):
    """
    Read the (usually normally-distributed) alpha parameters from the datablock.
    """
    # Read the alpha parameters from the datablock.
    # We start these at zero instead of one to match SACC convention
    nparam = len(tracer_uncertainty.tracer_names) * tracer_uncertainty.nparams
    alpha = np.zeros(nparam)
    for i in range(nparam):
        alpha[i] = block[values_section, f"alpha_{i}"]
    return alpha


def determine_tracer_uncertainty_type(tracer_uncertainty):
    """
    Determine what type of tracer uncertainty we have been given, from
    the types available in Sacc.
    """
    # check what type of uncertainty we have - this determmines how we change the n(z).
    if isinstance(tracer_uncertainty, sacc.tracer_uncertainty.NZShiftUncertainty):
        mode = "shift"
    elif isinstance(tracer_uncertainty, sacc.tracer_uncertainty.NZShiftStretchUncertainty):
        mode = "shift_stretch"
    elif isinstance(tracer_uncertainty, sacc.tracer_uncertainty.NZLinearUncertainty):
        mode = "linear_combination"
    else:
        raise ValueError("Unsupported tracer uncertainty type")
    return mode

# This is currently stolen directly from nz_prior
def shift_and_width_model(z, nz, shift, width):
    """
    Aplies a shift and a width to the given p(z) distribution.
    This is done by evluating the n(z) distribution at
    p((z-mu)/width + mu + shift) where mu is the mean redshift
    of the fiducial n(z) distribution and the rescaling by the width.
    Finally the distribution is normalized.
    """
    nz_i = scipy.interpolate.interp1d(z, nz, kind="linear", fill_value=0.0, bounds_error=False)
    mu = np.average(z, weights=nz)
    pdf = nz_i((z - mu - shift) * width + mu)
    # The normnalization does not matter in cosmosis
    # as we always renormalize later, but so we do not normalize here.
    return pdf
