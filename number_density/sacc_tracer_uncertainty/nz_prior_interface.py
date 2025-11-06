from cosmosis.datablock import option_section
import sacc
from scipy.interpolate import interp1d
import numpy as np

def setup(options):
    # Load the SACC file specified
    sacc_file = options.get_string(option_section, "sacc_file")    
    section_names = options.get_string(option_section, "section_names").split()
    values_section = options.get_string(option_section, "values_section", default="nz_uncertainty")
    uncertainty_name = options.get_string(option_section, "uncertainty_name", default="")
    sacc_data = sacc.Sacc.load_fits(sacc_file)

    found_uncertainties = list(sacc_data.tracer_uncertainties.keys())
    # Read all the tracer uncertainty objects from the SACC file.
    # There should only be one.
    if uncertainty_name:
        if uncertainty_name not in sacc_data.tracer_uncertainties:
            raise ValueError(f"Requested tracer uncertainty {uncertainty_name} not found in SACC file {sacc_file}, available uncertainties: {found_uncertainties}")
        tracer_uncertainty = sacc_data.tracer_uncertainties[uncertainty_name]
    else:
        n_uncertainty = len(sacc_data.tracer_uncertainties)
        if n_uncertainty != 1:
            raise ValueError(f"Only one tracer uncertainty is supported currently, found {n_uncertainty}, called: {found_uncertainties}. Set uncertainty_name option to select one.")
        tracer_uncertainty = list(sacc_data.tracer_uncertainties.values())[0]


    # check what type of uncertainty we have - this determmines how we change the n(z).
    if isinstance(tracer_uncertainty, sacc.tracer_uncertainty.NZShiftUncertainty):
        mode = "shift"
    elif isinstance(tracer_uncertainty, sacc.tracer_uncertainty.NZShiftStretchUncertainty):
        mode = "shift_stretch"
    elif isinstance(tracer_uncertainty, sacc.tracer_uncertainty.NZLinearUncertainty):
        mode = "linear_combination"
    else:
        raise ValueError("Unsupported tracer uncertainty type")
    
    print("Uncertainty is expecting to apply to tracers:")
    print(tracer_uncertainty.tracer_names)
    print("mean = ", tracer_uncertainty.mean)
    print("linear transformation shape = ", tracer_uncertainty.linear_transformation.shape)

        
    return {"mode": mode, "section_names": section_names, "tracer_uncertainty": tracer_uncertainty, "values_section": values_section}

def read_nzs(block, section_names):
    # We don't have any way to check that the user specified the section names
    # in the same order. This is bad.

    zs = []
    nzs = []
    for section_name in section_names:
        z = block[section_name, "z"]

        # CosmoSIS bin numbers start at one.
        # Read all the n(z) bins for this section.
        i = 1
        while block.has_value(section_name, f"bin_{i}"):
            nz = block[section_name, f"bin_{i}"]
            zs.append(z)
            nzs.append(nz)
            i += 1

    return zs, nzs

def update_nzs(zs, nzs, tracer_uncertainty, alpha, mode):
    ntracer = len(tracer_uncertainty.tracer_names)
    M = tracer_uncertainty.linear_transformation
    mu = tracer_uncertainty.mean

    # In the linear model case the values that are stored in the uncertainty
    # object are the vectors that apply to the n(z) itself, so we just need to do
    # a linear combination of these.
    if mode == "linear_combination":
        new_nzs = np.concatenate(nzs) + M @ alpha
        new_nzs = np.split(new_nzs, ntracer)
    else:
        # Otherwise, the values stored are shift (and perhaps stretch) values.
        # We need to generate them and then apply them to each n(z).
        if mode == "shift":
            shifts = mu + M @ alpha
            width = 1 + np.zeros_like(shifts)
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

    return new_nzs

def replace_nzs(block, section_names, new_nzs):
    # Save the new n(z) back to the datablock
    i = 0
    for section_name in section_names:
        j = 1
        while block.has_value(section_name, f"bin_{j}"):
            block[section_name, f"bin_{j}"] = new_nzs[i]
            j += 1
            i += 1


def execute(block, config):
    mode = config["mode"]
    tracer_uncertainty = config["tracer_uncertainty"]
    section_names = config["section_names"]
    values_section = config["values_section"]

    # Work out number of parameters that we need
    ntracer = len(tracer_uncertainty.tracer_names)
    zs, nzs = read_nzs(block, section_names)

    if len(nzs) != ntracer:
        raise ValueError(f"Number of tracers in SACC file ({ntracer}) does not match number of bins in datablock ({len(nzs)})")

    # Read the alpha parameters from the datablock.
    # We start these at zero instead of one to match SACC convention
    nparam = ntracer * tracer_uncertainty.nparams
    alpha = np.zeros(nparam)
    for i in range(nparam):
        alpha[i] = block[values_section, f"alpha_{i}"]

    # Get the new n(z)s, depending on the specific method
    new_nzs = update_nzs(zs, nzs, tracer_uncertainty, alpha, mode)

    # Update the data block with the new n(z)s
    replace_nzs(block, section_names, new_nzs)

    return 0


# This is currently stolen directly from nz_prior
def shift_and_width_model(z, nz, shift, width):
    """
    Aplies a shift and a width to the given p(z) distribution.
    This is done by evluating the n(z) distribution at
    p((z-mu)/width + mu + shift) where mu is the mean redshift
    of the fiducial n(z) distribution and the rescaling by the width.
    Finally the distribution is normalized.
    """
    nz_i = interp1d(z, nz, kind="linear", fill_value="extrapolate")
    mu = np.average(z, weights=nz)
    pdf = nz_i((z - mu + shift) / width + mu)
    norm = np.sum(pdf)
    return pdf / norm
