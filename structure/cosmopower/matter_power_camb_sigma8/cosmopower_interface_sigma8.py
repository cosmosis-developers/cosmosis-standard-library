import os
from cosmosis.datablock import names, option_section
from scipy.interpolate import CubicSpline
import numpy as np
import sys

# This tells CUDA always to use the CPU, not the GPU
# This is no slower.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cosmopower as cp

# These are pre-defined strings we use as datablock section names
cosmo = names.cosmological_parameters
distances = names.distances

dirname = os.path.split(__file__)[0]
default_emu_dir = os.path.join(dirname, "emu_files")


def setup(options):

    emulators = options.get_string(option_section, "emulators", default="both")
    z_section = options.get_string(option_section, "z_section", default="nz_source")
    use_specific_k_modes = options.get_bool(
        option_section, "use_specific_k_modes", default=False
    )

    config = {
        "z_section": z_section,
    }

    # user can ask for specific k values to be used, in which case we will rebin the
    # default cosmopower trained k values to them
    if use_specific_k_modes:
        kmin = options.get_double(option_section, "kmin", default=1e-5)
        kmax = options.get_double(option_section, "kmax", default=10.0)
        nk = options.get_int(option_section, "nk", default=200)
        config["k"] = np.logspace(np.log10(kmin), np.log10(kmax), num=nk)

    # Create the object that connects to cosmopower
    # load pre-trained NN model: maps cosmological parameters to linear log-P(k)
    emulator_dir = options.get_string(option_section, "emulator_dir", default=default_emu_dir)
    print(emulator_dir)
    if emulators not in ["nonlinear", "linear", "both"]:
        raise ValueError(
            f"Invalid value for 'emulators' option {emulators}. Should be one of: [linear, nonlinear, both]"
        )

    # Load the linear emulator object and reference spectrum, if requested
    if emulators == "both" or emulators == "linear":
        emu_filename = os.path.join(
            emulator_dir, "log10_reference_lin_matter_power_emulator_mead2020_feedback"
        )
        ref_filename = os.path.join(
            emulator_dir, "center_linear_matter_mead2020_feedback.npz"
        )
        print(emu_filename)

        config["lin_matter_power_cp"] = cp.cosmopower_NN(
            restore=True, restore_filename=emu_filename
        )
        config["reference_linear_spectra"] = np.log10(np.load(ref_filename)["features"])

    # Load the non-linear emulator object and reference spectrum, if requested
    if emulators == "both" or emulators == "nonlinear":
        emu_filename = os.path.join(
            emulator_dir,
            "log10_reference_non_lin_matter_power_emulator_mead2020_feedback",
        )
        ref_filename = os.path.join(
            emulator_dir, "center_non_linear_matter_mead2020_feedback.npz"
        )
        config["nonlin_matter_power_cp"] = cp.cosmopower_NN(
            restore=True, restore_filename=emu_filename
        )
        config["reference_nonlinear_spectra"] = np.log10(
            np.load(ref_filename)["features"]
        )

    # Return all this config information
    return config


def get_cosmopower_inputs(block, z, do_nonlinear):

    # Get the basic parameters we needf for both prediction sets (NL and Linear)
    # Note the slightly mixed naming conventeion - obh2 but omch2 - this is not an error.
    params = {
        "sigma8": block[cosmo, "sigma_8"],
        "n_s": block[cosmo, "n_s"],
        "h": block[cosmo, "h0"],
        "obh2": block[cosmo, "ombh2"],
        "omch2": block[cosmo, "omch2"],
    }

    # The NL prediction needs this one too.
    if do_nonlinear:
        params["log_T_AGN"] = block.get_double(names.halo_model_parameters, "logT_AGN")

    # These are the ranges of the parameters that the emulator was trained on
    ranges = {
        "sigma8": [0.39, 1.01],
        "n_s": [0.84, 1.1],
        "h": [0.64, 0.82],
        "obh2": [0.019, 0.026],
        "omch2": [0.051, 0.255],
        "z": [0, 6.0],
        "log_T_AGN": [6.5, 9.36],
    }

    # Check that all the parameters are in the required ranges
    for param in params:
        pmin, pmax = ranges[param]
        if params[param] < pmin or params[param] > pmax:
            raise ValueError(
                f"Cosmopower: {param} out of range: {params[param]} not in [{pmin}, {pmax}]"
            )

    # The redshifts must also be in the required range.
    zmin, zmax = ranges["z"]
    if z[0] < zmin or z[-1] > zmax:
        raise ValueError(
            f"Redshifts out of range: {z[0]} to {z[-1]} not in [{zmin}, {zmax}]"
        )

    # These parameters were fixed during the training of the emulator,
    # so we check they are the same here
    fixed = {"omega_k": 0.0, "w": -1.0, "wa": 0.0, "mnu": 0.06}
    for name, val in fixed.items():
        if block[cosmo, name] != val:
            raise ValueError(f"Parameter {name} must be fixed at {val}")

    # The cosmopower interface expects an array of parameter values for each redshift
    # that we are emulating, even when the parameters are all the same
    linear_params = {p: np.full(z.size, v) for p,v in params.items()}
    linear_params["z"] = z

    # The non-linear parameters are the same as the linear ones, with one added.
    if do_nonlinear:
        nl_params = linear_params.copy()
        nl_params["log_T_AGN"] = np.full(
            z.size, block.get_double(names.halo_model_parameters, "logT_AGN")
        )
    else:
        nl_params = {}

    return linear_params, nl_params


def rebin(P, k, k_new):
    """
    Re-bin a matter power spectrum to new k values
    using a separate cubic spline for each redshift

    Parameters
    ----------
    P : array
        The power spectrum to rebin
    k : array
        The original k values
    k_new : array
        The new k values
    """
    P_new = np.zeros(shape=(P.shape[0], len(k_new)))
    for i in range(P.shape[0]):
        P_spline = CubicSpline(k, P[i])
        P_new[i] = P_spline(k_new)
    return P_new


def get_predictions(params, network, reference, k_rebin):
    k = network.modes

    # Get the NL prediction, and convert to physical values
    P = network.predictions_np(params)
    for i in range(P.shape[0]):
        P[i] = P[i] + reference 
    P = 10 ** P

    # If necessary, rebin the power spectrum to the requested k values
    if k_rebin is not None:
        P = rebin(P, k, k_rebin)
        k = k_rebin

    return k, P


def execute(block, config):
    # For convenience we ensure that the S_8 parameter is saved here
    sigma8 = block[cosmo, "sigma_8"]
    omegam = block[cosmo, "omega_m"]
    s8 = sigma8 * np.sqrt(omegam / 0.3)
    block.put(cosmo, "S_8", s8)

    # H0 is needed for the k/h conversion
    h0 = block[cosmo, "h0"]

    z = block["nz_source", "z"]

    do_linear = "lin_matter_power_cp" in config
    do_nonlinear = "nonlin_matter_power_cp" in config

    try:
        linear_params, nl_params = get_cosmopower_inputs(block, z, do_nonlinear)
    except ValueError as e:
        sys.stderr.write(f"Cosmopower range error: {e}\n")
        return 1

    if do_nonlinear:
        # Get the cosmopower predictions for the non-linear power spectrum
        network = config["nonlin_matter_power_cp"]
        reference = config["reference_nonlinear_spectra"]
        k_nl, P_nl = get_predictions(nl_params, network, reference, config.get("k"))

        # Save to the block as a grid
        block.put_grid("matter_power_nl", "z", z, "k_h", k_nl / h0, "p_k", P_nl * h0**3)

    if do_linear:
        # Get the cosmopower predictions for the linear power spectrum
        network = config["lin_matter_power_cp"]
        reference = config["reference_linear_spectra"]
        k_lin, P_lin = get_predictions(linear_params, network, reference, config.get("k"))

        # Save to the block as a grid
        block.put_grid(
            "matter_power_lin", "z", z, "k_h", k_lin / h0, "p_k", P_lin * h0**3
        )

    return 0
