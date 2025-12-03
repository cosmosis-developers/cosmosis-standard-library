from cosmosis.datablock import names, option_section as opt
import numpy as np
import traceback
import sys
import pathlib
import pickle
from scipy.interpolate import CubicSpline, UnivariateSpline

# Finally we can now import camb
import camb

# Get the camb functions from the camb_interface module.
# Should really put this somewhere else!
camb_dir = pathlib.Path(__file__).parent.parent.resolve() / "camb"
sys.path.append(str(camb_dir))
from camb_interface import matter_power_section_names, be_quiet_camb, get_optional_params, get_choice, make_z_for_pk, extract_camb_params, extract_recombination_params, extract_reionization_params, extract_dark_energy_params, extract_initial_power_params, extract_nonlinear_params, save_derived_parameters, save_distances, setup as setup_camb

# TODO: maybe we import camb background calculations from camb_interface, so it is eaiser to update things?

import cosmopower as cp
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to use CPU if GPU is avalible otherwise the GPU memory runs out of memory. It also does not slower the predtiction.

cosmo = names.cosmological_parameters

MODE_BG = 'background'
MODE_THERM = 'thermal'
MODE_CMB = 'cmb'
MODE_POWER = 'power'
MODE_ALL = 'all'
MODES = [MODE_BG, MODE_THERM, MODE_CMB, MODE_POWER, MODE_ALL]

DEFAULT_A_S = 2.1e-9

C_KMS = 299792.458


camb_section_names = ['cosmological_parameters', 'halo_model_parameters', 'recfast', 'reionization', 'de_equation_of_state']

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

    # Get the prediction, and convert to physical values
    logP = network.predictions_np(params)
    for i in range(P.shape[0]):
        logP[i] = logP[i] + reference 
    P = 10 ** logP

    # If necessary, rebin the power spectrum to the requested k values
    if k_rebin is not None:
        P = rebin(P, k, k_rebin)
        k = k_rebin

    return k, P

def set_params(block, params_in, fixed_params, limits, z):
    """
    Get all parameters from block and give them the
    names and form that cosmopower expects.

    For free parameters, it checks that they are in the required ranges. 
    For fixed params, it checks that they are the same as used in training.
    """

    params = {}
    for param in params_in:
        for name in camb_section_names:
            if block.has_value(name, param):
                params[param] = [block.get(name, param)]

    # Check that all the parameters are in the required ranges
    for param in params:
        pmin, pmax = limits[param]
        if params[param][0] < pmin or params[param][0] > pmax:
            raise ValueError(
                f"Cosmopower: {param} out of range: {params[param][0]} not in [{pmin}, {pmax}]"
            )

    # The redshifts must also be in the required range.
    zmin, zmax = limits["z"]
    if z[0] < zmin or z[-1] > zmax:
        raise ValueError(
            f"Redshifts out of range: {z[0]} to {z[-1]} not in [{zmin}, {zmax}]"
        )

    # These parameters were fixed during the training of the emulator,
    # so we check they are the same here
    for name, val in fixed_params.items():
        for section in camb_section_names:
            if block.has_value(section, name):
                if block[section, name] != val:
                    raise ValueError(f"Parameter {name} must be fixed at {val}")
        
    return params

def set_params_cls(block):
    # Get parameters from block and give them the
    # names and form that cosmopower expects
    params = {
        'omega_b': [block[cosmo, 'ombh2']],
        'omega_cdm': [block[cosmo, 'omch2']],
        'h': [block[cosmo, 'h0']],
        'tau_reio': [block[cosmo, 'tau']],
        'n_s': [block[cosmo, 'n_s']],
        'ln10^{10}A_s': [np.log(block[cosmo, 'A_s'] * 1e10)]
    }
    limits  = {
        'omega_b': [0.005, 0.04],
        'omega_cdm': [0.001, 0.99],
        'h': [0.2, 1.0],
        'tau_reio': [0.01, 0.8],
        'n_s': [0.7, 1.3],
        'ln10^{10}A_s': [1.61, 5]
    }
    for param in params:
        pmin, pmax = limits[param]
        if params[param][0] < pmin or params[param][0] > pmax:
            raise ValueError(
                f"Cosmopower: {param} out of range: {params[param][0]} not in [{pmin}, {pmax}]"
            )
    # We have no idea what the fixed parameters were in training the CMB spectra, we hope for the best!
    return params

def setup(options):
    config, more_config = setup_camb(options)

    # Cosmopower specific settings!
    cosmopower_root_filename = options.get_string(opt, 'cosmopower_folder', default='')
    if config['WantTransfer']:
        more_config['cosmopower_k'] = options.get_bool(
            opt, 'use_cosmopower_kvec', default=False
        )

        more_config['fixed_params'] = pickle.load(
            open(
                f'{cosmopower_root_filename}/cosmopower_emulator_fixed_params.pkl',
                'rb'
            )
        )

        more_config['limits'] = pickle.load(
            open(
                f'{cosmopower_root_filename}/cosmopower_emulator_param_limits.pkl',
                'rb'
            )
        )

        # We require delta_tot case to be present in order to get the growth parameters!
        if 'delta_tot' not in more_config['power_spectra']:
            raise ValueError(
                "We require 'delta_tot' to be present in 'power_spectra'. "
                "Rerun the training with this option enabled in CAMB!"
            )

        for transfer_type in more_config['power_spectra']:
            name = matter_power_section_names[transfer_type]

            more_config[f'{name}_lin_cp'] = cp.cosmopower_NN(
                restore=True,
                restore_filename=(
                    f'{cosmopower_root_filename}/cosmopower_emulator_{name}_lin'
                )
            )

            more_config[f'{name}_nl_cp'] = (
                cp.cosmopower_NN(
                    restore=True,
                    restore_filename=(
                        f'{cosmopower_root_filename}/cosmopower_emulator_{name}_nl'
                    )
                )
                if config['NonLinear'] != 'NonLinear_none'
                else None
            )

            more_config[f'reference_{name}_lin_cp'] = np.log10(
                pickle.load(
                    open(
                        f'{cosmopower_root_filename}/'
                        f'cosmopower_emulator_{name}_lin_reference.pkl',
                        'rb'
                    )
                )
            )

            more_config[f'reference_{name}_nl_cp'] = (
                np.log10(
                    pickle.load(
                        open(
                            f'{cosmopower_root_filename}/'
                            f'cosmopower_emulator_{name}_nl_reference.pkl',
                            'rb'
                        )
                    )
                )
                if config['NonLinear'] != 'NonLinear_none'
                else None
            )

                            
    # CosmoPower cls
    if config['WantCls']:
        print(
            "WARNING: Using CosmoPower CMB emulators for TT, TE, EE and PP! "
            "These are assumed to be trained by Spurio Mancini et al. 2021 and may "
            "not be accurate for any use case! If you are using some other "
            "emulators for the CMB spectra, please be warned!"
        )

        more_config['TT_cp'] = cp.cosmopower_NN(
            restore=True,
            restore_filename=f'{cosmopower_root_filename}/cmb_TT_NN'
        )

        more_config['TE_cp'] = cp.cosmopower_PCAplusNN(
            restore=True,
            restore_filename=f'{cosmopower_root_filename}/cmb_TE_PCAplusNN'
        )

        more_config['EE_cp'] = cp.cosmopower_NN(
            restore=True,
            restore_filename=f'{cosmopower_root_filename}/cmb_EE_NN'
        )

        more_config['PP_cp'] = cp.cosmopower_PCAplusNN(
            restore=True,
            restore_filename=f'{cosmopower_root_filename}/cmb_PP_PCAplusNN'
        )


    return [config, more_config]

def compute_growth_factor(P_tot, z):
    P_kmin = P_tot[:,0]
    D = np.sqrt(P_kmin / P_kmin[0]).squeeze()
    
    logD = np.log(D)

    a = 1.0/(1.0+z)
    loga = np.log(a)

    logD_spline = UnivariateSpline(loga[::-1], logD[::-1])
    f_spline = logD_spline.derivative()
    f = f_spline(loga)
    
    return D, f

def window(k_mode):
    R = 8 # in units Mpc/h which is correct since k is in h/Mpc
    return 3*(np.sin(k_mode*R)-k_mode*R*np.cos(k_mode*R))/(k_mode*R)**3

def save_matter_power(r, block, config, more_config):
    p = r.Params
    # Grids in k, z on which to save matter power.
    # There are two kmax values - the max one calculated directly,
    # and the max one extrapolated out too. We output to the larger of these.
    kmax_power = max(more_config['kmax'], more_config['kmax_extrapolate'])
    z = make_z_for_pk(more_config)[::-1]
    nz = len(z)

    k_rebin = None
    if not more_config['cosmopower_k']:
        k_rebin = np.logspace(
            np.log10(6.712847971357405e-05),
            np.log10(kmax_power),
            more_config['nk']
        )
        # For some reason the kmin is in CAMB set to 1e-4 * 0.671284...
        # and it doesn't seem to change with varying h0.

    P_tot = None
    for transfer_type in more_config['power_spectra']:
        name = matter_power_section_names[transfer_type]

        params = set_params(
            block,
            more_config[f'{name}_lin_cp'].parameters,
            more_config['fixed_params'],
            more_config['limits'],
            z
        )

        # The cosmopower interface expects an array of parameter values for
        # each redshift that we are emulating, even when parameters are the same.
        params = {par: np.full(z.size, v) for par, v in params.items()}
        params['z'] = z

        k, P_lin = get_predictions(
            params,
            more_config[f'{name}_lin_cp'],
            more_config[f'reference_{name}_lin_cp'],
            k_rebin
        )

        # Save matter power as a grid
        block.put_grid(f'{name}_lin', 'z', z, 'k_h', k, 'p_k', P_lin)

        if config['NonLinear'] != 'NonLinear_none':
            k, P_nl = get_predictions(
                params,
                more_config[f'{name}_nl_cp'],
                more_config[f'reference_{name}_nl_cp'],
                k_rebin
            )
            block.put_grid(f'{name}_nl', 'z', z, 'k_h', k, 'p_k', P_nl)

        # Save this for the growth rate later
        if transfer_type == 'delta_tot':
            P_tot = P_lin

        # Assumes we have P_lin at z=0!
        # matter_power = primordial_PK * transfer**2 * k**4 /
        #                (k**3 / (2 * np.pi**2))
        primordial_PK = p.scalar_power(
            k * block[names.cosmological_parameters, 'h0']
        )
        transfer = np.sqrt(
            P_lin[0, :] / (primordial_PK * k * 2.0 * np.pi**2.0)
        )
        block.put_double_array_1d(f'{name}_transfer_func', 'k_h', k)
        block.put_double_array_1d(f'{name}_transfer_func', 't_k', transfer)

    # Get growth rates and sigma_8
    rs_DV, H, DA, F_AP = r.get_BAO(z, p).T
    D, f = compute_growth_factor(P_tot, z)

    # Save growth rates and sigma_8
    block[names.growth_parameters, 'z'] = z
    block[names.growth_parameters, 'a'] = 1 / (1 + z)
    block[names.growth_parameters, 'rs_DV'] = rs_DV
    block[names.growth_parameters, 'H'] = H
    block[names.growth_parameters, 'DA'] = DA
    block[names.growth_parameters, 'F_AP'] = F_AP
    block[names.growth_parameters, 'd_z'] = D
    block[names.growth_parameters, 'f_z'] = f

    if not block.has_value(names.cosmological_parameters, 'sigma_8'):
        sigma_sq_cs = CubicSpline(
            k,
            k**2 * window(k)**2 * P_tot[0] / (2 * np.pi**2)
        )
        sigma_8 = np.sqrt(sigma_sq_cs.integrate(k.min(), k.max()))
        block[names.cosmological_parameters, 'sigma_8'] = sigma_8
        block[names.cosmological_parameters, 'S_8'] = (
            sigma_8 * np.sqrt(p.omegam / 0.3)
        )
    else:
        block[names.cosmological_parameters, 'S_8'] = (
            block[names.cosmological_parameters, 'sigma_8']
            * np.sqrt(p.omegam / 0.3)
        )


def save_cls(r, block, more_config):
    """
    Save CMB power spectra to the block.

    For now this only uses the pre-trained models from Spurio Mancini et al. 2021, 
    as the CosmoPower training in CosmoSIS is not yet set up to do the training for the CMB spectra.
    """
    
    params = set_params_cls(block)
    cmb_unit = (2.7255e6)**2 #muK
    tt_spectra = more_config['TT_cp'].ten_to_predictions_np(params) * cmb_unit
    te_spectra = more_config['TE_cp'].predictions_np(params) * cmb_unit
    ee_spectra = more_config['EE_cp'].ten_to_predictions_np(params) * cmb_unit
    pp_spectra = more_config['PP_cp'].ten_to_predictions_np(params)
    ell = more_config['TT_cp'].modes
    # Planck likelihood requires (ell*(ell+1))/(2*pi) C_ell
    block[names.cmb_cl, 'TT'] = tt_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block[names.cmb_cl, 'TE'] = te_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block[names.cmb_cl, 'EE'] = ee_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block[names.cmb_cl, 'PP'] = pp_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block[names.cmb_cl, 'ell'] = ell


def execute(block, config):
    config, more_config = config
    p = "<Error occurred during parameter setup>"
    try:
        p = extract_camb_params(block, config, more_config)
        r = camb.get_background(p)

    except camb.CAMBError:
        if more_config['n_printed_errors'] <= more_config['max_printed_errors']:
            print("CAMB error caught: for these parameters")
            print(p)
            print(traceback.format_exc())
            if more_config['n_printed_errors'] == more_config['max_printed_errors']:
                print("\nFurther errors will not be printed.")
            more_config['n_printed_errors'] += 1
        return 1

    with be_quiet_camb():
        save_derived_parameters(r, block)
        save_distances(r, block, more_config)
    
    if p.WantTransfer:
        with be_quiet_camb():
            save_matter_power(r, block, config, more_config)

    if p.WantCls:
        with be_quiet_camb():
            save_cls(r, block, more_config)
    
    return 0