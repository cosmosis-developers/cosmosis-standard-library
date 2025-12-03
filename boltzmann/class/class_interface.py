from builtins import str
import os
from cosmosis.datablock import names, option_section
import sys
import traceback
import warnings

# add class directory to the path
dirname = os.path.split(__file__)[0]
# enable debugging from the same directory
if not dirname.strip():
    dirname = '.'

import numpy as np

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters
distances = names.distances
cmb_cl = names.cmb_cl


def setup(options):
    class_version = options.get_string(option_section, "version", "3.2.5")
    install_dir = dirname + f"/class_v{class_version}/classy_install/"
    sys.path.insert(0, install_dir)

    import classy
    print(f"Loaded classy from {classy.__file__}")

    # Read options from the ini file which are fixed across
    # the length of the chain
    config = {
        'lmax': options.get_int(option_section, 'lmax', default=2000),
        'zmax': options.get_double(option_section, 'zmax', default=3.0),
        'kmax': options.get_double(option_section, 'kmax', default=10.0),
        'nk': options.get_int(option_section, 'nk', default=100),
        'debug': options.get_bool(option_section, 'debug', default=False),
        'lensing': options.get_bool(option_section, 'lensing', default=True),
        'cmb': options.get_bool(option_section, 'cmb', default=True),
        'mpk': options.get_bool(option_section, 'mpk', default=True),
        'save_matter_power_lin': options.get_bool(option_section, 'save_matter_power_lin', default=True),
        'save_cdm_baryon_power_lin': options.get_bool(option_section, 'save_cdm_baryon_power_lin', default=False),
        'background_only': options.get_bool(option_section, "background_only", default=False)
    }

    # cannot request cmb or mpk when background_only=True
    assert not (
        config['background_only'] and (config['cmb'] or config['mpk'] or config['lensing'])
    ), "Cannot request cmb or mpk when background_only=True"

    for _, key in options.keys(option_section):
        if key.startswith('class_'):
            config[key] = options[option_section, key]


    # Create the object that connects to Class
    config['cosmo'] = classy.Class()

    # Return all this config information
    return config

def choose_outputs(config):
    outputs = []
    if config['cmb']:
        outputs.append("tCl")
        outputs.append("pCl")
    if config['lensing']:
        outputs.append("lCl")
    if config["mpk"]:
        outputs.append("mPk")
    return ",".join(outputs)

def get_N_ur(nmassive):
    """
    Returns the number of massless neutrino species
        (ultra relativistic species).
    Values are taken from the CLASS documentation: 
    https://github.com/lesgourg/class_public/blob/0ceb7a9a4c1e444ef5d5d56a8328a0640be91b18/explanatory.ini#L304
    and reflect a  neff = 3.044 for the standard 
    model of cosmology.
    """
    if nmassive == 0:
        return 3.044
    elif nmassive == 1:
        return 2.0308
    elif nmassive == 2:
        return 1.0176
    elif nmassive == 3:
        return 0.00441
    else:
        raise ValueError(f"Invalid number of massive neutrinos: {nmassive}. "
                         "Must be 0, 1, 2, or 3.")

def get_class_inputs(block, config):

    # Get parameters from block and give them the
    # names and form that class expects
    nnu = block.get_double(cosmo, 'nnu', 3.044)
    nmassive = block.get_int(cosmo, 'num_massive_neutrinos', default=0)
    params = {
        'output': choose_outputs(config),
        'lensing':   'yes' if config['lensing'] else 'no',
        'A_s':       block[cosmo, 'A_s'],
        'n_s':       block[cosmo, 'n_s'],
        'h':        block[cosmo, 'h0'],
        'omega_b':   block[cosmo, 'ombh2'],
        'omega_cdm': block[cosmo, 'omch2'],
        'tau_reio':  block[cosmo, 'tau'],
        'T_cmb':     block.get_double(cosmo, 'TCMB', default=2.726),
        'N_ur':      block.get_double(cosmo, 'N_ur', default=get_N_ur(nmassive)),
        'N_ncdm':    nmassive,
        'l_switch_limber': 9,
    }


    neutrino_mass_total = block.get_double(cosmo, 'mnu', default=0.06)
    if nmassive == 0:
        if neutrino_mass_total == 0.06:
            assert neutrino_mass_total == 0, f"mnu must equal 0 when num_massive_neutrinos=0, but you had {neutrino_mass_total} - you may have left it at the default value"
        else:
            assert neutrino_mass_total == 0, f"mnu must equal 0 when num_massive_neutrinos=0, but you had {neutrino_mass_total}"
    else:
        neutrino_masses = np.repeat(neutrino_mass_total / nmassive, nmassive).astype(str)
        params['m_ncdm'] = ", ".join(neutrino_masses)


    if config["cmb"] or config["lensing"]:
        params.update({
          'l_max_scalars': config["lmax"],
        })
  

    if config["mpk"]:
        params.update({
            'P_k_max_h/Mpc':  config["kmax"],
            # 'z_pk': ', '.join(str(z) for z in np.arange(0.0, config['zmax'], 0.01)),
            'z_max_pk': config['zmax'],
        })



    if block.has_value(cosmo, "massless_nu"):
        warnings.warn("Parameter massless_nu is being ignored. Set nnu, the effective number of relativistic species in the early Universe.")

    if (block.has_value(cosmo, "omega_nu") or block.has_value(cosmo, "omnuh2")) and not (block.has_value(cosmo, "mnu")):
        warnings.warn("Parameter omega_nu and omnuh2 are being ignored. Set mnu and num_massive_neutrinos instead.")


    for key,val in config.items():
        if key.startswith('class_'):
            params[key[6:]] = val

    return params




def get_class_outputs(block, c, config):
    ##
    # Derived cosmological parameters
    ##

    h0 = block[cosmo, 'h0']

    ##
    # Matter power spectrum
    ##

    # Ranges of the redshift and matter power
    dz = 0.01
    kmin = 1e-4
    kmax = config['kmax'] * h0
    nk = config["nk"]

    # Define k,z we want to sample
    z = np.arange(0.0, config["zmax"] + dz, dz)
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    nz = len(z)

    if not config['background_only']:
        # Extract (interpolate) P(k,z) at the requested
        # sample points.
        if 'mPk' in c.pars['output']:
            block[cosmo, 'sigma_8'] = c.sigma8()

            # Total matter power spectrum (saved as grid)
            if config['save_matter_power_lin']:
                P = np.zeros((k.size, z.size))
                for i, ki in enumerate(k):
                    for j, zi in enumerate(z):
                        P[i, j] = c.pk_lin(ki, zi)
                # We transpose P here to match the camb convention
                block.put_grid("matter_power_lin", "z", z, "k_h", k / h0, "p_k", P.T * h0**3)

            # CDM+baryons power spectrum
            if config['save_cdm_baryon_power_lin']:
                P = np.zeros((k.size, z.size))
                for i, ki in enumerate(k):
                    for j, zi in enumerate(z):
                        P[i, j] = c.pk_cb_lin(ki, zi)
                block.put_grid('cdm_baryon_power_lin', 'z', z, 'k_h', k/h0, 'p_k', P.T * h0 **3)

            # Get growth rates and sigma_8
            D = [c.scale_independent_growth_factor(zi) for zi in z]
            f = [c.scale_independent_growth_factor_f(zi) for zi in z]
            fsigma = [c.effective_f_sigma8(zi) for zi in z]
            sigma_8_z = [c.sigma(8.0, zi, h_units=True) for zi in z]
            block[names.growth_parameters, "z"] = z
            block[names.growth_parameters, "sigma_8"] = np.array(sigma_8_z)
            block[names.growth_parameters, "fsigma_8"] = np.array(fsigma)
            block[names.growth_parameters, "d_z"] = np.array(D)
            block[names.growth_parameters, "f_z"] = np.array(f)
            block[names.growth_parameters, "a"] = 1/(1+z)

            if c.nonlinear_method != 0:
                for i, ki in enumerate(k):
                    for j, zi in enumerate(z):
                        P[i, j] = c.pk(ki, zi)

                block.put_grid("matter_power_nl", "z", z, "k_h", k / h0, "p_k", P.T * h0**3)

        ##
        # Now the CMB C_ell
        ##
        if config["cmb"]:
            c_ell_data = c.lensed_cl() if config['lensing'] else c.raw_cl()
            ell = c_ell_data['ell']
            ell = ell[2:]

            # Save the ell range
            block[cmb_cl, "ell"] = ell

            # t_cmb is in K, convert to mu_K, and add ell(ell+1) factor
            tcmb_muk = block[cosmo, 'tcmb'] * 1e6
            f = ell * (ell + 1.0) / 2 / np.pi * tcmb_muk**2

            # Save each of the four spectra
            for s in ['tt', 'ee', 'te', 'bb']:
                block[cmb_cl, s] = c_ell_data[s][2:] * f

            # save the lensing potential power spectrum
            if config['lensing']:
                f_lens = ell * (ell + 1.0) / 2 / np.pi
                for s in ['pp', 'tp']:
                    block[cmb_cl, s] = c_ell_data[s][2:] * f_lens

    ##
    # Distances and related quantities
    ##

    # save redshifts of samples
    block[distances, 'z'] = z
    block[distances, 'a'] = 1 / (1 + z)
    block[distances, 'nz'] = nz

    # Save distance samples
    d_l = np.array([c.luminosity_distance(zi) for zi in z])
    block[distances, 'd_l'] = d_l
    d_a = np.array([c.angular_distance(zi) for zi in z])
    block[distances, 'd_a'] = d_a
    block[distances, 'd_m'] = d_a * (1 + z)

    # Save some auxiliary related parameters
    block[distances, 'age'] = c.age()
    block[distances, 'rs_zdrag'] = c.rs_drag()

    # Save H(z), which is also in Mpc^-1 units, like in camb
    h_z = np.array([c.Hubble(zi) for zi in z])
    block[distances, 'H'] = h_z

    # Save the comoving distance
    d_c = np.array([c.comoving_distance(zi) for zi in z])
    block[distances, 'd_c'] = d_c

    # Save angular average of comoving angular diameter and line of sight distance, like in camb
    d_v = ((1 + z)**2 * z * d_a**2 / h_z)**(1./3.)
    block[distances, 'd_v'] = d_v

    mu = np.zeros(z.size)
    mu[0] = -np.inf
    mu[1:] = 5.0 * np.log10(d_l[1:]) + 25.0
    block[distances, 'mu'] = mu





def execute(block, config):
    import classy
    c = config['cosmo']

    try:
        # Set input parameters
        params = get_class_inputs(block, config)
        c.set(params)

        # saves neff if we are using nur, for cases where nur is sampled:
        block[cosmo, 'nnu'] = c.Neff()

        if not config['background_only']:
            # Run perturbation calculations
            c.compute()

        # Extract outputs
        get_class_outputs(block, c, config)
    except classy.CosmoError as error:
        if config['debug']:
            sys.stderr.write("Error in class. You set debug=T so here is more debug info:\n")
            traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write("Error in class. Set debug=T for info: {}\n".format(error))
        return 1
    finally:
        # Reset for re-use next time
        c.struct_cleanup()
    return 0


def cleanup(config):
    config['cosmo'].empty()
