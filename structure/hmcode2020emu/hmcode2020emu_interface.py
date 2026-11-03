# Felipe Andrade-Oliveira 2024
# This module is a wrapper for the HMcode2020Emu emulator model_v2.zip 06-2024
# Known Issues: emulator was trained with a single massive neutrino species 
# https://github.com/MariaTsedrik/HMcode2020Emu
from cosmosis.datablock import names, option_section
import warnings
import numpy as np
import HMcode2020Emu as hmcodeemu
import pdb

emulator = hmcodeemu.Matter_powerspectrum()
cosmo = names.cosmological_parameters
hmpar = names.halo_model_parameters



def set_params(As=1.8418e-9, ns=0.949, hubble=0.615, log10TAGN=7.7, neutrino_mass=0.0773062, 
                omega_baryon=0.045, omega_cdm=0.2908032395143077, w0=-1.0, wa=0.0,
                zvals=0.0, k=None):
    """
    Generates a dicionary for HMcode2020Emu input, for different z's but same
    cosmo/astro parameters
    zvals: list of z
    """
    
    if not hasattr(zvals, "__len__"):
        #fao: Warning it should be a zvals should be a list
        zvals = [zvals]
        
    n_zs = len(zvals)    
    params=    {'As': [As]*n_zs,
                'hubble': [hubble]*n_zs,
                'log10TAGN': [log10TAGN]*n_zs,
                'neutrino_mass': [neutrino_mass]*n_zs, 
                'ns': [ns]*n_zs,
                'omega_baryon': [omega_baryon]*n_zs,
                'omega_cdm': [omega_cdm]*n_zs,
                'w0': [w0]*n_zs,
                'wa': [wa]*n_zs,
                'z': zvals}
    
    if k is not None:
        params.update({'k':k})
    return params


def setup(options):
    # options store all the modules, options for the ini's
    config = {}
    config['verbose'] = options.get_bool(option_section, 'verbose', default=False)

    config['zmin'] = options.get_double(option_section, 'zmin', default=0.0)
    config['zmax'] = options.get_double(option_section, 'zmax', default=3.01)
    config['nz'] = options.get_int(option_section, 'nz', default=150)
    config['kmin'] = options.get_double(option_section, 'kmin', default=1e-4)
    config['kmax'] = options.get_double(option_section, 'kmax', default=49.99999)
    config['nk'] = options.get_int(option_section, 'nk', default=701) # FAO checkme
    config['kmax_extrapolate'] = options.get_double(option_section, 'kmax_extrapolate', default=50.) # FAO checkme

    # From 
    if config['zmax']>3.01:
        raise ValueError("z >3.0 is out of emulator range!")
    elif config['zmax']>3.0 and config['zmax'] <= 3.01:
        warnings.warn("3.0> z >=3.1 is already larger than emulator zmax but used for P(k) validation.")
        
    if config['kmin']<1e-4:
        raise ValueError("k <1e-4 is out of emulator range!")
    if config['kmax']>49.99999+1 : # FAO: checkme
        raise ValueError(f"kmax={config['kmax']} is out of emulator range!")
    if config['kmax_extrapolate'] is not None:
        warnings.warn(f"kmax_extrapolate={config['kmax_extrapolate']} is not implemented.")

    return config

def execute(block, config):
    import pdb; 
    warnings.warn("Emulator:\n DO NOT USE sigma_12.\n Check if sigma_8 can be sampled ")

    kmax = config['kmax']
    kmin = config['kmin']
    nk = config['nk'] # FAO checkme
    k_arr = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    if np.max(k_arr)>49.99999999999996:
        warnings.warn(f"Maximum k={np.max(k_arr)} is out of emulator range! Trimming to k<49.99999")
        #k_arr = k_arr[k_arr<50.]

    zmin = config['zmin'] #config['zmin']
    zmax = config['zmax']
    nz = config['nz'] # FAO: padrao config['nz'] # FAO checkme
    # creating this grid and removing the last point 
    # to make a grid compatible with cosmosis/camb and emulator
    z_arr = np.linspace(zmin, zmax, nz)
    
    # FAO: check here Only for comparison
    z_arr = np.linspace(zmin,3.01, nz)
    z_arr = z_arr[z_arr<3.0]
    # import pdb; pdb.set_trace()

    params= set_params(omega_cdm  = block.get_double(cosmo, "omega_c"),
                        As            = block.get_double(cosmo, "a_s"),
                        omega_baryon  = block.get_double(cosmo,"omega_b"),
                        ns            = block.get_double(cosmo, 'n_s'),
                        hubble        = block.get_double(cosmo,'h0'),
                        neutrino_mass = block.get_double(cosmo, 'mnu'), #FAO CHECK!
                        w0            = block.get_double(cosmo, 'w'),
                        wa            = block.get_double(cosmo, 'wa', default=0.0),
                        log10TAGN     = block.get_double(hmpar, "logt_agn"),
                        zvals         = z_arr,
                        k             = k_arr #FAO: checkme
                    )

    params_nl= set_params(omega_cdm  =  block.get_double(cosmo, "omega_c"),
                        As            = block.get_double(cosmo, "a_s"),
                        omega_baryon  = block.get_double(cosmo,"omega_b"),
                        ns            = block.get_double(cosmo, 'n_s'),
                        hubble        = block.get_double(cosmo,'h0'),
                        neutrino_mass = block.get_double(cosmo, 'mnu'), #FAO CHECK!
                        w0            = block.get_double(cosmo, 'w'),
                        wa            = block.get_double(cosmo, 'wa', default=0.0),
                        log10TAGN     = block.get_double(hmpar, "logt_agn"),
                        zvals         = z_arr,
                        k             = k_arr[k_arr>=0.01] #FAO: checkme
                    )


    klemu, plin = emulator.get_linear_pk(**params, no_nu=False)
    knemu, pnemu = emulator.get_nonlinear_pk(baryonic_boost=True, **params_nl)

    # concatenation of PkLin, Pk_NL
    pl_left = plin[:, klemu<knemu[0]]
    pnl  = np.concatenate((pl_left, pnemu),axis=1)
    k_h   = np.concatenate((klemu[klemu<knemu[0]], knemu))
    # k_h = []
    
    # Saving sigma_8(z)
    omega_m = block[cosmo, "omega_m"]
    sigma_8, fsigma_8 = emulator.get_sigma8(**params)
    block[names.growth_parameters, "sigma_8"] = sigma_8
    block[names.growth_parameters, "fsigma_8"] = fsigma_8
    block[cosmo, "sigma_8"] = sigma_8[0]
    block[cosmo, "S_8"] = sigma_8[0]*np.sqrt(omega_m/0.3)


   
    block.put_grid("matter_power_lin", "z", z_arr, "k_h", k_h, "P_k", plin)
    block.put_grid("matter_power_nl", "z",  z_arr, "k_h", k_h, "P_k", pnl)

    # FAO debug
    # print(k_h.shape)
    # # saving data:
    # np.save(f"/pscratch/sd/f/faoli/emu_cosmosis/y6-3x2pt-methods/cosmosis/chains/plinemu.npy", plin)
    # np.save("/pscratch/sd/f/faoli/emu_cosmosis/y6-3x2pt-methods/cosmosis/chains/pnlemu.npy", pnl)
    # np.save("/pscratch/sd/f/faoli/emu_cosmosis/y6-3x2pt-methods/cosmosis/chains/z_arr.npy", z_arr)
    # np.save("/pscratch/sd/f/faoli/emu_cosmosis/y6-3x2pt-methods/cosmosis/chains/k_h.npy", k_h)
    # print("=====END EMU FAO+=====")

    return 0

if __name__=="__main__":
    print("Executing example case")



"""
[('bias_lens', 'b1'),
 ('bias_lens', 'b1e_sig8_bin1'),
 ('bias_lens', 'b1e_sig8_bin2'),
 ('bias_lens', 'b1e_sig8_bin3'),
 ('bias_lens', 'b1e_sig8_bin4'),
 ('bias_lens', 'b1e_sig8_bin5'),
 ('bias_lens', 'b1e_sig8_bin6'),
 ('bias_lens', 'b2'),
 ('bias_lens', 'b2e_sig8sq_bin1'),
 ('bias_lens', 'b2e_sig8sq_bin2'),
 ('bias_lens', 'b2e_sig8sq_bin3'),
 ('bias_lens', 'b2e_sig8sq_bin4'),
 ('bias_lens', 'b2e_sig8sq_bin5'),
 ('bias_lens', 'b2e_sig8sq_bin6'),
 ('bias_lens', 'b3'),
 ('bias_lens', 'b4'),
 ('bias_lens', 'b5'),
 ('bias_lens', 'b6'),
 ('cosmological_parameters', 'a_s'),
 ('cosmological_parameters', 'a_s_1e9'),
 ('cosmological_parameters', 'baryon_fraction'),
 ('cosmological_parameters', 'consistency_module_was_used'),
 ('cosmological_parameters', 'h0'),
 ('cosmological_parameters', 'hubble'),
 ('cosmological_parameters', 'k'),
 ('cosmological_parameters', 'mnu'),
 ('cosmological_parameters', 'n_s'),
 ('cosmological_parameters', 'nnu'),
 ('cosmological_parameters', 'num_massive_neutrinos'),
 ('cosmological_parameters', 'ombh2'),
 ('cosmological_parameters', 'omch2'),
 ('cosmological_parameters', 'omega_b'),
 ('cosmological_parameters', 'omega_c'),
 ('cosmological_parameters', 'omega_k'),
 ('cosmological_parameters', 'omega_lambda'),
 ('cosmological_parameters', 'omega_m'),
 ('cosmological_parameters', 'omega_nu'),
 ('cosmological_parameters', 'omlamh2'),
 ('cosmological_parameters', 'ommh2'),
 ('cosmological_parameters', 'omnuh2'),
 ('cosmological_parameters', 'standard_neutrino_neff'),
 ('cosmological_parameters', 'tau'),
 ('cosmological_parameters', 'tcmb'),
 ('cosmological_parameters', 'w'),
 ('cosmological_parameters', 'wa'),
 ('cosmological_parameters', 'yhe'),
 ('halo_model_parameters', 'logt_agn'),
 ('intrinsic_alignment_parameters', 'a1'),
 ('intrinsic_alignment_parameters', 'a2'),
 ('intrinsic_alignment_parameters', 'alpha1'),
 ('intrinsic_alignment_parameters', 'alpha2'),
 ('intrinsic_alignment_parameters', 'bias_ta'),
 ('intrinsic_alignment_parameters', 'z_piv'),
 ('lens_photoz_errors', 'bias_1'),
 ('lens_photoz_errors', 'bias_2'),
 ('lens_photoz_errors', 'bias_3'),
 ('lens_photoz_errors', 'bias_4'),
 ('lens_photoz_errors', 'bias_5'),
 ('lens_photoz_errors', 'bias_6'),
 ('lens_photoz_errors', 'width_1'),
 ('lens_photoz_errors', 'width_2'),
 ('lens_photoz_errors', 'width_3'),
 ('lens_photoz_errors', 'width_4'),
 ('lens_photoz_errors', 'width_5'),
 ('lens_photoz_errors', 'width_6'),
 ('mag_alpha_lens', 'alpha_1'),
 ('mag_alpha_lens', 'alpha_2'),
 ('mag_alpha_lens', 'alpha_3'),
 ('mag_alpha_lens', 'alpha_4'),
 ('mag_alpha_lens', 'alpha_5'),
 ('mag_alpha_lens', 'alpha_6'),
 ('shear_calibration_parameters', 'm1'),
 ('shear_calibration_parameters', 'm2'),
 ('shear_calibration_parameters', 'm3'),
 ('shear_calibration_parameters', 'm4'),
 ('wl_photoz_errors', 'bias_1'),
 ('wl_photoz_errors', 'bias_2'),
 ('wl_photoz_errors', 'bias_3'),
 ('wl_photoz_errors', 'bias_4')]
"""
