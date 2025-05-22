from cosmosis.datablock import names, option_section
import numpy as np
from scipy import interpolate
import os

cosmo = names.cosmological_parameters
dirname = os.path.split(__file__)[0]


def setup(options):
    default_datafile = os.path.join(dirname, "PRIMAT_Yp_DH_ErrorMC_2021.dat")
    # File name for BBN data
    datafile = options.get_string(
        option_section, "data", default=default_datafile)

    # verbosity setting
    verbose = options.get_bool(option_section, "verbose", default=False)

    # Load and spline data
    dat = np.genfromtxt(datafile, names=True, comments='#')
    spline = interpolate.bisplrep(dat['ombh2'], dat['DeltaN'], dat['Yp'])

    input_name = options.get_string(
        option_section, "input_name", default="nnu")

    if input_name == "massless_nu":
        raise ValueError("The input_name=massless_nu option in bbn_consistency.py "
                         "has been replaced with input_name=nnu. This relects the change " 
                         "in the parameter name used in camb and class.")

    if input_name not in ["delta_neff", "nnu"]:
        raise ValueError("The input_name option in bbn_consistency.py must be either "
                         f"delta_neff or nnu. The current value is {input_name}")

    # Saved data will be returned later
    return {
        'spline': spline,
        'ombh2_min': np.min(dat['ombh2']),
        'ombh2_max': np.max(dat['ombh2']),
        'DeltaN_min': np.min(dat['DeltaN']),
        'DeltaN_max': np.max(dat['DeltaN']), 
        'input_name':input_name,
        'verbose':verbose
    }


def execute(block, config):
    input_name = config['input_name']
    ombh2 = block[cosmo, "ombh2"]

    if input_name == "delta_neff":
        if block.has_value(cosmo, "delta_neff"):
            DeltaN = block[cosmo, "delta_neff"]
        else:
            DeltaN = 0.0

    elif input_name == "nnu":
        standard_neutrino_neff = block.get_double(cosmo, "standard_neutrino_neff", default=3.044)
        DeltaN = block[cosmo, "nnu"]  - standard_neutrino_neff

    else:
        raise ValueError(f"Unknown input_name {input_name} in bbn_consistency.py")


    # Check for out-of-range parameters
    ombh2_min = config["ombh2_min"]
    ombh2_max = config["ombh2_max"]
    DeltaN_min = config["DeltaN_min"]
    DeltaN_max = config["DeltaN_max"]

    if (ombh2 < ombh2_min) or (ombh2 > ombh2_max) or (DeltaN < DeltaN_min) or (DeltaN > DeltaN_max):
        if config['verbose']:

            print("Parameter(s) out of range for BBN Consistency:")

            if ombh2 < ombh2_min or ombh2 > ombh2_max:
                print(f"ombh2 = {ombh2}  which is outside ({ombh2_min}, {ombh2_max})")

            if DeltaN < config['DeltaN_min'] or DeltaN > config['DeltaN_max']:
                print(f"DeltaN = {DeltaN}  which is outside ({DeltaN_min}, {DeltaN_max})")

        return 1

    # save the helium fraction back to the block
    block[cosmo, "yhe"] = interpolate.bisplev(ombh2, DeltaN, config['spline'])

    # We tell CosmoSIS that everything went fine by returning zero
    return 0


def cleanup(config):
    # Python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
