import numpy as np

from cosmosis.datablock import option_section


def setup(options):
    input_parameters = options.get_string(option_section, "uncorrelated_parameters").split()
    input_parameters = [p.strip().split("/") for p in input_parameters]
    output_parameters = options.get_string(option_section, "output_parameters").split()
    output_parameters = [p.strip().split("/") for p in output_parameters]
    covariance_file = options[option_section, "covariance"]

    cov = np.loadtxt(covariance_file)
    L = np.linalg.cholesky(cov) 

    if options.has_value(option_section, "mean"):
        mean = options[option_section, "mean"]
    else:
        mean = np.zeros(L.shape[0])

    # Allow loading from a text file
    if isinstance(mean, str):
        mean = np.loadtxt(mean)

    nmean = len(mean)
    ncov = len(cov)
    if ncov != nmean:
        raise ValueError(f"Covariance matrix and mean vector have different lengths, {ncov} and {nmean}")

    return input_parameters, output_parameters, L, mean

def execute(block, config):
    input_parameters, output_parameters, L, mu = config
    
    p = []
    for section, name in input_parameters:
        p.append(block[section, name])
    p = L @ np.array(p) + mu

    for i, (section, name) in enumerate(output_parameters):
        block[section, name] = p[i]

    return 0

def clean(config):
    pass
