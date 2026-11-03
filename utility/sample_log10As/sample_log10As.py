"""
Sample log10(10^10 A_s) instead of A_s.

Originally written by David Sanchez Cid for the HSC re-analysis in DESC.
"""
from cosmosis.datablock import option_section, names


def setup(options):
    return options.get_string(option_section, "log10As_name", default="log10As")


def execute(block, config):
    log10As_name = config
    log10As = block[names.cosmological_parameters, log10As_name]
    block[names.cosmological_parameters, "A_s"] = 10 ** log10As / 1e10
    return 0


def cleanup(config):
    pass
