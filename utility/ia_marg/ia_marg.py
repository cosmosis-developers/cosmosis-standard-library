"""
Convert sampled IA amplitude * redshift-dependence products into the redshift
power-law indices used by the tidal alignment models.
"""
from cosmosis.datablock import option_section


def setup(options):
    section = options.get_string(option_section, "ia_section", "intrinsic_alignment_parameters")
    return section


def execute(block, config):
    section = config

    for i in [1, 2]:
        amplitude = block[section, f"A{i}"]
        if amplitude == 0.0:
            raise ValueError(f"Cannot compute alpha{i} because A{i} is zero")
        block[section, f"alpha{i}"] = block[section, f"alphaA{i}"] / amplitude

    return 0


def cleanup(config):
    pass
