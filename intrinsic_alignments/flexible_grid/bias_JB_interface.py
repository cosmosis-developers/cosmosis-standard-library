from cosmosis.datablock import names, option_section
import bias_grid as functions

import numpy as np
import matplotlib.pyplot as plt


def setup(options):
    nknodes = options[option_section, 'nknodes']
    nznodes = options[option_section, 'nznodes']
    ia = options[option_section, 'intrinsic_alignments']
    gb = options[option_section, 'galaxy_bias']
    opt = {'nznodes': nznodes, 'nknodes': nknodes,
           'intrinsic_alignments': ia, 'galaxy_bias': gb}

    grid_generator = functions.flexible_grid(opt)
    return opt, grid_generator


def execute(block, config):

    options, grid = config

    # Define datablock section names
    nl = names.matter_power_nl
    cospar = names.cosmological_parameters

    # Use the grid object created during startup
    # with the specific realisation of the nodes

    grid.setup_grid_nodes(block)
    grid.interpolate_grid()
    grid.evaluate_and_save_bias(block)
    return 0


def cleanup(config):
    pass
