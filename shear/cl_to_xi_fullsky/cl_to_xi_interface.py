#coding: utf-8
#import cl_to_xi_full
from __future__ import print_function
from builtins import range
import numpy as np
from cosmosis.datablock import option_section, names as section_names
from cl_to_xi import save_xi_00_02, save_xi_22, arcmin_to_radians, SpectrumInterp, cl_to_xi_to_block, cl_to_xi_to_block_eb
from legendre import get_legfactors_00, get_legfactors_02, get_legfactors_22, precomp_GpGm, apply_filter, get_legfactors_02_binav, get_legfactors_00_binav, get_legfactors_22_binav
import sys
import os
dirname = os.path.split(__file__)[0]
twopoint_path = os.path.join(dirname,"..","..","likelihood","2pt")
sys.path.insert(0, twopoint_path)
import twopoint
import warnings

def read_theta(filename, xi_type_2pt, theta_type = 'centers', desired_units = 'arcmin'):
    """
    Short function to read in theta values from a specified fits file.
    Desired angle units in 'rad', 'arcmin', 'arcsec', 'deg
    """
    T = twopoint.TwoPointFile.from_fits(filename)
    xi = T.get_spectrum(xi_type_2pt)
    # make sure the units are in arcmin (or whatever else you want)
    xi.convert_angular_units(desired_units)
    # get the theta values for a single bin pair.
    # This method currently assumes that the same binning is used for all bin pairs (in a given corr funct).
    # scale cuts are applied later and can be different for each bin pair. 
    pairs = xi.get_bin_pairs()
    indices = xi.get_pair_mask(pairs[0][0], pairs[0][1]) #use the first appearing pair, since all should be the same.
    if theta_type == 'centers':
        thetas = xi.angle[indices]
        return thetas
    elif theta_type == 'edges':
        theta_min = xi.angle_min[indices]  # array of min values
        theta_max = xi.angle_max[indices] # array of max values
        # We currently assume adjacent and non-overlapping bins.
        np.testing.assert_allclose(theta_min[1:], theta_max[:-1], rtol=1e-03,
            err_msg='theta bin min and max values do not match.', verbose=True)
        theta_bins = np.append(theta_min, theta_max[-1])
        return theta_bins
    else:
        raise ValueError("Unknown theta_type in read_theta")



def setup(options):

    xi_type = options.get_string(option_section, 'xi_type')

    # Check xi_type is one of the allowed values
    if xi_type not in ['22', '22+', '22-', '02', '02+', '00', 'EB']:
        raise ValueError(
            "xi_type should be '22', '22+' or '22-' for spin 2 autocorrelations e.g. xi+/-(theta), "
            "'EB' for the pair of E+B and E-B calculations for xi+/- respectively, "
            "00 for scalar autocorrelations e.g. w(theta), "
            "or '02' or '02+' for spin 0 x spin 2 correlations e.g. gamma_t(theta)")

    # Choices about how we do the calculation
    ell_max = options.get_int(option_section, "ell_max")
    bin_avg = options.get_bool(option_section, "bin_avg", False)

    # Where we get the theta values from
    theta_file = options.get_string(option_section, "theta_file", "")
    default_file_section = {
        'EB': 'xip',
        '22': 'xip',
        '22+': 'xip',
        '22-': 'xim',
        '02': 'gammat',
        '02+': 'gammat',
        '00': 'wtheta',
    }[xi_type]
    theta_section = options.get_string(option_section, "theta_section", default_file_section)
    
    #Filter the Cls at high ell to reduce ringing:
    high_l_filter = options.get_double(option_section, "high_l_filter", 0.75)

    # How to get the theta mid-points from the edges.
    two_thirds_midpoint = options.get_bool(option_section, "two_thirds_midpoint", True)

    # Naming choices for the input and outut
    cl_section = options.get_string(option_section, "input_section_name", "")
    output_section = options.get_string(option_section, "output_section_name", "")
    save_name = options.get_string(
        option_section, "save_name", "")

    if theta_file:
        print("*** Reading in theta values from data file {} ***".format(theta_file))

    # There are quite a lot of options here on how to choose theta values.
    # If bin_avg is True then we need the edge values.
    # We can either get those from a file or from specified edge values
    if bin_avg:
        print("*** Using bin averaged Legendre coefficients ***")

        if theta_file:
            print("Note: we are assuming that the theta values for all bin pairs are "
                "the same. If this is not true, you need to modify cl_to_xi_interface")
            # We read both the edge values and separately the mid points.
            # The edges are used for the actual calculation.  The former are used
            # for plotting and scale cuts.
            theta = read_theta(theta_file, theta_section, theta_type = 'centers', desired_units = 'rad')
            theta_edges = read_theta(theta_file, theta_section, theta_type = 'edges', desired_units = 'rad')
        else:
            # Add a couple of warnings in case people have messed uup
            if options.has_value(option_section, "theta"):
                print('WARNING: Specify `theta_edges` instead of `theta` when using bin averaging. `theta` is ignored')

            if options.has_value(option_section, "n_theta"):
                print('WARNING: Specify `n_theta_bins` instead of `n_theta` when using bin averaging. `n_theta` is ignored')

            # People may specify theta edges directly, or they can specify min/max/n values
            if options.has_value(option_section, "theta_edges"):
                print('*** Note: Using `theta_edges` values specified in ini file ***')
                theta_edges = options[option_section, 'theta_edges']

                # In order to be "edges" we need two values.
                if len(theta_edges) < 2:
                    raise ValueError("theta_edges has only one value - need more")

                theta_edges = arcmin_to_radians(theta_edges)
            else:
                print('** NOTE that theta_min and theta_max refer to the min and max of the first and last angular bin, respectively **')
                n_theta_bins = options.get_int(option_section, "n_theta_bins", 20)
                theta_min = options.get_double(option_section, "theta_min", 1.) 
                theta_max = options.get_double(option_section, "theta_max", 300.)
                theta_min = arcmin_to_radians(theta_min)
                theta_max = arcmin_to_radians(theta_max)
                theta_edges = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta_bins + 1)

            # We also get theta from theta_edges, for plotting and scale cuts mostly.
            # There are a few ways we can do this.
            if two_thirds_midpoint:
                # this is currently the default
                theta = (2./3.) * (theta_edges[1:]**3 - theta_edges[:-1]**3) / (theta_edges[1:]**2 - theta_edges[:-1]**2)
            else:
                # geometric mean
                theta = (theta_edges[:-1]*theta_edges[1:])**0.5
    else:
        theta_edges = None
        print("*** Using Legendre coefficients at individual theta values ***")

        if theta_file:
            # We can read the mid-point theta values from the file if we want
            print("*** Reading in central theta values from datafile ***")
            theta = read_theta(theta_file, theta_section, theta_type = 'centers', desired_units = 'arcmin')
            theta = arcmin_to_radians(theta)

        elif options.has_value(option_section, "theta"):
            # or secondarily use the theta value specified directly
            print('*** Note: Using `theta` values specified in ini file ***')
            theta = options[option_section, 'theta']
            # Allow single theta values
            if np.isscalar(theta):
                theta = np.array([theta])
            theta = arcmin_to_radians(theta)
        
        else:
            # or finally fall back to calculating  from min/max/n
            n_theta = options.get_int(option_section, "n_theta", 20)
            theta_min = options.get_double(option_section, "theta_min", 1.) 
            theta_max = options.get_double(option_section, "theta_max", 300.)
            theta_min = arcmin_to_radians(theta_min)
            theta_max = arcmin_to_radians(theta_max)
            theta = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta)

    e_plus_b_name = None

    # setup precompute functions and I/O sections
    if xi_type in ["22", "22+", "22-"]:

        # Function to use to pre-compute the Legnendre factors
        if bin_avg:
            precomp_func = get_legfactors_22_binav
        else:    
            precomp_func = get_legfactors_22
        
        # The default section to read the C_ell values from
        if not cl_section:
            cl_section = "shear_cl"

        # where to save the output values.
        # Default is to just change cl->xipm
        if output_section == "":
            output_section = (cl_section.replace("cl","xi_plus"), cl_section.replace("cl","xi_minus"))
        else:
            output_section = (output_section + "_plus", output_section + "_minus")
    elif xi_type == "EB":
        # Function to use to pre-compute the Legnendre factors
        if bin_avg:
            precomp_func = get_legfactors_22_binav
        else:    
            precomp_func = get_legfactors_22
        
        # The default section to read the C_ell values from
        if cl_section:
            cl_section = tuple(cl_section.split())
        else:
            cl_section = ("shear_cl", "shear_cl_bb")

        # where to save the output values.
        # Default is to just change cl->xipm
        if output_section:
            output_section = tuple(output_section.split())
        else:
            output_section = ("shear_xi_plus", "shear_xi_minus")

        e_plus_b_name = options.get_string(option_section, "e_plus_b_name", "shear_cl")

    elif xi_type == "00":
        # Same for wtheta - function that makes the big transfer matrix
        if bin_avg:
            precomp_func = get_legfactors_00_binav
        else:    
            precomp_func = get_legfactors_00

        # Default C_ell
        if not cl_section:
            cl_section = "galaxy_cl"

        # Default output
        if output_section == "":
            output_section = cl_section.replace("cl", "xi")

    elif xi_type in ["02", "02+"]:
        # Finally the same for gamma_t
        if bin_avg:
            precomp_func = get_legfactors_02_binav
        else:    
            precomp_func = get_legfactors_02

        # Default C_ell
        if not cl_section:
            cl_section = "galaxy_shear_cl"

        # Default output
        if output_section == "":
            output_section = cl_section.replace("cl", "xi")

    # Useful for assigning different names to same type of correlation function
    # For instance, if we have measurements with two different samples
    if save_name != "":
        add = "_" + save_name
        if isinstance(output_section, str):
            output_section += add
        else:
            output_section = ( output_section[0]+add, output_section[1]+add )

    print("Computing coefficients to transform {} -> {}".format(cl_section, output_section))

    # This is the (potentially) slow bit - actually work out the Legendre coefficients
    if bin_avg:
        legfacs = precomp_func(np.arange(ell_max + 1), theta_edges)
    else:
        legfacs = precomp_func(np.arange(ell_max + 1), theta)


    if high_l_filter>0:
        if isinstance(legfacs, tuple):
            legfacs = (
                apply_filter(ell_max, high_l_filter, legfacs[0]), 
                apply_filter(ell_max, high_l_filter, legfacs[1])
            )
        else:
            legfacs = apply_filter( ell_max, high_l_filter, legfacs )

    return xi_type, theta, theta_edges, ell_max, legfacs, cl_section, output_section, save_name, bin_avg, e_plus_b_name

def combine_eb(block, ee_section, bb_section, e_plus_b_name):
    nbin_shear = block[ee_section, 'nbin']
    p_section, m_section = e_plus_b_name + "_eplusb", e_plus_b_name + "_eminusb"

    # clone the EE section, such that we retain all of the metadata
    block._copy_section(ee_section,p_section)
    block._copy_section(ee_section,m_section)

    ell = block[ee_section, "ell"]
    block[p_section, "ell"] = ell
    block[m_section, "ell"] = ell
    block[p_section, "nbin"] = nbin_shear
    block[m_section, "nbin"] = nbin_shear
    for i in range(nbin_shear):
        for j in range(0,i+1):
            bin_ij = 'bin_{0}_{1}'.format(i+1,j+1)
            ee = block[ee_section, bin_ij]
            bb = block[bb_section, bin_ij]
            block[p_section, bin_ij] = ee+bb
            block[m_section, bin_ij] = ee-bb

    return p_section, m_section


def execute(block, config):

    xi_type, thetas, theta_edges, ell_max, legfacs, cl_section, output_section, save_name, bin_avg, e_plus_b_name = config

    # If we are doing the special EB mode then we first want to combine
    # the E and B modes into a E+B and E-B sections
    if xi_type == "EB":
        p_section, m_section = combine_eb(block, cl_section[0], cl_section[1], e_plus_b_name)
        cl_section, bb_section = cl_section

    ell = block[cl_section, "ell"]
    min_ell = ell[0]
    if min_ell > 1:
        raise ValueError('Min ell value must be 1 - your pipeline only supplied ell from {}'.format(min_ell))

    nbina, nbinb = block[cl_section, 'nbin_a'], block[cl_section, 'nbin_b']

    if not isinstance(output_section, tuple):
        output_section = (output_section,)
        legfacs = (legfacs,)

    for i in range(1, nbina + 1):
        for j in range(1, nbinb + 1):
            name = 'bin_%d_%d' % (i, j)
            if not block.has_value(cl_section, name):
                continue

            if xi_type == "EB":
                # Get the E+B and E-B separately.
                # These were just calculated above when we called combine_eb 
                e_plus_b = block[p_section, name]
                e_minus_b = block[m_section, name]
                e_plus_b_interp = SpectrumInterp(ell, e_plus_b)
                e_minus_b_interp = SpectrumInterp(ell, e_minus_b)
                cl_to_xi_to_block_eb(block, output_section, name,
                              e_plus_b_interp, e_minus_b_interp, thetas, legfacs)
            else:
                c_ell = block[cl_section, name]
                cl_interp = SpectrumInterp(ell, c_ell)
                cl_to_xi_to_block(block, output_section, name,
                              cl_interp, thetas, legfacs)

    if isinstance(output_section, str):
        output_section = (output_section,)
        
    for o in output_section:
        block[o, "nbin_a"] = nbina
        block[o, "nbin_b"] = nbinb
        block[o, "sample_a"] = block[cl_section, "sample_a"]
        block[o, "sample_b"] = block[cl_section, "sample_b"]
        block[o, "is_auto"] = block[cl_section, "is_auto"]
        block[o, "cl_section"] = cl_section
        block[o, "theta"] = thetas
        block.put_metadata(o, "theta", "unit", "radians")
        block[o, "sep_name"] = "theta"
        block[o, "save_name"] = save_name
        block[o, "bin_avg"] = bin_avg
        if theta_edges is not None:
            block[o, "theta_edges"] = theta_edges
            block.put_metadata(o, "theta_edges", "unit", "radians")

    
    
    return 0

def cleanup(config):
    # nothing to do here!  We just include this
    # for completeness.  The joy of python.
    return 0
