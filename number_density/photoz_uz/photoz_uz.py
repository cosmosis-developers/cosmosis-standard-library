from builtins import range
from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import interp1d

def build_degauss(pctile, gmax, dg):
    '''Return a transformation function from the gaussian 
    to the non-gaussian u distribution.
    Arguments:
    `pctile`:   array of target output values for the lookup table (LUT)
    `gmax, dg`: input values for the LUT will be (-gmax, -gmax+dg, ..., +gmax)
    The returned object will be a function  mapping  g->u.'''

    # Establishing matching percentile points
    gg = np.arange(-gmax,gmax+dg/2, dg)
    uu = np.copy(pctile)
    # Clean out any duplicates in the tables.  This is hacky.
    dup = uu[1:]==uu[:-1]
    keep = np.concatenate( (np.logical_not(dup), [True]))
    gg = gg[keep]
    uu = uu[keep]
    # The following function will complain if dimension of pctile array does not match
    # the number of needed g values.
    return interp1d(gg,uu,kind="linear", bounds_error=False, fill_value=(uu[0], uu[-1]))

def setup(options):
    '''The available options:
    `sample`:     either `lens` or `source`
    `basis_file`: A numpy `npz` file with contents listed below, containing all info 
                  needed to create n(z)'s
    `n_modes`:    Number of compressed modes to use (per redshift bin, if each
                  bin has its own compression).  If omitted or set to <=0, all modes
                  in the `basis_file` will be used.
    `no_degaussbins` Lens redshift bin to not apply degaussianization

    The `basis_file` should contain the following arrays:
    `U`:          An array of shape (n_modes, n_bins, n_z) giving the basis vectors (across n_z redshifts)
                  for the n(z) functions.  If `perbin==True` then each (imode,ibin) combinations will 
                  have a distinct coefficient `u_{ibin}_{imode}` as a free parameter.  If `perbin` is
                  `False`, then there are `u_{imode}` parameters giving a common coefficient to all
                  bins' n(z)'s.
    `perbin`:     A scalar that is converted to bool specifying whether modes are distinct per bin.
    `g_convention`: If this is given, it means that we will construct a lookup table that converts the 
                  Cosmosis `u` parameter, with a unit Gaussian prior, into the true `u` value used
                  as basis coefficient.  This array has two entries, `gmax, dg` The lookup
                  table will have as its input value an array [-gmax, -gmax+dg, ...0, +gmax].  The
                  output of the lookup table will be specified by the `pctile` array, and any input
                  values outside the [-gmax,gmax] interval will be assigned to the min and max values
                  of the pctile array.

                  If `g_convention` is not present, then the Cosmosis `u` parameters will be used 
                  directly as coefficients, with unit normal priors.
    `pctile`:     This array must given if a `g_convention` is present, and it gives the lookup table
                  values for the de-gaussianized coefficients.  If `perbin` is `True`, then this array
                  will have shape `(n_modes, n_bins, n_g)` where `n_g` is the number of g values.  
                  If `perbin` is `False`, the the pctile array is 2d, `(n_modes, n_g)`.'''

    # Read the configuration variables:
    sample = options.get_string(option_section, "sample", "")
    basis_file = options.get_string(option_section, "basis_file", "")
    n_modes = options.get_int(option_section, "n_modes", 0)
    # only lens modes are degauss
    if sample == 'lens':
        # no_degaussbins = options.get_int(option_section, "no_degaussbins", 9999)
        no_degaussbins = options.get_string(option_section, "no_degaussbins", "")
        # convert from string to array of int
        if no_degaussbins != "":
            no_degaussbins = np.array([int(x.strip()) for x in no_degaussbins.split(",")])
        else: 
            no_degaussbins = np.array([])
    else:
        no_degaussbins = np.array([])
    print('Lens z-bins degauss = ', no_degaussbins)

    # Read the basis vectors and record sizes
    npzfile = np.load(basis_file)
    U=npzfile['U']
    n_bins = U.shape[1]
    n_z = U.shape[2]
    # Decide how many modes to use
    if n_modes>0:
        # Check that requested number of modes are available
        if n_modes>U.shape[0]:
            raise ValueError('More nz modes requested than available in ' + basis_file)
        U = U[:n_modes,:,:]
    else:
        # Use all available modes
        n_modes = U.shape[0]

    # Will we be doing one set of modes per bin?
    perbin = bool(npzfile['perbin'])
    
    # Now build degaussianizing functions if specified
    degauss = []   # Empty list if there is no degaussianizing needed.
    if 'g_convention' in npzfile:
        [gmax,dg]=npzfile['g_convention']
        # Read the (required) pctile values for lookup tables
        pctile=npzfile['pctile']
        
        if perbin:
            # Build 2d array of degauss functions, indexed as [i_bin, i_mode]
            degauss = [[build_degauss(pctile[jmode,jbin], gmax, dg) for jmode in range(n_modes)] \
                       for jbin in range(n_bins)]
        else:
            # Build 1d array of degauss functions
            degauss = [build_degauss(pctile[jmode], gmax, dg) for jmode in range(n_modes)]         

    # Return extracted information

    # Where to find the parameters for this:
    bias_section = sample+"_photoz_u"
    return {"sample": sample,
            "bias_section":bias_section,
            "basis":U,
            "degauss":degauss,
            "n_modes":n_modes,
            "no_degaussbins":no_degaussbins,
            "n_bins":n_bins,
            "perbin":perbin}

def execute(block, config):
    # Read own info
    pz = 'nz_'+config['sample']
    # nz0 = config['nz0']
    U = config['basis']
    degauss = config['degauss']
    n_modes = config['n_modes']
    if config['sample'] == 'lens':
        no_degaussbins = config['no_degaussbins']
    else: 
        no_degaussbins = np.array([])
    perbin = config['perbin']
    # Read z values from pz block
    nbin = block[pz, "nbin"]
    z = block[pz, "z"]

    # Check for match between the basis functions' dimensions and what pz has
    if config['n_bins'] != nbin:
        raise ValueError('Bin count in uz config does not match that in pz block')
    if U.shape[2] != len(z)-1:
        print("Shape of U:", U.shape[2])
        print("Length of z vector:", len(z)-1)
        raise ValueError('Length of basis vectors does not match length of z vector in pz block')

    # Read u values
    uvals = config['bias_section']
    u_ = np.zeros((nbin,n_modes))

    for i in range(nbin):
        if perbin:
            u_[i,:] = np.array([block[uvals, "u_{0}_{1}".format(i,j) ] for j in range(n_modes)])
            # Apply the degaussianization, if any:
            if degauss and i not in no_degaussbins:
                for j in range(n_modes):
                    u_[i,j] = degauss[i][j](u_[i,j])
            else:
                print(f'Not applying degauss to lens z-bin {i}')
        else:
            u_[i,:] = np.array([block[uvals, "u_{0}".format(j) ] for j in range(n_modes)])
            if degauss:
                for j in range(n_modes):
                    u_[i,j] = degauss[j](u_[i,j])
  
    # Make the n(z)'s:
    # U's are indexed as [mode,bin,z], u's are indexed as [bin,mode], nz indexed as (bin,z)
    dz = np.einsum('ijk,ji->jk',U, u_)

    # Save n(z)'s in the pz block.
    for i in range(nbin):
        bin_name = "bin_%d" % (i+1)  # Cosmosis labels are 1-indexed
        nz = block[pz, bin_name] 
        nz[1:]+=dz[i]   # Because Cosmosis version should have a zero prepended, for z=0 value
        nz /= np.trapz(nz, z)
        block[pz, bin_name] = nz
    return 0

def cleanup(config):
    pass
