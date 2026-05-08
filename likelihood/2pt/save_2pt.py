"""
This module saves cosmosis output to a 2-pt file after interpolating it 
to specified ell values. It is useful for making simulations, but
is not yet fully supported so please use it with care and check 
the results carefully.
"""
from __future__ import print_function

from builtins import range
from builtins import object
from cosmosis.datablock import option_section, names
import numpy as np
import twopoint
from twopoint_cosmosis import type_table
import gaussian_covariance
from spec_tools import TheorySpectrum, real_space_cov, perarcmin2_to_perrad2, ClCov, arcmin_to_rad, convert_angle


def get_scales( x_min, x_max, nbins, logspaced=True, integer_lims=False, two_thirds_midpoint=False):
    """
    Get scales
    """
    if logspaced:
        log_lims = np.linspace( np.log(x_min), np.log(x_max), nbins+1 )
        lims = np.exp(log_lims)
        if two_thirds_midpoint:
            xmin = lims[:-1]
            xmax = lims[1:]
            mids = (2./3.) * (xmax**3 - xmin**3) / (xmax**2 - xmin**2)
        else:
            xmin = log_lims[:-1]
            xmax = log_lims[1:]
            log_mids = 0.5*(xmin + xmax)
            mids = np.exp(log_mids)
    else:
        lims = np.linspace(x_min, x_max, nbins+1)
        xmin = lims[:-1]
        xmax = lims[1:]
        if two_thirds_midpoint:
            mids = (2./3.) * (xmax**3 - xmin**3) / (xmax**2 - xmin**2)
        else:
            mids = 0.5 * (xmin + xmax)

    return lims, mids

def get_ell_scales( ell_min, ell_max, nbins, logspaced=True, two_thirds_midpoint=False):
    #ells should be integers. So get scales using get_scales and then 
    #convert
    lims, mids = get_scales( ell_min, ell_max, nbins, logspaced=logspaced, two_thirds_midpoint=two_thirds_midpoint)
    ell_lims = (np.floor(lims)).astype(int)
    ell_mids = np.exp( 0.5 * ( np.log(ell_lims[1:]) + np.log(ell_lims[:-1]) ) )
    return ell_lims, ell_mids

def setup(options):
    """
    This needs to specify:
    - The section names of the projected power spectra or correlation functions 
    required:
    spectrum_sections = shear_cl galaxy_cl shear_galaxy_cl
    or spectrum_sections = shear_xi galaxy_xi galaxy_shear_xi
    - Binning (assumed the same for all spectra) - see e.g. theta_min,
    theta_max, n_theta options...
    - If you want a covariance, we also need shot/shape noise. Provide this as 
    sigma_e^2/n for shape noise or 1/n for shot noise where n is the number of 
    galaxies per sq. arcminute:
    noise_shear = 
    noise_galaxy = 
    """
    #First of all get the angular scales/binning
    logspaced = options.get_bool(option_section, "logspaced", True)
    make_covariance = options.get_bool(option_section, "make_covariance", False)
    copy_covariance = options.get_string(option_section, "copy_covariance", "")
    angle_units = options.get_string(option_section, "angle_units", "")
    two_thirds_midpoint = options.get_bool(option_section, "two_thirds_midpoint", False)

    if copy_covariance and make_covariance:
        raise ValueError("You can only set one of make_covariance and copy_covariance")

    config = {
        "make_covariance": make_covariance,
        "copy_covariance": copy_covariance,
        "logspaced": logspaced, 
        "angle_units": angle_units,
    }

    def read_list(key, default=""):
        try:
            s = options.get_string(option_section, key, default)
        except Exception:
            # Values like "00 00 00" (all integer-like tokens) get stored as
            # an int array in the CosmoSIS datablock. Convert back to strings.
            val = options[option_section, key]
            if hasattr(val, '__iter__') and not isinstance(val, str):
                s = " ".join(str(int(v)).zfill(2) for v in val)
            else:
                s = str(int(val)).zfill(2)
        return s.split()

    #Get the spectrum section names to be saved
    config["spectrum_sections"] = read_list("spectrum_sections")

    # Can optionally set a list of spectra where we should not save cross-correlations,
    # only auto-correlations
    auto_only_sections = read_list("auto_only")
    config["auto_only"] = [(s in auto_only_sections) for s in config["spectrum_sections"]]

    # And output names (name of extensions in output fits files).
    # By default we save everything to the same name it has in cosmosis sections,
    # but we can also rename things.
    if options.has_value(option_section, "output_extensions" ):
        config["output_extensions"] = read_list("output_extensions")
    else:
        config["output_extensions"] = config["spectrum_sections"]


    # This module can either save real-space or fourier-space quantities.
    # We decide which based on whether the user sets the parameters
    # (theta_min, theta_max, n_theta) or (ell_min, ell_max, n_ell)
    if options.has_value(option_section, "theta_min"):
        config['real_space'] = True
        theta_min = options.get_double(option_section, "theta_min")
        theta_max = options.get_double(option_section, "theta_max")
        n_theta = options.get_int(option_section, "n_theta")

        # There are a few different choices for how to select theta values
        # from min/max/n values. We could do log-spaced or linear-spaced,
        # or (2./3.) * (xmax**3 - xmin**3) / (xmax**2 - xmin**2) which is
        # a good approximation to the correct area-weighted answer.
        theta_lims, theta_mids = get_scales(theta_min, theta_max, n_theta,
            logspaced=logspaced, two_thirds_midpoint=two_thirds_midpoint)

        #Work out what units we're working with
        if angle_units == "":
            print("No angle_units provided, assuming arcminutes")
            angle_units = twopoint.ANGULAR_UNITS["arcmin"]
        else:
            angle_units = twopoint.ANGULAR_UNITS[angle_units]
        config['angle_units'] = angle_units

        # We also retain the theta limits in user-supplied units
        # so that we can use them in feedback later.
        config['angle_lims_userunits'] = theta_lims
        config['angle_mids_userunits'] = theta_mids

        #convert to radians
        config['angle_lims'] = convert_angle(theta_lims, config['angle_units'], twopoint.ANGULAR_UNITS['rad'])
        config['angle_mids'] = convert_angle(theta_mids, config['angle_units'], twopoint.ANGULAR_UNITS['rad'])
        
        print("Saving at these theta values (in %s):"%config['angle_units'])
        print(config['angle_mids_userunits'])


        # If we are making covariances then we need the C_ell to compute them,
        # even though we are only saving real-space quantities. In that case
        # the user needs to supply the section names for C_ell values and also
        # the corresponding xi section names to which they are transformed.
        # TODO: Ideally this would be tracked earlier.
        if config['make_covariance']:
            config["cl_sections"] = read_list("cl_sections")
            config["cl_to_xi_types"] = read_list("cl_to_xi_types")
    else:
        # The Fourier-space variant of the same thing.
        if not options.has_value(option_section, "ell_min"):
            raise ValueError("Must provide either (theta_min, theta_max and n_theta) "
                             "or (ell_min, ell_max, n_ell)")
        config['real_space'] = False
        ell_min = options.get_int(option_section, "ell_min")
        ell_max = options.get_int(option_section, "ell_max")
        n_ell = options.get_int(option_section, "n_ell")

        # The choices for ell values are similar to those for real-space values,
        # except that we the two_thirds_midpoint is a lot less useful.
        ell_lims, ell_mids = get_ell_scales( ell_min, ell_max, n_ell,
            logspaced=logspaced, two_thirds_midpoint=two_thirds_midpoint)
        config['angle_lims'] = ell_lims
        config['angle_mids'] = ell_mids
        config['angle_lims_userunits'] = config['angle_lims']
        config['angle_mids_userunits'] = config['angle_mids']
        config['angle_units'] = None
        print("Saving at these ell values:")
        print(config['angle_mids'])

    def get_array(name):
        val = options[option_section, name]
        val = np.atleast_1d(np.array(val, dtype=float))
        return val

    if config['make_covariance']:
        # Read in noise ingredients. If there is only a single bin these need to be
        # converted from scalars to arrays for later.
        #Convert per arcmin^2 quantities to per radian^2
        config["number_density_shear_arcmin2"] = get_array("number_density_shear_arcmin2")
        config["number_density_lss_arcmin2"] = get_array("number_density_lss_arcmin2")
        config["number_density_shear_rad2"] = perarcmin2_to_perrad2(config["number_density_shear_arcmin2"])
        config["number_density_lss_rad2"] = perarcmin2_to_perrad2(config["number_density_lss_arcmin2"])        

        # Previously user specified the parameter sigma_e, but it was never clear
        # if this was total or per-component.  So check explicitly for users still
        # setting that and explain the change.
        if (options.has_value(option_section, 'sigma_e') 
            and not options.has_value(option_section, 'sigma_e_total')):
            raise ValueError("The parameter sigma_e should now be specified as "
                             "'sigma_e_total' instead of the old (ambiguous) 'sigma_e', "
                             "i.e. sigma_e_total = sqrt(2) * sigma_e_per_component")
        config['sigma_e_total'] = get_array("sigma_e_total")

        # Other covariance computation parameters
        config['fsky'] = options[option_section, "fsky"] 
        config['upsample_cov'] = options.get_int(option_section, "upsample_cov", 10)
        if config['upsample_cov'] < 2:
            config['upsample_cov'] = None
        config['ell_max'] = options.get_int(option_section, "ell_max")
        config['high_l_filter'] = options.get_double(option_section, "high_l_filter", 0.75)
        cmb_lensing_noise_file = options.get_string(option_section, 'cmb_lensing_noise_file', '')
        config['cmb_lensing_noise_interp'] = None
        if cmb_lensing_noise_file:
            from scipy.interpolate import interp1d
            if cmb_lensing_noise_file.endswith('.npy'):
                # Dict-based npy file (e.g. SO/Planck noise files stored as pickled dicts).
                # Use cmb_lensing_ells_key and cmb_lensing_noise_key to select the arrays.
                noise_dict = np.load(cmb_lensing_noise_file, allow_pickle=True,
                                     encoding='latin1').item()
                ells_key = options.get_string(option_section, 'cmb_lensing_ells_key', 'els')
                noise_key = options.get_string(option_section, 'cmb_lensing_noise_key', 'Nl_MV')
                ells_nl = np.array(noise_dict[ells_key]).real
                nl = np.array(noise_dict[noise_key]).real
            else:
                # Plain text file: if 1D (ell = index), or 2D (first column is ell)
                noise_data = np.loadtxt(cmb_lensing_noise_file)
                if noise_data.ndim == 1:
                    ells_nl = np.arange(len(noise_data), dtype=float)
                    nl = noise_data
                else:
                    n_cols = noise_data.shape[1]
                    if not options.has_value(option_section, 'cmb_lensing_noise_col'):
                        raise ValueError(
                            "cmb_lensing_noise_file '%s' has %d columns. "
                            "You must set cmb_lensing_noise_col to select the noise column "
                            "(0-indexed, excluding col 0 which is ell). "
                            "E.g. for the SO nlkk files the MV estimator is col 7."
                            % (cmb_lensing_noise_file, n_cols))
                    cmb_lensing_noise_col = options.get_int(option_section, 'cmb_lensing_noise_col')
                    if cmb_lensing_noise_col == 0:
                        raise ValueError(
                            "cmb_lensing_noise_col=0 would select the ell column. "
                            "Use a non-zero column index for the noise.")
                    if cmb_lensing_noise_col >= n_cols or cmb_lensing_noise_col < -n_cols:
                        raise ValueError(
                            "cmb_lensing_noise_col=%d is out of range for file '%s' "
                            "which has %d columns (0-indexed)."
                            % (cmb_lensing_noise_col, cmb_lensing_noise_file, n_cols))
                    print("Reading CMB lensing noise from column %d of %d in '%s'"
                          % (cmb_lensing_noise_col % n_cols, n_cols, cmb_lensing_noise_file))
                    ells_nl = noise_data[:, 0]
                    nl = noise_data[:, cmb_lensing_noise_col]
            # Remove NaN/inf entries (e.g. below lmin where noise is undefined)
            valid = np.isfinite(ells_nl) & np.isfinite(nl)
            ells_nl = ells_nl[valid]
            nl = nl[valid]
            # Raise at setup time if the file does not cover the required ell range
            if config['ell_max'] > ells_nl[-1]:
                raise ValueError(
                    "cmb_lensing_noise_file '%s' only provides noise up to ell=%d, "
                    "but ell_max=%d is required for the covariance. "
                    "Either use a file with higher ell coverage or reduce ell_max."
                    % (cmb_lensing_noise_file, int(ells_nl[-1]), config['ell_max']))
            # fill_value for ell below the file's lmin (e.g. ell=0,1 below lmin=100):
            # use nl[0]. Values above ell_max are guaranteed covered by the check above.
            config['cmb_lensing_noise_interp'] = interp1d(ells_nl, nl, kind='cubic',
                                                           bounds_error=False, fill_value=(nl[0], nl[-1]))

    # name of the output file and whether to overwrite it.
    config['filename'] = options.get_string(option_section, "filename")
    config['overwrite'] = options.get_bool(option_section, "overwrite", False)


    # Scale selections in the saved data and bins to remove completely are specified as
    # angle_range_{bin1}_{bin2} = ang_min, ang_max
    # and, for example
    # cut_gamma_t = (bin1,bin2) (bin3,bin4)
    # to delete the pairs (1,2) and (3,4) of the gamma_t output.
    scale_cuts = {}
    bin_cuts = []
    range_token = "angle_range_"
    cut_token = "cut_"
    for _, key in options.keys(option_section):
        if key.startswith(range_token):
            bits = key[len(range_token):].split("_")
            name = "_".join(bits[:-2])
            if name not in config["output_extensions"]:
                raise ValueError("You set %s but there's no output extension named %s"%(key, name))
            bin1 = int(bits[-2])
            bin2 = int(bits[-1])
            scale_cuts[(name, bin1, bin2)] = options.get_double_array_1d(
                option_section, key)
        elif key.startswith(cut_token):
            name = key[len(cut_token):]
            if name not in config["output_extensions"]:
                raise ValueError("You set %s but there's no output extension named %s"%(key, name))
            cuts = options[option_section, key].split()
            cuts = [eval(cut) for cut in cuts]
            for b1, b2 in cuts:
                bin_cuts.append((name, b1, b2))
    config['bin_cuts'] = bin_cuts
    config['scale_cuts'] = scale_cuts

    return config

def execute(block, config):

    # Get various output bits
    real_space = config['real_space']
    fourier_space = not real_space
    filename = config['filename']
    overwrite = config['overwrite']
    make_covariance = config['make_covariance']
    copy_covariance = config['copy_covariance']

    # This module is not designed to be used during long-running
    # MCMC analyses, so it's fine to have verbose output here.
    print("Saving two-point data to {}".format(filename))

    theory_spec_list = []
    cl_theory_spec_list = []
    cl_to_xi_type_list = []
    spec_meas_list = []
    kernels = []
    no_kernel_found = []

    #Loop through spectrum_sections, generating a SpectrumMeasurement
    #for each.
    #If we're generating the covariance for real space spectra, also 
    #generate a TheorySpectrum for the corresponding Cl.
    print("Generating twopoint file with the following spectra:")
    print("    ", config['spectrum_sections'])
    for i_spec in range( len(config["spectrum_sections"]) ):
        auto_only = config["auto_only"][i_spec]
        spectrum_section = config["spectrum_sections"][i_spec]
        output_extension = config["output_extensions"][i_spec]

        #Read in sample information from block
        sample_a, sample_b = ( block[spectrum_section, "sample_a"], 
                               block[spectrum_section, "sample_b"] )
        kernel_name_a = "nz_"+sample_a
        kernel_name_b = "nz_"+sample_b
        
        # Get n(z) kernels required, as these are saved in the output
        # file.
        if ((kernel_name_a not in [ k.name for k in kernels ]) 
            and (kernel_name_a not in no_kernel_found)):
            if block.has_section(kernel_name_a):
                kernels.append( 
                    twopoint.NumberDensity.from_block( 
                        block, kernel_name_a ) )
            else:
                no_kernel_found.append(kernel_name_a)

        # And for the second kernel
        if kernel_name_b not in [ k.name for k in kernels ]:
            if block.has_section(kernel_name_b):
                kernels.append( 
                    twopoint.NumberDensity.from_block( 
                        block, kernel_name_b ) )
            else:
                no_kernel_found.append(kernel_name_b)

        # Report any missing kernels, but this is not necessarily an error.
        if len(no_kernel_found)>0:
            print("No kernel found for kernel names:", no_kernel_found)
            print("This might not be a problem e.g. for CMB lensing.")

        theory_spec = TheorySpectrum.from_block( block, 
            spectrum_section, auto_only=auto_only )
        theory_spec_list.append(theory_spec)

        #get angle_units
        if config["angle_units"] is not None:
            angle_units = config['angle_units'].name
        else:
            angle_units = None

        spec_meas_list.append( 
            theory_spec.to_twopoint_object(config['angle_mids_userunits'], 
            (kernel_name_a, kernel_name_b), output_extension, 
            angle_lims=config['angle_lims_userunits'], 
            angle_units=angle_units))
        
        if make_covariance:
            if real_space:
                #In this case we also need the corresponding Cl spectra to generate the covariance
                cl_section = config["cl_sections"][i_spec]
                cl_spec = TheorySpectrum.from_block(block, cl_section, auto_only=False)
                if cl_spec.is_bin_averaged:
                    raise ValueError("We need interpolated C_ell values for covariances, but your pipeline supplied bin-averaged")
                cl_theory_spec_list.append( cl_spec )
                #Check cls have the same bin pairings as their corresponding real-space spectra
                try:
                    assert cl_spec.bin_pairs == theory_spec_list[i_spec].bin_pairs
                except AssertionError as e:
                    print( "cl and xi specs have different bin_pairs:" )
                    print( "sections were %s and %s"%(cl_section, spectrum_section))
                    print( "cl bin pairs:", cl_spec.bin_pairs )
                    print( "xi bin pairs:", theory_spec_list[i_spec].bin_pairs)
                    raise(e)

    if not real_space:
        cl_theory_spec_list = theory_spec_list

    if make_covariance:
        #First we need to get the ClCov
        #For the covariance matrix, we may need to read in more Cls - e.g. if we want 
        #a covariance for Cl_ab and Cl_cd, we require Cl_ad, Cl_ac, Cl_bc and Cl_bd for 
        #the covariance calculation.
        #So the following checks the types of Cl_ab and Cl_cd, and from that infers
        #the required spectra.
        types = []
        for spec in cl_theory_spec_list:
            type_i, type_j = spec.types
            if type_i not in types:
                types.append(type_i)
            if type_j not in types:
                types.append(type_j)
        cl_specs = []
        for (i,type_1) in enumerate(types):
            for (j,type_2) in enumerate(types[i:]):
                print("Getting cls for cov:")
                print("type_1:", type_1)
                print("type_2:", type_2)
                #Get the cl section name for these types from the type_table (man
                #we need to sort out easy access to this type_table info...it sucks right now)
                try:
                    cl_section = type_table[(type_1.name, type_2.name)][0]
                except KeyError:
                    cl_section = type_table[(type_2.name, type_1.name)][0]
                assert cl_section not in [s.name for s in cl_specs]
                cl_spec = TheorySpectrum.from_block( block, cl_section )
                #Add noise if necessary
                if (cl_spec.types[0] == cl_spec.types[1]):
                    if cl_spec.types[0].name == "galaxy_shear_emode_fourier":
                        noise = ([ (s**2 / 2 / n) for (s,n) in 
                                 zip(config['sigma_e_total'],config['number_density_shear_rad2']) ])
                    elif cl_spec.types[0].name == "galaxy_position_fourier":
                        noise = [ 1./n for n in config['number_density_lss_rad2'] ]
                    elif cl_spec.types[0].name == "cmb_kappa_fourier":
                        if config['cmb_lensing_noise_interp'] is None:
                            raise ValueError("Set cmb_lensing_noise_file in [save_2pt] to compute covariance with CMB kappa")
                        noise = [config['cmb_lensing_noise_interp']]
                    else:
                        print("Tried to, but can't generate noise for spectrum %s"%cl_section)
                        raise ValueError
                    cl_spec.set_noise(noise)

                cl_specs.append( cl_spec )

        cl_cov = ClCov(cl_specs, fsky=config['fsky'])

        # If saving real-space data we have to convert he Fourier C_ell covariance
        # to the real-space version.  There are various ways to do this; the one
        # here should be accurate enough.
        if real_space:

            #If requested, apply bin cuts now - this will speed up the covariance calculation
            #Need to apply cuts to real space spectra and cls
            for (name, b1, b2) in config['bin_cuts']:
                print("cutting %d,%d from %s"%(b1,b2,name))
                spec_index = config['output_extensions'].index(name)
                spec_meas_list[spec_index].cut_bin_pair( (b1,b2), complain=True )
                cl_theory_spec_list[spec_index].cut_bin_pair( (b1,b2) )

            cov_blocks, covmat, xi_starts, xi_lengths = real_space_cov( cl_cov, 
                cl_theory_spec_list, config['cl_to_xi_types'], 
                config['ell_max'], config['angle_lims'], 
                upsample=config['upsample_cov'], 
                high_l_filter = config['high_l_filter'] )
            covmat_info = twopoint.CovarianceMatrixInfo( 'COVMAT', [s.name for s in spec_meas_list], 
                                                         xi_lengths, covmat )

        else:
            # Otherwise just convert directly to 2pt format
            covmat, cl_lengths = cl_cov.get_binned_cl_cov(config['angle_lims'])
            assert covmat.shape[0] == sum([len(s.value) for s in spec_meas_list])
            covmat_info = twopoint.CovarianceMatrixInfo( 'COVMAT', [s.name for s in spec_meas_list],
                                                         [len(s.value) for s in spec_meas_list], covmat )
    
    elif copy_covariance:
        parent_file = twopoint.TwoPointFile.from_fits(copy_covariance)
        # check that covariances match
        for s1, s2 in zip(spec_meas_list, parent_file.spectra):
            if s1.name != s2.name:
                raise ValueError("Different spectrum order in parent file")
            if len(s1) != len(s2):
                print ("s1, s2",s1, s2)
                print ("len s1, s2",len(s1), len(s2))
                raise ValueError("Different spectrum lengths in parent file")
            if not ((s1.bin1 == s2.bin1).all() and (s1.bin2 == s2.bin2).all()):
                raise ValueError("Different tomo bin order in parent file")
            if not (s1.angular_bin == s2.angular_bin).all():
                raise ValueError("Different angular bin order in parent file")
        covmat_info = parent_file.covmat_info
    else:
        covmat_info = None

    if not spec_meas_list:
        raise ValueError("Sorry - I couldn't find any spectra to save.")

    windows = []

    data = twopoint.TwoPointFile(spec_meas_list, kernels, windows, covmat_info)

    # Apply cuts
    scale_cuts = config['scale_cuts']
    bin_cuts = config['bin_cuts']
    if scale_cuts or bin_cuts:
        data.mask_scales(scale_cuts, bin_cuts)

    data.to_fits(filename, overwrite=overwrite)

    return 0

def convert_nz_steradian(n):
    return n * (41253.0 * 60. * 60.) / (4 * np.pi)



