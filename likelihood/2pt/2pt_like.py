from cosmosis.gaussian_likelihood import GaussianLikelihood
from cosmosis.datablock import names
from cosmosis import __version__ as cosmosis_version

from twopoint_cosmosis import theory_names, type_table
from astropy.io import fits
from scipy.interpolate import interp1d
import numpy as np
import twopoint
import gaussian_covariance
import os
from spec_tools import TheorySpectrum
default_array = np.repeat(-1.0, 99)

# There is a bug in cosmosis 3.19 and below that causes a crash in the
# logging mechanism if an array of strings is saved to the block.
# This only appears if you try to run debugging, but is a hard crash
# so we really want to avoid it.
try:
    cosmosis_version = cosmosis_version.split('.')
    cosmosis_version = int(cosmosis_version[0]), int(cosmosis_version[1])
    is_newer_than_3_19 = (cosmosis_version[0] > 3) or \
        (cosmosis_version[0] == 3 and cosmosis_version[1] > 19)
except (ValueError, IndexError):
    # fall back to the old version
    is_newer_than_3_19 = False

def is_default(x):
    return len(x) == len(default_array) and (x == default_array).all()


def convert_nz_steradian(n):
    return n * (41253.0 * 60. * 60.) / (4 * np.pi)


class TwoPointLikelihood(GaussianLikelihood):
    # This is a sub-class of the class GaussianLikelihood
    # which can be found in the file ${COSMOSIS_SRC_DIR}/cosmosis/gaussian_likelihood.py
    # That super-class implements the generic behaviour that all Gaussian likelihoods
    # follow - the basic form of the likelihoods, inverting covariance matrices, saving
    # results, etc.  This sub-clas does the parts that are specific to this 2-pt
    # likelihood - loading data from a file, getting the specific theory prediction
    # to which to compare it, etc.
    like_name = "2pt"

    def __init__(self, options):
        # We may decide to use an analytic gaussian covariance
        # in that case we won't load the covmat.
        self.gaussian_covariance = options.get_bool(
            "gaussian_covariance", False)
        if self.gaussian_covariance:
            self.constant_covariance = False

        self.moped = options.get_string('moped', default='')

        super(TwoPointLikelihood, self).__init__(options)

        if self.moped:
            print("Using compressed data from MOPED algorithm: {} data points".format(len(self.moped_data)))
            if self.sellentin:
                raise ValueError("Sellentin mode is incompatible with Moped mode in 2pt like")

    def build_data(self):
        filename = self.options.get_string('data_file')

        # Suffixes to added on to two point data from e.g. different experiments
        suffix_string = self.options.get_string('suffixes', default="")
        if suffix_string == "":
            # If there are no suffixes provided, then we create a list of a single empty suffix
            suffixes = [""]
        else:
            suffixes_temp = suffix_string.split()
            suffixes = []
            for suffix in suffixes_temp:
                if (suffix.lower() == 'none'):
                    suffixes.append('')
                else:
                    suffixes.append("_" + suffix)
        self.suffixes = suffixes

        if self.gaussian_covariance:
            covmat_name = None
            area = self.options.get_double("survey_area")  # in square degrees
            self.sky_area = area * (np.pi * np.pi) / (180 * 180)

            def get_arr(x):
                if self.options.has_value(x):
                    a = self.options[x]
                    if not isinstance(a, np.ndarray):
                        a = [a]
                else:
                    a = default_array
                return a

            self.number_density_shear_bin = get_arr("number_density_shear_bin")
            self.number_density_lss_bin = get_arr("number_density_lss_bin")
            self.sigma_e_bin = get_arr("sigma_e_bin")

        else:
            covmat_name = self.options.get_string("covmat_name", "COVMAT")

        # This is the main work - read data in from the file
        self.two_point_data = twopoint.TwoPointFile.from_fits(
            filename, covmat_name)

        # Potentially cut out lines. For some reason one version of
        # this file used zeros to mark masked values.
        if self.options.get_bool("cut_zeros", default=False):
            print("Removing 2-point values with value=0.0")
            self.two_point_data.mask_bad(0.0)

        if self.options.get_bool("cut_cross", default=False):
            print("Removing 2-point values from cross-bins")
            self.two_point_data.mask_cross()

        # All the names of two-points measurements that were found in the data
        # file
        all_names = [spectrum.name for spectrum in self.two_point_data.spectra]

        # We may not want to use all the likelihoods in the file.
        # We can set an option to only use some of them
        data_sets = self.options.get_string("data_sets", default="all")
        if data_sets != "all":
            data_sets = data_sets.split()
            self.two_point_data.choose_data_sets(data_sets)

        # The ones we actually used.
        self.used_names = [
            spectrum.name for spectrum in self.two_point_data.spectra]

        # Check for scale cuts. In general, this is a minimum and maximum angle for
        # each spectrum, for each redshift bin combination. Which is clearly a massive pain...
        # but what can you do?

        scale_cuts = {}
        for name in self.used_names:
            s = self.two_point_data.get_spectrum(name)
            for b1, b2 in s.bin_pairs:
                option_name = "angle_range_{}_{}_{}".format(name, b1, b2)
                if self.options.has_value(option_name):
                    r = self.options.get_double_array_1d(option_name)
                    scale_cuts[(name, b1, b2)] = r

        # Now check for completely cut bins
        # example:
        # cut_wtheta = 1,2  1,3  2,3
        bin_cuts = []
        for name in self.used_names:
            s = self.two_point_data.get_spectrum(name)
            option_name = "cut_{}".format(name)
            if self.options.has_value(option_name):
                cuts = self.options[option_name].split()
                cuts = [eval(cut) for cut in cuts]
                for b1, b2 in cuts:
                    bin_cuts.append((name, b1, b2))

        if scale_cuts or bin_cuts:
            self.two_point_data.mask_scales(scale_cuts, bin_cuts)
        else:
            print("No scale cuts mentioned in ini file.")

        # Info on which likelihoods we do and do not use
        print("Found these data sets in the file:")
        total_data_points = 0
        final_names = [
            spectrum.name for spectrum in self.two_point_data.spectra]
        for name in all_names:
            if name in final_names:
                data_points = len(self.two_point_data.get_spectrum(name))
            else:
                data_points = 0
            if name in self.used_names:
                print("    - {}  {} data points after cuts {}".format(name,  data_points, "  [using in likelihood]"))
                total_data_points += data_points
            else:
                print("    - {}  {} data points after cuts {}".format(name, data_points, "  [not using in likelihood]"))
        print("Total data points used = {}".format(total_data_points))

        # Convert all units to radians.  The units in cosmosis are all
        # in radians, so this is the easiest way to compare them.
        for spectrum in self.two_point_data.spectra:
            if spectrum.is_real_space():
                spectrum.convert_angular_units("rad")
                # if self.options.get_bool("print physical scale",False):
                #	section,_,_=theory_names(spectrum)
                #	chi_peak =
                #	for ang in spectrum.angle:

        # build up the data vector from all the separate vectors.
        # Just concatenation
        data_vector = np.concatenate(
            [spectrum.value for spectrum in self.two_point_data.spectra])

        # Make sure
        if len(data_vector) == 0:
            raise ValueError(
                "No data was chosen to be used from 2-point data file {0}. It was either not selectedin data_sets or cut out".format(filename))

        if self.moped:
            data_file = fits.open(filename)
            self.moped_data = data_file['MOPED-DATA-{}'.format(self.moped)].data['moped']
            self.moped_transform = data_file['MOPED-TRANSFORM-{}'.format(self.moped)].data
            data_file.close()

            return None, self.moped_data

        # The x data is not especially useful here, so return None.
        # We will access the self.two_point_data directly later to
        # determine ell/theta values
        return None, data_vector

    def build_covariance(self):

        
        C = np.array(self.two_point_data.covmat)
        r = self.options.get_int('covariance_realizations', default=-1)
        self.sellentin = self.options.get_bool('sellentin', default=False)

        if self.moped:
            return np.identity(len(self.moped_data))


        if self.sellentin:
            if not self.constant_covariance:
                print()
                print("You asked for the Sellentin-Heavens correction to be applied")
                print("But also asked for a non-constant (maybe Gaussian?) covariance")
                print("matrix.  I think that probably suggests you have made a mistake")
                print("somewhere unless you have thought about this quite carefully.")
                print()
            if r < 0:
                print()
                print("ERROR: You asked for the Sellentin-Heavens corrections")
                print("by setting sellentin=T, but you did not set covariance_realizations")
                print("If you want covariance_realizations=infinity you can use 0")
                print("(unlikely, but it's also possible you were super-perverse and set it negative?)")
                print()
                raise ValueError(
                    "Please set covariance_realizations for 2pt like. See message above.")
            elif r == 0:
                print()
                print("NOTE: You asked for the Sellentin-Heavens corrections")
                print("but set covariance_realizations=0. I am assuming you want")
                print("the limit of an infinite number of realizations, so we will just go back")
                print("to the original Gaussian model")
                print()
                self.sellentin = False
            else:
                # use proper correction
                self.covariance_realizations = r
                print()
                print("You set sellentin=T so I will apply the Sellentin-Heavens correction")
                print("for a covariance matrix estimated from Monte-Carlo simulations")
                print("(you told us it was {} simulations in the ini file)".format(r))
                print("This analytic marginalization converts the Gaussian distribution")
                print("to a multivariate student's t distribution instead.")
                print()

        elif r > 0:
            # Just regular increase in covariance size, no Sellentin change.
            p = C.shape[0]
            # This x is the inverse of the alpha used in the old code
            # because that applied to the weight matrix not the covariance
            x = (r - 1.0) / (r - p - 2.0)
            C = C * x
            print()
            print("You set covariance_realizations={} in the 2pt likelihood parameter file".format(r))
            print("So I will apply the Anderson-Hartlap correction to the covariance matrix")
            print("The covariance matrix is nxn = {}x{}".format(p, p))
            print("So the correction scales the covariance matrix by (r - 1) / (r - n - 2) = {}".format(x))
            print()
        return C

    def extract_theory_points(self, block):
        theory = []
        # We may want to save these splines for the covariance matrix later
        self.theory_splines = {}

        # We have a collection of data vectors, one for each spectrum
        # that we include. We concatenate them all into one long vector,
        # so we do the same for our theory data so that they match

        # We will also save angles and bin indices for plotting convenience,
        # although these are not actually used in the likelihood
        angle = []
        bin1 = []
        bin2 = []
        dataset_name = []

        # Get appropriate suffixes
        # If only a single suffix is provided, assume this applies to all data sets
        if len(self.suffixes) == 1:
            suffixes = np.tile(self.suffixes[0], len(self.two_point_data.spectra))
        elif len(self.suffixes) > 1 and len(self.suffixes) == len(self.two_point_data.spectra):
            suffixes = self.suffixes
        else:
            raise ValueError("The number of suffixes supplied does not match the number of two point spectra.")
        
        # Now we actually loop through our data sets    
        for ii, spectrum in enumerate(self.two_point_data.spectra):
            theory_vector, angle_vector, bin1_vector, bin2_vector = self.extract_spectrum_prediction(
                block, spectrum, suffixes[ii])
            theory.append(theory_vector)
            angle.append(angle_vector)
            bin1.append(bin1_vector)
            bin2.append(bin2_vector)
            dataset_name.append(np.repeat(spectrum.name, len(bin1_vector)))

        # We also collect the ell or theta values.
        # The gaussian likelihood code itself is not expecting these,
        # so we just save them here for convenience.
        angle = np.concatenate(angle)
        bin1 = np.concatenate(bin1)
        bin2 = np.concatenate(bin2)
        dataset_name = np.concatenate(dataset_name)
        block[names.data_vector, self.like_name + "_angle"] = angle
        block[names.data_vector, self.like_name + "_bin1"] = bin1
        block[names.data_vector, self.like_name + "_bin2"] = bin2
        if is_newer_than_3_19:
            block[names.data_vector, self.like_name+"_name"] = dataset_name
        else:
            block[names.data_vector, self.like_name+"_name"] = "Value not saved in block in cosmosis 3.19 or below due to a bug."

        # the thing it does want is the theory vector, for comparison with
        # the data vector
        theory = np.concatenate(theory)

        if self.moped:
            return np.dot(self.moped_transform, theory)

        return theory

    def do_likelihood(self, block):
        # Run the
        super(TwoPointLikelihood, self).do_likelihood(block)

        if self.sellentin:
            # The Sellentin-Heavens correction from arxiv 1511.05969
            # accounts for a finite number of Monte-Carlo realizations
            # being used to estimate the covariance matrix.

            # Note that this invalidates the saved simulation used for
            # the ABC sampler.  I can't think of a better way of doing this
            # than overwriting the whole things with NaNs - that will at
            # least make clear there is a problem somewhere and not
            # yield misleading results.
            block[names.data_vector, self.like_name + "_simulation"] = (
                np.nan * block[names.data_vector, self.like_name + "_simulation"])

            # It changes the Likelihood from Gaussian to a multivariate
            # student's t distribution.  Here we will have to do a little
            # hack and overwrite the stuff that the original Gaussian
            # method did above
            N = self.covariance_realizations
            chi2 = block[names.data_vector, self.like_name + "_CHI2"]

            # We might be using a cosmologically varying
            # covariance matrix, though I'm not sure what that would mean.
            # There is a warning about this above.
            if self.constant_covariance:
                log_det = 0.0
            else:
                log_det = block[names.data_vector, self.like_name + "_LOG_DET"]

            like = -0.5 * log_det - 0.5 * N * np.log(1 + chi2 / (N - 1.))

            # overwrite the log-likelihood
            block[names.likelihoods, self.like_name + "_LIKE"] = like

    #Should suffix be made into a keyword?
    def extract_spectrum_prediction(self, block, spectrum, suffix):

        # We may need theory predictions for multiple different
        # types of spectra: e.g. shear-shear, pos-pos, shear-pos.
        # So first we find out from the spectrum where in the data
        # block we expect to find these - mapping spectrum types
        # to block names
        section, x_name, y_name = theory_names(spectrum)
        
        # To handle multiple different data sets we allow a suffix
        # to be applied to the section names, so that we can look up
        # e.g. "shear_cl_des" instead of just "shear_cl".
        section += suffix
        
        # Initialize TheorySpectrum class from block
        bin_pairs = spectrum.get_bin_pairs()
        theory_spec = TheorySpectrum.from_block(block, 
            section, bin_pairs=bin_pairs)

        # If the theory spectrum has been bin-averaged then we expect the
        # data to be so also.  We check this by ensuring that angle_min is specified
        # Based on this, we also generate the angle argument passed to the spectrum
        # differently.  The bin-averaged version expects a tuple angle_min and angle_max,
        # whereas the interpolated version just wants a single angle.
        if theory_spec.is_bin_averaged:
            if spectrum.angle_min is None:
                raise ValueError("Your theory pipeline produced angle-binnned values, but your data it not binned.")
            angles = list(zip(spectrum.angle_min, spectrum.angle_max))
        else:
            angles = spectrum.angle
        
        # We store the nominal mid-points for plotting later on, etc.
        angle_mids = spectrum.angle

        # This is a bit of a hack, but later on if we are making a covariance
        # we need all the splines, so pull them out here.
        bin_splines = {}

        # We build up these vectors from all the data points.
        # Only the theory vector is needed for the likelihood - the others
        # are for convenience, debugging, etc.
        theory_vector = []
        angle_vector = []
        bin1_vector = []
        bin2_vector = []

        for (b1, b2, angle, angle_mid) in zip(spectrum.bin1, spectrum.bin2, angles, angle_mids):
            # The extra object will either be a spline (for interpolated spectra)
            # or theta mid-point values (for bin-averaged ones, e.g. for plotting)
            theory, extra = theory_spec.get_spectrum_value(b1, b2, angle)

            # We can only record the splines for non-bin-averaged values
            if not theory_spec.is_bin_averaged:
                bin_splines[y_name.format(b1, b2)] = extra

            # Build up the vector - we make this into an array later
            theory_vector.append(theory)
            angle_vector.append(angle_mid)
            bin1_vector.append(b1)
            bin2_vector.append(b2)

        self.theory_splines[section] = bin_splines

        # Return the whole collection as an array
        theory_vector = np.array(theory_vector)

        # For convenience we also save the angle vector (ell or theta)
        # and bin indices
        angle_vector = np.array(angle_vector)
        bin1_vector = np.array(bin1_vector, dtype=int)
        bin2_vector = np.array(bin2_vector, dtype=int)

        return theory_vector, angle_vector, bin1_vector, bin2_vector


    def extract_covariance(self, block):
        assert self.gaussian_covariance, "Set constant_covariance=F but somehow not with Gaussian covariance.  Internal error - please open an issue on the cosmosis site."

        C = []
        # s and t index the spectra that we have. e.g. s or t=1 might be the full set of
        # shear-shear measuremnts
        for s, AB in enumerate(self.two_point_data.spectra[:]):
            M = []
            for t, CD in enumerate(self.two_point_data.spectra[:]):
                print("Looking at covariance between {} and {} (s={}, t={})".format(AB.name, CD.name, s, t))
                # We only calculate the upper triangular.
                # Get the lower triangular here. We have to
                # transpose it compared to the upper one.
                if s > t:
                    MI = C[t][s].T
                else:
                    MI = gaussian_covariance.compute_gaussian_covariance(self.sky_area,
                                                                         self._lookup_theory_cl, block, AB, CD)
                M.append(MI)
            C.append(M)

        # C is now a list of lists of 2D arrays.
        # Now turn C into a big 2D array by stacking
        # the arrays
        C = np.vstack([np.hstack(CI) for CI in C])

        return C

    def _lookup_theory_cl(self, block, A, B, i, j, ell):
        """
        This is a helper function for the compute_gaussian_covariance code.
        It looks up the theory value of C^{ij}_{AB}(ell) in the 
        """
        # We have already saved splines into the theory space earlier
        # when constructing the theory vector.
        # So now we just need to look those up again, using the same
        # code we use in the twopoint library.
        section, ell_name, value_name = type_table[A, B]
        assert ell_name == "ell", "Gaussian covariances are currently only written for C_ell, not other 2pt functions"
        d = self.theory_splines[section]

        # We save the splines with these names when we extract the theory vector
        name_ij = value_name.format(i, j)
        name_ji = value_name.format(j, i)

        # Hopefully we already have the theory spline extracted
        if name_ij in d:
            spline = d[name_ij]
        # For symmetric spectra (not just auto-correlations, but any thing like C_EE or C_NN where
        # we cross-correlate something with itself) we can use ji for ij as it is the same. This is
        # not true for cross spectra
        elif name_ji in d and (A == B):
            spline = d[name_ji]
        else:
            # It's possible too that we need something for the covariance that we didn't need for the
            # data vector - for example to got the covariance between C^EE and C^NN we need C^NE even
            # if we don't have any actual measurements of NE. In that case we have to g
            angle_theory = block[section, ell_name]
            if block.has_value(section, name_ij):
                theory = block[section, name_ij]
            # The same symmetry argument as above applies
            elif block.has_value(section, name_ji) and A == B:
                theory = block[section, name_ji]
            else:
                raise ValueError("Could not find theory prediction {} in section {}".format(
                    value_name.format(i, j), section))

            spline = interp1d(angle_theory, theory)
            # Finally cache this so we don't have to do this again.
            d[name_ij] = spline

        obs_cl = spline(ell)

        # For shear-shear the noise component is sigma^2 / number_density_bin
        # and for position-position it is just 1/number_density_bin
        if (A == B) and (A == twopoint.Types.galaxy_shear_emode_fourier.name) and (i == j):
            if i > len(self.number_density_shear_bin) or i > len(self.sigma_e_bin) or is_default(self.sigma_e_bin) or is_default(self.number_density_shear_bin):
                raise ValueError(
                    "Not enough number density bins for shear specified")
            noise = self.sigma_e_bin[i - 1]**2 / \
                convert_nz_steradian(self.number_density_shear_bin[i - 1])
            obs_cl += noise
        if (A == B) and (A == twopoint.Types.galaxy_position_fourier.name) and (i == j):
            if i > len(self.number_density_lss_bin) or is_default(self.number_density_lss_bin):
                raise ValueError(
                    "Not enough number density bins for lss specified")
            noise = 1.0 / \
                convert_nz_steradian(self.number_density_lss_bin[i - 1])
            obs_cl += noise

        return obs_cl


setup, execute, cleanup = TwoPointLikelihood.build_module()
