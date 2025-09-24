import sacc
from cosmosis.gaussian_likelihood import GaussianLikelihood
from cosmosis.datablock import names, BlockError
import numpy as np
import re
import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
twopoint_dir = os.path.join(parent_dir, "2pt")
sys.path.append(twopoint_dir)
from spec_tools import SpectrumInterp

default_sections = {
    "cl_ee": "shear_cl",
    "galaxy_shear_cl_ee": "shear_cl",
    "cl_bb": "shear_cl_bb",
    "galaxy_shear_cl_bb": "shear_cl_bb",
    "cl_eb": "shear_cl_eb",
    "galaxy_shear_cl_eb": "shear_cl_eb",
    "cl_be": "shear_cl_be",
    "galaxy_shear_cl_be": "shear_cl_be",
    "galaxy_density_cl": "galaxy_cl",
    "galaxy_shearDensity_cl_e": "galaxy_shear_cl",
}




class SaccClLikelihood(GaussianLikelihood):
    # This is a sub-class of the class GaussianLikelihood
    # which can be found in the file ${COSMOSIS_SRC_DIR}/cosmosis/gaussian_likelihood.py
    # That super-class implements the generic behaviour that all Gaussian likelihoods
    # follow - the basic form of the likelihoods, inverting covariance matrices, saving
    # results, etc.  This sub-clas does the parts that are specific to this 2-pt
    # likelihood - loading data from a file, getting the specific theory prediction
    # to which to compare it, etc.
    like_name = "2pt"

    def __init__(self, options):
        self.save_theory = options.get_string("save_theory", "")
        self.save_realization = options.get_string("save_realization", "")
        self.flip = options.get_string("flip", "").split()

        super().__init__(options)


    def build_data(self):
        filename = self.options.get_string('data_file')

        s = sacc.Sacc.load_fits(filename)
        self.sacc_data = s

        # All the names of two-points measurements that were found in the data
        # file
        all_names = s.get_data_types()

        # We may not want to use all the likelihoods in the file.
        # We can set an option to only use some of them
        data_sets = self.options.get_string("data_sets", default="all").split()
        if data_sets != ["all"]:
            for d in all_names:
                if d not in data_sets:
                    s.remove_selection(d)

        # Allow user to list a series of regular expression patterns, presumably
        # simple ones like source* lens* to be kept in the file
        tracer_patterns = self.options.get_string("keep_tracers", default=".*").split()
        tracer_re = [re.compile(pattern) for pattern in tracer_patterns]
        tracer_pairs = s.get_tracer_combinations()
        for t1, t2 in tracer_pairs:
            match1 = any([p.match(t1) for p in tracer_re])
            match2 = any([p.match(t2) for p in tracer_re])
            if not match1:
                miss = t1
            elif not match2:
                miss = t2
            else:
                miss = ""

            if miss:
                print(f"Removing tracer pair {t1}, {t2} because {miss} does not match any pattern in {tracer_patterns}")
                s.remove_selection(tracers=(t1,t2))

        # The ones we actually used.
        self.used_names = s.get_data_types()

        # Check for scale cuts. In general, this is a minimum and maximum angle for
        # each spectrum, for each redshift bin combination. Which is clearly a massive pain...
        # but what can you do?
        import warnings
        warnings.filterwarnings("ignore", "empty index selected", category=UserWarning)

        for name in self.used_names:
            for t1, t2 in s.get_tracer_combinations(name):
                option_name = "angle_range_{}_{}_{}".format(name, t1, t2)
                if self.options.has_value(option_name):
                    r = self.options.get_double_array_1d(option_name)
                    # TODO: Update for theta limits on xi(theta)

                    s.remove_selection(name, (t1, t2), ell__lt=r[0])
                    s.remove_selection(name, (t1, t2), ell__gt=r[1])

        for name in self.used_names:
            option_name = "cut_{}".format(name)
            if self.options.has_value(option_name):
                cuts = self.options[option_name].split()
                for cut in cuts:
                    t1, t2 = cut.split(",")
                    s.remove_selection(name, (t1, t2))


        # Info on which likelihoods we do and do not use
        print("Found these data sets in the file:")
        total_data_points = 0
        final_names = s.get_data_types()
        for name in all_names:
            if name in final_names:
                data_points = len(s.indices(name))
            else:
                data_points = 0
            if name in self.used_names:
                print("    - {}  {} data points after cuts {}".format(name,  data_points, "  [using in likelihood]"))
                total_data_points += data_points
            else:
                print("    - {}  {} data points after cuts {}".format(name, data_points, "  [not using in likelihood]"))
        print("Total data points used = {}".format(total_data_points))

        self.sections_for_names = {}
        for name in final_names:
            if self.options.has_value(f"{name}_section"):
                section = self.options[f"{name}_section"]
            elif name in default_sections:
                section = default_sections[name]
            else:
                raise ValueError(f"SACC likelihood does not yet understand data type {name}")
            print(f"Will look for theory prediction for data set {name} in section {section}")
            self.sections_for_names[name] = section


        # build up the data vector from all the separate vectors.
        # Just concatenation
        data_vector = s.get_mean()

        # Make sure
        if len(data_vector) == 0:
            raise ValueError(
                "No data was chosen to be used from 2-point data file {0}. It was either not selectedin data_sets or cut out".format(filename))

        # The x data is not especially useful here, so return None.
        # We will access the self.sacc_data directly later to
        # determine ell/theta values
        return None, data_vector

    def build_covariance(self):
        
        C = self.sacc_data.covariance.dense
        r = self.options.get_int('covariance_realizations', default=-1)
        self.sellentin = self.options.get_bool('sellentin', default=False)

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
                print(f"(you told us it was {r} simulations in the ini file)")
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

        # Now we actually loop through our data sets
        for spectrum in self.sacc_data.get_data_types():
            theory_vector, angle_vector, bin1_vector, bin2_vector = self.extract_spectrum_prediction(
                block, spectrum)
            theory.append(theory_vector)
            angle.append(angle_vector)
            bin1.append(bin1_vector)
            bin2.append(bin2_vector)


        # We also collect the ell or theta values.
        # The gaussian likelihood code itself is not expecting these,
        # so we just save them here for convenience.
        angle = np.concatenate(angle)
        bin1 = np.concatenate(bin1)
        bin2 = np.concatenate(bin2)
        # dataset_name = np.concatenate(dataset_name)
        block[names.data_vector, self.like_name + "_angle"] = angle
        block[names.data_vector, self.like_name + "_bin1"] = bin1
        block[names.data_vector, self.like_name + "_bin2"] = bin2
        # block[names.data_vector, self.like_name+"_name"] = dataset_name

        # the thing it does want is the theory vector, for comparison with
        # the data vector
        theory = np.concatenate(theory)

        return theory

    def do_likelihood(self, block):
        # Run the
        super().do_likelihood(block)

        # We can optionally save realizations or pure-theory sacc files here
        # for subsequent testing
        if self.save_theory:
            output_sacc = self.sacc_data.copy()
            theory = block[names.data_vector, self.like_name + "_theory"]
            for t, d in zip(theory, output_sacc.data):
                d.value = t
            output_sacc.save_fits(self.save_theory, overwrite=True)

        if self.save_realization:
            output_sacc = self.sacc_data.copy()
            theory = block[names.data_vector, self.like_name + "_simulation"]
            for t, d in zip(theory, output_sacc.data):
                d.value = t
            output_sacc.save_fits(self.save_realization, overwrite=True)


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

    def extract_spectrum_prediction(self, block, data_type):

        s = self.sacc_data
        section = self.sections_for_names[data_type]

        ell_theory = block[section, "ell"]
        is_auto = block[section, "is_auto"]

        # We build up these vectors from all the data points.
        # Only the theory vector is needed for the likelihood - the others
        # are for convenience, debugging, etc.
        theory_vector = []
        angle_vector = []
        bin1_vector = []
        bin2_vector = []


        # Because we called to_canonical_order when we loaded the data,
        # we know that the data is grouped by data type, and then by tracers (tomo bins).
        # So that means we can do a data type at a time and then concatenate them, and
        # within this do a bin pair at a time, and concatenate them too.
        for b1, b2 in s.get_tracer_combinations(data_type):
            # Here we assume that the bin names are formatted such that
            # they always end with _1, _2, etc. That isn't always true in
            # sacc, but is somewhat baked into cosmosis in other modules.
            # It would be nice to update that throughout, but that will
            # have to wait. Also, cosmosis bins start from 1 not 0.
            # We need to make sure that's fixed in the 
            i = int(b1.split("_")[-1]) + 1
            j = int(b2.split("_")[-1]) + 1

            if data_type in self.flip:
                i, j = j, i

            try:
                cl_theory = block[section, f"bin_{i}_{j}"]
            except BlockError:
                if is_auto:
                    cl_theory = block[section, f"bin_{j}_{i}"]
                else:
                    raise

            # check that all the data points share the same window
            # object (window objects contain weights for a set of ell values,
            # as a matrix), or that none have windows.
            window = None

            for d in s.get_data_points(data_type, (b1, b2)):
                w = d.get_tag('window')
                if (window is not None) and (w is not window):
                    raise ValueError("Sacc likelihood currently assumes data types share a window object")
                window = w

            # We need to interpolate between the sample ell values
            # onto all the ell values required by the weight function
            # This will give zero outside the range where we have
            # calculated the theory
            cl_theory_spline = SpectrumInterp(ell_theory, cl_theory)
            if window is not None:
                ell_window = window.values
                cl_theory_interpolated = cl_theory_spline(ell_window)

            for d in s.get_data_points(data_type, (b1, b2)):
                ell_nominal = d['ell']
                if window is None:
                    binned_cl_theory = cl_theory_spline(ell_nominal)
                else:
                    index = d['window_ind']
                    weight = window.weight[:, index]

                    # We don't automatically renormalize the weights.
                    # Some contexts, like the output from NaMaster,
                    # use non-unit-sum weights
                    binned_cl_theory = (weight @ cl_theory_interpolated)

                theory_vector.append(binned_cl_theory)
                angle_vector.append(ell_nominal)
                bin1_vector.append(i - 1)
                bin2_vector.append(j - 1)

        # Return the whole collection as a single array
        theory_vector = np.array(theory_vector)

        # For convenience we also save the angle vector (ell or theta)
        # and bin indices
        angle_vector = np.array(angle_vector)
        bin1_vector = np.array(bin1_vector, dtype=int)
        bin2_vector = np.array(bin2_vector, dtype=int)

        return theory_vector, angle_vector, bin1_vector, bin2_vector


setup, execute, cleanup = SaccClLikelihood.build_module()
