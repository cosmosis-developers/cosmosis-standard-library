import sacc
from cosmosis.gaussian_likelihood import GaussianLikelihood
from cosmosis.datablock import names
import numpy as np
import re
import sacc_likelihoods

import os
import sys

default_sections = {
    # Spectrum default sections
    "cl_ee": ("spectrum", "shear_cl",),
    "galaxy_shear_cl_ee": ("spectrum", "shear_cl",),
    "cl_bb": ("spectrum", "shear_cl_bb",),
    "galaxy_shear_cl_bb": ("spectrum", "shear_cl_bb",),
    "cl_eb": ("spectrum", "shear_cl_eb",),
    "galaxy_shear_cl_eb": ("spectrum", "shear_cl_eb",),
    "cl_be": ("spectrum", "shear_cl_be",),
    "galaxy_shear_cl_be": ("spectrum", "shear_cl_be",),
    "galaxy_density_cl": ("spectrum", "galaxy_cl",),
    "galaxy_shearDensity_cl_e": ("spectrum", "galaxy_shear_cl",),
    "galaxy_shearDensity_cl_b": ("spectrum", "galaxy_shear_cl",),

    # Real default sections
    "xi_ee": ("real", "shear_xi",),
    "galaxy_shear_xi_ee": ("real", "shear_xi",),
    "xi_bb": ("real", "shear_xi_bb",),
    "galaxy_shear_xi_bb": ("real", "shear_xi_bb",),
    "xi_eb": ("real", "shear_xi_eb",),
    "galaxy_shear_xi_eb": ("real", "shear_xi_eb",),
    "xi_be": ("real", "shear_xi_be",),
    "galaxy_shear_xi_be": ("real", "shear_xi_be",),
    "galaxy_shear_xi_minus": ("real", "shear_xi_minus",),
    "galaxy_shear_xi_plus": ("real", "shear_xi_plus",),
    "galaxy_shear_xi_imagMinus": ("real", "shear_xi_minus",),
    "galaxy_shear_xi_imagPlus": ("real", "shear_xi_plus",),
    "galaxy_density_xi": ("real", "galaxy_xi",),
    "galaxy_shearDensity_xi_t": ("real", "galaxy_shear_xi",),
    "galaxy_shearDensity_xi_x": ("real", "galaxy_shear_xi",),

    # COSEBI's and Psi-stats default sections
    "galaxy_shear_cosebi_bb": ("cosebis", "cosebis_b"),
    "galaxy_shear_cosebi_ee": ("cosebis", "cosebis"),
    "galaxy_shearDensity_cosebi_e": ("cosebis", "psi_stats_gm"),
    "galaxy_density_cosebi": ("cosebis", "psi_stats_gg"),

    # One-point default sections
    # Come back and think about the naming here later:
    "galaxy_stellarmassfunction": ("one_point_mass", "smf",),
    "galaxy_luminosityfunction": ("one_point_luminosity", "lf",),
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
        self.sacc_like = options.get_string("sacc_like", "2pt")

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
        for tracers in tracer_pairs:
            matches = [any([p.match(t) for p in tracer_re]) for t in tracers]
            miss = ""
            for t, m in zip(tracers, matches):
                if not m:
                    miss = t

            if miss:
                print(f"Removing tracer combination {tracers} because {miss} does not match any pattern in {tracer_patterns}")
                s.remove_selection(tracers=(t1,t2))

        # The ones we actually used.
        self.used_names = s.get_data_types()

        # Check for scale cuts. In general, this is a minimum and maximum angle for
        # each spectrum, for each redshift bin combination. Which is clearly a massive pain...
        # but what can you do?

        valid_tags = ["ell", "theta", "n", "mass", "luminosity"]
        # We only support these tags for now, but we could add more
        for name in self.used_names:
            for tracers in s.get_tracer_combinations(name):
                if len(tracers) == 2:
                    t1, t2 = tracers
                    option_name = f"angle_range_{name}_{t1}_{t2}"
                else:
                    t1 = tracers[0]
                    option_name = f"onepoint_range_{name}_{t1}"
                    # also allow the invalid "angle_range" form
                    invalid_name = f"angle_range_{name}_{t1}"

                if self.options.has_value(option_name) or (len(tracers) == 1 and self.options.has_value(invalid_name)):
                    # if invalid_name is used, warn and substitute
                    if len(tracers) == 1 and self.options.has_value(invalid_name):
                        print(f"**** WARNING: '{invalid_name}' is not a valid option name. "
                              f"Use '{option_name}' instead.")
                        option_name = invalid_name
                    r = self.options.get_double_array_1d(option_name)
                    tags = np.unique([k for row in s.get_data_points(name) for k in row.tags])
                    for tag in valid_tags:
                        if tag in tags:
                            # Determine the tracer tuple based on the tag and data type
                            tracer_tuple = (t1, t2) if tag in ["ell", "theta", "n"] else (t1,)
                            # Create keyword arguments for lt and gt
                            kwargs_lt = {f"{tag}__lt": r[0]}
                            kwargs_gt = {f"{tag}__gt": r[1]}
                            # Call the remove_selection method
                            s.remove_selection(name, tracer_tuple, **kwargs_lt)
                            s.remove_selection(name, tracer_tuple, **kwargs_gt)

        for name in self.used_names:
            option_name = f"cut_{name}"
            if self.options.has_value(option_name):
                cuts = self.options[option_name].split()
                for cut in cuts:
                    tracers = cut.split(",")
                    s.remove_selection(name, tracers)

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
                print(f"    - {name}  {data_points} data points after cuts   [using in likelihood]")
                total_data_points += data_points
            else:
                print(f"    - {name}  {data_points} data points after cuts   [not using in likelihood]")
        print(f"Total data points used = {total_data_points}")

        self.sections_for_names = {}
        for name in final_names:
            if self.options.has_value(f"{name}_section"):
                section = self.options[f"{name}_section"]
            elif name in default_sections:
                section = default_sections[name][1]
            else:
                raise ValueError(f"You need to specify {name}_section in the ini file as the data type {name} is not known.")
            print(f"Will look for theory prediction for data set {name} in section {section}")
            if self.options.has_value(f"{name}_category"):
                category = self.options[f"{name}_category"]
            elif name in default_sections:
                category = default_sections[name][0]
            else:
                raise ValueError(f"You need to specify {name}_category in the ini file as the data type {name} is not known.")
            self.sections_for_names[name] = (category, section)


        # build up the data vector from all the separate vectors.
        # Just concatenation
        data_vector = s.get_mean()

        # Make sure
        if len(data_vector) == 0:
            raise ValueError(
                f"No data was chosen to be used from 2-point data file {filename}. It was either not selectedin data_sets or cut out")

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
            print(f"You set covariance_realizations={r} in the 2pt likelihood parameter file")
            print("So I will apply the Anderson-Hartlap correction to the covariance matrix")
            print(f"The covariance matrix is nxn = {p}x{p}")
            print(f"So the correction scales the covariance matrix by (r - 1) / (r - n - 2) = {x}")
            print()
        return C

    def extract_theory_points(self, block):
        theory = []
        dataset_names = []

        # We have a collection of data vectors, one for each spectrum
        # that we include. We concatenate them all into one long vector,
        # so we do the same for our theory data so that they match


        # Now we actually loop through our data sets
        for data_type in self.sacc_data.get_data_types():
            category, section = self.sections_for_names[data_type]
            if "one_point" in category:
                extract_prediction = sacc_likelihoods.onepoint
                extract_kwargs = {}
            else:
                extract_prediction = getattr(sacc_likelihoods, self.sacc_like, None)
                if extract_prediction is None:
                    raise ValueError(f"2pt likelihood requires the {self.sacc_like} method to be defined")
                extract_kwargs = {"flip": self.flip, "options": self.options}

            theory_vector, metadata_vectors = extract_prediction(
                self.sacc_data, block, data_type, section, category=category, **extract_kwargs
            )

            # We also save metadata vectors such as the bin indices
            # and angles, so that we can use them in plotting etc.
            for name, vector in metadata_vectors.items():
                block[names.data_vector, f"{data_type}_{name}"] = vector

            theory.append(theory_vector)
            dataset_names.append(np.repeat(data_type, len(theory_vector)))


        dataset_names = np.concatenate(dataset_names)
        block[names.data_vector, self.like_name+"_name"] = dataset_names

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

setup, execute, cleanup = SaccClLikelihood.build_module()
