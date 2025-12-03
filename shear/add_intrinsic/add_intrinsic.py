from __future__ import print_function
from builtins import range
from cosmosis.datablock import option_section, names

def setup(options):
    do_shear_shear = options.get_bool(option_section, "shear-shear", True)
    do_position_shear = options.get_bool(
        option_section, "position-shear", True)
    do_shear_cmbkappa=options.get_bool(option_section,"shear-cmbkappa",False)
    perbin = options.get_bool(option_section, "perbin", False)

    suffix = options.get_string(option_section, "suffix", "")

    print()
    print("The add_intrinsic module will try to combine IA terms into these spectra:")
    if do_shear_shear:
        print(" - shear-shear.")
    if do_position_shear:
        print(" - position-shear.")
    if do_shear_cmbkappa:
        print(" - shear-CMB kappa ")
    if not (do_shear_cmbkappa or do_position_shear or do_shear_shear):
        print("... actually not into anything. You set shear-shear=F and position-shear=F  shear-cmbkappa=F")
        print("Ths module will not do anything in this configuration")
    print()

    if suffix:
        suffix = "_" + suffix

    sec_names = {
        "shear_shear": "shear_cl" + suffix,
        "shear_shear_bb": "shear_cl_bb" + suffix,
        "shear_shear_gg": "shear_cl_gg" + suffix,
        "galaxy_shear": "galaxy_shear_cl" + suffix,
        "shear_intrinsic": "shear_cl_gi" + suffix,
        "galaxy_intrinsic": "galaxy_intrinsic_cl"  + suffix,
        "intrinsic_intrinsic": "shear_cl_ii"  + suffix,
        "intrinsic_intrinsic_bb": "shear_cl_ii_bb"  + suffix,
        "parameters": "intrinsic_alignment_parameters" + suffix,
        "shear_cmbkappa": "shear_cmbkappa_cl" + suffix,
        "intrinsic_cmbkappa": "intrinsic_cmbkappa_cl" + suffix,
    }   

    return do_shear_shear, do_position_shear, do_shear_cmbkappa, perbin, sec_names


def execute(block, config):
    do_shear_shear, do_position_shear, do_shear_cmbkappa, perbin, sec_names = config

    shear_shear = sec_names['shear_shear']
    shear_shear_bb = sec_names['shear_shear_bb']
    shear_shear_gg = sec_names['shear_shear_gg']
    galaxy_shear = sec_names['galaxy_shear']
    galaxy_intrinsic = sec_names['galaxy_intrinsic']
    shear_intrinsic = sec_names['shear_intrinsic']
    parameters = sec_names['parameters']
    intrinsic_intrinsic = sec_names['intrinsic_intrinsic']
    intrinsic_intrinsic_bb = sec_names['intrinsic_intrinsic_bb']
    shear_cmbkappa = sec_names['shear_cmbkappa']
    intrinsic_cmbkappa = sec_names['intrinsic_cmbkappa']

    if do_shear_shear:
        nbin_shear = block[shear_shear, 'nbin']
    elif do_position_shear:
        nbin_shear = block[galaxy_intrinsic, 'nbin_b']
    elif do_shear_cmbkappa:
        nbin_shear = block[shear_cmbkappa, 'nbin_a']
    if do_position_shear:
        nbin_pos = block[galaxy_shear, 'nbin_a']

    if perbin:
        A = [block[parameters, "A1_{}".format(i + 1)]
             for i in range(nbin_shear)]
    else:
        A = [1 for i in range(nbin_shear)]

    if do_shear_shear:
        # for shear-shear, we're replacing 'shear_cl' (the GG term) with GG+GI+II...
        # so in case useful, save the GG term to shear_cl_gg.
        # also check for a b-mode contribution from IAs
        block[shear_shear_gg, 'ell'] = block[shear_shear, 'ell']

        # Add metadata to the backup gg-only section
        for key in ["nbin_a", "nbin_b", "nbin", "sample_a", "sample_b", "is_auto", "auto_only", "sep_name"]:
            if block.has_value(shear_shear, key):
                block[shear_shear_gg, key] = block[shear_shear, key]

        for i in range(nbin_shear):
            for j in range(i + 1):
                bin_ij = 'bin_{0}_{1}'.format(i + 1, j + 1)
                bin_ji = 'bin_{1}_{0}'.format(i + 1, j + 1)
                block[shear_shear_gg, bin_ij] = block[shear_shear, bin_ij]
                block[shear_shear, bin_ij] += (
                    A[i] * A[j] * block[intrinsic_intrinsic, bin_ij]  # II
                    + A[j] * block[shear_intrinsic, bin_ij]  # The two GI terms
                    + A[i] * block[shear_intrinsic, bin_ji]
                )
                if block.has_section(intrinsic_intrinsic_bb):
                    block[shear_shear_bb, bin_ij] = block[intrinsic_intrinsic_bb, bin_ij]
    if do_position_shear:
        for i in range(nbin_pos):
            for j in range(nbin_shear):
                bin_ij = 'bin_{0}_{1}'.format(i + 1, j + 1)
                block[galaxy_shear, bin_ij] += A[j] * \
                    block[galaxy_intrinsic, bin_ij]

    if do_shear_cmbkappa:
        for i in range(nbin_shear):
            bin_ij = 'bin_{0}_{1}'.format(i + 1, 1)
            block[shear_cmbkappa, bin_ij] += A[i] * \
                    block[intrinsic_cmbkappa, bin_ij]


    return 0
