import numpy as np
from cosmosis.datablock import names, option_section
import sacc

def setup(options):
    nz_file = options.get_string(option_section, "nz_file")
    data_sets = options.get_string(option_section, "data_sets")
    prefix_section = options.get_bool(option_section, "prefix_section", True)
    data_sets = data_sets.split()
    if not data_sets:
        raise RuntimeError(
            "Option data_sets empty; please set the option data_sets=name1 name2 etc and I will search the sacc file for those tracers")

    print("Loading number density data from {0}:".format(nz_file))

    s = sacc.Sacc.load_fits(nz_file)

    data = {}

    for tracer in s.tracers.values():
        # Check if we want to use this tracer
        for d in data_sets:
            if tracer.name.startswith(d):
                data_set = d
                break
        else:
            continue

        if data_set not in data:
            data[data_set] = {}
        index = int(tracer.name.split("_")[-1])
        z = tracer.z
        nz = tracer.nz
        data[data_set][index] = (z, nz)

    output = {}
    # CosmoSIS is currently expecting all the bins to have the
    # same z grid. Changing this is possible but would require
    # some changes later in the pipeline, so we check here.
    # We also check for a few other possible issues, and slightly
    # reorder the data. This isn't critical, we could do this better.
    for data_set in data_sets:
        if data_set not in data:
            raise ValueError(f"n(z) data called {data_set} not found in file {nz_file}")
        d = data[data_set]
        n = len(data[data_set])
        z = None
        nzs = {}
        for i in range(n):
            if i not in d:
                raise ValueError(f"n(z) data in {data_set} not contiguous bins in file {nz_file}")
            zi, nz = d[i]
            if (z is not None) and (not np.allclose(zi, z)):
                raise ValueError(f"z values different for different bins in {nz_file}")
            z = zi
            nzs[i] = nz
        name = f"nz_{data_set}" if prefix_section else data_set
        output[name] = (z, nzs)

    return output


def execute(block, config):
    data_sets = config
    for name, (z, nz) in list(data_sets.items()):
        nbin = len(nz)
        ns = len(z)
        block[name, "nbin"] = nbin
        block[name, "nz"] = ns
        block[name, "z"] = z
        for i, n in nz.items():
            block[name, "bin_{0}".format(i + 1)] = n
    return 0
