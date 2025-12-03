from cosmosis import option_section
import numpy as np
def setup(options):
    f = options.get_double(option_section, "fraction", 0.1)
    seed = options.get_int(option_section, "seed", 6789)
    verbose = options.get_bool(option_section, "verbose", False)
    rng = np.random.RandomState(seed)
    return {"fraction": f, "rng": rng, "verbose": verbose}

def execute(block, config):
    if config["rng"].rand() < config["fraction"]:
        if config["verbose"]:
            print("Randomly failing")
        return 1
    return 0
