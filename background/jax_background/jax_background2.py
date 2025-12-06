from cosmosis.datablock import option_section, names
import jax_cosmo
import jax.numpy as jnp
import jax

cosmosis_jax = True

def setup(options):
    zmax = options.get_double(option_section, "zmax", default=3.0)
    nz = options.get_int(option_section, "nz", default=301)
    return {
        "zmax": zmax,
        "nz": nz,
    }

def execute(block, config):
    omega_m = block.get_double("cosmological_parameters", "omega_m")
    omega_k = block.get_double("cosmological_parameters", "omega_k")
    w = block.get_double("cosmological_parameters", "w")
    h0 = block.get_double("cosmological_parameters", "h0")
    omega_b = 0.044
    omega_c = omega_m - omega_b
    cosmo = jax_cosmo.Cosmology(Omega_c=omega_c, Omega_b=omega_b, h=h0, n_s=0.96, sigma8=0.8, Omega_k=omega_k, w0=w, wa=0.0)
    dz = config["zmax"] / (config["nz"] - 1)
    z = jnp.linspace(dz, config["zmax"], config["nz"])
    a = 1 / (1 + z)
    d_a = jax_cosmo.background.angular_diameter_distance(cosmo, a) / h0
    d_m = jax_cosmo.background.transverse_comoving_distance(cosmo, a) / h0
    d_l = (1 + z)**2 * d_a
    mu = 5 * jnp.log10(d_l) + 25
    # return {
    block["distances", "z"] = z
    block["distances", "a"] = a
    block["distances", "mu"] = mu
    block["distances", "d_a"] = d_a
    block["distances", "d_m"] = d_m
    block["distances", "d_l"] = d_l
    return 0
    # could change the return value to "block" maybe?

