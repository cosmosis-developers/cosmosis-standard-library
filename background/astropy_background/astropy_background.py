from cosmosis.datablock import option_section, names
import astropy.cosmology
import numpy as np

#Cosmologies currently in astropy.cosmology


def setup(options):
	zmax = options[option_section, "zmax"]
	nz = options[option_section, "nz"]
	model = options[option_section, "model"]

	# Models available in astropy.cosmology
	astropy_models = {
		"flatlambdacdm": astropy.cosmology.FlatLambdaCDM,
		"flatw0wacdm": astropy.cosmology.Flatw0waCDM,
		"flatwcdm": astropy.cosmology.FlatwCDM,
		"lambdacdm": astropy.cosmology.LambdaCDM,
		"w0wacdm": astropy.cosmology.w0waCDM,
		"w0wzcdm": astropy.cosmology.w0wzCDM,
		"wcdm": astropy.cosmology.wCDM,
		"wpwacdm": astropy.cosmology.wpwaCDM,
	}

	# Finf the model the user has specified
	model_class = astropy_models.get(model.lower())
	if model_class is None:
		raise ValueError("Unknown astropy model {}".format(model))

	return {"zmax":zmax, "nz":nz, "model_class":model_class}



def set_params(block, model_class):
	# These pairs are the astropy parameter name and the cosmosis param name
	params_needed_by_class = {  
		astropy.cosmology.FlatLambdaCDM: [('H0', 'hubble'), ('Om0', 'omega_m')],


		astropy.cosmology.FlatwCDM:      [('H0', 'hubble'), ('Om0', 'omega_m'), 
										  ('w0', 'w')],

		astropy.cosmology.Flatw0waCDM:   [('H0', 'hubble'), ('Om0', 'omega_m'), 
										  ('w0', 'w'), ('wa', 'wa')],

		astropy.cosmology.LambdaCDM:     [('H0', 'hubble'), ('Om0', 'omega_m'), ('Ode0', 'omega_lambda')],


		astropy.cosmology.wCDM:          [('H0', 'hubble'), ('Om0', 'omega_m'), ('Ode0', 'omega_lambda'),
										  ('w0', 'w')],

		astropy.cosmology.w0waCDM:       [('H0', 'hubble'), ('Om0', 'omega_m'), ('Ode0', 'omega_lambda'), 
										  ('w0', 'w'), ('wa', 'wa')],

		astropy.cosmology.w0wzCDM:       [('H0', 'hubble'), ('Om0', 'omega_m'), ('Ode0', 'omega_lambda'),
										  ('w0', 'w'), ('wz', 'wz')],

		astropy.cosmology.w0wzCDM:       [('H0', 'hubble'), ('Om0', 'omega_m'), ('Ode0', 'omega_lambda'),
										  ('w0', 'w'), ('wz', 'wz')],

		astropy.cosmology.wpwaCDM:       [('H0', 'hubble'), ('Om0', 'omega_m'), ('Ode0', 'omega_lambda'), 
										  ('wp', 'wp'),('wa', 'wa'), ('zp', 'zp')],

	}

	params_needed = params_needed_by_class[model_class]

	# Pull the parameters we need out from cosmosis
	params = {}
	for astropy_name, cosmosis_name in params_needed:
		params[astropy_name] = block[names.cosmological_parameters, cosmosis_name]
	
	# Create the astropy object that does the calculations
	model = model_class(**params)

	return model


def execute(block, config):
	zmax = config["zmax"]
	nz = config["nz"]

	#Create our cosmological model
	model_class = config["model_class"]
	model = set_params(block, model_class)

	#Calculate distances
	z = np.linspace(0.0, zmax, nz)
	a = 1/(1.+z)

	# comoving radial
	D_C = model.comoving_distance(z).to_value(astropy.units.Mpc)
	try:
		Ok0 = model._Ok0
	except AttributeError:
		Ok0 = model.Ok0

	# comoving transverse
	if Ok0 == 0:
		D_M = D_C
	else:
		sqrtOk0 = np.sqrt(abs(Ok0))
		dh = model._hubble_distance.value
		if Ok0 > 0:
			D_M = dh / sqrtOk0 * np.sinh(sqrtOk0 * D_C / dh)
		else:
			D_M = dh / sqrtOk0 * np.sin(sqrtOk0 * D_C / dh)

	D_L = D_M * (1 + z)
	D_A = D_M / (1 + z)
	mu = np.zeros(z.size)
	mu[0] = -np.inf
	mu[1:] = 5.0 * np.log10(D_L[1:]) + 25.0	

	H_z = (model.H(z) / astropy.constants.c).to_value(1/astropy.units.Mpc)
	D_V = ((1 + z)**2 * z * D_A**2 / H_z )**(1./3.)


	#Save results
	block[names.distances, "z"] = z
	block[names.distances, "a"] = a
	block[names.distances, "mu"] = mu
	block[names.distances, "D_L"] = D_L
	block[names.distances, "D_A"] = D_A
	block[names.distances, "D_V"] = D_V
	block[names.distances, "D_M"] = D_M
	block[names.distances, "D_C"] = D_C
	block[names.distances, "H"] = H_z

	# Save the units in the metadata as well
	for name in ("D_L", "D_A", "D_M", "D_C", "D_V"):
		block.put_metadata(names.distances, name, "unit", "Mpc")
	block.put_metadata(names.distances, "H", "unit", "1.0/Mpc")

	return 0