import cosmosis.campaign
from cosmosis.postprocessing.v2 import Chain
import getdist.plots
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# The campaign file defines all the chains that we want to generate
campaign, _ = cosmosis.campaign.parse_yaml_run_file("./examples/desi-figure-2-campaign.yaml")


# First make all the chains, by looping through the campaign.
# The next major version of CosmoSIS will do this for you more easily.
for name, run in campaign.items():
    if name != "base":
        cosmosis.campaign.launch_run(run)

# These colors match those used in the DESI paper as
# best I could
colors = {
    'BGS': [(0.729, 0.811, 0.552), (0.517, 0.745, 0.012)],
    'LRG1': [(0.996, 0.854, 0.596), (0.956, 0.627, 0.086)],
    'LRG2': [(0.960, 0.678, 0.580), (0.929, 0.262, 0.094)],
    'LRG3+ELG1': [(0.733, 0.603, 0.741), (0.341, 0.211, 0.701)],
    'ELG2': [(0.584, 0.725, 0.772), (0.219, 0.447, 0.6)],
    'Lya': [(0.733, 0.556, 0.749), (0.439, 0.0, 0.474)],
    'QSO': [(0.662, 0.815, 0.725), (0.247, 0.552, 0.360)],
    'All': [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
 }


# This is the order we want the contours to be plotted in. It's not the same
# as the legend order. This is because we don't want the big contours to 
# cover up the small ones
order = ["BGS", "LRG1", "QSO", "ELG2", "Lya", "LRG3+ELG1", "LRG2", "All"]


# The chains are complete now so load them all in
chain_data = {}
for i, (name, run) in enumerate(campaign.items()):
    # The base parent run it not used, it's just a template.
    # We will soon be able to specify this in the yaml file
    if name == "base" :
        continue

    # Load the chain using a chain object. This is in the upcoming
    # v2 api for cosmosis postprocessing, which will soon replace the
    # main version. It uses GetDist to do everything.
    chain_file = campaign[name]['params']['output', 'filename']
    chain = Chain.load(chain_file, name=name, burn=0.2)

    # Store the GetDist MCSamples object in a dictionary
    chain_data[name] = chain.mcsamples


# split the chains into two groups, the single-data set chains
# and the combined one. This is because the contours are all filled
# for the single data sets, but not for the combined one.
combined_sample = chain_data["All"]
single_data_samples = {name: chain_data[name] for name in order if name != "All"}

# This is the order we want the labels to appear in. 
new_order = [order.index(name) for name in colors.keys()]

# The rest of this is standard GetDist calls.
plotter = getdist.plots.get_single_plotter()

xmin = 87.
xmax = 115.0
ymin = 0.18
ymax = 0.55
lims = [xmin, xmax, ymin, ymax]

plotter.plot_2d(list(single_data_samples.values()),
                "DISTANCES--H0RD", 
                "cosmological_parameters--omega_m", 
                filled=True, 
                colors=[colors[name] for name in single_data_samples.keys()],
                add_legend_proxy=True,
                lims=lims,)

plotter.plot_2d(combined_sample,
                "DISTANCES--H0RD",
                "cosmological_parameters--omega_m", 
                filled=False, 
                colors=["k", "k"],
                lims=lims,
                add_legend_proxy=True)


ax=plotter.get_axes()
plotter.add_legend(list(single_data_samples.keys()) + ["All"], label_order=new_order, legend_loc="upper right")
ax.set_xlabel(r"$\mathrm{H}_0 r_d \, [100 \mathrm{km}\, \mathrm{s}^{-1}]$")
ax.set_ylabel(r"$\Omega_m$")
plotter.export("output/desi.pdf")
