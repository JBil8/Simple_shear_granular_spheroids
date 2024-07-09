import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import argparse
import os
from DataExporter import DataExporter
from matplotlib.ticker import NullFormatter, ScalarFormatter, StrMethodFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec

parser = argparse.ArgumentParser(description='Process granular simulation.')
parser.add_argument('-a', '--aspect_ratio', type=float, help='aspect ratio')
parser.add_argument('-t', '--type', type=str, help='simulation type: either I or phi')

args = parser.parse_args()

#parsing command line arguments
cofs = [0.4]
simulation_type = args.type
ap = args.aspect_ratio
#aspect_ratios = [1.5]

if simulation_type == "I":
    values = [0.1, 0.0398, 0.0158, 0.0063, 0.0025, 0.001]
    #values = [0.1, 0.0398, 0.0158, 0.0063, 0.0025]
    n_plots = int(8)    
    ylabels = ['$D_y$ [m$^2$]']
    key_list = ['msd']
    simulation_name = '$I$'

elif simulation_type == "phi":
    #values = [0.5, 0.6, 0.7, 0.8, 0.9]
    values = [0.5, 0.6, 0.7]
    n_plots = int(6)
    ylabels = ['$Z$', '$\\theta_x$', '$\%$ aligned', '$S_2$', '$\mu_{eff}$', '$D_y$ [m$^2$]']
    key_list = ['Z', 'theta_x', 'percent_aligned', 'S2', 'mu_effective', 'msd']
    simulation_name = '$\phi$'

# loop over the values of the parameter

avg_dict = {'msd': np.zeros((len(cofs), len(values)))}

font_properties = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=20)
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
cmap = plt.get_cmap('cividis', len(values)+1)

for j, value in enumerate(values):
    # loop over the aspect ratios
    for i, cof in enumerate(cofs):
    # open pickel file from output_data
        with open('output_data2/simple_shear_ap' + str(ap) + '_cof_' + str(cof) + '_' + simulation_type + '_' + str(value) + '.pkl', 'rb') as f:
            data_vtk = pkl.load(f)   

        with open('output_data2/simple_shear_ap' + str(ap) + '_cof_' + str(cof) + '_' + simulation_type + '_' + str(value) + '_dump.pkl', 'rb') as f:
            data_dump = pkl.load(f)         


        msd = data_vtk['msd']
        indx = np.where(msd == 0.0)[0][0]
        msd = msd[indx:]
        strain = np.arange(0, msd.shape[0])*18/msd.shape[0]

        v_shearing = data_vtk['v_shearing']

        #plot the msd gainst the strain for all values of the parameter
        ax.plot(strain, msd, label=f'$I$= {value},', color=cmap(j), linewidth=2)

ax.set_xlabel('$\dot{\gamma} t$', fontsize=18)
ax.set_ylabel('$\\langle y^2 \\rangle$ [m$^2$]', fontsize=18)
ax.set_title('$\\mu_p$ = ' + str(cof) + ', $\\alpha$ = ' + str(ap), fontsize=18)
ax.legend(fontsize=16)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(16)
# show the plot
plt.show()

        # average the data over the last 50% of the simulation (after strain = 10)
a= b       
# avg_dict['eulerian_vx'].shape == (10, 5, 6)

font_properties = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=16)

# Plot the data for all values of the parameter on the same plot
fig, axes = plt.subplots(int(n_plots/2), 2, figsize=(8, 10), sharex='col', gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, constrained_layout=True)
cmap = plt.get_cmap('plasma', len(cofs)+1)

# Flatten the 2D array of subplots into a 1D array
axes = axes.flatten()

# Iterate over properties and plot on each subplot
for property_index, property_name in enumerate(key_list):
    ax = axes[property_index]  # Get the current subplot

    # Iterate over aspect ratios and plot on the same subplot
    for i, cof in enumerate(cofs):
        ax.semilogx(values, avg_dict[property_name][i, :], label=f'$\\mu_p$= {cof}', color=cmap(i), linewidth=2)

    if property_name == 'msd':
        # Set scientific notation for y-axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0e}'))
        ax.text(-0.32, 0.5, ylabels[property_index], rotation="vertical", va="center", ha="center", transform=ax.transAxes, fontproperties=font_properties)
    else:
        # Use text function to add LaTeX-formatted y-axis label
        ax.text(-0.27, 0.5, ylabels[property_index], rotation="vertical", va="center", ha="center", transform=ax.transAxes, fontproperties=font_properties)

        

    # Hide x-axis values for some subplots
    if property_index != len(key_list) - 1 and property_index != len(key_list) - 2:
        ax.xaxis.set_major_formatter(NullFormatter())

# Set xlabel only on the bottom plot of each column
for ax in axes[-2:]:
    if simulation_type == "I":
        ax.set_xlabel('$I$')
    elif simulation_type == "phi":
        ax.set_xlabel('$\phi$')
# Customize the plot (labels, legend, etc.)
axes[1].legend(fontsize=16)


for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

#plt.show()
fig.suptitle('$\\alpha$ = ' + str(ap), fontsize=20)
fig.savefig('output_plots/parametric_plots/simple_shear_ap' + str(ap) + '_' + simulation_type + '_all_cofs.png')
