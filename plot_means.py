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
parser.add_argument('-c', '--cof', type=float, help='coefficient of friction')
parser.add_argument('-t', '--type', type=str, help='simulation type: either I or phi')

args = parser.parse_args()

#parsing command line arguments
cof = args.cof
simulation_type = args.type
aspect_ratios = [1.0, 1.5, 2.0, 2.5, 3.0]
aspect_ratios = [1.5, 2.0, 2.5, 3.0]


if simulation_type == "I":
    values = [0.1, 0.0398, 0.0158, 0.0063, 0.0025, 0.001]
    #values = [0.1, 0.0398, 0.0158, 0.0063, 0.0025]
    n_plots = int(8)    
    ylabels = ['$Z$', '$\omega_z$ [rad/s]', '$\\theta_x$', '$\%$ aligned', '$S_2$', '$\phi$', '$\mu_{eff}$', '$D_y$ [m$^2$]']
    key_list = ['Z', 'omega_z', 'theta_x', 'percent_aligned', 'S2', 'phi', 'mu_effective', 'msd']
    simulation_name = '$I$'

elif simulation_type == "phi":
    #values = [0.5, 0.6, 0.7, 0.8, 0.9]
    values = [0.5, 0.6, 0.7]
    n_plots = int(6)
    ylabels = ['$Z$', '$\\theta_x$', '$\%$ aligned', '$S_2$', '$\mu_{eff}$', '$D_y$ [m$^2$]']
    key_list = ['Z', 'theta_x', 'percent_aligned', 'S2', 'mu_effective', 'msd']
    simulation_name = '$\phi$'

# loop over the values of the parameter

avg_dict = {'Z': np.zeros((len(aspect_ratios), len(values))),
            'omega_z': np.zeros((len(aspect_ratios), len(values))),
            'theta_x': np.zeros((len(aspect_ratios), len(values))),
            'percent_aligned': np.zeros((len(aspect_ratios), len(values))),
            'S2': np.zeros((len(aspect_ratios), len(values))),
            'phi': np.zeros((len(aspect_ratios), len(values))),
            'autocorrelation_v': np.zeros((len(aspect_ratios), len(values))),
            'mu_effective': np.zeros((len(aspect_ratios), len(values))),
            'msd': np.zeros((len(aspect_ratios), len(values))), 
            'eulerian_vx': np.zeros((10, len(aspect_ratios), len(values))), 
            'vel_fluct': np.zeros((10, len(aspect_ratios), len(values)))}
plt.ioff()

# Data averaging

for j, value in enumerate(values):
    # loop over the aspect ratios
    for i, ap in enumerate(aspect_ratios):
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
        #box_height = data_vtk['box_height']
        # Perform the curve fit
        slope = np.polyfit(strain, msd, 1)[0]/2 # divide by 2 to get the diffusion coefficient in 1D

        # average the data over the last 50% of the simulation (after strain = 10)
        index_half = int(data_dump['Z'].shape[0]/2)
        avg_dict['Z'][i,  j] = (np.mean(data_dump['Z'][index_half:]))
        avg_dict['omega_z'][i,  j] = (np.mean(data_vtk['omega_z'][index_half:]))
        avg_dict['theta_x'][i,  j] = np.rad2deg(np.mean(data_vtk['theta_x'][index_half:]))
        avg_dict['percent_aligned'][i,  j] = 100*(np.mean(data_vtk['percent_aligned'][index_half:]))
        avg_dict['S2'][i,  j] = (np.mean(data_vtk['S2'][index_half:]))
        avg_dict['phi'][i,  j] = (np.mean(data_vtk['phi'][index_half:]))
        avg_dict['autocorrelation_v'][i,  j] = (np.mean(data_vtk['autocorrelation_v'][index_half:]))
        avg_dict['mu_effective'][i,  j] = (np.mean(data_vtk['mu_effective'][index_half:]))
        avg_dict['msd'][i,  j] = (slope)
        eulerian_vx = (np.mean(data_vtk['eulerian_vx'][index_half:, :], axis=0))
        avg_dict['eulerian_vx'][:, i,  j] = eulerian_vx/v_shearing

        avg_dict['vel_fluct'][:, i,  j] = np.sqrt(np.mean(
            (data_vtk['eulerian_vx'][index_half:, :]-eulerian_vx)**2, axis=0))/v_shearing


font_properties = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=16)

# Plot the data for all values of the parameter on the same plot

fig, axes = plt.subplots(int(n_plots/2), 2, figsize=(8, 10), sharex='col', gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, constrained_layout=True)
cmap = plt.get_cmap('viridis', len(aspect_ratios))

# Flatten the 2D array of subplots into a 1D array
axes = axes.flatten()

# Iterate over properties and plot on each subplot
for property_index, property_name in enumerate(key_list):
    ax = axes[property_index]  # Get the current subplot

    # Iterate over aspect ratios and plot on the same subplot
    for i, ap in enumerate(aspect_ratios):
        ax.semilogx(values, avg_dict[property_name][i, :], label=f'$\\alpha$= {ap}', color=cmap(i), linewidth=2)

    if property_name == 'msd':
        # Set scientific notation for y-axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0e}'))
        ax.text(-0.32, 0.5, ylabels[property_index], rotation="vertical", va="center", ha="center", transform=ax.transAxes, fontproperties=font_properties)
    else:
        # Use text function to add LaTeX-formatted y-axis label
        ax.text(-0.27, 0.5, ylabels[property_index], rotation="vertical", va="center", ha="center", transform=ax.transAxes, fontproperties=font_properties)

    # Check if it's the bottom plot and adjust x-axis settings
    # if property_index != len(key_list) - 1 and property_index != len(key_list) - 2:
    #     ax.xaxis.set_major_formatter(NullFormatter())

    if simulation_type == "I":
        ax.set_xlabel('$I$')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0e}'))
    elif simulation_type == "phi":
        ax.set_xlabel('$\phi$')
# # Set xlabel only on the bottom plot of each column
# for ax in axes[-2:]:
#     if simulation_type == "I":
#         ax.set_xlabel('$I$')
#     elif simulation_type == "phi":
#         ax.set_xlabel('$\phi$')
        
# Customize the plot (labels, legend, etc.)
axes[1].legend(fontsize=16)

for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

fig.suptitle('$\mu_p$ = ' + str(cof), fontproperties=font_properties, fontsize=20)
fig.savefig('output_plots/parametric_plots/simple_shear_ap_cof_' + str(cof) + '_' + simulation_type + '_all_values.png')


# plot average eulerian velocity
y = np.linspace(0, 1, 10)

# Plot the data for all values of the parameter on the same plot
fig, axes = plt.subplots(2, len(values), figsize=(12, 10), sharey='row', gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, constrained_layout=True)
cmap = plt.get_cmap('viridis', len(aspect_ratios))

# Flatten the 2D array of subplots into a 1D array
axes = axes.flatten()

for j, value in enumerate(values):
    # Iterate over aspect ratios and plot on the same subplot
    for i, ap in enumerate(aspect_ratios):
        axes[j].plot(avg_dict['eulerian_vx'][:, i, j], y, label=f'$\\alpha$ = {ap}', color=cmap(i), linewidth=2)
        axes[j].set_xlabel('$<V_x>/V_w$')
        axes[j+len(values)].plot(avg_dict['vel_fluct'][:, i, j], y, label=f'$\alpha$ = {ap}', color=cmap(i), linewidth=2)
        axes[j+len(values)].set_xlabel('$V_{rms}\'/V_w$')
        
    axes[j].set_title(f'{simulation_name} = {value}')

axes[0].set_ylabel('$y/H$')
axes[len(values)].set_ylabel('$y/H$')
axes[0].legend(fontsize=16)
fig.suptitle('$\mu_p$ = ' + str(cof), fontproperties=font_properties, fontsize=20)
# increase the font size
for ax in axes:
    ax.tick_params(axis='x', rotation=45)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

#plt.show()
fig.savefig('output_plots/parametric_plots/simple_shear_cof_' + str(cof) + '_' + simulation_type + '_eulerian_statistics.png')