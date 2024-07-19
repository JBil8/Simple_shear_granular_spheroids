import os
import pickle
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter, MaxNLocator

formatter = FuncFormatter(lambda x, pos: f'{x:.2f}')


# Define the parameters
aspect_ratios = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
cofs = [0.0, 0.4, 1.0]
Is = [0.0316, 0.01, 0.00316, 0.001,0.000316, 0.0001]

# Define the keys of interest for subplots
keys_of_interest = ['phi', 'mu', 'S2', 'Omega_z']
additional_keys = ['p_xy', 'p_yy']

# Initialize a data holder
data = {key: [] for key in keys_of_interest + additional_keys}
data['inertialNumber'] = []
data['cof'] = []
data['ap'] = []

os.chdir("./output_data_stress_updated")

# Load the data from files
for ap in aspect_ratios:
    for cof in cofs:
        for I in Is:
            filename = f'simple_shear_ap{ap}_cof_{cof}_I_{I}.pkl'
            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    file_data = pickle.load(file)
                    inertial_number = file_data.get('inertialNumber', None)
                    if inertial_number is not None:
                        data['inertialNumber'].append(inertial_number)
                        data['cof'].append(cof)
                        data['ap'].append(ap)
                        for key in keys_of_interest + additional_keys:
                            data[key].append(file_data.get(key, None))
            else:
                print(f"File {filename} does not exist.")

# Ensure the output directory exists
output_dir = "../parametric_plots_stress_updated"
os.makedirs(output_dir, exist_ok=True)

# Function to create subplots
def create_subplots(data):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create a colormap
    colormap = plt.cm.plasma
    num_colors = len(aspect_ratios)
    colors = [colormap(i / num_colors) for i in range(num_colors)]

    fig, axs = plt.subplots(4, 3, figsize=(15, 20), sharex='col')
    fig.subplots_adjust(right=0.85)
    labels = {
        'phi': '$\phi$',
        'mu': '$\\mu = \sigma_{xy} / \sigma_{yy}$',
        'S2': '$S_2$',
        'Omega_z': '$2\\langle \omega_z \\rangle / \dot{\gamma}$'
    }
    for ax in axs.flat:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=4))

    for row, key in enumerate(keys_of_interest):
        for col, cof in enumerate(cofs):
            ax = axs[row, col]
            for ap, color in zip(aspect_ratios, colors):
                x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                y_values = [value for value, aspect_ratio, coef in zip(data[key], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]

                if key == 'mu':
                    p_xy_values = [value for value, aspect_ratio, coef in zip(data['p_xy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                    p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                    y_values = [pxy / pyy if pyy != 0 else None for pxy, pyy in zip(p_xy_values, p_yy_values)]

                if x_values and y_values:  # Ensure that the lists are not empty
                    ax.plot(x_values, y_values, label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')

            ax.set_xscale('log')
            if row == 3:
                ax.set_xlabel('$I$', fontsize=25)
            if col == 0:
                ax.set_ylabel(labels[key], fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=18)
            if col == 2 and row == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)

    plt.savefig(os.path.join(output_dir, 'combined_plot.png'), bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

# Create the subplots
create_subplots(data)
os.chdir("..")

print("Combined plot saved to parametric_plots")
