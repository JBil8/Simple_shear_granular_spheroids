import os
import pickle
import matplotlib.pyplot as plt

# Define the parameters
#aspect_ratios = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
cofs = [0.0, 0.4, 1.0]
Is = [0.0316, 0.01, 0.00316, 0.001,0.000316, 0.0001]
#aspect_ratios = [0.33, 0.40, 0.50, 0.56, 0.67, 0.83, 1.0]
#aspect_ratios = [0.83]
aspect_ratios = [0.33, 0.40, 0.50, 0.56, 0.67, 0.83, 1.0,
                  1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

# cofs = [0.0, 0.4, 1.0, 10.0]
# Is = [1.0, 0.158, 0.025, 0.0063, 0.00398, 0.0001]

# Define the keys of interest
keys_of_interest = ['theta_x', 'percent_aligned', 'S2', 'Z', 'phi', 'Nx_diff', 'Nz_diff', 'Omega_z', 'p_yy', 'p_xy']

# Initialize a data holder
data = {key: [] for key in keys_of_interest}
data['inertialNumber'] = []
data['cof'] = []
data['ap'] = []

os.chdir("./output_data_stress_updated")
#os.chdir("./output_data")

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
                        for key in keys_of_interest:
                            data[key].append(file_data.get(key, None))
            else:
                print(f"File {filename} does not exist.")
#print(file_data.keys())

# Ensure the output directory exists
output_dir = "../parametric_plots_oblate_prolate"
#output_dir = "../parametric_plots"
os.makedirs(output_dir, exist_ok=True)

# Function to create plots
def create_plots(data):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

     # Create a colormap
    colormap = plt.cm.coolwarm
    num_colors = len(aspect_ratios)
    colors = [colormap(i / num_colors) for i in range(num_colors)]


    for cof in cofs:
        # Plot mu = p_xy / p_yy
        plt.figure(figsize=(10, 8))
        for ap, color in zip(aspect_ratios, colors):
            x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
            p_xy_values = [value for value, aspect_ratio, coef in zip(data['p_xy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
            p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
            if x_values and p_xy_values and p_yy_values:  # Ensure that the lists are not empty
                mu_values = [pxy / pyy if pyy != 0 else None for pxy, pyy in zip(p_xy_values, p_yy_values)]
                plt.plot(x_values, mu_values, label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')
        #print("INERTIAL NUMBER", x_values)  
        plt.xscale('log')
        plt.legend()
        plt.title(f'$\\mu_p={cof}$', fontsize=20)
        #increase font size
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.xlabel('$I$', fontsize=20)
        plt.ylabel('$\\mu = \sigma_{xy} /\sigma_{yy}$', fontsize=20)
        filename = f'mu_cof_{cof}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


    # Plot other keys
    for key in keys_of_interest:
        if key in ['p_yy', 'p_xy']:
            continue  # Skip p_yy and p_xy since we already used them to calculate mu
        if key == 'contact_angle':
            label = '$\\theta_c$'
        elif key == 'percent_aligned':
            label = '$\%_z$'
        elif key == 'theta_x':
            label = '$\\theta_x$'
        elif key == 'S2':
            label = '$S_2$'
        elif key == 'Z':
            label = '$Z$'
        elif key == 'phi':
            label = '$\phi$'
        elif key == 'Nx_diff':
            label = '$N_x/P$'
        elif key == 'Nz_diff':
            label = '$N_z/P$'
        elif key == 'Omega_z':
            label = '$2\\langle \omega_z \\rangle /\dot{\gamma}$'
        else:       
            label = key

        for cof in cofs:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                y_values = [value for value, aspect_ratio, coef in zip(data[key], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                if key == 'theta_x':
                    y_values = [value * 180 / 3.20159 for value in y_values] # Convert to degrees
                elif key == 'Nx_diff':
                    p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                    y_values = [nx / pyy if pyy != 0 else None for nx, pyy in zip(y_values, p_yy_values)]
                elif key == 'Nz_diff':
                    p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                    y_values = [-nz / pyy if pyy != 0 else None for nz, pyy in zip(y_values, p_yy_values)]

                elif key == 'percent_aligned':
                    y_values = [value * 100 for value in y_values] # Convert to percentageS

                if x_values and y_values:  # Ensure that the lists are not empty
                    plt.plot(x_values, y_values, label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')

            plt.xscale('log')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
            plt.xlabel('$I$', fontsize=20)
            plt.ylabel(label, fontsize=20)
            plt.legend()
            plt.title(f'$\\mu_p={cof}$', fontsize=20)
            filename = f'{key}_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

# Create the plots
create_plots(data)
os.chdir("..")

print("Plots saved to parametric_plots")
