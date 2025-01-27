import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse
from matplotlib.transforms import ScaledTranslation
import matplotlib as mpl

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}'

def exp_corr_strain(x, gamma_v, nu, eta):
    return (1 + (x/gamma_v)**nu)**(-eta)

def exp_corr_length(x, length):
    return np.exp(-x/length)

def phiI(x, phic, cphi, betaphi):
    return phic - cphi * x**betaphi

def muI(x, muc, cmu, I0):
    return muc + cmu / (1+ (I0/x))

# Set the ticks as functions of pi
def pi_formatter(x, pos):
    fractions = {0: '0', np.pi/4: r'$\pi/4$', np.pi/2: r'$\pi/2$', 
                 3*np.pi/4: r'$3\pi/4$', np.pi: r'$\pi$'}
    return fractions.get(x, f'{x/np.pi:.2g}π')

def load_data(aspect_ratios, cofs, Is, keys_of_interest):
    # Initialize a data holder
    data = {key: [] for key in keys_of_interest}
    data['inertialNumber'] = []
    data['cof'] = []
    data['ap'] = []
    data['shear_rate'] = [] 
    data['I_nominal'] = []
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
                            data['I_nominal'].append(I)
                            data['cof'].append(cof)
                            data['ap'].append(ap)
                            data['shear_rate'].append(file_data.get('shear_rate', None))  # Append shear_rate
                            for key in keys_of_interest:
                                data[key].append(file_data.get(key, None))
                else:
                    print(f"File {filename} does not exist.")
    return data
# Define the parameters
# cofs = [0.0, 0.01, 0.1, 0.4, 1.0, 10.0]
cofs = [0.0, 0.001, 0.01, 0.1, 0.4, 1.0, 10.0]
# aspect_ratios = [1.0]
Is = [0.1, 0.046, 0.022, 0.01, 0.0046, 0.0022, 0.001] 
# Is = [0.1, 0.046, 0.022, 0.01] #, 0.0046, 0.0022, 0.001] 
# aspect_ratios = [0.4, 0.50, 0.67,  1.0, 1.5, 2.5]
aspect_ratios = [0.33, 0.4, 0.5, 0.56, 0.67, 0.83, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
# aspect_ratios = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
aspect_ratios_to_show = aspect_ratios
keys_of_interest = ['thetax_mean', 'percent_aligned', 'S2', 'Z', 'phi', 'Nx_diff', 'Nz_diff',
                     'Omega_z', 'p_yy', 'p_xy', 'Dyy', 'Dzz', 'total_normal_dissipation',
                       'total_tangential_dissipation', 'percent_sliding', 'vx_fluctuations', 'vy_fluctuations', 'vz_fluctuations',
                       'S2', 'c_delta_vy', 'c_r_values', 'strain', 'auto_corr']
keys_of_interest = ['S2']
# keys_of_interest = ['strain', 'auto_corr', 'c_delta_vy', 'c_r_values',] #'rke', 'tke', 'rke', 'tke'

# os.chdir("./output_data_hertz")
os.chdir("./output_data_final")


data = load_data(aspect_ratios, cofs, Is, keys_of_interest)
output_dir = "../parametric_plots_final"
#output_dir = "../parametric_plots"
os.makedirs(output_dir, exist_ok=True)


# Function to create plots
def create_plots(data):

    # allocate list for critical values of mu
    mu_c = []

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

     # Create a colormap
    colormap = plt.cm.RdYlBu
    num_colors = len(aspect_ratios)
    extreme_indices = np.concatenate([
        np.linspace(0, 0.3, num_colors // 2, endpoint=False),  # Lower 30%
        np.linspace(0.7, 1.0, num_colors - num_colors // 2)    # Upper 30%
    ])
    colors = [colormap(i) for i in extreme_indices]

    # Insert a unique color for the central aspect ratio
    central_color = 'black'
    central_ap = 1.0  # Replace with the actual central value of aspect ratio

    # Identify the index of the central aspect ratio
    central_index = aspect_ratios.index(central_ap)

    # Insert the central color
    colors.insert(central_index, central_color)

    colormap_cof = plt.cm.viridis
    num_colors_cof = len(cofs)
    colors_cof = [colormap_cof(i) for i in np.linspace(0, 1, num_colors_cof)]


    if 'p_xy' in keys_of_interest and 'p_yy' in keys_of_interest:
        intial_guess = [1.0, 1.0, 1.0]
        for cof in cofs:
            # Plot mu = p_xy / p_yy
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            for ap, color in zip(aspect_ratios, colors):
                x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                p_xy_values = [value for value, aspect_ratio, coef in zip(data['p_xy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                if x_values and p_xy_values and p_yy_values:  # Ensure that the lists are not empty
                    mu_values = [pxy / pyy if pyy != 0 else None for pxy, pyy in zip(p_xy_values, p_yy_values)]
                if ap in aspect_ratios_to_show:
                    plt.plot(x_values, mu_values, label=f'$\\alpha={ap}$', color=color, linestyle='None', marker='o')
                    popt, pcov = curve_fit(muI, x_values, mu_values, p0=intial_guess, method='trf', x_scale = [1, 1, 1], bounds=([0, 0, 0], [1, 1, 1]))
                    # print(f"cof={cof}, ap={ap}, popt={popt}")
                    mu_c.append(popt[0])
                    plt.plot(x_values, muI(x_values, *popt), color=color, linestyle='--')
                            
            plt.xscale('log')
            # plt.title(f'$\\mu_p={cof}$', fontsize=20)
            plt.title("Effective Friction / Stress Anisotropy", fontsize=20)
            #increase font size
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$I$', fontsize=20)
            plt.ylabel('$\\mu = \\sigma_{xy} /\\sigma_{yy}$', fontsize=20)
            filename = f'mu_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

        # Plot mu_c vs ap for the different cofs
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        # split mu_c into s many lists as the cofs I have
        mu_c = [mu_c[i:i + len(aspect_ratios)] for i in range(0, len(mu_c), len(aspect_ratios))]
        for i, cof in enumerate(cofs):
            plt.plot(aspect_ratios, mu_c[i], label=f'$\\mu_p={cof}$', color=colors_cof[i], linestyle='--', marker='o', linewidth=2)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('$\\alpha$', fontsize=20)
        plt.ylabel('$\\mu_c$', fontsize=20)
        plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)
        filename = 'mu_c.png'
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()


    phi_c = []

    # Plot other keys
    for key in keys_of_interest:
        if key in ['p_yy', 'p_xy']:
            continue  # Skip p_yy and p_xy since we already used them to calculate mu
        elif key == 'percent_aligned':
            label = '$\\%_z$'
        elif key == 'thetax_mean':
            label = '$\\theta_x \, [^\\circ]$'
        elif key == 'S2':
            label = '$S_2$'
        elif key == 'Z':
            label = '$Z$'
        elif key == 'phi':
            label = '$\\phi$'
        elif key == 'Nx_diff':
            label = '$N_1/P$'
        elif key == 'Nz_diff':
            label = '$N_2/P$'
        elif key == 'Omega_z':
            label = '$2\\langle \\omega_z \\rangle /\\dot{\\gamma}$'
        elif key == 'Dyy':
            label = '$D_{yy}/\\dot{\\gamma}d^2$'
        elif key == 'Dzz':
            label = '$D_{zz}/\\dot{\\gamma}d^2$'
        elif key == 'vx_fluctuations':
            label = '$\\delta v_x / \\dot{\\gamma}d$'
        elif key == 'vy_fluctuations':
            label = '$\\delta v_y / \\dot{\\gamma}d$'
        elif key == 'vz_fluctuations':
            label = '$\\delta v_z / \\dot{\\gamma}d$'
        elif key == 'omega_fluctuations':
            label = '$\\delta \\omega_z / \\dot{\\gamma}$'
        elif key == 'percent_sliding':
            label = '$\\chi$'
        elif key == 'total_normal_dissipation' or key == 'total_tangential_dissipation' or key == 'c_delta_vy' or key == 'c_r_values' or key == 'strain' or key == 'auto_corr':
            continue  # Skip these keys for now
        else: 
            label = key

        for cof in cofs:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)   
            plt.xticks(fontsize=20)
            plt.xscale('log')
            # plt.yscale('log')

            for ap, color in zip(aspect_ratios, colors):
                
                x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                y_values = [value for value, aspect_ratio, coef in zip(data[key], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                if key == 'thetax_mean':
                    y_values = [value * 180 / 3.14 for value in y_values] # Convert to degrees
                    if ap < 1:
                        y_values = [value + 90 for value in y_values]
                    if ap == 1:
                        y_values = [45 for value in y_values]
                elif key == 'Nx_diff':
                    p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                    y_values = [nx / pyy if pyy != 0 else None for nx, pyy in zip(y_values, p_yy_values)]
                elif key == 'Nz_diff':
                    p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                    y_values = [nz / pyy if pyy != 0 else None for nz, pyy in zip(y_values, p_yy_values)]
                elif key == 'Dyy' or key == 'Dzz':
                    d_eq = (2+ap)/3 # Equivalent diameter
                    # Dynamically get the shear_rate for each data point
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    # Ensure that the shear_rate and y_values are aligned in size
                    if len(y_values) == len(shear_rate_values):
                        y_values = [value / (shear_rate*d_eq ** 2) for value, shear_rate in zip(y_values, shear_rate_values)]
                    else:
                        print(f"Warning: Mismatched lengths for key '{key}' and shear_rate values for aspect_ratio={ap}, coef={cof}.")
                    plt.yscale('log')
                elif key == 'percent_aligned':
                    y_values = [value * 100 for value in y_values] # Convert to percentageS
                elif key == 'Omega_z':
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [-2*value / shear_rate for value, shear_rate in zip(y_values, shear_rate_values)]
                    plt.yscale('log')
                    # plt.ylim(-0.1, 1.1)
                elif key == 'vx_fluctuations' or key == 'vy_fluctuations' or key == 'vz_fluctuations':
                    d_eq = (2+ap)/3
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [value / (shear_rate*d_eq) for value, shear_rate in zip(y_values, shear_rate_values)]
                    if ap> 1:
                        y_values = [value*ap**(1/3) for value in y_values]
                    else:
                        y_values = [value*ap**(-1/3) for value in y_values]
                    plt.yscale('log')
                elif key == "omega_fluctuations":
                    if cof == 0.0 and ap == 1.0:
                        continue
                    print(f"key={key}, ap={ap}, cof={cof}")
                    print(f"x_values={x_values}")
                    print(f"y_values={y_values}")
                    y_values_new = [np.sqrt(value[2]) for value in y_values]
                    # y_values_new = [np.sqrt(np.linalg.norm(value[1, :])) for value in y_values]
                    # y_values_new = [value[1, 2] for value in y_values]
                    y_values = y_values_new
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [1*value / shear_rate for value, shear_rate in zip(y_values, shear_rate_values)]
                    plt.yscale('log')
                elif key == 'percent_sliding':
                    plt.yscale('log')
                if x_values and y_values :  # Ensure that the lists are not empty
                    if key == 'phi':
                        # plt.plot(x_values, y_values, label=f'$\\alpha={ap}$', color=color, linestyle='--')
                        bounds = ([0, 0, 0], [0.75, 100, 100])
                        intial_guess = [0.6, 0.5, 0.5]
                        popt, pcov = curve_fit(phiI, x_values, y_values, p0=intial_guess, maxfev=20000, bounds=bounds)
                        phi_c.append(popt[0])
                        if ap == 1.0:
                            plt.plot(x_values, phiI(x_values, *popt), color=color, linestyle='--', linewidth=2)
                        else:
                            plt.plot(x_values, phiI(x_values, *popt), color=color, linestyle='--')
                        plt.plot(x_values, y_values, label=f'$\\alpha={ap}$', color=color, linestyle='None', marker='o')

                    else:
                        plt.plot(x_values, y_values, label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')
            
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$I$', fontsize=20)
            plt.ylabel(label, fontsize=20)
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)   
            filename = f'{key}_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

    if 'phi' in keys_of_interest:
        # Plot phi_c vs ap for the different cofs
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        # split mu_c into s many lists as the cofs I have
        phi_c = [phi_c[i:i + len(aspect_ratios)] for i in range(0, len(phi_c), len(aspect_ratios))]
        for i, cof in enumerate(cofs):
            plt.plot(aspect_ratios, phi_c[i], label=f'$\\mu_p={cof}$', color=colors_cof[i], linestyle='--', marker='o', linewidth=3)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('$\\alpha$', fontsize=20)
        plt.ylabel('$\\phi_c$', fontsize=20)
        plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.2), ncols = 4)
        filename = 'phi_c.png'
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()

    if 'omega_fluctuations' in keys_of_interest and 'vy_fluctuations' in keys_of_interest:
        # print the ratio of omega times d over v
        for cof in cofs:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                omega_values = [value for value, aspect_ratio, coef in zip(data['omega_fluctuations'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                vy_values = [value for value, aspect_ratio, coef in zip(data['vy_fluctuations'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                if x_values and omega_values and vy_values:
                    # if ap > 1:
                    #     d_eq = ap # longest dimension for prolate
                    # else:
                    #     d_eq = 1
                    d_eq = 1
                    ratio = [np.sqrt(delta_omegaz[2])*d_eq / delta_vy for delta_omegaz, delta_vy in zip(omega_values, vy_values)]
                    plt.plot(x_values, ratio, label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')
            plt.xscale('log')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)
            plt.xlabel('$I$', fontsize=20)
            plt.ylabel('$\\delta \\omega_z \ell / \\delta v_y$', fontsize=20)
            plt.title(f'$\\mu_p={cof}$', fontsize=20)
            filename = f'omega_vy_ratio_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()


    if 'total_normal_dissipation' in keys_of_interest and 'total_tangential_dissipation' in keys_of_interest:
        #Plot total_tangential_dissipation / (total_normal_dissipation + total_tangential_dissipation)
        for cof in cofs:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                r = (ap-1)/(ap+1)   
                # Extract inertial numbers, total_normal_dissipation, and total_tangential_dissipation values
                x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                normal_dissipation_values = [value for value, aspect_ratio, coef in zip(data['total_normal_dissipation'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                tangential_dissipation_values = [value for value, aspect_ratio, coef in zip(data['total_tangential_dissipation'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                # Ensure all the lists have the same length
                if len(x_values) == len(normal_dissipation_values) == len(tangential_dissipation_values):
                    # Compute the ratio of tangential to total dissipation
                    dissipation_ratios = [tangential / (normal + tangential) if (normal + tangential) != 0 else None for tangential, normal in zip(tangential_dissipation_values, normal_dissipation_values)]
                    
                # Plot the values
                if x_values and dissipation_ratios:  # Ensure that the lists are not empty
                    plt.plot(x_values, dissipation_ratios, label=f'$\\alpha={ap:.2f}, \, r={r:.2f}$', color=color, linestyle='--', marker='o')

            # Set plot properties
            plt.xscale('log')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1.4, 1.4), ncols = 2)   
            plt.xlabel('$I$', fontsize=20)
            plt.ylabel("$W_t/W_{tot}$", fontsize=20)  # Y-axis label for the ratio
            plt.title(f'$\\mu_p={cof}$', fontsize=20)
            
            # Save the plot
            filename = f'power_dissipation_ratio_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

        # Plot total_tangential_dissipation / total_normal_dissipation as a function of cof for different I and fixed ap
        # one plot for each I
        for I in Is:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                # Extract the inertial numbers, total_normal_dissipation, and total_tangential_dissipation values
                x_values = [cof for inertial_number, aspect_ratio, cof in zip(data['I_nominal'], data['ap'], data['cof']) if aspect_ratio == ap  and inertial_number == I]
                normal_dissipation_values = [value for value, aspect_ratio, Inertial in zip(data['total_normal_dissipation'], data['ap'], data['I_nominal']) if aspect_ratio == ap and Inertial == I and value is not None]
                tangential_dissipation_values = [value for value, aspect_ratio, Inertial in zip(data['total_tangential_dissipation'], data['ap'], data['I_nominal']) if aspect_ratio == ap and Inertial == I and value is not None]
                # Ensure all the lists have the same length
                if len(x_values) == len(normal_dissipation_values) == len(tangential_dissipation_values):
                    # Compute the ratio of tangential to total dissipation
                    dissipation_ratios = [tangential / normal  if normal != 0 else None for tangential, normal in zip(tangential_dissipation_values, normal_dissipation_values)]
                    
                # Plot the values
                if x_values and dissipation_ratios:  # Ensure that the lists are not empty
                    if x_values[0] == 0.0:
                        x_values[0] = 0.001
                    plt.plot(x_values, dissipation_ratios, label=f'$\\alpha={ap:.2f}$', color=color, linestyle='--', marker='o')

            # Set plot properties
            plt.xscale('log')
            plt.yscale('log')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 2)
            plt.xlabel('$\\mu_p$', fontsize=20)
            plt.ylabel("$W_t/W_n$", fontsize=20)  # Y-axis label for the ratio
            plt.title(f'$I={I}$', fontsize=20)
            plt.savefig(os.path.join(output_dir, f'power_dissipation_ratio_I_{I}.png'), bbox_inches='tight')
            plt.close()

        # Initialize an empty dictionary to store crossing points
        crossing_points_low = {}
        crossing_points_high = {}
        plt.figure(figsize=(10, 8))
        for ap, color in zip(aspect_ratios, colors):
            mu_p_crossings_low = []
            mu_p_crossings_high = []
            I_crossings_low = []
            I_crossings_high = []

            for I in Is:
                # Extract mu_p, I, and P_t / P_n values for this aspect ratio
                x_values = [cof for inertial_number, aspect_ratio, cof in zip(data['I_nominal'], data['ap'], data['cof']) if aspect_ratio == ap and inertial_number == I]
                normal_dissipation_values = [value for value, aspect_ratio, Inertial in zip(data['total_normal_dissipation'], data['ap'], data['I_nominal']) if aspect_ratio == ap and Inertial == I and value is not None]
                tangential_dissipation_values = [value for value, aspect_ratio, Inertial in zip(data['total_tangential_dissipation'], data['ap'], data['I_nominal']) if aspect_ratio == ap and Inertial == I and value is not None]

                if len(x_values) == len(normal_dissipation_values) == len(tangential_dissipation_values):
                    # Compute the dissipation ratio
                    dissipation_ratios = [tangential / normal if normal != 0 else None for tangential, normal in zip(tangential_dissipation_values, normal_dissipation_values)]
                    
                    # Find where P_t / P_n crosses 1
                    for i in range(len(dissipation_ratios) - 1):
                        if dissipation_ratios[i] is not None and dissipation_ratios[i + 1] is not None:
                            if (dissipation_ratios[i] < 1 and dissipation_ratios[i + 1] > 1):
                               # Logarithmic transformation
                                log_x1, log_x2 = np.log(x_values[i]), np.log(x_values[i + 1])
                                log_y1, log_y2 = np.log(dissipation_ratios[i]), np.log(dissipation_ratios[i + 1])

                                # Interpolation in log-log space
                                log_crossing_mu_p = log_x1 + (np.log(1) - log_y1) * (log_x2 - log_x1) / (log_y2 - log_y1)

                                # Transform back to the original scale
                                crossing_mu_p = np.exp(log_crossing_mu_p)
                                mu_p_crossings_low.append(crossing_mu_p)
                                I_crossings_low.append(I)

                            elif (dissipation_ratios[i] > 1 and dissipation_ratios[i + 1] < 1):
                                # Logarithmic transformation
                                log_x1, log_x2 = np.log(x_values[i]), np.log(x_values[i + 1])
                                log_y1, log_y2 = np.log(dissipation_ratios[i]), np.log(dissipation_ratios[i + 1])

                                # Interpolation in log-log space
                                log_crossing_mu_p = log_x1 + (np.log(1) - log_y1) * (log_x2 - log_x1) / (log_y2 - log_y1)

                                # Transform back to the original scale
                                crossing_mu_p = np.exp(log_crossing_mu_p)
                                mu_p_crossings_high.append(crossing_mu_p)
                                I_crossings_high.append(I)
            
            # Store the crossing points for this aspect ratio
            crossing_points_low[ap] = (mu_p_crossings_low, I_crossings_low)
            crossing_points_high[ap] = (mu_p_crossings_high, I_crossings_high)

            # Plot the level set for this aspect ratio
            plt.plot(mu_p_crossings_low, I_crossings_low, label=f'$\\alpha={ap:.2f}$', color=color, linestyle='-', linewidth=2)
            plt.plot(mu_p_crossings_high, I_crossings_high, color=color, linestyle='--', linewidth=2)

        #plot a power law 1 to 2 starting at the bottom left corner
        plt.plot([0.001, 0.01], [0.001, 0.1], color='black', linestyle='--') 

        # plot should be cut exactly at the limits
        plt.xlim(0.001, 10)
        plt.ylim(0.001, 0.1)

        # Plot properties
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.xlabel('$\\mu_p$', fontsize=20)
        plt.ylabel('$I$', fontsize=20)
        plt.legend(fontsize=15, loc='upper right', bbox_to_anchor=(1.2, 1.0), ncols = 1)
        plt.title('Level Set of $P_t/P_n = 1$', fontsize=20)
        plt.savefig(os.path.join(output_dir, 'level_set_power_dissipation_ratio.png'), bbox_inches='tight')
        plt.show()

    if 'c_delta_vy' in keys_of_interest and 'c_r_values' in keys_of_interest:
        # Plot c_delta_vy vs c_r_values for different I and fixed ap and mu_p   
        I = 0.001
        for cof in cofs:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                x_values = [
                    value for value, aspect_ratio, Inertial, coef in zip(
                        data['c_r_values'], data['ap'], data['I_nominal'], data['cof']
                    ) if aspect_ratio == ap and Inertial == I and coef == cof and value is not None
                ]
                y_values = [
                    value for value, aspect_ratio, Inertial, coef in zip(
                        data['c_delta_vy'], data['ap'], data['I_nominal'], data['cof']
                    ) if aspect_ratio == ap and Inertial == I and coef == cof and value is not None
                ]
                if x_values and y_values: 
                    # Convert to numpy arrays for easier plotting
                    d_eq = (2+ap)/3
                    x_values = np.asarray(x_values)/d_eq
                    y_values = np.asarray(y_values)

                    # fit exponential curve to find correlation length
                    popt, pcov = curve_fit(exp_corr_length, x_values[0], y_values[0], p0=[3], method='lm', maxfev=20000)
                    print(f"ap={ap}, popt={popt}, pcov={pcov}")
                    plt.plot(
                        x_values[0], y_values[0], 
                        label=f'$\\alpha={ap}$', color=color, 
                        linestyle='none', marker='o'
                    )
                    plt.plot(
                        x_values[0], exp_corr_length(x_values[0], *popt), 
                        color=color, linestyle='--'
                    )

            # Adjust plot settings
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$r/d_{eq}$', fontsize=20)
            plt.ylabel('$\\tilde{{C}}(r)$', fontsize=20)
            plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
            plt.title(f'$I={I}, \mu_p={cof}$', fontsize=20)

            # Save the figure
            filename = f'c_delta_vy_I_{I}_mup_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

    if 'strain' in keys_of_interest and 'auto_corr' in keys_of_interest:
        # plot auto_corr vs strain for different I and fixed ap and mu_p
        I = 0.001
        for cof in cofs:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                x_values = [
                    value for value, aspect_ratio, Inertial, coef in zip(
                        data['strain'], data['ap'], data['I_nominal'], data['cof']
                    ) if aspect_ratio == ap and coef == cof and Inertial == I and value is not None
                ]
                y_values = [
                    value for value, aspect_ratio, Inertial, coef in zip(
                        data['auto_corr'], data['ap'], data['I_nominal'], data['cof']
                    ) if aspect_ratio == ap and coef == cof and Inertial == I and value is not None
                ]
                if x_values and y_values:
                    plt.plot(x_values[0], y_values[0], label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim(0.009, 0.5)
            plt.ylim(0.01, 1.1)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('$\\gamma$', fontsize=20)
            plt.ylabel('$\\tilde{{C}}(\\gamma)$', fontsize=20)
            plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
            plt.title(f'$\\mu_p={cof}$', fontsize=20)
            filename = f'auto_corr_strain_cof_{cof}_I_{I}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()


    # plot sliding percentage as a function of ap for different I and fixed cof
    if 'percent_sliding' in keys_of_interest:
        for I in Is:
            plt.figure(figsize=(10, 8))
            for cof, color in zip(cofs, colors_cof):
                # x_values = [ap for inertial_number, aspect_ratio, coef in zip(data['I_nominal'], data['ap'], data['cof']) if inertial_number == I and coef == cof]
                x_values = aspect_ratios
                sliding_values = [value for value, inertial_number, coef in zip(data['percent_sliding'], data['I_nominal'], data['cof']) if inertial_number == I and coef == cof and value is not None]
                if x_values and sliding_values:
                    plt.plot(x_values, sliding_values, label=f'$\\mu_p={cof}$', color=color, linestyle='--', marker='o')
            x_values = np.asarray(x_values)
            plt.plot(x_values[:7],0.1*x_values[:7]**(-2/3), label=f'$\\alpha^{{-2/3}}$', color='black', linestyle='--')
            plt.plot(x_values[6:],0.1*x_values[6:]**(2/3), label=f'$\\alpha^{{2/3}}$', color='black', linestyle='--')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim(0.3, 3.0)
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('$\\alpha$', fontsize=20)
            plt.ylabel('$\\chi$', fontsize=20)
            plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)
            plt.title(f'$I={I}$', fontsize=20)
            plt.savefig(os.path.join(output_dir, f'sliding_percentage__mu_p_I_{I}.png'), bbox_inches='tight')
            plt.close()

    # Plot sliding percentage as a function of cof (mu_p) for different I
    if 'percent_sliding' in keys_of_interest:
        for I in Is:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                # x_values = [coef for inertial_number, aspect_ratio, coef in zip(data['I_nominal'], data['ap'], data['cof']) if inertial_number == I and aspect_ratio == ap]
                x_values = cofs  # Now x-values correspond to cof (mu_p)
                sliding_values = [
                    value for value, inertial_number, aspect_ratio in zip(data['percent_sliding'], data['I_nominal'], data['ap']) 
                    if inertial_number == I and aspect_ratio == ap and value is not None
                ]
                if x_values and sliding_values:
                    plt.plot(
                        x_values, sliding_values, 
                        label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o'
                    )
            x_values = np.asarray(x_values)
            # Add any specific relationships to plot, for example, trends for cof
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim(min(cofs), max(cofs))
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('$\\mu_p$', fontsize=20)
            plt.ylabel('$\\chi$', fontsize=20)
            plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
            plt.title(f'$I={I}$', fontsize=20)
            plt.savefig(os.path.join(output_dir, f'sliding_percentage_I_{I}_mu_p.png'), bbox_inches='tight')
            plt.close()


def plot_polar_histograms_ap(bins, histograms, title, labels, symmetry=False):
    """
    Plot multiple polar histograms as lines with varying ap.
    
    Args:
    - bins (array): The bin edges for the histogram (global or local).
    - histograms (list of arrays): A list of histogram values (heights of bins) to plot.
    - title (str): Title for the plot.
    - labels (list of str): Labels for each histogram (e.g., different ap values).
    - symmetry (bool): Flag to indicate if the histogram should be symmetric.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(13, 8))
    ax.set_title(title, va='bottom')

    # Define a color map to differentiate lines
    colormap = plt.cm.RdYlBu
    num_colors = len(aspect_ratios)
    extreme_indices = np.concatenate([
        np.linspace(0, 0.3, num_colors // 2, endpoint=False),  # Lower 30%
        np.linspace(0.7, 1.0, num_colors - num_colors // 2)    # Upper 30%
    ])
    colors = [colormap(i) for i in extreme_indices]

    # Insert a unique color for the central aspect ratio
    central_color = 'black'
    central_ap = 1.0  # Replace with the actual central value of aspect ratio

    # Identify the index of the central aspect ratio
    central_index = aspect_ratios.index(central_ap)

    # Insert the central color
    colors.insert(central_index, central_color)


    for idx, (histogram, label, color) in enumerate(zip(histograms, labels, colors)):
        if symmetry:
            # Local histograms: Assuming the bins and histogram are for 0 to 90 degrees (10 bins)
            assert len(histogram) == len(bins) - 1, "The length of histogram should match the number of bins - 1."

            # Reverse the histogram for the 90 to 180 degrees
            histogram_mirror_90_180 = histogram[::-1]

            # Combine 0-90, 90-180, 180-270 (same as 90-0), and 270-360 (same as 0-90)
            extended_histogram = np.concatenate((histogram, histogram_mirror_90_180, histogram, histogram_mirror_90_180))
            
            # Adjust bins to match the extended histogram
            bin_width = bins[1] - bins[0]
            extended_bins = np.arange(0, 360 + bin_width, bin_width)
        else:
            # Global histograms: Use original histogram and bins for non-symmetric data (144 bins)
            assert len(histogram) == len(bins) - 1, "The length of histogram should match the number of bins - 1."
            extended_histogram = histogram
            extended_bins = bins

        # Calculate the bin centers for the extended bins
        bin_centers = (extended_bins[:-1] + extended_bins[1:]) / 2
        angles_rad = np.radians(bin_centers)  # Convert bin centers to radians

        # Ensure that the number of angles and histogram values match
        if len(angles_rad) != len(extended_histogram):
            raise ValueError(f"Mismatch in dimensions: angles_rad has {len(angles_rad)} elements, but extended_histogram has {len(extended_histogram)} elements")

        # Plot data as lines on the polar plot
        ax.plot(angles_rad, extended_histogram, label=label, color=color)

    max_hist = max([np.max(hist) for hist in histograms])
    ax.set_ylim(0, max_hist * 1.1)

    plt.legend(bbox_to_anchor=(1.4, 1), loc='upper right', fontsize=12)
    # print(output_dir + '/polar_histogram_{title}_lines.png')
    plt.savefig(f"{output_dir}/polar_histogram_{title}_lines.png", bbox_inches='tight')
    plt.close()

def create_polar_plots_varying_ap(data, fixed_cof, fixed_I, histogram_keys, pdf_keys, local_histogram_keys):
    bins_global = np.linspace(-180, 180, 145)  # Define your bins for global data (144 values)
    bins_local = np.linspace(0, 90, 11)  # Define your bins for local data (10 values)

    # Loop over the histogram keys (force histograms to be normalized by p * A)
    for hist_key in histogram_keys:
        histograms = []
        labels = []
        
        # Loop over the aspect ratios (ap) while keeping cof and I fixed
        for ap in aspect_ratios:
            filename = f'simple_shear_ap{ap}_cof_{fixed_cof}_I_{fixed_I}.pkl'
            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    file_data = pickle.load(file)
                    histogram = file_data.get(hist_key, None)
                    box_x_length = file_data.get('box_x_length', None)
                    box_z_length = file_data.get('box_z_length', None)
                    total_area = file_data.get('total_area', None)
                    avg_pressure = file_data.get('p_yy', None)

                    if histogram is not None and box_x_length is not None and box_z_length is not None:
                        # Calculate the area A and normalize by p * A
                        #area = box_x_length * box_z_length
                        area = total_area
                        normalized_histogram = histogram / (avg_pressure * area)  # p is always 50

                        # Check if the histogram length matches the bins for global data
                        if len(normalized_histogram) == len(bins_global) - 1:
                            histograms.append(normalized_histogram)
                            labels.append(f'$\\alpha={ap}$')
                        else:
                            print(f"Warning: Histogram length does not match global bins for {filename}")

        # Call the plot function if there are histograms to plot
        if histograms:
            plot_polar_histograms_ap(bins_global, histograms, f'{hist_key} ($\\mu_p={fixed_cof}$, $I={fixed_I}$)', labels)

    # Handle PDF histograms (no normalization)
    for pdf_key in pdf_keys:
        pdf_histograms = []
        labels = []
        
        # Loop over aspect ratios (ap) for PDF histograms
        for ap in aspect_ratios:
            filename = f'simple_shear_ap{ap}_cof_{fixed_cof}_I_{fixed_I}.pkl'
            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    file_data = pickle.load(file)
                    pdf_histogram = file_data.get(pdf_key, None)
                    
                    if pdf_histogram is not None:
                        # Check if the pdf histogram length matches the bins for global data
                        if len(pdf_histogram) == len(bins_global) - 1:
                            pdf_histograms.append(pdf_histogram)
                            labels.append(f'$\\alpha={ap}$')
                        else:
                            print(f"Warning: PDF histogram length does not match global bins for {filename}")

        # Call the plot function if there are histograms to plot
        if pdf_histograms:
            plot_polar_histograms_ap(bins_global, pdf_histograms, f'{pdf_key} ($\\mu_p={fixed_cof}$, $I={fixed_I}$)', labels)

    # Handle local force histograms (normalized by p * A)
    for local_hist_key in local_histogram_keys:
        local_histograms = []
        labels = []
        
        # Loop over aspect ratios (ap) for local data
        for ap in aspect_ratios:
            filename = f'simple_shear_ap{ap}_cof_{fixed_cof}_I_{fixed_I}.pkl'
            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    file_data = pickle.load(file)
                    local_histogram = file_data.get(local_hist_key, None)
                    box_x_length = file_data.get('box_x_length', None)
                    box_z_length = file_data.get('box_z_length', None)
                    area_particle = file_data.get('total_area', None)
                    avg_pressure = file_data.get('p_yy', None)
                    
                    if local_histogram is not None and box_x_length is not None and box_z_length is not None:
                        # Calculate the area A and normalize by p * A
                        # area = box_x_length * box_z_length
                        normalized_local_histogram = local_histogram / (avg_pressure * area_particle)  # p is always 50
                        # normalized_local_histogram = local_histogram 

                        # Check if the histogram length matches the bins for local data
                        if len(normalized_local_histogram) == len(bins_local) - 1:
                            local_histograms.append(normalized_local_histogram)
                            labels.append(f'$\\alpha={ap}$')
                        else:
                            print(f"Warning: Local histogram length does not match local bins for {filename}")
        
        # Call the plot function if there are histograms to plot
        if local_histograms:
            plot_polar_histograms_ap(bins_local, local_histograms, f'Normalized {local_hist_key} ($\\mu_p={fixed_cof}$, $I={fixed_I}$)', labels, symmetry=True)

def set_axes_equal(ax, max_length):
    """
    Set equal scaling (aspect ratio) for all axes of a 3D plot.
    This ensures that arrows have the correct proportions in all dimensions.
    """
    # Set the limits for all axes based on the max length
    ax.set_xlim([-max_length, max_length])
    ax.set_ylim([0, max_length])
    ax.set_zlim([0, max_length])

def plot_fabric_eigenvectors_ap(data, fixed_cof, fixed_I, aspect_ratios):
    """
    Plot 3D arrows representing the eigenvectors of the fabric tensor for different aspect ratios.
    
    Args:
    - data: Dictionary containing the simulation data.
    - fixed_cof: The coefficient of friction to be fixed.
    - fixed_I: The inertial number to be fixed.
    - aspect_ratios: List of aspect ratios to loop through.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90)

    # Define a colormap to differentiate the arrows by aspect ratio
    colormap = plt.cm.RdYlBu
    num_colors = len(aspect_ratios)
    extreme_indices = np.concatenate([
        np.linspace(0, 0.3, num_colors // 2, endpoint=False),  # Lower 30%
        np.linspace(0.7, 1.0, num_colors - num_colors // 2)    # Upper 30%
    ])
    colors = [colormap(i) for i in extreme_indices]

    # Insert a unique color for the central aspect ratio
    central_color = 'black'
    central_ap = 1.0  # Replace with the actual central value of aspect ratio

    # Identify the index of the central aspect ratio
    central_index = aspect_ratios.index(central_ap)

    # Insert the central color
    colors.insert(central_index, central_color)


    # Initialize the max length to find the maximum arrow length
    max_arrow_length = 0

    # Loop over the aspect ratios (ap)
    for idx, ap in enumerate(aspect_ratios):
        filename = f'simple_shear_ap{ap}_cof_{fixed_cof}_I_{fixed_I}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                file_data = pickle.load(file)
                
                # Extract the fabric tensor
                fabric_tensor = file_data.get('fabric', None)
                
                if fabric_tensor is not None:
                    # eigenvalues, eigenvectors = np.linalg.eig(fabric_tensor)
                    # print(f"Eigenvectors for ap={ap}:\n{eigenvectors}\n{eigenvalues}")
                    # for i in range(3):
                    #     if eigenvectors[0][i] < 0:
                    #         eigenvectors[:, i] = -eigenvectors[:, i]
                    #     if eigenvectors[1][i] < 0:
                    #         eigenvectors[:, i] = -eigenvectors[:, i]
                    # # Identify the eigenvector closest to (0, 0, 1)
                    # z_index = np.argmax(np.abs(eigenvectors[:, 2]))  # Index of the eigenvector aligned with z-axis
                    # xy_indices = [i for i in range(3) if i != z_index]  # Indices of the eigenvectors in the x-y plane

                    # # Prepare the plot
                    # fig, ax = plt.subplots(figsize=(6, 6))

                    # # Loop over the eigenvectors in the x-y plane
                    # for idx in xy_indices:
                    #     eig_val = eigenvalues[idx]
                    #     eig_vec = eigenvectors[:, idx]

                    #     # Project the eigenvector into the x-y plane
                    #     arrow_vector = eig_vec[:2] * eig_val  # Take only the x and y components
                    #     max_arrow_length = max(max_arrow_length, np.linalg.norm(arrow_vector))

                    #     # Plot the 2D arrow
                    #     ax.quiver(0, 0,  # Origin (0, 0)
                    #             arrow_vector[0], arrow_vector[1],  # Arrow direction and length
                    #             angles='xy', scale_units='xy', scale=1, color=colors[idx], label=f'$\\alpha={ap}$')

                    # Compute the eigenvalues and eigenvectors of the fabric tensor
                    eigenvalues, eigenvectors = np.linalg.eig(fabric_tensor)
                    print(f"Eigenvectors for ap={ap}:\n{eigenvectors}\n{eigenvalues}")
                    for i in range(3):
                        # if eigenvectors[0][i] < 0:
                        #     eigenvectors[:, i] = -eigenvectors[:, i]
                        if eigenvectors[1][i] < 0:
                            eigenvectors[:, i] = -eigenvectors[:, i]

                    # Loop over each eigenvalue-eigenvector pair
                    for i in range(3):  # Assuming 3x3 fabric tensor
                        eig_val = eigenvalues[i]
                        eig_vec = eigenvectors[:, i]

                        # Rescale the eigenvector by the corresponding eigenvalue
                        arrow_vector = eig_vec * eig_val

                        # Update max_arrow_length
                        max_arrow_length = max(max_arrow_length, np.linalg.norm(arrow_vector))

                        # Plot the arrow (using quiver for 3D arrows)
                        ax.quiver(0, 0, 0,  # Origin (0, 0, 0)
                                  arrow_vector[0], arrow_vector[1], arrow_vector[2],  # Arrow direction and length
                                  color=colors[idx], label=f'$\\alpha={ap}$' if i == 0 else "", arrow_length_ratio=0.1)

    # Set equal aspect ratio for all axes and adjust the limits based on the maximum arrow length
    set_axes_equal(ax, max_arrow_length * 1.1)  # Add 10% extra space for visual clarity
    
    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    # Add a legend (showing the aspect ratios)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')

    plt.title(f'Fabric Eigenvectors ($\\mu_p={fixed_cof}$, $I={fixed_I}$)')
    plt.savefig(output_dir + '/fabric_eigenvectors_cof_{fixed_cof}_I_{fixed_I}.png', bbox_inches='tight')
    plt.show()

def plot_pdf_thetax(fixed_cof, fixed_I, aspect_ratios):
    """
    Plot the PDF of angular distribution for different aspect ratios as a histogram.
    
    Args:
    - data: Dictionary containing the simulation data.
    - fixed_cof: The coefficient of friction to be fixed.
    - fixed_I: The inertial number to be fixed.
    - aspect_ratios: List of aspect ratios to loop through.
    """
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Define a colormap to differentiate the histograms by aspect ratio
    colormap = plt.cm.RdYlBu
    num_colors = len(aspect_ratios)
    extreme_indices = np.concatenate([
        np.linspace(0, 0.3, num_colors // 2, endpoint=False),  # Lower 30%
        np.linspace(0.7, 1.0, num_colors - num_colors // 2)    # Upper 30%
    ])
    colors = [colormap(i) for i in extreme_indices]

    # Insert a unique color for the central aspect ratio
    central_color = 'black'
    central_ap = 1.0  # Replace with the actual central value of aspect ratio

    # Identify the index of the central aspect ratio
    central_index = aspect_ratios.index(central_ap)

    # Insert the central color
    colors.insert(central_index, central_color)

    # Loop over the aspect ratios (ap)
    for idx, ap in enumerate(aspect_ratios):
        filename = f'simple_shear_ap{ap}_cof_{fixed_cof}_I_{fixed_I}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                file_data = pickle.load(file)
                
                # Extract the PDF data
                pdf_data = file_data.get('pdf_thetax', None)
                
                if pdf_data is not None and pdf_data.shape == (180, 2):
                    pdf_values = pdf_data[:, 1]  # The PDF values
                    bin_centers = pdf_data[:, 0]  # The bin centers

                    if ap<1:
                        bin_centers = bin_centers + np.pi/2
                        bin_centers = np.where(bin_centers>np.pi/2, bin_centers-np.pi, bin_centers)
                          # Sort the data to avoid crossing boundaries
                        sorted_indices = np.argsort(bin_centers)
                        bin_centers = bin_centers[sorted_indices]
                        pdf_values = pdf_values[sorted_indices]
                    # Plot the histogram as a line plot
                    print(bin_centers, pdf_values)
                    plt.plot(bin_centers, pdf_values, label=f'$\\alpha={ap}$', color=colors[idx], linewidth=2)

    # Customize the plot
    plt.xlabel('$\\theta_x$ [rad]', fontsize=18)
    plt.ylabel('$\\psi[\\theta_x]$', fontsize=18)
    plt.gca().xaxis.set_major_locator(MultipleLocator(base=np.pi/4))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    plt.gca().tick_params(axis='x', labelsize=14)
    plt.gca().tick_params(axis='y', labelsize=14)

    plt.title(f'Circular PDF of Angular Distribution ($\\mu_p={fixed_cof}$, $I={fixed_I}$)', fontsize=16)
    plt.legend(fontsize=16, ncol=2)
    # plt.grid(True)
    
    # Save and show the plot
    plt.savefig(output_dir + "/pdf_thetax_cof_" + str(fixed_cof) + "_I_" + str(fixed_I) + '.png', bbox_inches='tight')
    plt.show()

def plot_ellipsoids_with_combined_data_ap(
    data, fixed_cof, fixed_I, aspect_ratios, variable_keys, operation, label, bins_local
):
    """
    Plot ellipsoids with combined data (from multiple keys) mapped onto their surfaces.

    Args:
    - data: Dictionary containing the simulation data.
    - fixed_cof: The coefficient of friction to be fixed.
    - fixed_I: The inertial number to be fixed.
    - aspect_ratios: List of aspect ratios to loop through.
    - variable_keys: List of keys to access the data for the operation.
    - operation: A function (or lambda) to combine the data from multiple keys.
    - label: Label for the color bar.
    - bins_local: Bin edges for the local data (0 to 90 degrees).
    """
    assert len(bins_local) - 1 > 0, "bins_local must have at least two edges."
    assert callable(operation), "Operation must be a callable function or lambda."
    assert len(variable_keys) > 1, "Provide at least two variable keys for operations."

    # Collect all data to determine global min and max for colormap normalization
    all_combined_data = []
    for ap in aspect_ratios:
        filename = f'simple_shear_ap{ap}_cof_{fixed_cof}_I_{fixed_I}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                file_data = pickle.load(file)
                
                # Extract data for all specified keys
                data_values = []
                for key in variable_keys:
                    hist_data = file_data.get(key, None)
                    if hist_data is not None:
                        data_values.append(hist_data)
                
                if len(data_values) == len(variable_keys):
                    # Combine data using the specified operation
                    combined_data = operation(*data_values)
                    all_combined_data.extend(combined_data)

    # Determine global min and max for normalization
    vmin = min(all_combined_data)
    vmax = max(all_combined_data)

    # Use a consistent colormap
    cmap = plt.cm.GnBu    
    # cmap = plt.cm.YlOrBr
    for ap in aspect_ratios:
        filename = f'simple_shear_ap{ap}_cof_{fixed_cof}_I_{fixed_I}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                file_data = pickle.load(file)

                # Extract data for all specified keys
                data_values = []
                for key in variable_keys:
                    hist_data = file_data.get(key, None)
                    
                    if hist_data is not None:
                        data_values.append(hist_data)

                    if len(data_values) == len(variable_keys):
                        # Combine data using the specified operation
                        combined_data = operation(*data_values)
                        all_combined_data.extend(combined_data)
                        print(f"Aspect ratio: {ap}, Combined data sum: {np.sum(combined_data)}")
                    # Create a grid of angles
                    u = np.linspace(0, 2 * np.pi, 50)  # azimuthal angle
                    v = np.linspace(0, np.pi, 2 * len(bins_local) - 1)  # polar angle based on bins_local

                    # Parametric equations for the ellipsoid
                    x = np.outer(np.sin(v), np.cos(u))
                    y = np.outer(np.sin(v), np.sin(u))
                    z = float(ap) * np.outer(np.cos(v), np.ones_like(u))

                    # Calculate the bin centers
                    bin_centers = (bins_local[:-1] + bins_local[1:]) / 2

                    # Create a color array for the ellipsoid surface
                    colors = np.zeros_like(z)

                    # Normalize hist_data for colormap
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)

                    # Map hist_data onto the ellipsoid surface
                    for i in range(len(bin_centers)):
                        # Upper hemisphere (0 to 90 degrees)
                        indices_upper = (v >= np.radians(bins_local[i])) & (v < np.radians(bins_local[i + 1]))
                        colors[indices_upper, :] = combined_data[i] 
                       
                        # Mirror to the lower hemisphere (90 to 180 degrees)
                        indices_lower = (v >= np.radians(180 - bins_local[i + 1])) & (v < np.radians(180 - bins_local[i]))
                        colors[indices_lower, :] = combined_data[i]

                    # Plot the ellipsoid
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='3d')

                    # Use a colormap to plot the surface
                    ax.plot_surface(x, y, z, facecolors=cmap(norm(colors)), rstride=1, cstride=1, antialiased=True, alpha=1.0)

                    # Set the view and remove axis labels for a cleaner look
                    ax.set_aspect('equal')
                    ax.axis('off')
                    ax.view_init(elev=-26, azim=-6, roll=30)

                    # Add alpha value as a legend
                    ax.text2D(0.05, 0.95, f"$\\alpha = {ap}$", transform=ax.transAxes, fontsize=14)

                    # Add color bar
                    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    mappable.set_array(hist_data)
                    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
                    cbar.ax.tick_params(labelsize=20)
                    cbar.set_label(label=label, fontsize=20)

                    # Save the figure
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(f'{output_dir}/{label}_ellipsoid_ap_{ap}_cof_{fixed_cof}_I_{fixed_I}.png',
                                bbox_inches='tight', dpi=300)
                    plt.close()



# Force histograms to be normalized by p * A
histogram_keys = [
    'global_normal_force_hist',  # normal force average by its global direction, if present
    'global_tangential_force_hist',# tangential force average by its global direction, if present
    'global_normal_force_hist_cp',  # normal force average by its contact point direction in the global frame
    'global_tangential_force_hist_cp'# tangential force average by its contact point direction in the global frame
]

# PDF histograms (no normalization needed)
pdf_keys = [
    'contacts_hist_global_normal',  
    'contacts_hist_global_tangential',
    'contacts_hist_cont_point_global',  
    'contacts_hist_cont_point_local'
]

# Local force histograms to be normalized by p * A
local_histogram_keys = [
    'local_normal_force_hist_cp',  
    'local_tangential_force_hist_cp',
    'contacts_hist_cont_point_local'
]

ellipsoids_keys = ['contacts_hist_cont_point_local', 'total_area']
operation = lambda local_hist, area:  local_hist / max(local_hist)

# ellipsoids_keys = ['local_tangential_force_hist_cp', 'local_normal_force_hist_cp']

# ellipsoids_keys = ['power_dissipation_normal', 'power_dissipation_tangential', 'total_area', 'area_adjustment_ellipsoid', 'total_normal_dissipation', 'total_tangential_dissipation', 'contacts_hist_cont_point_local']
# operation = lambda normal_hist, tangential_hist,  total_area, area_adjustment, total_normal, total_tangential, pdf_contact: (
#     (normal_hist + tangential_hist)* pdf_contact/2 * total_area /
#     (area_adjustment * (total_normal + total_tangential))
# )

# ellipsoids_keys = ['power_dissipation_normal', 'power_dissipation_tangential', 'total_area', 'area_adjustment_ellipsoid', 'total_normal_dissipation', 'total_tangential_dissipation', 'contacts_hist_cont_point_local']
# operation = lambda normal_hist, tangential_hist,  total_area, area_adjustment, total_normal, total_tangential, pdf_contact: (
#     area_adjustment
# )

ellipsoids_keys = ['local_normal_force_hist_cp', 'total_area', 'p_yy']
operation = lambda normal_hist, total_area, pyy: ( normal_hist / (total_area * pyy))

# ellipsoids_keys = ['contacts_hist_cont_point_local', 'area_adjustment_ellipsoid']
# operation = lambda local_hist, area: 2* area *local_hist 

# ellipsoids_keys = ['power_dissipation_normal', 'power_dissipation_tangential', 'contacts_hist_cont_point_local', 'total_normal_dissipation', 'total_tangential_dissipation']
# operation = lambda normal_hist, tangential_hist, pdf_events, total_normal, total_tangential: (
#     (normal_hist + tangential_hist) * pdf_events/
#     ((total_normal + total_tangential))
# )

# ellipsoids_keys = ['power_dissipation_normal', 'power_dissipation_tangential']
# operation = lambda normal_hist, tangential_hist,: (
#     (normal_hist + tangential_hist)/ np.sum(normal_hist + tangential_hist)
# )


# Create the plots
cof = 10.0
I = 0.1

# create_polar_plots_varying_ap(data, fixed_cof=cof, fixed_I=I, histogram_keys=histogram_keys, pdf_keys=pdf_keys, local_histogram_keys=local_histogram_keys)
# plot_fabric_eigenvectors_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios)
bins_local = np.linspace(0, 90, 10)  # Define your bins for local data (10 values)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=lambda x, y: x / y, label='Mobilized Friction', bins_local=bins_local)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=operation, label='Surface Pdf', bins_local=bins_local)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=operation, label='Total_Power_dissipation', bins_local=bins_local)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=operation, label='surface_pdf_normalized_max', bins_local=bins_local)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=operation, label='surface_pdf_normalized_max', bins_local=bins_local)

create_plots(data)

# plot_pdf_thetax(fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios)

os.chdir("..")

# print("Plots saved to parametric_plots")
