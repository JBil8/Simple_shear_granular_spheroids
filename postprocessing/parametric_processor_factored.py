import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from matplotlib.patches import Ellipse
from matplotlib.transforms import ScaledTranslation
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

np.set_printoptions(legacy='1.25')
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}'

def exp_corr_strain(x, gamma_v, eta):
    return (1 + (x/gamma_v)**1.0)**(-eta)

def exp_corr_length(x, length):
    return np.exp(-x/length)

def phiI(x, phic, cphi, betaphi):
    return phic - cphi * x**betaphi

def muI(x, muc, cmu, I0):
    return muc + cmu / (1+ (I0/x))

def power_law(x, a, b):
    return a * x**b

# Set the ticks as functions of pi
def pi_formatter(x, pos):
    fractions = {0: '0', np.pi/4: r'$\pi/4$', np.pi/2: r'$\pi/2$', 
                 3*np.pi/4: r'$3\pi/4$', np.pi: r'$\pi$'}
    return fractions.get(x, f'{x/np.pi:.2g}π')

def coordination_I(x, Z_c, c, a):
    return Z_c - c*I**a

def create_colormap_ap(aspect_ratios, central_ap=1.0, central_color='black', colormap=plt.cm.RdYlBu):
    """
    Create a colormap with a unique color for the central aspect ratio.

    Parameters:
    - aspect_ratios (list): List of aspect ratios.
    - central_ap (float): The central aspect ratio value (default is 1.0).
    - central_color (str): The color to use for the central aspect ratio (default is 'black').
    - colormap: The colormap to use (default is plt.cm.RdYlBu).

    Returns:
    - colors (list): List of colors corresponding to the aspect ratios.
    """
    num_colors = len(aspect_ratios)
    
    # Generate extreme indices for the colormap
    extreme_indices = np.concatenate([
        np.linspace(0, 0.3, num_colors // 2, endpoint=False),  # Lower 30%
        np.linspace(0.7, 1.0, num_colors - num_colors // 2)    # Upper 30%
    ])
    
    # Get colors from the colormap
    colors = [colormap(i) for i in extreme_indices]
    
    if central_ap in aspect_ratios:
        # Identify the index of the central aspect ratio
        central_index = aspect_ratios.index(central_ap)
        
        # Insert the central color
        colors.insert(central_index, central_color)

    return colors

def load_data(aspect_ratios, cofs, Is, keys_of_interest):
    # Initialize a data holder
    data = {key: [] for key in keys_of_interest}
    data['inertialNumber'] = []
    data['cof'] = []
    data['ap'] = []
    data['shear_rate'] = [] 
    data['I_nominal'] = []
    data['box_size'] = []
    data['pressure_yy'] = []
    data['pressure_xy'] = []
    # Load the data from files
    for ap in aspect_ratios:
        for cof in cofs:
            for I in Is:
                filename = f'simple_shear_ap{ap}_cof_{cof}_I_{I}.pkl'
                if os.path.exists(filename):
                    with open(filename, 'rb') as file:
                        file_data = pickle.load(file)
                        print(file_data.keys())
                        inertial_number = file_data.get('inertialNumber', None)
                        if inertial_number is not None:
                            # if the measured inertial number relative to the nominal one is more than 10% off, skip the data
                            if abs(inertial_number - I) / I > 0.08:
                                continue
                            else:
                                data['inertialNumber'].append(inertial_number)
                                data['I_nominal'].append(I)
                                data['cof'].append(cof)
                                data['ap'].append(ap)
                                data['shear_rate'].append(file_data.get('shear_rate', None))  # Append shear_rate
                                data['box_size'].append(file_data.get('box_x_length', None)*file_data.get('box_z_length', None)*file_data.get('box_y_length', None))
                                data['pressure_yy'].append(file_data.get('p_yy', None))   
                                data['pressure_xy'].append(file_data.get('p_xy', None))
                        for key in keys_of_interest:
                            data[key].append(file_data.get(key, None))
                else:
                    print(f"File {filename} does not exist.")
    return data

# Define the parameters
cofs = [0.0, 0.4, 10.0]
cofs = [0.0, 0.001, 0.01, 0.1, 0.4, 1.0, 10.0]
# aspect_ratios = [1.0]
Is = [0.1, 0.046, 0.022, 0.01, 0.0046, 0.0022, 0.001] 
aspect_ratios = [0.33, 0.40, 0.50, 0.56, 0.67, 0.83, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

Y_modulus = 5.0e6  # Young's modulus in Pa
# aspect_ratios = [0.33, 0.50, 0.83, 1.0,  1.2,  2.0, 3.0]
# aspect_ratios = [0.33, 0.50, 1.0,  1.5, 3.0]

# Is = [0.1]
# cofs = [0.4, 0.4]
# aspect_ratios = [1.0]


# Is = [0.01]
# cofs =[0.4]
# aspect_ratios = [0.50, 0.67, 1.0, 1.5, 2.0]
# aspect_ratios = [0.33, 3.0]
aspect_ratios_to_show = aspect_ratios
keys_of_interest = ['thetax_mean', 'percent_aligned', 'S2', 'Z', 'phi', 'Nx_diff', 'Nz_diff',
                     'Omega_z', 'p_yy', 'p_xy', 'Dyy', 'Dzz', 'total_normal_dissipation',
                       'total_tangential_dissipation', 'percent_sliding', 'vx_fluctuations', 'vy_fluctuations', 'vz_fluctuations',
                       'S2', 'c_delta_vy', 'c_r_values', 'shear_stress_normal', 'shear_stress_tangential', 
                       'box_x_length', 'box_y_length', 'box_z_length']      # 'strain', 'auto_corr_vel', 'auto_corr_omega']
# keys_of_interest = ['thetax_mean', 'percent_aligned', 'S2', 'Z', 'phi', 'Nx_diff', 'Nz_diff',
#                      'Omega_z', 'p_yy', 'p_xy', 'Dyy', 'Dzz', 'percent_sliding', 'vx_fluctuations', 'vy_fluctuations', 'vz_fluctuations',
#                        'S2', 'c_delta_vy', 'c_r_values']
keys_of_interest = ['vx_fluctuations', 'vy_fluctuations', 'vz_fluctuations', 'rke', 'tke', 'vx_div_vy', 'vx_div_vz', 'rke_div_tke']
os.chdir("./output_data_dataset_v2")
# os.chdir("./output_data_26_02_2025")
# os.chdir("./output_data_dt_0.08")

data = load_data(aspect_ratios, cofs, Is, keys_of_interest)
output_dir = "../parametric_plots_v2"
# output_dir = "../parametric_plots_dt_0.08"
os.makedirs(output_dir, exist_ok=True)


# Function to create plotsn
def create_plots(data):

    # allocate list for critical values of mu
    mu_c = []

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create a colormap for the aspect ratios
    colors = create_colormap_ap(aspect_ratios, central_ap=1.0, central_color='black', colormap=plt.cm.RdYlBu)

    colormap_cof = plt.cm.viridis
    num_colors_cof = len(cofs)
    colors_cof = [colormap_cof(i) for i in np.linspace(0, 1, num_colors_cof)]

    # Create a figure for the subplots
    fig, axes = plt.subplots(1, len(cofs), figsize=(10, 7), sharey=True)

    if 'p_xy' in keys_of_interest and 'p_yy' in keys_of_interest:
        intial_guess = [1.0, 1.0, 1.0]
        for i, cof in enumerate(cofs):
            x_fit = np.logspace(-3.2, -0.8, 100)
            # Plot mu = p_xy / p_yy
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111)
            for ap, color in zip(aspect_ratios, colors):
                x_values = [inertial_number*3/(ap+2)*ap**(1/3) for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                p_xy_values = [value for value, aspect_ratio, coef in zip(data['p_xy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                if x_values and p_xy_values and p_yy_values:  # Ensure that the lists are not empty
                    mu_values = [pxy / pyy if pyy != 0 else None for pxy, pyy in zip(p_xy_values, p_yy_values)]
                if ap in aspect_ratios_to_show:
                    plt.plot(x_values, mu_values, label=f'$\\alpha={ap}$', color=color, linestyle='None', marker='o')
                    popt, pcov = curve_fit(muI, x_values, mu_values, p0=intial_guess, method='trf', x_scale = [1, 1, 1], bounds=([0, 0, 0], [1, 1, 1]))
                    # print(f"cof={cof}, ap={ap}, popt={popt}")
                    mu_c.append(popt[0])
                    plt.plot(x_fit, muI(x_fit, *popt), color=color, linestyle='--')
            plt.xscale('log')
            # plt.title(f"$\\mu_p={cof}$", fontsize=20)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.legend(fontsize=30, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)
            plt.xlabel('$I$', fontsize=30)
            plt.ylabel('$\\mu = \\sigma_{xy} /\\sigma_{yy}$', fontsize=30)
            filename = f'mu_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()
            ax = axes[i]
            for ap, color in zip(aspect_ratios, colors):
                x_values = [inertial_number*3/(ap+2)*ap**(1/3) for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                p_xy_values = [value for value, aspect_ratio, coef in zip(data['p_xy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                
                if x_values and p_xy_values and p_yy_values:  # Ensure that the lists are not empty
                    mu_values = [pxy / pyy if pyy != 0 else None for pxy, pyy in zip(p_xy_values, p_yy_values)]
                
                if ap in aspect_ratios_to_show:
                    ax.plot(x_values, mu_values, label=f'$\\alpha={ap}$', color=color, linestyle='None', marker='o')
                    popt, pcov = curve_fit(muI, x_values, mu_values, p0=intial_guess, method='trf', x_scale=[1, 1, 1], bounds=([0, 0, 0], [1, 1, 1]))
                    # print(f"cof={cof}, ap={ap},mu_popt={popt}")
                    # print(x_values, mu_values)
                    # mu_c.append(popt[0])
                    ax.plot(x_fit, muI(x_fit, *popt), color=color, linestyle='--', linewidth=2)
       
           # Set the title, labels, and other properties for the subplot
            ax.set_xscale('log')
            # ax.set_title(f"$\\mu_p={cof}$", fontsize=30)
            # ax.set_xlabel('$I$', fontsize=30)
            # ax.set_xticks([])
            ax.set_ylabel('$\\mu = \\sigma_{xy} /P $', fontsize=30 if i == 0 else 0)  # Only show ylabel on the first plot
            ax.tick_params(axis='both', which='major', labelsize=30, length=8, width=2)
        # ax.legend(fontsize=15, loc='upper right', bbox_to_anchor=(1.0, 1.35), ncols=7)

        # Adjust layout to make sure subplots fit without overlap
        # plt.tight_layout()
        for ax in axes:
            ax.tick_params(axis='both', length=8, width=2)
        # Save the figure
        filename = 'mu_cof_subplots.png'
        # plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()

    phi_c = []

    # Plot other keys
    for key in keys_of_interest:
        if key in ['p_yy', 'p_xy']:
            continue  # Skip p_yy and p_xy since we already used them to calculate mu
        elif key == 'percent_aligned':
            label = '$\\%_z$'
        elif key == 'thetax_mean':
            label = '$\\theta_x \\, [^\\circ]$'
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
            label = '$\\delta \\omega / \\dot{\\gamma}$'
            # label = '$\\delta \\omega/ \\dot{\\gamma}$'
        elif key == 'percent_sliding':
            label = '$\\chi$'
        elif key == 'tke':
            label = '$\\delta E_{transl}/\Omega$'
        elif key == 'rke':
            label = '$\\delta E_{rot}/\Omega$'
        elif key == 'c_delta_vy' or key == 'c_r_values' or key == "c_delta_omega_z" or key == 'strain' or key == 'auto_corr_vel' or key == 'auto_corr_omega' or key == 'shear_stress_normal' or key == 'shear_stress_tangential' or key == 'total_normal_dissipation' or key == 'total_tangential_dissipation' or key == 'c_y_values' or key == 'c_density_y':
            continue  # Skip these keys for now
        else: 
            label = key


        # fig, axes = plt.subplots(1, len(cofs), figsize=(15, 4), sharey=True)
        fig, axes = plt.subplots(1, len(cofs), figsize=(10, 7), sharey=True)
        for i, cof in enumerate(cofs):
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111)   
            # plt.xticks(fontsize=20)
            # plt.xscale('log')
            # plt.yscale('log')

            for ap, color in zip(aspect_ratios, colors):
                # if ap <= 1.0:
                #     ax = axes[0, i]
                # elif ap >= 1.0:
                #     ax = axes[1, i]   
                ax = axes[i]
                x_values = [inertial_number*3/(ap+2)*ap**(1/3) for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
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
                    # plt.yscale('log')
                elif key == 'percent_aligned':
                    y_values = [value * 100 for value in y_values] # Convert to percentageS
                elif key == 'Omega_z':
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [-2*value / shear_rate for value, shear_rate in zip(y_values, shear_rate_values)]
                    # plt.yscale('log')
                    # plt.ylim(-0.1, 1.1)
                elif key == 'vx_fluctuations' or key == 'vy_fluctuations' or key == 'vz_fluctuations':
                    d_eq = 2*ap**(1/3)
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    # print(shear_rate_values)
                    y_values = [value / (shear_rate*d_eq) for value, shear_rate in zip(y_values, shear_rate_values)]
                    # if key == 'vx_fluctuations':
                    #     y_values = [value**2 for value in y_values]
                    # if ap> 1:
                    #     y_values = [value*ap**(1/3) for value in y_values]
                    # else:
                    #     y_values = [value*ap**(-1/3) for value in y_values]
                    plt.yscale('log')
                    # if ap == 1.0:
                    #     # plot best fit line in loglog space
                    #     slope, intercept, r_value, p_value, std_err =  linregress(np.log(x_values), np.log(y_values))
                    #     # plt.plot(x_values, np.exp(intercept) * x_values**slope, color=color, linestyle='--', linewidth=2, label=f"$\\sim I^{{{slope:.2f}}}$")
                    #     # plt.plot(x_values, 3**(0.5)*np.exp(intercept) * x_values**slope, color=color, linestyle='--', linewidth=2, label=f"$\\sim \\alpha^{{1/2}} I^{{{slope:.2f}}}$")
                    #     ax.plot(x_values, np.exp(intercept) * x_values**slope, color=color, linestyle='--', linewidth=2, label=f"$\\sim I^{{{slope:.2f}}}$")
                    #     # ax.plot(x_values, 3**(0.5)*np.exp(intercept) * x_values**slope, color=color, linestyle='--', linewidth=2, label=f"$\\sim \\alpha^{{1/2}} I^{{{slope:.2f}}}$")

                elif key == "omega_fluctuations":
                    if cof == 0.0 and ap == 1.0:
                        continue
                    # print(f"key={key}, ap={ap}, cof={cof}")
                    # print(f"omega_fluctuations={y_values}")
                    # y_values_new = [value[2] for value in y_values] # not need to take sqrt it is already the fluctuation
                    y_values_new = [np.linalg.norm(value) for value in y_values]
                    # y_values_new = [value[0] for value in y_values] 
                    y_values = y_values_new
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    # y_values = [1*value*max(ap, 1/ap)**(2.3) / shear_rate for value, shear_rate in zip(y_values, shear_rate_values)]
                    y_values = [1*value / shear_rate for value, shear_rate in zip(y_values, shear_rate_values)]
                    ax.set_yscale('log')
                    ax.set_xscale('log')
                    if ap == 1.0 and cof != 0.0:
                        # plot best fit line in loglog space
                        slope, intercept, r_value, p_value, std_err =  linregress(np.log(x_values), np.log(y_values))
                        ax.plot(x_values, np.exp(intercept) * x_values**slope, color=color, linestyle='-', linewidth=2, label=f"$\\sim I^{{{slope:.2f}}}$")
                        axes[1, i].plot(x_values, np.exp(intercept) * x_values**slope, color=color, linestyle='-', linewidth=2, label=f"$\\sim I^{{{slope:.2f}}}$", zorder=10)
                    if ap == 1.5 and cof == 0.0:
                        # plot best fit line in loglog space
                        slope, intercept, r_value, p_value, std_err =  linregress(np.log(x_values), np.log(y_values))
                        # plt.plot(x_values, np.exp(intercept) * x_values**slope, color=color, linestyle='--', linewidth=2, label=f"$\\sim I^{{{slope:.2f}}}$")
                        # plt.plot(x_values, np.exp(intercept) * x_values**slope/((3/1.5)**2), color=color, linestyle='--', linewidth=2, label=f"$\\sim \\alpha^{{-1.5}} I^{{{slope:.2f}}}$")
                        ax.plot(x_values, np.exp(intercept) * x_values**slope, color='black', linestyle='-', linewidth=2, label=f"$\\sim I^{{{slope:.2f}}}$")
                        # ax.plot(x_values, np.exp(intercept) * x_values**slope/((3/1.5)**1.5), color='gray', linestyle='--', linewidth=2, label=f"$\\sim \\alpha^{{-1.5}} I^{{{slope:.2f}}}$")

                elif key == 'rke' or key == 'tke':
                    volume_values = [volume for volume, aspect_ratio, coef in zip(data['box_size'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [value / volume for value, volume in zip(y_values, volume_values)]
                    plt.yscale('log')
                    if key == 'rke' and ap == 1.0 and cof == 0.0:
                        continue
                
                elif key == 'Z':
                    ax.axhline(3, color='black', linestyle='--', linewidth=1, label='$Z=3$')
                    ax.axhline(10, color='black', linestyle=':', linewidth=1, label='$Z=4$')
                    ax.axhline(6, color='black', linestyle='-.', linewidth=1, label='$Z=6$')
                    ax.set_xscale('log')
                    ax.set_xlim(0.00047, 0.21)

                elif key == 'percent_sliding':
                    plt.yscale('log')
                if x_values and y_values :  # Ensure that the lists are not empty
                    if key == 'phi':
                        # plt.plot(x_values, y_values, label=f'$\\alpha={ap}$', color=color, linestyle='--')
                        bounds = ([0, 0, 0], [0.75, 100, 100])
                        intial_guess = [0.6, 0.5, 0.5]
                        popt, pcov = curve_fit(phiI, x_values, y_values, p0=intial_guess, maxfev=20000, bounds=bounds)
                        if pcov[0, 0] < 1e-3:
                            phi_c.append(popt[0])
                        x_fit = np.logspace(-3.2, -0.8, 100)
                        if ap == 1.0:
                            ax.plot(x_fit, phiI(x_fit, *popt), color=color, linestyle='--', linewidth=2)
                        else:
                            ax.plot(x_fit, phiI(x_fit, *popt), color=color, linestyle='--', linewidth=2)
                        ax.plot(x_values, y_values, label=f'$\\alpha={ap}$', color=color, linestyle='None', marker='o')
                        ax.set_xscale('log')

                    else:
                        ax.plot(x_values, y_values, label=f'$\\alpha={ap}$', color=color, linestyle=':', marker='o', markersize=4, linewidth=2)   
                        ax.set_xscale('log') 

            # plt.xticks(fontsize=20)
            # plt.yticks(fontsize=20)
            # plt.xlabel('$I$', fontsize=20)
            # plt.ylabel(label, fontsize=20)
            # # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)   
            # filename = f'{key}_cof_{cof}.png'
            # plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            # plt.close()

            # Set the title, labels, and other properties for the subplot
            # ax.set_xscale('log')
            # for j, ax in enumerate(axes[0]):
            #     cof = cofs[j]
            #     ax.set_title(f"$\\mu_p={cof}$", fontsize=20)
            #     ax.set_xticklabels([])

            # # ax.set_xticks([])
            # # ax.set_xlabel('$I$', fontsize=30)
            # for ax in axes[:, 0]:
            #     ax.set_ylabel(label, fontsize=20) 
            for ax in np.ravel(axes):
                ax.tick_params(axis='both', which='major', labelsize=30, length=8, width=2)
        axes[0].set_ylabel(label, fontsize=30)  # Set ylabel for the first subplot only
            # ax.legend(fontsize=15, loc='upper right', bbox_to_anchor=(1, 2), ncols=1)
            
        # Save the figure
        filename = f'{key}_subplots.png'
        fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
       
    if 'rke' in keys_of_interest and 'tke' in keys_of_interest:
        # plot the ratio of rke to tke
        fig, axes = plt.subplots(1, len(cofs), figsize=(15, 4), sharey=True)
        for i, cof in enumerate(cofs):
            # plt.figure(figsize=(10, 8))
            ax = axes[i]
            for ap, color in zip(aspect_ratios, colors):
                x_values = [inertial_number*3/(ap+2)*ap**(1/3) for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                rke_values = [value for value, aspect_ratio, coef in zip(data['rke'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                tke_values = [value for value, aspect_ratio, coef in zip(data['tke'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                if x_values and rke_values and tke_values:
                    if ap == 1.0 and cof == 0.0:
                        continue
                    ratio = [rke / tke for rke, tke in zip(rke_values, tke_values)]
                    ax.plot(x_values, ratio, label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f"$\\mu_p={cof}$", fontsize=15)
            ax.set_xlabel('$I$', fontsize=15)
            ax.set_ylabel('$\\delta E_{rot} / \\delta E_{transl}$', fontsize=15 if i == 0 else 0)  # Only show ylabel on the first plot
            ax.tick_params(axis='both', which='major', labelsize=15, length=8, width=2)
            # plt.xscale('log')
            # plt.xticks(fontsize=20)
            # plt.yticks(fontsize=20)
            # # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)
            # plt.xlabel('$I$', fontsize=20)
            # plt.ylabel('$\delta E_{rot} / \delta E_{transl}$', fontsize=20)
            # plt.title(f'$\\mu_p={cof}$', fontsize=20)
            # filename = f'rke_tke_ratio_cof_{cof}.png'
            # plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            # plt.close
        # ax.legend(fontsize=15, loc='upper right', bbox_to_anchor=(2, 1.2), ncols=1)
        # Save the figure
        filename = f'rke_tke_subplots.png'
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300) 

    # if 'vx_fluctuations' in keys_of_interest and 'vy_fluctuations' in keys_of_interest and 'vz_fluctuations' in keys_of_interest:
    #     # plot the magnitude of the velocity fluctuations taken as the square root of the sum of the squares of the components
    #     # normalized by the shear rate and the equivalent diameter
    #     fig, axes = plt.subplots(2, len(cofs), figsize=(10, 4), sharey=True, sharex=True)

    #     for i, cof in enumerate(cofs):
    #         for ap, color in zip(aspect_ratios, colors):
    #             if ap <= 1.0:
    #                 ax = axes[0, i]
    #             elif ap >= 1.0:
    #                 ax = axes[1, i]

    #             x_values = [inertial_number*3/(ap+2)*ap**(1/3) for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
    #             vx_values = [value for value, aspect_ratio, coef in zip(data['vx_fluctuations'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
    #             vy_values = [value for value, aspect_ratio, coef in zip(data['vy_fluctuations'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
    #             vz_values = [value for value, aspect_ratio, coef in zip(data['vz_fluctuations'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
    #             if x_values and vx_values and vy_values and vz_values:
    #                 d_eq = ap**(1/3) 
    #                 shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
    #                 v_value = [np.sqrt(vx**2 + vy**2 + vz**2) for vx, vy, vz in zip(vx_values, vy_values, vz_values)]
    #                 v_values = [value / (shear_rate*d_eq) for value, shear_rate in zip(v_value, shear_rate_values)]
    #                 # v_values = [vx_values/vy_values for vx_values, vy_values in zip(vx_values, vy_values)]
    #                 ax.loglog(x_values, v_values, label=f'$\\alpha={ap}$', color=color, linestyle=':', marker='o')
    #                 ax.set_xscale('log')
    #                 if ap == 1.0:
    #                     # plot best fit line in loglog space
    #                     slope, intercept, r_value, p_value, std_err =  linregress(np.log(x_values), np.log(v_values))
    #                     ax.plot(x_values, np.exp(intercept) * x_values**slope, color=color, linestyle='-', linewidth=2, label=f"$\\sim I^{{{slope:.2f}}}$")
    #                     # axes[1, i].loglog(x_values, v_values, label=f'$\\alpha={ap}$', color=color, linestyle=':', marker='o', zorder=10)   
    #                     # if cof == 0.0:
    #                     #     ax.plot(x_values, 3**(-0.33)*np.exp(intercept) * x_values**slope, color='gray', linestyle='-', linewidth=2, label=f"$\\sim \\alpha^{{-1/4}} I^{{{slope:.2f}}}$")
    #         # ax.set_xscale('log')
    #         # ax.set_yscale('log')
    #         for j, ax in enumerate(axes[0]):
    #             cof = cofs[j]
    #             ax.set_title(f"$\\mu_p={cof}$", fontsize=20)
    #             ax.set_xticklabels([])

    #         # ax.set_xticks([])
    #         # ax.set_xlabel('$I$', fontsize=30)
    #         label = '$\\delta v / \\dot{\\gamma}d_{eq}$'
    #         for ax in axes[:, 0]:
    #             ax.set_ylabel(label, fontsize=20) 
    #         for ax in np.ravel(axes):
    #             ax.tick_params(axis='both', which='major', labelsize=20, length=8, width=2)


        # # Save the figure
        # filename = f'v_fluctuations_subplots.png'
        # plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        # plt.close()   

        # plot the velocity fluctuations wrt alpha for fixed I
        # fig, axes = plt.subplots(1, len(cofs), figsize=(15, 4), sharey=True)
        # for i, cof in enumerate(cofs):
        #     ax = axes[i]
        #     for I_target in Is:  # Example fixed I values
        #         alphas = []
        #         v_values = []
        #         for ap in aspect_ratios:
        #             # Find all data points matching the current aspect ratio and coefficient of friction
        #             filtered = [
        #                 (inertial_number, vx, vy, vz, shear_rate)
        #                 for inertial_number, a, c, vx, vy, vz, shear_rate in zip(
        #                     data['inertialNumber'], data['ap'], data['cof'],
        #                     data['vx_fluctuations'], data['vy_fluctuations'], data['vz_fluctuations'],
        #                     data['shear_rate']
        #                 )
        #                 if a == ap and c == cof and inertial_number is not None and abs(inertial_number - I_target) < 0.001
        #             ]
        #             if filtered:
        #                     # Compute and collect individual normalized values
        #                     norm_values = []
        #                     for vx, vy, vz, shear_rate in [(vx, vy, vz, shear) for _, vx, vy, vz, shear in filtered]:
        #                         # print(f"vx={vx}, vy={vy}, vz={vz}, shear_rate={shear_rate}")
        #                         delta_v = np.sqrt(vx**4 + vy**2 + vz**2)
        #                         d_eq = 2*ap**(1/3)
        #                         v_normalized = delta_v / (shear_rate * d_eq)
        #                         norm_values.append(v_normalized)

        #                     # Take mean of normalized values if multiple match
        #                     v_values.append(np.mean(norm_values))
        #                     alphas.append(ap)
        #         if alphas:
        #             ax.plot(alphas, v_values, marker='o', linestyle='-', label=f"$I={I_target}$")
        #             #PLOT BEST FIT LINE
        #             alphas_prolate = [ap for ap in alphas if ap > 1]
        #             v_values_prolate = [v for ap, v in zip(alphas, v_values) if ap > 1]
        #             slope, intercept, r_value, p_value, std_err =  linregress(np.log(alphas_prolate),np.log(v_values_prolate))
        #             # print(f"cof={cof}, I_target={I_target}, slope={slope}")
        #             ax.plot(alphas_prolate, np.exp(intercept) * np.array(alphas_prolate)**slope, color='black', linestyle='--', linewidth=2, label=f"$\\sim \\alpha^{{{slope:.2f}}}$")  

        #     ax.set_xscale('log')
        #     ax.set_yscale('log')
        #     ax.set_title(f"$\\mu_p={cof}$", fontsize=15)
        #     ax.set_xlabel('$\\alpha$', fontsize=15)
        #     if i == 0:
        #         ax.set_ylabel('$\\delta v / \\dot{\\gamma}d$', fontsize=15)
        #     ax.tick_params(axis='both', which='major', labelsize=15, length=8, width=2)
        #     ax.legend(fontsize=12)

        # # Save the plot
        # filename = f'v_fluctuations_vs_alpha_fixed_I.png'
        # plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        # plt.close()
 
    if 'phi' in keys_of_interest and 'p_xy' in keys_of_interest and 'p_yy' in keys_of_interest:
        # Plot mu_c and phi_c vs alpha
        fig, axes = plt.subplots(2, figsize=(7, 8), sharex=True)
        fig.subplots_adjust(hspace=0.1)

        # split lists by number of aspect ratios
        phi_c = [phi_c[i:i + len(aspect_ratios)] for i in range(0, len(phi_c), len(aspect_ratios))]
        mu_c  = [mu_c[i:i + len(aspect_ratios)]  for i in range(0, len(mu_c), len(aspect_ratios))]

        #read csv file as nump array, skip first row
        phi_nagy = np.genfromtxt('../nagy_phi_c.csv', delimiter=',', skip_header=1)
        mu_nagy = np.genfromtxt('../nagy_mu_c.csv', delimiter=',', skip_header=1)


        for i, (mu_vals, phi_vals) in enumerate(zip(mu_c, phi_c)):
            cof = cofs[i]
            aps = aspect_ratios

            aps_phi, phi_vals_plot = aps, phi_vals
            if np.isclose(cof, 1.0):
                filtered = [(ap, val) for ap, val in zip(aps, phi_vals) if not np.isclose(ap, 0.4)]
                if filtered:
                    aps_phi, phi_vals_plot = zip(*filtered)

            aps_mu, mu_vals_plot = aps, mu_vals

            # Always plot mu_c
            axes[0].plot(aps_mu, mu_vals_plot, label=f'$\\mu_p={cof}$',
                        color=colors_cof[i], linestyle='--', marker='o', linewidth=3)

            # Plot phi_c unless cof == 10.0
            if not np.isclose(cof, 10.0):
                axes[1].plot(aps_phi, phi_vals_plot, label=f'$\\mu_p={cof}$',
                            color=colors_cof[i], linestyle='--', marker='o', linewidth=3)

            # Each cof uses columns 2*i (x) and 2*i + 1 (y)
            if 2*i + 1 < mu_nagy.shape[1]:
                x_nagy_mu = mu_nagy[:, 2*i]
                y_nagy_mu = mu_nagy[:, 2*i + 1]
                # Filter NaNs if any
                mask = ~np.isnan(x_nagy_mu) & ~np.isnan(y_nagy_mu)
                axes[0].plot(x_nagy_mu[mask], y_nagy_mu[mask], label=f'Nagy $\\mu_p={cof}$',
                            color=colors_cof[i], linestyle='-', marker='x', linewidth=2)

            # Similarly, plot Nagy data for phi plot (axes[1]) if cof != 10.0
            if not np.isclose(cof, 10.0) and 2*i + 1 < phi_nagy.shape[1]:
                x_nagy_phi = phi_nagy[:, 2*i]
                y_nagy_phi = phi_nagy[:, 2*i + 1]
                mask = ~np.isnan(x_nagy_phi) & ~np.isnan(y_nagy_phi)
                axes[1].plot(x_nagy_phi[mask], y_nagy_phi[mask], label=f'Nagy $\\mu_p={cof}$',
                            color=colors_cof[i], linestyle='-', marker='x', linewidth=2)

        # Donev data on bottom plot
        data_donev = np.genfromtxt('../donev_jamming_packing_fraction.csv', delimiter=',')
        alphas_donev = data_donev[:, 0]
        phi_c_donev = data_donev[:, 1]

        prolate_donev = np.where(alphas_donev > 1.0)[0]
        alphas_donev = alphas_donev[prolate_donev]
        phi_c_donev = phi_c_donev[prolate_donev]

        axes[1].plot(alphas_donev, phi_c_donev, label='RCP', color='black',
                    linestyle='-', marker='o', linewidth=3, markersize=10, markerfacecolor='none')

        # Formatting
        for ax in axes:
            ax.tick_params(labelsize=20, length=8, width=2)

        axes[1].set_xlabel('$\\alpha$', fontsize=20)
        axes[0].set_ylabel('$\\mu_c$', fontsize=20)
        axes[1].set_ylabel('$\\phi_c$', fontsize=20)
        # axes[1].legend(fontsize=16, loc='upper right')  # optional
        # axes[1].set_xticks([0.33, 1.0, 2.0, 3.0])
        axes[1].set_xlim(0.9, 3.1)
        axes[0].set_xlim(0.9, 3.1)
        

        filename = 'phi_mu_c_vs_alpha.png'
        fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()



    if 'omega_fluctuations' in keys_of_interest and 'vy_fluctuations' in keys_of_interest:
        # print the ratio of omega times d over v
        for cof in cofs:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                x_values = [inertial_number*3/(ap+2)*ap**(1/3) for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                omega_values = [value for value, aspect_ratio, coef in zip(data['omega_fluctuations'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                vy_values = [value for value, aspect_ratio, coef in zip(data['vy_fluctuations'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                if x_values and omega_values and vy_values:
                    if ap > 1:
                        d_eq = ap # longest dimension for prolate
                    else:
                        d_eq = 1
                    d_eq = 1
                    # ratio = [delta_omegaz[2]*d_eq / delta_vy for delta_omegaz, delta_vy in zip(omega_values, vy_values)]
                    ratio = [np.linalg.norm(delta_omegaz)*d_eq / delta_vy for delta_omegaz, delta_vy in zip(omega_values, vy_values)]
                    plt.plot(x_values, ratio, label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')
            plt.xscale('log')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)
            plt.xlabel('$I$', fontsize=20)
            plt.ylabel('$\\delta \\omega_z \\ell / \\delta v_y$', fontsize=20)
            plt.title(f'$\\mu_p={cof}$', fontsize=20)
            filename = f'omega_vy_ratio_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

    I_fixed = 0.01
    # Find indices where data['I_nominal'] matches I_fixed
    indices_I_fixed = [i for i, I in enumerate(data['I_nominal']) if I == I_fixed]

    # plot_keys = ['S2', 'thetax_mean', 'Omega_z']  
    plot_keys = ['vx_div_vy', 'vx_div_vz', 'rke_div_tke']
    print(f"Plotting keys: {plot_keys}")
    make_subplot = all(k in keys_of_interest for k in plot_keys)
    print(f"Making subplots: {make_subplot}")
    if make_subplot:
        # fig, axes = plt.subplots(4, 1, figsize=(8, 20), sharex=True) 
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)  
        axes = axes.flatten()

    for key_index, key in enumerate(plot_keys):
        if key in keys_of_interest or (key == 'rke_div_tke' and 'rke' in keys_of_interest and 'tke' in keys_of_interest):
            if make_subplot:
                ax = axes[key_index]
            else:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)

            # For each coefficient of friction
            for i, cof in enumerate(cofs):
                ap_vals = []
                key_vals = []
                shear_rate_vals = []

                for idx in indices_I_fixed:
                    if data['cof'][idx] == cof:
                        ap_vals.append(data['ap'][idx])
                        shear_rate_vals.append(data['shear_rate'][idx])

                        if key == 'rke_div_tke':
                            rke = data['rke'][idx]
                            tke = data['tke'][idx]
                            ratio = rke / tke if tke != 0 else None
                            if cof == 0.0 and ap_vals[-1] == 1.0:
                                ratio = 0.0
                            key_vals.append(ratio)
                            label = '$\delta E_{rot}/\delta E_{transl}$'
                            plt.axhline(1, color='black', linestyle='--', linewidth=1) # label='$\\delta E_{rot}/\\delta E_{transl} = 1$')

                        elif key == 'vx_div_vy':
                            vx = data['vx_fluctuations'][idx]
                            vy = data['vy_fluctuations'][idx]
                            ratio = vx / vy if vy != 0 else None
                            key_vals.append(ratio)
                            label = '$\\delta v_x/ \\delta  v_y$'
                            if cof == 0.001:
                                # set the value for ap=1.0 to none
                                if ap_vals[-1] == 1.0:
                                    key_vals[-1] = None
                        elif key == 'vx_div_vz':
                            vx = data['vx_fluctuations'][idx]
                            vz = data['vz_fluctuations'][idx]
                            ratio = vx / vz if vz != 0 else None
                            key_vals.append(ratio)
                            label = '$\\delta  v_x/ \\delta  v_z$'
                            if cof == 0.001:
                                # set the value for ap=1.0 to none
                                if ap_vals[-1] == 1.0:
                                    key_vals[-1] = None
                        else:
                            key_vals.append(data[key][idx])

                # Sort by aspect ratio
                ap_vals, key_vals, shear_rate_vals = zip(*sorted(zip(ap_vals, key_vals, shear_rate_vals)))

                # Special handling
                if key == 'thetax_mean':
                    key_vals = [v * 180 / np.pi for v in key_vals]
                    key_vals = [45 if np.isclose(ap, 1.0, atol=1e-3) else v + 90 if ap < 1 else v
                                for ap, v in zip(ap_vals, key_vals)]
                    label = '$\\theta_x \\, [^\\circ]$'
                elif key == 'Omega_z':
                    normalized_vals = [-2 * omega / sr for omega, sr in zip(key_vals, shear_rate_vals)]
                    key_vals = normalized_vals
                    label = '$2\\langle \\omega_z \\rangle /\\dot{\\gamma}$'
                elif key == 'S2':
                    label = '$S_2$'

                rg_vals = [(value - 1) / (value + 1) for value in ap_vals]
                rg_vals, key_vals = zip(*sorted(zip(rg_vals, key_vals)))

                rg_vals = np.array(rg_vals)
                key_vals = np.array(key_vals)

                rg_oblate = rg_vals[rg_vals < 0]
                rg_prolate = rg_vals[rg_vals > 0]
                rgs_close_zero = [rg for rg in rg_vals if np.isclose(rg, 0, atol=3e-1)]

                vals_oblate = [key_vals[i] for i, rg in enumerate(rg_vals) if rg < 0]
                vals_prolate = [key_vals[i] for i, rg in enumerate(rg_vals) if rg > 0]
                vals_close_zero = [key_vals[i] for i, rg in enumerate(rg_vals) if np.isclose(rg, 0, atol=3e-1)]

                # if key == 'thetax_mean' or key == 'S2': 
                #     ax.plot(
                #         rg_oblate,
                #         vals_oblate,
                #         label=f'${cof}$',
                #         color=colors_cof[i],
                #         linestyle='--',
                #         marker='o',
                #         linewidth=3
                #     )
                    
                #     ax.plot(
                #         rg_prolate,
                #         vals_prolate,
                #         label=f'${cof}$',
                #         color=colors_cof[i],
                #         linestyle='--',
                #         marker='o',
                #         linewidth=3
                #     )

                #     ax.plot(
                #         rgs_close_zero,
                #         vals_close_zero,
                #         label=f'${cof}$',
                #         color=colors_cof[i],
                #         linestyle=':',
                #         marker='o',
                #         linewidth=1
                #     )

                if key != 'Omega_z':
                    ax.plot(
                        rg_vals,
                        key_vals,
                        label=f'${cof}$',
                        color=colors_cof[i],
                        linestyle='--',
                        marker='o',
                        linewidth=3
                    )
                
                # if key == 'Omega_z':
                #     # everything as normal except for mu_p = 0.0
                #     if cof == 0.0:
                #         vals_close_zero[3] = 1.0 # I set it to one or zero, it is ill-defined
                #         ax.plot(rg_oblate,
                #             vals_oblate,
                #             label=f'${cof}$',
                #             color=colors_cof[i],
                #             linestyle='--',
                #             marker='o',
                #             linewidth=3)
                #         ax.plot(rg_prolate,
                #             vals_prolate,
                #             color=colors_cof[i],
                #             linestyle='--',
                #             marker='o',
                #             linewidth=3)
                #         ax.plot(rgs_close_zero,
                #             vals_close_zero,
                #             color=colors_cof[i],
                #             linestyle=':',
                #             marker='o',
                #             linewidth=1
                #         )
                #     else:
                #         ax.plot(
                #             rg_vals,
                #             key_vals,
                #             label=f'${cof}$',
                #             color=colors_cof[i],
                #             linestyle='--',
                #             marker='o',
                #             linewidth=3
                #         )
                

            ax.set_xticks(ax.get_xticks())
            ax.tick_params(axis='both', labelsize=20, length=8, width=2)
            ax.set_ylabel(label, fontsize=24)

            if not make_subplot:
                filename = f'{key}_I_{I_fixed}.png'
                plt.legend(fontsize=18, loc='upper right', bbox_to_anchor=(1.0, 1.2), ncol=4)
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
                plt.close()
    axes[-1].set_xlabel('$r_g$', fontsize=24)
    # Save the 4-subplot figure
    if make_subplot:
        handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center', fontsize=18, ncol=len(cofs), bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout()
        filename = f'v_fluct_ratios_I_{I_fixed}.png'
        # filename = f'orientation_I_{I_fixed}.png'
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()


    # # For each key to be plotted:
    # for key in ['S2', 'thetax_mean', 'Omega_z', 'rke_div_tke']:
    #     if key in keys_of_interest or (key == 'rke_div_tke' and 'rke' in keys_of_interest and 'tke' in keys_of_interest):
    #         fig = plt.figure(figsize=(10, 8))
    #         ax = fig.add_subplot(111)

    #         # For each coefficient of friction
    #         for i, cof in enumerate(cofs):

    #             # Extract values for this cof and fixed I
    #             ap_vals = []
    #             key_vals = []
    #             shear_rate_vals = []
    #             for idx in indices_I_fixed:
    #                 if data['cof'][idx] == cof:
    #                     ap_vals.append(data['ap'][idx])
    #                     shear_rate_vals.append(data['shear_rate'][idx])

    #                     if key == 'rke_div_tke':
    #                         rke = data['rke'][idx]
    #                         tke = data['tke'][idx]
    #                         ratio = rke / tke if tke != 0 else None
    #                         if cof ==0.0 and ap_vals[-1] == 1.0:
    #                             ratio = 0.0
    #                         key_vals.append(ratio)
    #                         label = '$\delta E_{rot}/\delta E_{transl}$'
    #                     else:
    #                         key_vals.append(data[key][idx])
    #             # Sort by aspect ratio
    #             ap_vals, key_vals, shear_rate_vals = zip(*sorted(zip(ap_vals, key_vals, shear_rate_vals)))

    #             # Special handling for 'thetax_mean'
    #             if key == 'thetax_mean':
    #                 key_vals = [value * 180 / 3.14 for value in key_vals]  # Convert to degrees
    #                 key_vals = [45 if np.isclose(ap, 1.0, atol=1e-3) else value + 90 if ap < 1 else value
    #                             for ap, value in zip(ap_vals, key_vals)]
    #                 label = '$\\theta_x \\, [^\\circ]$'
    #             elif key == 'Omega_z':
    #                 normalized_vals = []
    #                 for ap_val, omega_val, shear_rate_val in zip(ap_vals, key_vals, shear_rate_vals):
    #                     # Search for the matching shear_rate in the data
    #                     normalized_vals.append(-2 * omega_val / shear_rate_val)
    #                 label = '$2\\langle \\omega_z \\rangle /\\dot{\\gamma}$'
    #                 key_vals = normalized_vals
    #             elif key == 'S2':
    #                 label = '$S_2$'
                
    #             rg_vals = [(value-1)/(value+1) for value in ap_vals]
    #             # Sort by aspect ratio for plotting
    #             rg_vals, key_vals = zip(*sorted(zip(rg_vals, key_vals)))
    #             plt.plot(
    #                 rg_vals,
    #                 key_vals,
    #                 label=f'$\\mu_p={cof}$',
    #                 color=colors_cof[i],
    #                 linestyle='--',
    #                 marker='o',
    #                 linewidth=3
    #             )

    #         plt.xticks(fontsize=30)
    #         plt.yticks(fontsize=30)
    #         # plt.xlabel('$\\alpha$', fontsize=30)
    #         plt.xlabel('$r_g$', fontsize=30)
    #         plt.ylabel(label, fontsize=30)
    #         # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1.35, 1.0))

    #         filename = f'{key}_I_{I_fixed}.png'
    #         plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    #         plt.close()

    if 'total_normal_dissipation' in keys_of_interest and 'total_tangential_dissipation' in keys_of_interest:
        #Plot total_tangential_dissipation / (total_normal_dissipation + total_tangential_dissipation)
        for cof in cofs:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                r = (ap-1)/(ap+1)   
                # Extract inertial numbers, total_normal_dissipation, and total_tangential_dissipation values
                x_values = [inertial_number*3/(ap+2)*ap**(1/3) for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
                normal_dissipation_values = [value for value, aspect_ratio, coef in zip(data['total_normal_dissipation'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                tangential_dissipation_values = [value for value, aspect_ratio, coef in zip(data['total_tangential_dissipation'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                # print(f"cof={cof}, ap={ap}, x_values={x_values}, normal_dissipation_values={normal_dissipation_values}, tangential_dissipation_values={tangential_dissipation_values}")
                # Ensure all the lists have the same length
                if len(x_values) == len(normal_dissipation_values) == len(tangential_dissipation_values):
                    # Compute the ratio of tangential to total dissipation
                    dissipation_ratios = [tangential / (normal + tangential) if (normal + tangential) != 0 else None for tangential, normal in zip(tangential_dissipation_values, normal_dissipation_values)]

                # Plot the values
                if x_values and dissipation_ratios:  # Ensure that the lists are not empty
                    plt.plot(x_values, dissipation_ratios, label=f'$\\alpha={ap:.2f}, \\, r={r:.2f}$', color=color, linestyle='--', marker='o')

            # Set plot properties
            plt.xscale('log')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1.4, 1.4), ncols = 2)   
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
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 2)
            plt.xlabel('$\\mu_p$', fontsize=20)
            plt.ylabel("$W_t/W_n$", fontsize=20)  # Y-axis label for the ratio
            plt.title(f'$I={I}$', fontsize=20)
            # plt.savefig(os.path.join(output_dir, f'power_dissipation_ratio_I_{I}.png'), bbox_inches='tight')
            plt.close()

        # Initialize an empty dictionary to store crossing points
        crossing_points_low = {}
        crossing_points_high = {}
        plt.figure(figsize=(5, 4))
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
                                # I_new = I* ap**(4/3)
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
            mu_p_crossings_high = np.array(mu_p_crossings_high)
            # mu_p_crossings_high = mu_p_crossings_high/ap**(0.9) if ap > 1 else mu_p_crossings_high*ap**(1.2)
            # Plot the level set for this aspect ratio
            plt.plot(mu_p_crossings_low, I_crossings_low, label=f'$\\alpha={ap:.2f}$', color=color, linestyle='-', linewidth=2)
            plt.plot(mu_p_crossings_high, I_crossings_high, color=color, linestyle='-', linewidth=2)

        #plot a power law 1 to 2 starting at the bottom left corner
        # plt.plot([0.002, 0.02], [0.001, 0.1], color='black', linestyle='--') 

        # plot should be cut exactly at the limits
        plt.xlim(0.001, 10)
        plt.ylim(0.001, 0.1)

        # for the inset
        # plt.xlim(0.002, 0.01)
        # plt.ylim(0.006, 0.02)

        # plt.xlim(0.9, 10)
        # plt.ylim(0.001, 0.1)

        # Plot properties
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.xticks([])
        # plt.yticks([])
        plt.xlabel('$\\mu_p$', fontsize=14)
        plt.ylabel('$I$', fontsize=14)
        # plt.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.05, 1.3), ncols = 3)
        # plt.title('Level Set of $P_t/P_n = 1$', fontsize=20)
        plt.savefig(os.path.join(output_dir, 'level_set_power_dissipation_ratio_rolling.png'), bbox_inches='tight', dpi = 300)#, transparent=True)
        plt.close()


    #    # Target mu_p value (doesn't need to exist in your data)
    #     target_mu_p =  0.01

    #     # Initialize storage for results
    #     alpha_values = []
    #     I_c_values = []

    #     for ap in crossing_points_low:
    #         # Get all crossing points for this aspect ratio
    #         mu_p_points, I_points = crossing_points_low[ap]
            
    #         # Skip if not enough points
    #         if len(mu_p_points) < 2:
    #             continue
            
    #         # Find closest points below and above target
    #         mu_p_array = np.array(mu_p_points)
    #         idx_below = np.where(mu_p_array <= target_mu_p)[0]
    #         idx_above = np.where(mu_p_array >= target_mu_p)[0]
            
    #         if len(idx_below) > 0 and len(idx_above) > 0:
    #             # Get closest points
    #             mu_p_below = mu_p_array[idx_below[-1]]  # Highest value below target
    #             I_below = I_points[idx_below[-1]]
                
    #             mu_p_above = mu_p_array[idx_above[0]]   # Lowest value above target
    #             I_above = I_points[idx_above[0]]
                
    #             # Logarithmic interpolation
    #             log_mu = [np.log(mu_p_below), np.log(mu_p_above)]
    #             log_I = [np.log(I_below), np.log(I_above)]
                
    #             # Interpolate to find log(I_c) at target log(mu_p)
    #             log_I_c = np.interp(np.log(target_mu_p), log_mu, log_I)
    #             I_c = np.exp(log_I_c)
                
    #             alpha_values.append(ap)
    #             I_c_values.append(I_c)

    #     # Sort by alpha for clean plotting
    #     sort_idx = np.argsort(alpha_values)
    #     alpha_values = np.array(alpha_values)[sort_idx]
    #     I_c_values = np.array(I_c_values)[sort_idx]

    #     # Plotting
    #     plt.figure(figsize=(5, 2.5))

    #     # Create log-log plot with custom formatting
    #     plt.plot(np.log10(alpha_values), np.log10(I_c_values), 'o-', color='black', markersize=8, linewidth=2)

    #     # plot best fit straight line in log-log space for alpha > 1 and alpha < 1
    #     alpha_prolate = alpha_values[alpha_values > 0.9]
    #     I_c_prolate = I_c_values[alpha_values > 0.9]
    #     alpha_oblate = alpha_values[alpha_values < 1.1]
    #     I_c_oblate = I_c_values[alpha_values < 1.1]
    #     # if len(alpha_prolate) > 1:
    #     #     slope_prolate, intercept_prolate, r_value_prolate, p_value_prolate, std_err_prolate = linregress(np.log10(alpha_prolate), np.log10(I_c_prolate))
    #     #     plt.plot(np.log10(alpha_prolate), intercept_prolate + slope_prolate * np.log10(alpha_prolate), color='red', linestyle='--', linewidth=2, label=f'Prolate Fit: $I_c \\sim \\alpha^{{{slope_prolate:.2f}}}$')
    #     # if len(alpha_oblate) > 1:
    #     #     slope_oblate, intercept_oblate, r_value_oblate, p_value_oblate, std_err_oblate = linregress(np.log10(alpha_oblate), np.log10(I_c_oblate))
    #     #     plt.plot(np.log10(alpha_oblate), intercept_oblate + slope_oblate * np.log10(alpha_oblate), color='blue', linestyle='--', linewidth=2, label=f'Oblate Fit: $I_c \\sim \\alpha^{{{slope_oblate:.2f}}}$')

    #     # Font sizes
    #     plt.xlabel('$\\log_{{10}}\\alpha$', fontsize=20)
    #     plt.ylabel('$\\log_{{10}} I_c$', fontsize=20)
    #     plt.xticks(fontsize=16)
    #     plt.yticks(fontsize=16)
        # plt.title(f'Critical Inertial Number vs Aspect Ratio\n$\\mu_p = {target_mu_p:.2f}$', fontsize=16)
        # plt.legend(fontsize=12, loc='upper right')
        
      
        # output_filename = f'critical_I_vs_alpha_mu_p_{target_mu_p:.3f}.png'.replace('.', 'p')
        # plt.savefig(os.path.join(output_dir, output_filename), 
        #             bbox_inches='tight', 
        #             dpi=300, 
        #             facecolor='white')
        # plt.close()

        # ap_target = 0.33  # Select the ap value you want to plot
        # # find index of ap_target
        # color = colors[aspect_ratios.index(ap_target)]  # Use same shade as in previous plot

        # # Extract data for the selected ap
        # filtered_data = [
        #     (mu, I, pt, pn)
        #     for mu, I, a, pt, pn in zip(data['cof'], data['I_nominal'], data['ap'],
        #                                 data['total_tangential_dissipation'], data['total_normal_dissipation'])
        #     if a == ap_target and pt is not None and pn is not None and pn > 0
        # ]

        # # Unpack
        # mu_values, I_values, pt_values, pn_values = zip(*filtered_data)

        # # Compute dissipation ratio
        # ratios = np.array(pt_values) / np.array(pn_values)

        # from scipy.interpolate import griddata
        # from matplotlib.colors import LogNorm
        # import matplotlib.colors as mcolorss
        
        # # Prepare log-log inputs
        # log_mu = np.log10(mu_values)
        # log_I = np.log10(I_values)
        # log_points = np.column_stack((log_mu, log_I))

        # # Interpolate the natural log of dissipation ratio
        # log_ratios = np.log(ratios)  # ln scale for better contrast
        # valid = np.isfinite(log_ratios)

        # # Create log-log grid
        # mu_grid = np.logspace(np.log10(min(mu_values)), np.log10(max(mu_values)), 40)
        # I_grid = np.logspace(np.log10(min(I_values)), np.log10(max(I_values)), 40)
        # MU, I = np.meshgrid(mu_grid, I_grid)
        # log_grid = np.column_stack((np.log10(MU).ravel(), np.log10(I).ravel()))

        # # Interpolate in log-log space
        # interpolated_ln_ratios = griddata(
        #     log_points[valid], log_ratios[valid], log_grid, method='linear'
        # )
        # LN_RATIO_GRID = interpolated_ln_ratios.reshape(MU.shape)
        # LN_RATIO_GRID = np.where(np.isfinite(LN_RATIO_GRID), LN_RATIO_GRID, np.nan)

        # # --- Plotting ---
        # plt.figure(figsize=(8, 6))
        # cmap = plt.cm.Reds

        # # Plot heatmap of ln(Pt/Pn)
        # heat = plt.pcolormesh(MU, I, LN_RATIO_GRID, shading='auto', cmap=cmap)

        # # Contour where ln(Pt/Pn) = 0 → Pt/Pn = 1
        # contour = plt.contour(
        #     MU, I, LN_RATIO_GRID, levels=[0], colors=[color], linewidths=2
        # )

        # # Log-log axes
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlim(0.001, 10)
        # plt.ylim(0.001, 0.1)
        # plt.xlabel('$\\mu_p$', fontsize=20)
        # plt.ylabel('$I$', fontsize=20)
        # plt.title(f'Dissipation Ratio Heatmap\n$\\alpha = {ap_target:.2f}$', fontsize=18)

        # # Colorbar (linear since data is ln-transformed)
        # # cbar = plt.colorbar(heat, orientation='horizontal', pad=0.1)
        # cbar = plt.colorbar(heat, orientation='vertical', pad=0.1)
        # cbar.set_label('$\ln(P_t / P_n)$', fontsize=20)

        # # Save figure
        # plt.tight_layout()
        # plt.savefig(os.path.join(output_dir, f'heatmap_ap_{ap_target:.2f}.png'), dpi=300, bbox_inches='tight')
        # plt.close()

    if 'c_density_y' in keys_of_interest and 'c_y_values' in keys_of_interest:
        # c_lengths = np.zeros((len(aspect_ratios), len(Is), len(cofs)))
        # Plot c_delta_vy vs c_r_values for different I and fixed ap and mu_p   
        for I in Is:
            for cof in cofs:
                plt.figure(figsize=(6, 4))
                for ap, color in zip(aspect_ratios, colors):
                    x_values = [
                        value for value, aspect_ratio, Inertial, coef in zip(
                            data['c_y_values'], data['ap'], data['I_nominal'], data['cof']
                        ) if aspect_ratio == ap and Inertial == I and coef == cof and value is not None
                    ]
                    y_values = [
                        value for value, aspect_ratio, Inertial, coef in zip(
                            data['c_density_y'], data['ap'], data['I_nominal'], data['cof']
                        ) if aspect_ratio == ap and Inertial == I and coef == cof and value is not None
                    ]

                    # box_lengths = [
                    #      value for value, aspect_ratio, Inertial, coef in zip(
                    #         data['box_y_length'], data['ap'], data['I_nominal'], data['cof']
                    #     ) if aspect_ratio == ap and Inertial == I and coef == cof and value is not None
                    # ]  

                    if x_values and y_values: 
                        # Convert to numpy arrays for easier plotting
                        d_eq = 2*ap**(1/3)
                        x_values = np.asarray(x_values)
                        y_values = np.asarray(y_values)
                        x_values = x_values / d_eq
                        plt.plot(
                            x_values[0], y_values[0], 
                            label=f'$\\alpha={ap}$', color=color, 
                            linestyle='-', linewidth=1
                        )
                       
                # Adjust plot settings
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('$y/d_{eq}$', fontsize=20)
                # plt.xlabel('$r/L_y$', fontsize=20)
                plt.ylabel('$C_n$', fontsize=20)
                # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
                # plt.title(f'$I={I}, \mu_p={cof}$', fontsize=20)
                plt.ylim(-1.5, 1.5)
                # Save the figure
                filename = f'c_density_y_I_{I}_mup_{cof}.png'
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi = 300)
                plt.close()

    if 'c_delta_vy' in keys_of_interest and 'c_r_values' in keys_of_interest:
        c_lengths = np.zeros((len(aspect_ratios), len(Is), len(cofs)))
        # Plot c_delta_vy vs c_r_values for different I and fixed ap and mu_p   
        for I in Is:
            for cof in cofs:
                plt.figure(figsize=(6, 4))
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

                    # box_lengths = [
                    #      value for value, aspect_ratio, Inertial, coef in zip(
                    #         data['box_y_length'], data['ap'], data['I_nominal'], data['cof']
                    #     ) if aspect_ratio == ap and Inertial == I and coef == cof and value is not None
                    # ]  

                    if x_values and y_values: 
                        # Convert to numpy arrays for easier plotting
                        d_eq = 2*ap**(1/3)
                        x_values = np.asarray(x_values)
                        y_values = np.asarray(y_values)
                        # box_lengths = np.asarray(box_lengths)
                        # x_values = x_values / box_lengths
                        x_values = x_values / d_eq
                        x_fit = np.linspace(0, np.max(x_values[0]), 100)
                        # fit exponential curve to find correlation length
                        popt, pcov = curve_fit(exp_corr_length, x_values[0], y_values[0], p0=[1], method='lm', maxfev=20000)
                        print(f"ap={ap}, popt={popt}, pcov={pcov}")
                        plt.plot(
                            x_values[0], y_values[0], 
                            label=f'$\\alpha={ap}$', color=color, 
                            linestyle='none', marker='o'
                        )
                        plt.plot(
                            x_fit, exp_corr_length(x_fit, *popt), 
                            color=color, linestyle='--'
                        )
                        c_lengths[aspect_ratios.index(ap), Is.index(I), cofs.index(cof)] = popt[0]

                # Adjust plot settings
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('$r/d_{eq}$', fontsize=20)
                # plt.xlabel('$r/L_y$', fontsize=20)
                plt.ylabel('$\\tilde{{C}}$', fontsize=20)
                # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
                # plt.title(f'$I={I}, \mu_p={cof}$', fontsize=20)

                # Save the figure
                filename = f'c_delta_vy_I_{I}_mup_{cof}.png'
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi = 300)
                plt.close()

        # Plot c_lengths vs ap for the different cofs at fixed I
        
        aps = np.array(aspect_ratios)
        r_gs = (aps - 1) / (aps + 1)
        
        # set values where c_lengths == 0 to none
        c_lengths = c_lengths.astype(float)  # ensure it can hold NaNs
        c_lengths[c_lengths == 0] = np.nan

        c_lengths[c_lengths == 0] = np.nan

        # Loop through each series to plot
        for j, I in enumerate(Is):
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)   
            for i, cof in enumerate(cofs):
                y = c_lengths[:, j, i]
                x = r_gs
                ax.plot(x, y, 
                        label=f'$\\mu_p={cof}$',  # avoid duplicate labels
                        color=colors_cof[i],
                        linestyle='--',
                        marker='o',
                        linewidth=2
                    )
                # Mask invalid (zero) values
                # mask = y != 0

                # # Find contiguous segments of valid data
                # idx = np.where(mask)[0]
                # if len(idx) == 0:
                #     continue

                # segments = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

                # for segment in segments:
                #     plt.plot(
                #         x[segment],
                #         y[segment],
                #         label=f'$\\mu_p={cof}$' if segment[0] == idx[0] else "",  # avoid duplicate labels
                #         color=colors_cof[i],
                #         linestyle='--',
                #         marker='o',
                #         linewidth=2
                #     )
            # ax.set_xticklabels(fontsize=20)
            # ax.set_yticklabels(fontsize=20)
            ax.set_xticks(ax.get_xticks())
            ax.set_yticks(ax.get_yticks())
            ax.tick_params(axis='both', labelsize=15)
            ax.set_xlabel('$r_g$', fontsize=20)
            # ax.set_ylabel('$\\ell/d_{{eq}}$', fontsize=20)
            ax.set_ylabel('$\\ell/d_{{eq}}$', fontsize=20)
            
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
            # plt.xscale('log')
            filename = f'c_lengths_I_{I}_ly.png'
            fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi = 300)
            plt.close()

        # # Plot c_lengths vs I for the different ap and fixed cof
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        for i, cof in enumerate(cofs):
            for j, ap in enumerate(aspect_ratios):
                    # set to none where c_lengths < 0.3
                    # c_lengths[j, :, i][c_lengths[j, :, i] < 0.3] = np.nan    
                    plt.loglog(Is, c_lengths[j, :, i], label=f'$\\alpha={ap}$', color=colors[j], linestyle='--', marker='o', linewidth=0.5)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$I$', fontsize=20)
            # plt.xscale('log')
            plt.ylabel('$\\ell/d_{avg}$', fontsize=20)
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
            filename = f'c_lengths_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()
     
    if 'c_delta_omega_z' in keys_of_interest and 'c_r_values' in keys_of_interest:
        c_lengths = np.zeros((len(aspect_ratios), len(Is), len(cofs)))
        # Plot c_delta_omega_z vs c_r_values for different I and fixed ap and mu_p   
        for I in Is:
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
                            data['c_delta_omega_z'], data['ap'], data['I_nominal'], data['cof']
                        ) if aspect_ratio == ap and Inertial == I and coef == cof and value is not None
                    ]
                    if x_values and y_values: 
                        # Convert to numpy arrays for easier plotting
                        d_eq = ap**(1/3)
                        x_values = np.asarray(x_values)/d_eq
                        y_values = np.asarray(y_values)
                        x_fit = np.linspace(0, 10, 50)
                        # fit exponential curve to find correlation length
                        popt, pcov = curve_fit(exp_corr_length, x_values[0], y_values[0], p0=[3], method='lm', maxfev=20000)
                        # print(f"ap={ap}, popt={popt}, pcov={pcov}")
                        plt.plot(
                            x_values[0], y_values[0], 
                            label=f'$\\alpha={ap}$', color=color, 
                            linestyle='none', marker='o'
                        )
                        plt.plot(
                            x_fit, exp_corr_length(x_fit, *popt), 
                            color=color, linestyle='--'
                        )
                        c_lengths[aspect_ratios.index(ap), Is.index(I), cofs.index(cof)] = popt[0]

                # Adjust plot settings
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('$r/d_{eq}$', fontsize=20)
                plt.ylabel('$\\tilde{{C}}$', fontsize=20)
                plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
                # plt.title(f'$I={I}, \mu_p={cof}$', fontsize=20)

                # Save the figure
                filename = f'c_delta_omega_z_I_{I}_mup_{cof}.png'
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi = 300)
                plt.close()

    # # Plot c_lengths vs ap for the different cofs at fixed I
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111)
    # for j, I in enumerate(Is):
    #     for i, cof in enumerate(cofs):
    #             plt.plot(aspect_ratios, c_lengths[:, j, i], label=f'$\\mu_p={cof}$', color=colors_cof[i], linestyle='--', marker='o', linewidth=2)
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     plt.xlabel('$\\alpha$', fontsize=20)
    #     plt.ylabel('$\\ell/d_{eq}$', fontsize=20)
    #     # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
    #     plt.xscale('log')
    #     filename = f'c_lengths_I_{I}.png'
    #     plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi = 300)
    #     plt.close()

    # # Plot c_lengths vs I for the different ap and fixed cof
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111)
    # for i, cof in enumerate(cofs):
    #     for j, ap in enumerate(aspect_ratios):
    #             plt.plot(Is, c_lengths[j, :, i], label=f'$\\alpha={ap}$', color=colors[j], linestyle='--', marker='o', linewidth=0.5)
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     plt.xlabel('$I$', fontsize=20)
    #     plt.xscale('log')
    #     plt.ylabel('$\\ell/d_{avg}$', fontsize=20)
    #     # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
    #     filename = f'c_lengths_cof_{cof}.png'
    #     plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    #     plt.close()

    if 'strain' in keys_of_interest and 'auto_corr_vel' in keys_of_interest:
        # plot auto_corr vs strain for different I and fixed ap and mu_p
        gamma_v = np.zeros((len(aspect_ratios), len(Is), len(cofs)))
        for I in Is:
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
                            data['auto_corr_vel'], data['ap'], data['I_nominal'], data['cof']
                        ) if aspect_ratio == ap and coef == cof and Inertial == I and value is not None
                    ]
                    print(f"ap={ap}, cof={cof}, I={I}, x_values={x_values}, y_values={y_values}")
                    if x_values and y_values:
                        plt.plot(x_values[0], y_values[0], label=f'$\\alpha={ap}$', color=color, linestyle='-', linewidth=2)#, marker='o')
                        # fit the shape of exponentrial decay with the data of first 10 data points
                        # popt, pcov = curve_fit(exp_corr_length, x_values[0][:10], y_values[0][:10], p0=[3], method='lm', maxfev=20000)
                        # plt.plot(x_values[0], exp_corr_length(x_values[0], *popt), color=color, linestyle='--')
                        # gamma_v[aspect_ratios.index(ap), Is.index(I), cofs.index(cof)] = popt[0]

                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlim(0.01, 1)
                plt.ylim(0.01, 1.1)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('$\\gamma$', fontsize=20)
                plt.ylabel('$\\tilde{{C}}(\\gamma)$', fontsize=20)
                # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
                plt.title(f'$\\mu_p={cof}$', fontsize=20)
                filename = f'auto_corr_vel_strain_cof_{cof}_I_{I}.png'
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
                plt.close()

        # plot gamma_v vs I for different ap and fixed cof
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        for i, cof in enumerate(cofs):
            for j, ap in enumerate(aspect_ratios):
                plt.loglog(Is, gamma_v[j, :, i], label=f'$\\alpha={ap}$', color=colors[j], linestyle='--', marker='o', linewidth=0.5)
                if ap == 1.0:
                    # # plot best fit power law for ap=1 only
                    # bounds = ([0, 0], [10, 10])
                    # intial_guess = [1, 1]
                    # popt, pcov = curve_fit(power_law, Is, gamma_v[j, :, i], p0=intial_guess, maxfev=20000, bounds=bounds)
                    # exponent = f"{popt[1]:.2f}"
                    # plt.plot(Is, power_law(Is, *popt), color=colors[j], linestyle='-.', linewidth=2, label=f'$\\sim I^{{{exponent}}}$') 
                    # plt.plot(Is, 2*power_law(Is, *popt), color=colors[j], linestyle='--', linewidth=2, label=f'$\\sim 2I^{{{exponent}}}$') 
                    ## plot best fit for linear regression in log-log space
                    slope, intercept, r_value, p_value, std_err =  linregress(np.log(Is), np.log(gamma_v[j, :, i]))
                    plt.plot(Is, np.exp(intercept)*Is**slope, color=colors[j], linestyle='-.', linewidth=2, label=f'$\\sim I^{{{slope:.2f}}}$')

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$I$', fontsize=20)
            plt.ylabel('$\\gamma_v$', fontsize=20)
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
            filename = f'gamma_v_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

    if 'strain' in keys_of_interest and 'auto_corr_omega' in keys_of_interest:
        # plot auto_corr vs strain for different I and fixed ap and mu_p
        gamma_omega = np.zeros((len(aspect_ratios), len(Is), len(cofs)))
        for I in Is:
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
                            data['auto_corr_omega'], data['ap'], data['I_nominal'], data['cof']
                        ) if aspect_ratio == ap and coef == cof and Inertial == I and value is not None
                    ]
                    if x_values and y_values:
                        plt.plot(x_values[0], y_values[0], label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')
                        # fit the shape of exponentrial decay with the data of first 10 data points
                        popt, pcov = curve_fit(exp_corr_length, x_values[0][:3], y_values[0][:3], p0=[3], method='lm', maxfev=20000)
                        plt.plot(x_values[0], exp_corr_length(x_values[0], *popt), color=color, linestyle='--')
                        gamma_omega[aspect_ratios.index(ap), Is.index(I), cofs.index(cof)] = popt[0]

                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlim(0.0009, 1.0)
                plt.ylim(0.001, 1.1)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('$\\gamma$', fontsize=20)
                plt.ylabel('$\\tilde{{C}}_{{\\omega_z}}(\\gamma)$', fontsize=20)
                # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
                plt.title(f'$\\mu_p={cof}$', fontsize=20)
                filename = f'auto_corr_omega_strain_cof_{cof}_I_{I}.png'
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
                plt.close()

        # plot gamma_omega vs I for different ap and fixed cof
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        for i, cof in enumerate(cofs):
            for j, ap in enumerate(aspect_ratios):
                if ap == 1.0 and cof == 0.0:
                    continue
                plt.loglog(Is, gamma_omega[j, :, i], label=f'$\\alpha={ap}$', color=colors[j], linestyle='--', marker='o', linewidth=0.5)
                if ap == 1.0 and cof != 0.0:
                    slope, intercept, r_value, p_value, std_err =  linregress(np.log(Is), np.log(gamma_omega[j, :, i]))
                    plt.plot(Is, np.exp(intercept)*Is**slope, color=colors[j], linestyle='-.', linewidth=2, label=f'$\\sim I^{{{slope:.2f}}}$')
                elif ap == 3.0 and cof == 0.0:
                    slope, intercept, r_value, p_value, std_err =  linregress(np.log(Is), np.log(gamma_omega[j, :, i]))
                    plt.plot(Is, np.exp(intercept)*Is**slope, color=colors[j], linestyle='-.', linewidth=2, label=f'$\\sim I^{{{slope:.2f}}}$')

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$I$', fontsize=20)
            plt.ylabel('$\\gamma_{{\\omega_z}}$', fontsize=20)
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
            filename = f'gamma_omega_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

    # plot sliding percentage as a function of ap for different I and fixed cof
    if 'percent_sliding' in keys_of_interest:
        for I in Is:
            plt.figure(figsize=(10, 8))
            for cof, color in zip(cofs, colors_cof):
                # Filter data together to ensure matching lengths
                filtered_data = [
                    (aspect_ratio, value)
                    for inertial_number, aspect_ratio, coef, value in zip(
                        data['I_nominal'], data['ap'], data['cof'], data['percent_sliding']
                    )
                    if inertial_number == I and coef == cof and value is not None
                ]

                # Unpack filtered data into x and y values
                if filtered_data:
                    x_values, sliding_values = zip(*filtered_data)
                else:
                    x_values, sliding_values = [], []

                plt.plot(x_values, sliding_values, label=f'$\\mu_p={cof}$', color=color, linestyle='--', marker='o')
            x_values = np.asarray(x_values)
            x_scaling_prolate = np.linspace(1.0, max(aspect_ratios), 10)
            x_scaling_oblate = np.linspace(min(aspect_ratios), 1.0, 10)
            plt.plot(x_scaling_oblate,0.1*x_scaling_oblate**(-2/3), label=f'$\\alpha^{{-2/3}}$', color='black', linestyle='--')
            plt.plot(x_scaling_prolate,0.1*x_scaling_prolate**(2/3), label=f'$\\alpha^{{2/3}}$', color='black', linestyle='--')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim(0.3, 3.0)
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('$\\alpha$', fontsize=20)
            plt.ylabel('$\\chi$', fontsize=20)
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)
            plt.title(f'$I={I}$', fontsize=20)
            plt.savefig(os.path.join(output_dir, f'sliding_percentage__mu_p_I_{I}.png'), bbox_inches='tight')
            plt.close()

    # Plot sliding percentage as a function of cof (mu_p) for different I
    if 'percent_sliding' in keys_of_interest:
        for I in Is:
            plt.figure(figsize=(10, 8))
            for ap, color in zip(aspect_ratios, colors):
                # x_values = [coef for inertial_number, aspect_ratio, coef in zip(data['I_nominal'], data['ap'], data['cof']) if inertial_number == I and aspect_ratio == ap]
                filtered_data = [
                    (coef, value) 
                    for value, inertial_number, aspect_ratio, coef in zip(
                        data['percent_sliding'], data['I_nominal'], data['ap'], data['cof']
                    )
                    if inertial_number == I and aspect_ratio == ap and value is not None
                ]

                # Unpack filtered data into x-values (cofs) and y-values (sliding_values)
                if filtered_data:
                    x_values, sliding_values = zip(*filtered_data)
                    # print(f"ap={ap}, I={I}, x_values={x_values}, sliding_values={sliding_values}")
                else:
                    x_values, sliding_values = [], []
                
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
            # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.4), ncols=4)
            plt.title(f'$I={I}$', fontsize=20)
            plt.savefig(os.path.join(output_dir, f'sliding_percentage_I_{I}_mu_p.png'), bbox_inches='tight')
            plt.close()

def plot_shear_stress_normal_tangential(data, Is, cofs, output_dir):
    # Plot shear stress (normal and tangential) as a function of aspect ratio for fixed I and mu_p
    colors_cof = plt.cm.viridis
    num_colors_cof = len(cofs)
    colors_cof = [colors_cof(i) for i in np.linspace(0, 1, num_colors_cof)]

    for I in Is:
        # fig, ax = plt.subplots(figsize=(10, 8))
        # for cof,color in zip(cofs, colors_cof):
        #     # Filter data together to ensure matching lengths
        #     filtered_data = [
        #         # (aspect_ratio, (normal_stress[0,1]+normal_stress[1,0])/(2*pyy) , (tangential_stress[0,1]+ tangential_stress[1,0])/(2*pyy))
        #         (aspect_ratio, normal_stress[1,0] / pyy , tangential_stress[1,0] / pyy)
        #         for inertial_number, aspect_ratio, coef, normal_stress, tangential_stress, pyy in zip(
        #             data['I_nominal'], data['ap'], data['cof'], data['shear_stress_normal'], data['shear_stress_tangential'], data['pressure_yy']
        #         )
        #         if inertial_number == I and coef == cof and normal_stress is not None and tangential_stress is not None
        #     ]

        #     # Unpack filtered data into x and y values
        #     if filtered_data:
        #         x_values, normal_stress_values, tangential_stress_values = zip(*filtered_data)
        #     else:
        #         x_values, normal_stress_values, tangential_stress_values = [], [], []

        #     total_stress = [normal + tangential for normal, tangential in zip(normal_stress_values, tangential_stress_values)]

        #     # Plot the data
        #     if x_values and normal_stress_values and tangential_stress_values:
        #         # plt.plot(x_values, normal_stress_values, label=f'N $\\mu_p={cof}$', color=color, linestyle='--', marker='o')
        #         # plt.plot(x_values, tangential_stress_values, label=f'T $\\mu_p={cof}$', color=color, linestyle=':', marker='o')
        #         # plt.plot(x_values, total_stress, label=f'Tot $\\mu_p={cof}$', color=color, linestyle='-', marker='o')
        #         x_values = np.asarray(x_values)
                
        #         shape_ratio = (x_values-1)/(x_values+1)

        #         ax.plot(shape_ratio, normal_stress_values, label=f'$\\sigma_{{xy}}^n/P$', linestyle=':', marker='o', linewidth=4, color= 'black')
        #         ax.plot(shape_ratio, tangential_stress_values, label=f'$\\sigma_{{xy}}^t/P$', linestyle='--', marker='o', linewidth=2, color= 'black')
        #         ax.plot(shape_ratio, total_stress, label=f'$\\sigma_{{xy}}/P$', linestyle='-', marker='o', linewidth=2, color= 'black')
        
        # # increase size of all x ticks both major and minor
        # import matplotlib.ticker as ticker
        # # Set tick sizes properly

        # # set x-axis to log scale
        # # ax.set_xscale('log')
        # ax.yaxis.set_ticks_position('both')
        # # ax.set_xlim(0.2, 3.1)
        # # ax.set_xlabel(r'$\alpha$', fontsize=20)
        # ax.set_xlabel(r'$r_g$', fontsize=20)
        # # ax.set_ylabel(r'$\sigma_{xy}/P$', fontsize=20)

        # # # Ensure both major and minor ticks are properly formatted
        # # ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, numticks=10))  # More major ticks
        # # ax.xaxis.set_minor_locator(ticker.LogLocator(base=3.0, subs=np.arange(3, 10) * 0.4, numticks=10))  # More minor ticks

        # # Set specific minor ticks
        # # minor_ticks = [0.33, 0.5, 2.0, 3.0] 
        # # ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))

        # # Increase size of all tick labels
        # ax.tick_params(axis='both', which='both', labelsize=20)

        # ax.tick_params(axis='both', labelsize=20)
        # ax.legend(fontsize=20, loc='best')
        # # Save figure correctly
        # fig.savefig(os.path.join(output_dir, f'shear_stress_I_{I}.png'), bbox_inches='tight')

        # # Close the figure properly
        # plt.close(fig)
        
        # from mpl_toolkits.mplot3d import Axes3D
        # from scipy.interpolate import griddata

        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')

        # # Flatten the dataset
        # X, Y, Z = [], [], []

        # for inertial_number, aspect_ratio, coef, normal_stress, tangential_stress, pyy in zip(
        #     data['I_nominal'], data['ap'], data['cof'], 
        #     data['shear_stress_normal'], data['shear_stress_tangential'], data['pressure_yy']
        # ):
        #     if inertial_number == I:
        #         X.append(aspect_ratio)
        #         Y.append(coef)
        #         Z.append((normal_stress[1,0] / pyy ))
        # # Convert to log scale (log10)
        # log_X = np.log10(X)
        # log_Y = np.log10(Y)
        # log_Z = np.log10(Z)
        # # Interpolate for a smoother surface
        # xi = np.linspace(min(log_X), max(log_X), 50)
        # yi = np.linspace(min(log_Y), max(log_Y), 50)
        # Xi, Yi = np.meshgrid(xi, yi)
        # Zi = griddata((log_X, log_Y), log_Z, (Xi, Yi), method='cubic')

        # # Plot surface
        # ax.plot_surface(Xi, Yi, Zi, cmap='plasma', edgecolor='none')

        # # Set axis labels with original (non-log) scale
        # ax.set_xlabel(r'$\log_{10}(\alpha)$', fontsize=14)
        # ax.set_ylabel(r'$\log_{10}(\mu_p)$', fontsize=14)
        # ax.set_zlabel(r'$\sigma_{xy}$', fontsize=14)
        # plt.title(f'Shear Stress Surface for I={I}', fontsize=16)
        # # Interpolate for smoother surface
        # xi = np.linspace(min(X), max(X), 50)
        # yi = np.linspace(min(Y), max(Y), 50)
        # Xi, Yi = np.meshgrid(xi, yi)
        # Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic')

        # ax.plot_surface(Xi, Yi, Zi, cmap='viridis', edgecolor='none')
        # ax.set_xlabel(r'$\alpha$', fontsize=14)
        # ax.set_ylabel(r'$\mu_p$', fontsize=14)
        # ax.set_zlabel(r'$\sigma_{xy}$', fontsize=14)
        # plt.title(f'Shear Stress Surface for I={I}', fontsize=16)
        # plt.show()
        # from scipy.interpolate import griddata

        # # Extract unique alpha and mu_p values
        # alpha_values = sorted(set(data['ap']))
        # cof_values = sorted(set(data['cof']))

        # # Convert to log scale (log10)
        # log_alpha_values = np.log10(alpha_values)
        # log_cof_values = np.log10(cof_values)

        # # Extract data points
        # X, Y, normal_Z, tangential_Z = [], [], [], []

        # for inertial_number, aspect_ratio, coef, normal_stress, tangential_stress, pyy in zip(
        #     data['I_nominal'], data['ap'], data['cof'], 
        #     data['shear_stress_normal'], data['shear_stress_tangential'], data['pressure_yy']
        # ):
        #     if inertial_number == I:
        #         X.append(np.log10(aspect_ratio))  # Log scale
        #         Y.append(np.log10(coef))  # Log scale
        #         normal_Z.append(normal_stress[1,0] / pyy)
        #         tangential_Z.append(tangential_stress[1,0] / pyy)

        # # Compute total shear stress
        # total_Z = [n + t for n, t in zip(normal_Z, tangential_Z)]

        # # Interpolation grid
        # xi = np.linspace(min(X), max(X), 30)
        # yi = np.linspace(min(Y), max(Y), 30)
        # Xi, Yi = np.meshgrid(xi, yi)

        # # Interpolate for smooth heatmaps
        # Zi_normal = griddata((X, Y), normal_Z, (Xi, Yi), method='cubic')
        # Zi_tangential = griddata((X, Y), tangential_Z, (Xi, Yi), method='cubic')
        # Zi_total = griddata((X, Y), total_Z, (Xi, Yi), method='cubic')

        # # Get common color range for all plots
        # vmin = min(np.nanmin(Zi_normal), np.nanmin(Zi_tangential), np.nanmin(Zi_total))
        # vmax = max(np.nanmax(Zi_normal), np.nanmax(Zi_tangential), np.nanmax(Zi_total))

        # # Create figure with 3 subplots
        # fig, axes = plt.subplots(3, 1, figsize=(6, 15), constrained_layout=True)

        # # Plot heatmaps
        # titles = ['Normal Shear Stress', 'Tangential Shear Stress', 'Total Shear Stress']
        # Z_data = [Zi_normal, Zi_tangential, Zi_total]

        # # reduce value of alpha for better visualization
        # log_alpha_values_pruned = log_alpha_values[::2]

        # for i, (ax, Zi, title) in enumerate(zip(axes, Z_data, titles)):
        #     c = ax.imshow(Zi, cmap="plasma", origin="lower", aspect="auto",
        #                 extent=[min(X), max(X), min(Y), max(Y)])#, vmin=vmin, vmax=vmax)  # Shared color scale
            
        #     ax.set_xticks(log_alpha_values)
        #     ax.set_xticklabels([f"{10**x:.2g}" for x in log_alpha_values])  # Convert back to original scale
            
        #     # Only add x-axis labels to the first subplot
        #     if i == 2:
        #         ax.set_xticks(log_alpha_values_pruned)
        #         ax.set_xticklabels([f"{10**x:.2g}" for x in log_alpha_values_pruned])  # Convert back to original scale
        #         # increase the font of the ticks on the x-axis
        #         ax.set_xlabel(r'$\log_{10}(\alpha)$', fontsize=16)
        #     else:
        #         ax.set_xticks([])  # Remove y-axis labels for the other plots
            
        #     ax.tick_params(axis='both', which='major', labelsize=16)   
        #     ax.set_ylabel(r'$\log_{10}(\mu_p)$', fontsize=16)
        #     # set colorbar for each plot 
        #     cbar = fig.colorbar(c, ax=ax, orientation="vertical")
        #     # increase the font of the ticks on the cbar
        #     cbar.ax.tick_params(labelsize=16)

        #     ax.set_title(title, fontsize=16)


        # # Add one colorbar for all subplots
        # fig.colorbar(c, ax=axes, orientation="vertical") #, fraction=0.2, pad=0.01) #,label=r'$\sigma_{xy}$')
        # plt.savefig(os.path.join(output_dir, f'shear_stress_map_I_{I}.png'), bbox_inches='tight')
        # plt.show()


        # Extract unique alpha and mu_p values
        alpha_values = sorted(set(data['ap']))
        cof_values = sorted(set(data['cof']))

        # Convert to log scale (log10)
        log_alpha_values = np.log10(alpha_values)
        log_cof_values = np.log10(cof_values)

        # Create an empty heatmap matrix
        heatmap_data = np.full((len(log_alpha_values), len(log_cof_values)), np.nan)

        # Populate heatmap matrix
        for i, alpha in enumerate(alpha_values):
            for j, cof in enumerate(cof_values):
                filtered = [
                    (normal_stress[1,0] / pyy)  
                    for inertial_number, aspect_ratio, coef, normal_stress, tangential_stress, pyy in zip(
                        data['I_nominal'], data['ap'], data['cof'], 
                        data['shear_stress_normal'], data['shear_stress_tangential'], data['pressure_yy']
                    )
                    if aspect_ratio == alpha and coef == cof and inertial_number == I
                ]
                heatmap_data[i, j] = np.mean(filtered) if filtered else np.nan

        # Convert shear stress to log scale (to match the 3D plot)
        log_heatmap_data = np.log10(heatmap_data)

        # Create the figure 
        fig, ax = plt.subplots(figsize=(8, 6))

        # Use imshow to plot the heatmap (log-scaled axes)
        c = ax.imshow(log_heatmap_data, cmap="plasma", origin="lower", aspect="auto",
                    extent=[min(log_cof_values), max(log_cof_values), min(log_alpha_values), max(log_alpha_values)])

        # Set log-scale tick labels
        ax.set_xticks(log_cof_values)
        ax.set_xticklabels([f"{10**x:.2g}" for x in log_cof_values])  # Convert back to original scale

        ax.set_yticks(log_alpha_values)
        ax.set_yticklabels([f"{10**x:.2g}" for x in log_alpha_values])  # Convert back to original scale

        # Add colorbar
        cbar = plt.colorbar(c, label=r'$\log_{10}(\sigma_{yx})$')

        # Set axis labels and title
        ax.set_xlabel(r'$\mu_p$', fontsize=16)
        ax.set_ylabel(r'$\alpha$', fontsize=16)
        ax.set_title(r'$\sigma_{yx}$ Heatmap (Log-Scaled Axes)', fontsize=18)

        plt.show()

        from mpl_toolkits.mplot3d import Axes3D
        from scipy.interpolate import griddata

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Flatten the dataset
        X, Y, normal_Z, tangential_Z = [], [], [], []

        for inertial_number, aspect_ratio, coef, normal_stress, tangential_stress, pyy in zip(
            data['I_nominal'], data['ap'], data['cof'], 
            data['shear_stress_normal'], data['shear_stress_tangential'], data['pressure_yy']
        ):
            if inertial_number == I:
                X.append(aspect_ratio)
                Y.append(coef)
                normal_Z.append(normal_stress[1,0] / pyy)
                tangential_Z.append(tangential_stress[1,0] / pyy)

        # Compute total stress
        total_Z = [n + t for n, t in zip(normal_Z, tangential_Z)]

        # Convert to log scale (log10)
        log_X = np.log10(X)
        log_Y = np.log10(Y)
        log_normal_Z = normal_Z # np.log10(normal_Z)
        log_tangential_Z = tangential_Z #np.log10(tangential_Z)
        log_total_Z = total_Z #np.log10(total_Z)

        # Interpolate for smoother surfaces
        xi = np.linspace(min(log_X), max(log_X), 50)
        yi = np.linspace(min(log_Y), max(log_Y), 50)
        Xi, Yi = np.meshgrid(xi, yi)

        Zi_normal = griddata((log_X, log_Y), log_normal_Z, (Xi, Yi), method='cubic')
        Zi_tangential = griddata((log_X, log_Y), log_tangential_Z, (Xi, Yi), method='cubic')
        Zi_total = griddata((log_X, log_Y), log_total_Z, (Xi, Yi), method='cubic')

        # Plot surfaces with different colors (no colormap)
        ax.plot_surface(Xi, Yi, Zi_normal, color='blue', alpha=0.5, edgecolor='none', label='Normal Stress')
        ax.plot_surface(Xi, Yi, Zi_tangential, color='red', alpha=0.5, edgecolor='none', label='Tangential Stress')
        ax.plot_surface(Xi, Yi, Zi_total, color='green', alpha=0.5, edgecolor='none', label='Total Stress')

        # Set axis labels with original (non-log) scale
        ax.set_xlabel(r'$\log_{10}(\alpha)$', fontsize=14)
        ax.set_ylabel(r'$\log_{10}(\mu_p)$', fontsize=14)
        ax.set_zlabel(r'$\sigma_{yx}$', fontsize=14)
        plt.title(f'Shear Stress Surfaces for I={I}', fontsize=16)

        # Manually create legend with color patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', edgecolor='black', label='Normal Stress'),
            Patch(facecolor='red', edgecolor='black', label='Tangential Stress'),
            Patch(facecolor='green', edgecolor='black', label='Total Stress')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.show()

def plot_dissipation_vs_aspect_ratio(data, keys_of_interest):
    """
    Plots the normalized dissipation (either tangential or normal) vs aspect ratio (ap)
    for fixed inertial number (I) and friction coefficient (μₚ).

    Parameters:
    - data: Dictionary containing simulation data
    - keys_of_interest: Set or list of keys to check for availability
    - key: Either 'total_tangential_dissipation' or 'total_normal_dissipation'
    """
    

    colormap_cof = plt.cm.viridis
    num_colors_cof = len(cofs)
    colors_cof = [colormap_cof(i) for i in np.linspace(0, 1, num_colors_cof)]

    for I in Is:  # Loop over inertial numbers
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        for i, key in enumerate(keys_of_interest):

            for cof in cofs:  # Loop over friction coefficients
            
                # up to cof <0.1 first subplot then second
                if cof< 0.4 and i == 0:
                    ax = axes[0, 0]
                elif cof >= 0.4 and i == 0:
                    ax = axes[0,1]
                elif cof < 0.4 and i == 1:
                    ax = axes[1, 0]
                elif cof >= 0.4 and i == 1:
                    ax = axes[1, 1]


                filtered_data = [
                    (ap, dissipation / (shear_rate * box_volume * p_yy))
                    for ap, dissipation, shear_rate, box_volume, p_yy, inertial_number, coef
                    in zip(
                        data['ap'], data[key], data['shear_rate'], 
                        data['box_size'], data['pressure_yy'], data['I_nominal'], data['cof']
                    )
                    if inertial_number == I and coef == cof and dissipation is not None
                ]

                # Unpack filtered values
                if filtered_data:
                    x_values, y_values = zip(*filtered_data)
                else:
                    x_values, y_values = [], []

                # Plot the data
                if x_values and y_values:
                    ax.loglog(x_values, y_values, label=f'$\\mu_p={cof}$', linestyle='--', marker='o', color=colors_cof[cofs.index(cof)], linewidth=3)
                    
                if (cof == 0.0 and key == 'total_normal_dissipation') or (cof == 0.1 and key == 'total_tangential_dissipation'):
                        aps = np.array(x_values)
                        y_values = np.array(y_values)
                        mask_oblate = aps < 0.8
                        mask_prolate = aps > 1.2

                        x_oblate = aps[mask_oblate]
                        y_oblate = y_values[mask_oblate]

                        x_prolate = aps[mask_prolate]
                        y_prolate = y_values[mask_prolate]

                        # Fit power-law model (linear regression in log-log space)
                        slope_oblate, intercept_oblate = np.polyfit(np.log(x_oblate), np.log(y_oblate), 1)
                        slope_prolate, intercept_prolate = np.polyfit(np.log(x_prolate), np.log(y_prolate), 1)

                        # Generate best-fit lines over the appropriate x ranges
                        x_fit_oblate = np.linspace(min(x_oblate), 0.8, 100)  # Smooth line for plotting
                        x_fit_prolate = np.linspace(1.3, max(x_prolate), 100)

                        y_fit_oblate = np.exp(intercept_oblate) * x_fit_oblate**slope_oblate
                        y_fit_prolate = np.exp(intercept_prolate) * x_fit_prolate**slope_prolate

                        # Plot best-fit lines
                        ax.loglog(x_fit_oblate, y_fit_oblate, label=f'Best fit (Oblate, $\\alpha^{{{slope_oblate:.2f}}}$)', linestyle='-', color='black', linewidth=3)
                        ax.loglog(x_fit_prolate, y_fit_prolate, label=f'Best fit (Prolate, $\\alpha^{{{slope_prolate:.2f}}}$)', linestyle='-', color='red', linewidth=3)
                        print(f"Oblate: slope={slope_oblate:.2f}, intercept={intercept_oblate:.2f}, I={I}, key={key}, cof={cof}")
                        print(f"Prolate: slope={slope_prolate:.2f}, intercept={intercept_prolate:.2f}")

        # plot power law alpha^-1/3
        x_values = np.asarray(x_values)
        x_scaling_prolate = np.linspace(1.5, max(aspect_ratios), 10)
        x_scaling_oblate = np.linspace(min(aspect_ratios), 0.67, 10)
        # plt.plot(x_scaling_oblate,0.21*x_scaling_oblate**(1), label=f'$\\alpha^{{1}}$', color='black', linestyle='--')
        # plt.plot(x_scaling_prolate,0.23*x_scaling_prolate**(-1), label=f'$\\alpha^{{-1}}$', color='black', linestyle='--')
        # Select x-values for oblate (x < 0.8) and prolate (x > 1.2)

        # Final plot settings
        axes[1, 0].set_xlabel(' $\\alpha$', fontsize=20)
        axes[1, 1].set_xlabel(' $\\alpha$', fontsize=20)
        axes[0, 0].set_ylabel(fr'$\mathcal{{P}}_n / (\dot{{\gamma}} P \Omega)$', fontsize=20)
        axes[1, 0].set_ylabel(fr'$\mathcal{{P}}_t / (\dot{{\gamma}} P \Omega)$', fontsize=20)

        for ax in np.ravel(axes):
            ax.tick_params(axis='both', length=10, width=2)
            for label in ax.get_yticklabels():
                label.set_fontsize(20)
            for label in ax.get_xticklabels():
                label.set_fontsize(20)

     

        custom_ticks = [0.33, 0.5, 1.0, 2.0, 3.0]
        ax.set_xticks(custom_ticks)  # Set custom ticks
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))

        ax.set_xticklabels([f"{tick:.1f}" for tick in custom_ticks], fontsize=20)  # Set explicit labels

        # Optional: fine-tune minor ticks and grid
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=12))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        # plt.yticks(fontsize=20)
        # plt.title(f'Normalized {key.replace("_", " ").title()} vs Aspect Ratio for I = {I}')
        # plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1.1, 2.3), ncols=3)
        plt.savefig(f"{output_dir}/normalized_dissipation_vs_aspect_ratio_I_{I}.png", bbox_inches='tight', dpi=300)
        plt.close()

def plot_dissipation_local_vs_global(data):
    """
    Plots the dissipation ratio (local/global) vs aspect ratio (ap)
    for fixed inertial number (I) and friction coefficient (μₚ).

    Parameters:
    - data: Dictionary containing simulation data
    - keys_of_interest: Set or list of keys to check for availability
    """

    colormap_cof = plt.cm.viridis
    num_colors_cof = len(cofs)
    colors_cof = [colormap_cof(i) for i in np.linspace(0, 1, num_colors_cof)]

    # Set label dynamically
    dissipation_label = r'\mathcal{P}'

    for I in Is:  # Loop over inertial numbers
        plt.figure(figsize=(10, 8))

        for cof in cofs:  # Loop over friction coefficients
            filtered_data = [
                (ap, (P_n +P_t) / (shear_rate * box_volume * p_xy))
                for ap, P_n, P_t, shear_rate, box_volume, p_xy, inertial_number, coef
                in zip(
                    data['ap'], data["total_normal_dissipation"], data["total_tangential_dissipation"], data['shear_rate'], 
                    data['box_size'], data['pressure_xy'], data['I_nominal'], data['cof']
                )
                if inertial_number == I and coef == cof and P_t is not None and P_n is not None
            ]

            # Unpack filtered values
            if filtered_data:
                x_values, y_values = zip(*filtered_data)
            else:
                x_values, y_values = [], []

            # Plot the data
            if x_values and y_values:
                plt.plot(x_values, y_values, label=f'$\\mu_p={cof}$', linestyle=':', marker='o', color=colors_cof[cofs.index(cof)], linewidth=3)

        # Final plot settings
        plt.xlabel('Aspect Ratio ($\\alpha$)', fontsize=20)
        plt.ylabel(fr'${dissipation_label} / (\dot{{\gamma}} \sigma_{{xy}} \Omega)$', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0.8, 1.2)
        # plt.title(f'Normalized {key.replace("_", " ").title()} vs Aspect Ratio for I = {I}')
        plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1.1, 1.3), ncols=3)
        plt.savefig(f"{output_dir}/ratio_dissipation_global_local_I_{I}.png", bbox_inches='tight', dpi=300)
        plt.close()

def plot_polar_histograms_ap(bins, histograms, title, labels, symmetry=False, ap=1.0):
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
    # ax.set_title(title, va='bottom')
    colors = create_colormap_ap(aspect_ratios, central_ap=1.0, central_color='black', colormap=plt.cm.RdYlBu)

    for idx, (histogram, label, color) in enumerate(zip(histograms, labels, colors)):
        if symmetry:
            # Local histograms: Assuming the bins and histogram are for 0 to 90 degrees (10 bins)
            assert len(histogram) == len(bins) - 1, "The length of histogram should match the number of bins - 1."

            # Reverse the histogram for the 90 to 180 degrees
            # histogram_mirror_90_180 = histogram[::-1]

            # Combine 0-90, 90-180, 180-270 (same as 90-0), and 270-360 (same as 0-90)
            # extended_histogram = np.concatenate((histogram, histogram_mirror_90_180, histogram, histogram_mirror_90_180))
            
            # Adjust bins to match the extended histogram 0 to 90
            bin_width = bins[1] - bins[0]
            # extended_bins = np.arange(0, 360 + bin_width, bin_width)
            extended_bins = np.arange(0, 90 + bin_width, bin_width)
            extended_histogram = histogram
            # if ap< 1, we need to shift the histogram by 90 degrees
            if ap < 1.0:
                # extended_histogram = np.roll(extended_histogram, len(histogram))
                # extended_histogram = np.roll(histogram, len(histogram)/2)
                print("Rolled")

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

        if not symmetry:
            angles_rad = np.append(angles_rad, angles_rad[0])
            extended_histogram = np.append(extended_histogram, extended_histogram[0])
        else:       # Add first and last points at 0° and 90° to close the loop
            angles_rad = np.insert(angles_rad, 0, 0.0)  # Insert 0° at the beginning
            angles_rad = np.append(angles_rad, np.radians(90))  # Append 90° at the end
            
            # Extend the histogram to match the boundaries
            extended_histogram = np.insert(extended_histogram, 0, extended_histogram[0])  # Duplicate first value at 0°
            extended_histogram = np.append(extended_histogram, extended_histogram[-1])  # Duplicate last value at 90°

        # Plot data as lines on the polar plot
        ax.plot(angles_rad, extended_histogram, label=label, color=color)


    if symmetry:
        ax.set_thetamin(0)   # Set minimum angle to 0 degrees
        ax.set_thetamax(90)  # Set maximum angle to 90 degrees

    max_hist = max([np.max(hist) for hist in histograms])
    ax.set_ylim(0, max_hist * 1.1)
    # increase the font size of the labels
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_rlabel_position(135)  # Moves radial labels

    # reduce number of radial labels to 4
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))



    # plt.legend(bbox_to_anchor=(1.4, 1), loc='upper right', fontsize=12)
    # print(output_dir + '/polar_histogram_{title}_lines.png')
    plt.savefig(f"{output_dir}/polar_histogram_{title}_lines.png", bbox_inches='tight')
    plt.close()

def create_polar_plots_varying_ap(fixed_cof, fixed_I, histogram_keys, pdf_keys, local_histogram_keys):
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
                            print(f"Warning: Histogram length does not match global bins for {filename, hist_key}")

        # Call the plot function if there are histograms to plot
        if histograms:
            plot_polar_histograms_ap(bins_global, histograms, f'{hist_key} (mu_p={fixed_cof}, I={fixed_I})', labels)

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
                            print(f"Warning: PDF histogram length does not match global bins for {filename, pdf_key}")

        # Call the plot function if there are histograms to plot
        if pdf_histograms:
            plot_polar_histograms_ap(bins_global, pdf_histograms, f'{pdf_key} (mu_p={fixed_cof}, I={fixed_I})', labels)

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
                    box_y_length = file_data.get('box_y_length', None)
                    box_z_length = file_data.get('box_z_length', None)
                    area_particle = file_data.get('total_area', None)
                    avg_pressure = file_data.get('p_yy', None)
                    area_adjustments = file_data.get('area_adjustment_ellipsoid', None)
                    # print(f"Ap={ap}, area_particle={area_particle}, avg_pressure={avg_pressure}, area_adjustments={area_adjustments}")
                    
                    
                    if local_histogram is not None and box_x_length is not None and box_z_length is not None:
                        # Calculate the area A and normalize by p * A
                        area = box_x_length * box_z_length
                        n_particles = 2000
                        avg_n_particles_layer = n_particles * 1/ box_y_length
                        if local_hist_key == 'local_normal_force_hist_cp' or local_hist_key == 'local_tangential_force_hist_cp':
                            #  normalized_local_histogram = avg_n_particles_layer*local_histogram / (avg_pressure * area) 
                             normalized_local_histogram = local_histogram / (avg_pressure * area_particle) 
                        else:
                            normalized_local_histogram = local_histogram
                       
                        # print(np.sum(local_histogram*area_adjustments))
                        # Check if the histogram length matches the bins for local data
                        if len(normalized_local_histogram) == len(bins_local) - 1:
                            # if ap < 1.0:
                            #     normalized_local_histogram = normalized_local_histogram[::-1]
                            local_histograms.append(normalized_local_histogram)
                            labels.append(f'$\\alpha={ap}$')
                        else:
                            print(f"Warning: Local histogram length does not match local bins for {filename}")
        
        # Call the plot function if there are histograms to plot
        if local_histograms:
            plot_polar_histograms_ap(bins_local, local_histograms, f'Normalized {local_hist_key} (mu_p={fixed_cof}, I={fixed_I})', labels, symmetry=True, ap=ap)

def set_axes_equal(ax, max_length):
    """
    Set equal scaling (aspect ratio) for all axes of a 3D plot.
    This ensures that arrows have the correct proportions in all dimensions.
    """
    # Set the limits for all axes based on the max length
    ax.set_xlim([-max_length, max_length])
    ax.set_ylim([0, max_length])
    ax.set_zlim([0, max_length])

def plot_fabric_eigenvectors_ap(fixed_cof, fixed_I, aspect_ratios):
    """
    Plot 3D arrows representing the eigenvectors of the fabric tensor for different aspect ratios.
    
    Args:
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
                    # Compute the eigenvalues and eigenvectors of the fabric tensor
                    eigenvalues, eigenvectors = np.linalg.eig(fabric_tensor)
                    print(f"Eigenvectors for ap={ap}:\n{eigenvectors}")
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
    plt.savefig(output_dir + "/fabric_eigenvectors_cof_" + str(fixed_cof) + "_I_" + str(fixed_I) + '.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_fabric_eigenvectors_2d(fixed_cof, fixed_I, aspect_ratios):
    """
    Plot 2D arrows representing the eigenvectors of the fabric tensor for different aspect ratios.
    Checks if the third eigenvector is close to (0,0,1) and plots only the XY-plane components.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    colormap = plt.cm.RdYlBu
    num_colors = len(aspect_ratios)
    extreme_indices = np.concatenate([
        np.linspace(0, 0.3, num_colors // 2, endpoint=False),
        np.linspace(0.7, 1.0, num_colors - num_colors // 2)
    ])
    colors = [colormap(i) for i in extreme_indices]
    
    central_color = 'black'
    central_ap = 1.0  # Replace with actual central aspect ratio
    central_index = aspect_ratios.index(central_ap)
    colors.insert(central_index, central_color)
    
    max_arrow_length = 0
    
    for idx, ap in enumerate(aspect_ratios):
        filename = f'simple_shear_ap{ap}_cof_{fixed_cof}_I_{fixed_I}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                file_data = pickle.load(file)
                fabric_tensor = file_data.get('fabric', None)
                
                if fabric_tensor is not None:
                    eigenvalues, eigenvectors = np.linalg.eig(fabric_tensor)
                    third_eigenvector = eigenvectors[:, 2]
                    dot_product = np.abs(np.dot(third_eigenvector, [0, 0, 1]))
                    
                    if dot_product > 0.99:  # Check if third eigenvector is nearly (0,0,1)
                        for i in range(2):  # Only plot first two eigenvectors in XY plane
                            if eigenvectors[1][i] < 0:
                                eigenvectors[:, i] = -eigenvectors[:, i]
                            # print(f"Angle of orientation of eigenvector {i} for ap={ap} is {np.degrees(np.arctan2(eigenvectors[1][i], eigenvectors[0][i]))}")
                            
                            eig_val = eigenvalues[i]
                            eig_vec = eigenvectors[:2, i]  # Take XY components
                            arrow_vector = eig_vec * eig_val
                            max_arrow_length = max(max_arrow_length, np.linalg.norm(arrow_vector))
                            ax.plot([0, arrow_vector[0]], [0, arrow_vector[1]], color=colors[idx], linewidth = 2)
    
    ax.set_aspect('equal')
    ax.set_xlabel('$x$', fontsize=20)
    ax.set_ylabel('$y$', fontsize=20)
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(0, 0.7)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)


    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')
    
    # plt.title(f'Fabric Eigenvectors ($\\mu_p={fixed_cof}$, $I={fixed_I}$)')
    plt.savefig(output_dir + "/fabric_eigenvectors_2d_cof_" + str(fixed_cof) + "_I_" + str(fixed_I) + '.png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

    def find_closest(value, array):
        array = np.array(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    alphas = [ap for ap in aspect_ratios if ap > 1]
    angle_diffs = []
    eigval_ratios = []
    valid_alphas = []

    def get_fabric_data(alpha):
        filename = f'simple_shear_ap{alpha}_cof_{fixed_cof}_I_{fixed_I}.pkl'
        if not os.path.exists(filename):
            return None, None
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        fabric_tensor = data.get('fabric', None)
        if fabric_tensor is None:
            return None, None

        eigenvalues, eigenvectors = np.linalg.eig(fabric_tensor)
        third_ev = eigenvectors[:, 2]
        if np.abs(np.dot(third_ev, [0, 0, 1])) < 0.99:
            return None, None
        
        # Ensure consistent orientation: flip if y component of dominant eigenvector < 0
        if eigenvectors[1, 0] < 0:
            eigenvectors[:, 0] = -eigenvectors[:, 0]

        # Dominant eigenvector and eigenvalue (index of max eigenvalue)
        idx_max = np.argmax(eigenvalues)
        dom_eigval = eigenvalues[idx_max]
        dom_eigvec = eigenvectors[:2, idx_max]  # XY components
        dom_eigvec /= np.linalg.norm(dom_eigvec)  # normalize

        # Angle in degrees w.r.t x-axis
        angle = np.degrees(np.arctan2(dom_eigvec[1], dom_eigvec[0]))

        return angle, dom_eigval

    for alpha in aspect_ratios:
        if alpha <= 1:
            continue
        inv_alpha = 1 / alpha
        closest_inv = find_closest(inv_alpha, aspect_ratios)

        # Optional: check if closest_inv is close enough to inv_alpha
        if abs(closest_inv - inv_alpha) > 0.1:  # tolerance, tweak as needed
            # skip if closest is too far
            continue

        angle_alpha, eigval_alpha = get_fabric_data(alpha)
        angle_inv, eigval_inv = get_fabric_data(closest_inv)


        if None in (angle_alpha, eigval_alpha, angle_inv, eigval_inv):
            continue

        diff = np.abs(angle_alpha - angle_inv)
        if diff > 180:
            diff = 360 - diff

        ratio = eigval_alpha / eigval_inv

        valid_alphas.append(alpha)
        angle_diffs.append(diff)
        eigval_ratios.append(ratio)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(3,2))    
    ax1.plot(alphas, angle_diffs, 'b-o', label='Angle Difference (degrees)')
    ax1.set_xlabel('$\\alpha$', fontsize=18)
    ax1.set_ylabel('$\\eta (°)$', color='b', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(alphas, eigval_ratios, 'r-s', label='Eigenvalue Ratio')
    ax2.set_ylabel('$\\lambda_1^p/\\lambda_1^o$', color='r', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='r')

    # Optional legends combined
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    # plot box encapulating the figure
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)


    # plt.title('Comparison of Dominant Eigenvectors and Eigenvalues for $\\alpha$ and $1/\\alpha$', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fabric_eigenvectors_comparison_cof_{fixed_cof}_I_{fixed_I}.png", bbox_inches='tight', dpi=300)
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
    plt.figure(figsize=(10, 7))

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
                    plt.plot(bin_centers, pdf_values, label=f'$\\alpha={ap}$', color=colors[idx], linewidth=2)

    # Customize the plot
    plt.xlabel('$\\theta_x$ [rad]', fontsize=18)
    plt.ylabel('$\\psi[\\theta_x]$', fontsize=18)
    plt.gca().xaxis.set_major_locator(MultipleLocator(base=np.pi/4))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    plt.gca().tick_params(axis='x', labelsize=14)
    plt.gca().tick_params(axis='y', labelsize=14)

    plt.title(f'Circular PDF of Angular Distribution ($\\mu_p={fixed_cof}$, $I={fixed_I}$)', fontsize=16)
    # plt.legend(fontsize=16, ncol=2)
    # plt.grid(True)
    
    # Save and show the plot
    plt.savefig(output_dir + "/pdf_thetax_cof_" + str(fixed_cof) + "_I_" + str(fixed_I) + '.png', bbox_inches='tight')
    plt.show()

def plot_ellipsoids_with_combined_data_ap(
    fixed_cof, fixed_I, aspect_ratios, variable_keys, operation, label, bins_local, output_dir="parametric_plots_new", cmap=plt.cm.GnBu, scale="linear"):
    """
    Plot multiple ellipsoids in a single figure with a shared color bar.

    Args:
    - data: Dictionary containing the simulation data.
    - fixed_cof: The coefficient of friction to be fixed.
    - fixed_I: The inertial number to be fixed.
    - aspect_ratios: List of aspect ratios to loop through.
    - variable_keys: List of keys to access the data for the operation.
    - operation: A function (or lambda) to combine the data from multiple keys.
    - label: Label for the color bar.
    - bins_local: Bin edges for the local data (0 to 90 degrees).
    - output_dir: Directory to save the output figure.
    """
    assert len(bins_local) - 1 > 0, "bins_local must have at least two edges."
    assert callable(operation), "Operation must be a callable function or lambda."
    # assert len(variable_keys) > 1, "Provide at least two variable keys for operations."

    # Collect all data to determine global min and max for colormap normalization
    all_combined_data = []
    ellipsoid_data = {}

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
                    # print(f"Data values: {data_values}")
                    combined_data = operation(*data_values)
                    all_combined_data.extend(combined_data)
                    ellipsoid_data[ap] = combined_data
    
    if label=='Mobilized Friction':
        # Normalize the combined data by dividing by the fixed_cof
        all_combined_data = np.array(all_combined_data) / fixed_cof
        ellipsoid_data = {ap: data / fixed_cof for ap, data in ellipsoid_data.items()}

    # Determine global min and max for normalization
    vmin = min(all_combined_data)
    vmax = max(all_combined_data)

    # print(f"Combined data (min: {np.min(combined_data)}, max: {np.max(combined_data)})")
    # print(f"vmin: {vmin}, vmax: {vmax}")
    # print(all_combined_data)
    # vmin=0.0
    # vmax = 3.0
    
    # Create a single figure with subplots
    fig, axes = plt.subplots(1, len(aspect_ratios), figsize=(2.5 * len(aspect_ratios), 3.5),
                             subplot_kw={'projection': '3d'})

    # If only one aspect ratio, make axes a list
    if len(aspect_ratios) == 1:
        axes = [axes]

    # Use a consistent colormap
    # cmap = plt.cm.inferno
    cmap = cmap    # cmap = cmap.reversed()
    if scale == "linear":
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax)

    for i, ap in enumerate(aspect_ratios):
        combined_data = ellipsoid_data[ap]
        ax = axes[i]

        # Create a grid of angles
        u = np.linspace(0, 2 * np.pi, 50)  # azimuthal angle
        v = np.linspace(0, np.pi, 2 * len(bins_local) - 1)  # polar angle based on bins_local

        # Parametric equations for the ellipsoid
        x = np.outer(np.sin(v), np.cos(u))
        y = np.outer(np.sin(v), np.sin(u))
        z = float(ap) * np.outer(np.cos(v), np.ones_like(u))

        if float(ap) < 1.0:
            # Rotate ellipsoid: move major axis (xy-plane) to z-axis
            # Rotate by 90° around x-axis: (y -> z, z -> -y)
            y_rot = y * np.cos(np.pi / 2) - z * np.sin(np.pi / 2)
            z_rot = y * np.sin(np.pi / 2) + z * np.cos(np.pi / 2)
            y = y_rot
            z = z_rot
            theta = -np.radians(30)
            x_rot = x * np.cos(theta) - y * np.sin(theta)
            y_rot = x * np.sin(theta) + y * np.cos(theta)
            x = x_rot
            y = y_rot

        # Calculate the bin centers
        bin_centers = (bins_local[:-1] + bins_local[1:]) / 2

        # Create a color array for the ellipsoid surface
        colors = np.zeros_like(z)

        # Assign colors to bins properly
        for j in range(len(bin_centers)):
            # Convert bin edges to radians
            lower_bound = np.radians(bins_local[j])
            upper_bound = np.radians(bins_local[j + 1])

            # Find indices in the latitude range
            indices = (v >= lower_bound) & (v < upper_bound)

            # Assign color values correctly
            colors[indices, :] = combined_data[j]

            # Mirror to the lower hemisphere
            lower_mirror_indices = (v >= np.radians(180 - bins_local[j + 1])) & (v < np.radians(180 - bins_local[j]))
            colors[lower_mirror_indices, :] = combined_data[j]

        # Convert colors to a colormap range
        face_colors = cmap(norm(colors))

        # Plot the ellipsoid
        ax.plot_surface(x, y, z, facecolors=face_colors, rstride=1, cstride=1, antialiased=True, alpha=1.0)

        # Set the view and remove axis labels for a cleaner look
        ax.set_aspect('equal')
        ax.axis('off')
        ax.view_init(elev=-26, azim=-6, roll=30)

        # Add alpha value as a legend
        # ax.text2D(0.25, 0.95, f"$\\alpha = {ap}$", transform=ax.transAxes, fontsize=14)

    # Add a single color bar at the bottom
    fig.subplots_adjust(wspace=0.0)  # Adjust layout to fit color bar
    cbar_ax = fig.add_axes([0.05, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # increase tick size in cbar to make them more visible

    cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(axis='y', labelsize=14, width=2, length=6)  # Major ticks
    cbar.ax.tick_params(axis='y', which='minor', width=1, length=6)  # Minor ticks

    # cbar.ax.tick_params(labelsize=40)
    # cbar.set_label(label=label, fontsize=16)

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{label}_ellipsoids_cof_{fixed_cof}_I_{fixed_I}.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_pdf_thetax_grid(cofs, Is, aspect_ratios, output_dir):
    """
    Create a grid of subplots showing PDF of angular distributions for different coefficients of friction and inertial numbers.
    
    Args:
    - cofs: List of coefficients of friction (one per row)
    - Is: List of inertial numbers (one per column)
    - aspect_ratios: List of aspect ratios to loop through
    - output_dir: Directory to save the figure
    """
    num_rows = len(cofs)
    num_cols = len(Is)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharex=True, sharey=True)
    
    colormap = plt.cm.RdYlBu
    num_colors = len(aspect_ratios)
    extreme_indices = np.concatenate([
        np.linspace(0, 0.3, num_colors // 2, endpoint=False),
        np.linspace(0.7, 1.0, num_colors - num_colors // 2)
    ])
    colors = [colormap(i) for i in extreme_indices]
    
    central_color = 'black'
    central_ap = 1.0
    
    if central_ap in aspect_ratios:
        central_index = aspect_ratios.index(central_ap)
        colors.insert(central_index, central_color)
    
    for row_idx, cof in enumerate(cofs):
        for col_idx, I in enumerate(Is):
            ax = axes[row_idx, col_idx] if num_rows > 1 and num_cols > 1 else (axes[row_idx] if num_cols > 1 else axes)
            
            for idx, ap in enumerate(aspect_ratios):
                filename = f'simple_shear_ap{ap}_cof_{cof}_I_{I}.pkl'
                if os.path.exists(filename):
                    with open(filename, 'rb') as file:
                        file_data = pickle.load(file)
                        pdf_data = file_data.get('pdf_thetax', None)
                        
                        if pdf_data is not None and pdf_data.shape == (180, 2):
                            pdf_values = pdf_data[:, 1]
                            bin_centers = pdf_data[:, 0]
                            
                            if ap < 1:
                                bin_centers += np.pi/2
                                bin_centers = np.where(bin_centers > np.pi/2, bin_centers - np.pi, bin_centers)
                                sorted_indices = np.argsort(bin_centers)
                                bin_centers = bin_centers[sorted_indices]
                                pdf_values = pdf_values[sorted_indices]
                            if ap != 1.0:
                                ax.plot(bin_centers, pdf_values, label=f'$\u03b1={ap}$', color=colors[idx], linewidth=2)
                            else:
                                # plot a constant homogneous distribution
                                pdf_values = 1/np.pi * np.ones_like(bin_centers)
                                ax.plot(bin_centers, pdf_values, label=f'$\u03b1={ap}$', color=colors[idx], linewidth=2)
            
            ax.set_title(f'$\\mu_p={cof}$, $I={I}$', fontsize=14)
            ax.xaxis.set_major_locator(MultipleLocator(base=np.pi/4))
            ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            if row_idx == num_rows - 1:
                ax.set_xlabel('$\\theta_x$ [rad]', fontsize=14)
            if col_idx == 0:
                ax.set_ylabel('$\\psi[\\theta_x]$', fontsize=14)
    
    # fig.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.05, 1.05), ncol=min(4, len(aspect_ratios)))
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'pdf_thetax_cof_subplots.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

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
]

# Local force histograms to be normalized by p * A
local_histogram_keys = [
    'local_normal_force_hist_cp',  
    'local_tangential_force_hist_cp',
    'contacts_hist_cont_point_local'
]

# ellipsoids_keys = ['contacts_hist_cont_point_local', 'total_area']
# operation = lambda local_hist, area:  local_hist / max(local_hist)

fric_mob_keys = ['local_tangential_force_hist_cp', 'local_normal_force_hist_cp']
ratio_nt_operation = lambda tangential, normal:  tangential / normal

pow_diss_keys = ['power_dissipation_normal', 'power_dissipation_tangential', 'total_area', 'area_adjustment_ellipsoid', 'total_normal_dissipation', 'total_tangential_dissipation', 'contacts_hist_cont_point_local', 'bin_counts_power']
pow_diss_operation = lambda normal_hist, tangential_hist,  total_area, area_adjustment, total_normal, total_tangential, pdf_contact, bin_count: (
    (normal_hist + tangential_hist)* pdf_contact/2 * total_area**2 /
    (area_adjustment* (total_normal + total_tangential)) )
    # (normal_hist + tangential_hist)* pdf_contact/2 * total_area / (total_normal + total_tangential))

pow_diss_ratio_keys = ['power_dissipation_tangential', 'power_dissipation_normal']


equal_op= lambda x: x 
multiply_op = lambda x, y: x*y

# Create the plots
# cmap for grayscale 
# cmap = plt.cm.Greys
# cmap = plt.cm.GnBu
cmap = plt.cm.inferno

# select a range in colormap

cmap = cmap.reversed()

create_plots(data)
# plot_shear_stress_normal_tangential(data, Is, cofs, output_dir)
# plot_fabric_eigenvectors_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios)
bins_local = np.linspace(0, 90, 10)  # Define your bins for local data (10 values)
cofs = [10.0]
Is = [0.01]
# aspect_ratios = [0.5, 0.67, 1.0, 1.5, 2.0]
# aspect_ratios = [0.5, 1.0, 2.0]
# for I in Is:
#     for cof in cofs:
#         pass
#         create_polar_plots_varying_ap(fixed_cof=cof, fixed_I=I, histogram_keys=histogram_keys, pdf_keys=pdf_keys, local_histogram_keys=local_histogram_keys)
#         plot_ellipsoids_with_combined_data_ap(cof, I, aspect_ratios, variable_keys=['contacts_hist_cont_point_local', 'total_area'], operation=multiply_op, label='surface_pdf', bins_local=bins_local, output_dir=output_dir, scale="log", cmap=cmap)
#         plot_ellipsoids_with_combined_data_ap(cof, I, aspect_ratios, variable_keys=fric_mob_keys, operation=ratio_nt_operation, label='Mobilized Friction', bins_local=bins_local, output_dir=output_dir, scale='linear', cmap=cmap)
#         plot_ellipsoids_with_combined_data_ap(cof, I, aspect_ratios, variable_keys=pow_diss_keys, operation=pow_diss_operation, label='Power area density', bins_local=bins_local, output_dir=output_dir, scale='log', cmap =cmap)
#         plot_ellipsoids_with_combined_data_ap(cof, I, aspect_ratios, variable_keys=pow_diss_ratio_keys, operation=ratio_nt_operation, label='Power tangential normal', bins_local=bins_local, output_dir=output_dir, scale="linear", cmap=cmap)  
        # # plot_ellipsoids_with_combined_data_ap(fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=operation, label='surface_pdf_normalized_max', bins_local=bins_local)
        # plot_fabric_eigenvectors_ap(fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios)
        # plot_fabric_eigenvectors_2d(fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios)
# plot_dissipation_vs_aspect_ratio(data, ['total_normal_dissipation', 'total_tangential_dissipation'])
# plot_dissipation_local_vs_global(data)
# plot_pdf_thetax(fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios)
# plot_pdf_thetax_grid(cofs, Is, aspect_ratios, output_dir)

os.chdir("..")

# print("Plots saved to parametric_plots")
