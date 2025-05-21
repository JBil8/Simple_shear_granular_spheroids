import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.colors import to_rgba
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse
from matplotlib.transforms import ScaledTranslation
import matplotlib as mpl
from matplotlib.legend_handler import HandlerPatch

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}'


def phiI(x, phic, cphi, betaphi):
    return phic + cphi * x**betaphi

def muI(x, muc, cmu, I0):
    return muc + cmu / (1+ (I0/x))

# Set the ticks as functions of pi
def pi_formatter(x, pos):
    fractions = {-np.pi/2: r'$-\pi/2$', -np.pi/4: r'$-\pi/4$', 0: '0', np.pi/4: r'$\pi/4$', np.pi/2: r'$\pi/2$'}                 
    return fractions.get(x, f'{x/np.pi:.2g}π')

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Get aspect ratio from the original ellipse
        orig_width = orig_handle.width
        orig_height = orig_handle.height
        ap = orig_height / orig_width if orig_width != 0 else 1

        # Apply your specific logic
        if ap == 1:
            ellipse_width = 13
            ellipse_height = 13
        elif ap > 1:
            ellipse_width = 13
            ellipse_height = 13 * ap
        else:  # ap < 1
            ellipse_width = 13 / ap
            ellipse_height = 13

        center = xdescent + width / 2, ydescent + height / 2
        p = Ellipse(xy=center,
                    width=ellipse_width,
                    height=ellipse_height,
                    facecolor=orig_handle.get_facecolor(),
                    edgecolor=orig_handle.get_edgecolor(),
                    linewidth=orig_handle.get_linewidth())
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

# Define the parameters
cofs = [0.4]
# cofs = [0.0, 0.01, 0.1, 0.4, 1.0,  10.0]
# aspect_ratios = [1.0]
Is = [0.1, 0.046, 0.022, 0.01, 0.0046, 0.0022, 0.001] 
# Is = [0.1, 0.046, 0.022, 0.01] #, 0.0046, 0.0022, 0.001] 
# aspect_ratios = [0.4, 0.50, 0.67,  1.0, 1.5, 2.5]
# aspect_ratios = [0.33, 0.4, 0.5, 0.56, 0.67, 0.83, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
aspect_ratios = [0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0]
# aspect_ratios_to_show = [0.33, 0.4, 0.5, 0.56, 0.67, 0.83, 1.0, 1.2 ,1.5, 1.8, 2.0, 2.5, 3.0]
aspect_ratios_to_show = [0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0]
# keys_of_interest = ['thetax_mean', 'percent_aligned', 'S2', 'Z', 'phi', 'Nx_diff', 'Nz_diff',
#                      'Omega_z', 'p_yy', 'p_xy', 'Dyy', 'Dzz', 'total_normal_dissipation',
#                        'total_tangential_dissipation', 'percent_sliding','omega_fluctuations', 'vx_fluctuations', 'vy_fluctuations', 'vz_fluctuations'] 
# keys_of_interest = ['p_yy', 'p_xy', 'phi', 'Z', 'Nx_diff', 'Nz_diff']
keys_of_interest = ['p_yy', 'p_xy', 'Nx_diff']
# keys_of_interest = ['vx_fluctuations','vy_fluctuations', 'vz_fluctuations', 'omega_fluctuations']

# Initialize a data holder
data = {key: [] for key in keys_of_interest}
data['inertialNumber'] = []
data['cof'] = []
data['ap'] = []
data['shear_rate'] = [] 

os.chdir("./output_data_26_02_2025")

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
                        data['shear_rate'].append(file_data.get('shear_rate', None))  # Append shear_rate
                        for key in keys_of_interest:
                            data[key].append(file_data.get(key, None))
            else:
                print(f"File {filename} does not exist.")

output_dir = "../parametric_plots_powder_grains"
#output_dir = "../parametric_plots"
os.makedirs(output_dir, exist_ok=True)

# Function to create plots
def create_plots(data):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

     # Create a colormap
    colormap = plt.cm.RdYlBu
    # colormap = plt.cm.viridis
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

    legend_elements = []
    intial_guess = [1.0, 1.0, 1.0]
    for cof in cofs:
        #get xlim and ylim
        global_xmin = float('inf')
        global_xmax = float('-inf')
        global_ymin = float('inf')
        global_ymax = float('-inf')
        # Plot mu = p_xy / p_yy
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        fig.set_size_inches(10, 8)
        fig.subplots_adjust(left=0.2)  # Ensures consistent left margin

        for ap, color in zip(aspect_ratios, colors):
            x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
            p_xy_values = [value for value, aspect_ratio, coef in zip(data['p_xy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
            p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
            if x_values and p_xy_values and p_yy_values:  # Ensure that the lists are not empty
                mu_values = [pxy / pyy if pyy != 0 else None for pxy, pyy in zip(p_xy_values, p_yy_values)]
                global_xmin = 0.99*min(global_xmin, min(x_values))
                global_xmax = 1.01*max(global_xmax, max(x_values))
                global_ymin = 0.99*min(global_ymin, min(mu_values))
                global_ymax = 1.01*max(global_ymax, max(mu_values))
                plt.xscale('log')
                if ap in aspect_ratios_to_show:
                    plt.plot(x_values, mu_values, label=f'$\\alpha={ap}$', color=color, linestyle='None')
                    popt, pcov = curve_fit(muI, x_values, mu_values, p0=intial_guess, method='trf', x_scale = [1, 1, 1], bounds=([0, 0, 0], [1, 1, 1]))
                    # print(f"cof={cof}, ap={ap}, popt={popt}")
                    plt.plot(x_values, muI(x_values, *popt), color=color, linestyle='--', linewidth=3)
        # plt.legend(fontsize=30, loc='upper right', bbox_to_anchor=(1.3, 1))
        for ap, color in zip(aspect_ratios, colors):
            x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
            p_xy_values = [value for value, aspect_ratio, coef in zip(data['p_xy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
            p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
            mu_values = [pxy / pyy if pyy != 0 else None for pxy, pyy in zip(p_xy_values, p_yy_values)]
            plt.xlim(global_xmin, global_xmax)
            plt.ylim(global_ymin, global_ymax)
                # use the axis scale tform to figure out how far to translate 
                # Add ellipses at each point   
                
            if ap in aspect_ratios_to_show:
                for x, y in zip(x_values, mu_values):
                    # log_x = np.log10(x)
                    if ap >1: 
                        ellipse_width = 0.02  # Width as a fraction of figure width
                        ellipse_height = ellipse_width*ap  # Height scaled by aspect ratio
                    else:
                        ellipse_height = 0.02
                        ellipse_width = ellipse_height/ap

                    trans = ax.transData.transform((x, y))  # Data to display coordinates
                    inv = fig.transFigure.inverted()       # Invert figure transform
                    fig_coords = inv.transform(trans)  # Data to figure coordinates

                    face_rgba = to_rgba(color, alpha=1.0)  # Convert color to RGBA with alpha

                    ellipse = Ellipse(
                        fig_coords, width=ellipse_width, height=ellipse_height,
                        transform=fig.transFigure, facecolor=face_rgba, edgecolor='black', linewidth=2
                    )
                    fig.patches.append(ellipse)  # Add ellipse to the figure patches
                legend_elements.append((ellipse, f"${ap}$"))

                        

        # plt.title(f'$\\mu_p={cof}$', fontsize=30)
        # plt.title("Effective Friction / Stress Anisotropy", fontsize=30)
        #increase font size


        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis='x', which='both', direction='in', length=6, width=2)
        ax.set_xlim(0.0009, 0.12)
        ax.set_xticklabels([])

        # plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        # plt.xlabel('$I$', fontsize=30)
        # plt.ylabel('$\\mu = \\sigma_{xy} /\\sigma_{yy}$', fontsize=30)
        plt.ylabel('$\\mu$', fontsize=30, labelpad=10)



        # add legend with the same ellipse shape as the symbol used. Put the legend outside on the top
        handles, lables = zip(*legend_elements)
        # ax.legend(handles, lables, fontsize=20, bbox_to_anchor=(1.0, 1.15), ncol = 4, frameon =False, handler_map={Ellipse: HandlerEllipse()})

        # fig.canvas.draw()  # Needed to compute label sizes
        # Get bounding box of the axis
        # bbox = ax.get_tightbbox(fig.canvas.get_renderer())

        # Compute adjusted width to keep x-axis consistent
        # fig_width = 10 + (bbox.x0 / fig.dpi)  # Adjust width by y-label width
        # fig.set_size_inches(fig_width, 8)  # Update figure size


        filename = f'mu_cof_{cof}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi = 300)
        plt.close()


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
            label = '$\\delta \\omega / \\dot{\\gamma}$'
        elif key == 'total_normal_dissipation' or key == 'total_tangential_dissipation':
            continue  # Skip these keys for now
        else: 
            label = key

        for cof in cofs:
            #get xlim and ylim
            global_xmin = float('inf')
            global_xmax = float('-inf')
            global_ymin = float('inf')
            global_ymax = float('-inf')
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
                elif key == 'percent_aligned':
                    y_values = [value * 100 for value in y_values] # Convert to percentageS
                elif key == 'Omega_z':
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [-2*value / shear_rate for value, shear_rate in zip(y_values, shear_rate_values)]
                    # print(y_values)
                    # plt.ylim(-0.1, 1.1)
                elif key == 'vx_fluctuations' or key == 'vy_fluctuations' or key == 'vz_fluctuations':
                    d_eq = (2+ap)/3
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [value / (shear_rate*d_eq) for value, shear_rate in zip(y_values, shear_rate_values)]
                elif key == "omega_fluctuations":
                    # print(y_values)
                    y_values_new = [np.sqrt(np.linalg.norm(value[1, :])) for value in y_values]
                    
                    y_values = y_values_new
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [1*value / shear_rate for value, shear_rate in zip(y_values, shear_rate_values)]

                #     y_values = y_values[0]
                elif key == 'phi':
                    # continue
                    intial_guess = [1, -1, 1]
                    popt, pcov = curve_fit(phiI, x_values, y_values, p0=intial_guess)
                
                global_xmin = 0.97*min(global_xmin, min(x_values))
                global_xmax = 1.03*max(global_xmax, max(x_values))
                global_ymin = min(global_ymin, min(y_values))
                global_ymax = max(global_ymax, max(y_values))
                global_ymin = global_ymin - 0.02*(global_ymax - global_ymin)
                global_ymax = global_ymax + 0.02*(global_ymax - global_ymin)
               
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)   
            plt.xticks(fontsize=30)
            plt.xscale('log')
            fig.set_size_inches(10, 8)
            fig.subplots_adjust(left=0.2)  # Ensures consistent left margin

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
                elif key == 'percent_aligned':
                    y_values = [value * 100 for value in y_values] # Convert to percentageS
                elif key == 'Omega_z':
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [-2*value / shear_rate for value, shear_rate in zip(y_values, shear_rate_values)]
                    # plt.ylim(-0.1, 1.1)
                elif key == 'vx_fluctuations' or key == 'vy_fluctuations' or key == 'vz_fluctuations':
                    d_eq = (2+ap)/3
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [value / (shear_rate*d_eq) for value, shear_rate in zip(y_values, shear_rate_values)]
                elif key == "omega_fluctuations":
                    # y_values_new = [np.sqrt(value[1, 2]) for value in y_values]
                    y_values_new = [np.sqrt(np.linalg.norm(value[1, :])) for value in y_values]
                    y_values = y_values_new
                    shear_rate_values = [shear_rate for shear_rate, aspect_ratio, coef in zip(data['shear_rate'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof]
                    y_values = [1*value / shear_rate for value, shear_rate in zip(y_values, shear_rate_values)]

                if x_values and y_values :  # Ensure that the lists are not empty
                    if key == 'phi':
                        intial_guess = [0.5, 0.5, 0.5]
                        popt, pcov = curve_fit(phiI, x_values, y_values, p0=intial_guess)
                        x_fit = np.linspace(min(x_values), max(x_values), 100)

                        plt.plot(x_fit, phiI(x_fit, *popt), color=color, linestyle='--', linewidth=3)

                    else:
                        aaa=1
                        plt.plot(x_values, y_values, label=f'$\\alpha={ap}$', color=color, linestyle='--', linewidth=3)
                    plt.xlim(global_xmin, global_xmax)
                    plt.ylim(global_ymin, global_ymax)
                    # use the axis scale tform to figure out how far to translate 
                    # Add ellipses at each point         
                    for x, y in zip(x_values, y_values):
                
                        if ap >1: 
                            ellipse_width = 0.02  # Width as a fraction of figure width
                            ellipse_height = ellipse_width*ap  # Height scaled by aspect ratio
                        else:
                            ellipse_height = 0.02
                            ellipse_width = ellipse_height/ap

                        trans = ax.transData.transform((x, y))  # Data to display coordinates
                        inv = fig.transFigure.inverted()       # Invert figure transform
                        fig_coords = inv.transform(trans)  # Data to figure coordinates

                        face_rgba = to_rgba(color, alpha=1.0)  # Convert color to RGBA with alpha

                        ellipse = Ellipse(
                            fig_coords, width=ellipse_width, height=ellipse_height,
                            transform=fig.transFigure, facecolor=color, alpha=1.0, edgecolor='black', linewidth=2
                        )
                        # fig.patches.append(ellipse)  # Add ellipse to the figure patches
                        ax.add_patch(ellipse)


            
            
            # Show ticks at the top and bottom
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(axis='x', which='both', direction='in', length=6, width=2)
            ax.tick_params(axis='y', direction='in', length=6, width=2)
            plt.yticks(fontsize=30)
            # ax.set_ylim(0.9*global_ymin, 1.2*global_ymax)


            # plt.legend(fontsize=30)
            plt.ylabel(label, fontsize=30, labelpad=10)

            # fig.canvas.draw()  # Needed to compute label sizes
            # Get bounding box of the axis

            ax.set_xlim(0.0009, 0.12)
            bbox = ax.get_tightbbox(fig.canvas.get_renderer())

            # Compute adjusted width to keep x-axis consistent
            # fig_width = 10 + (bbox.x0 / fig.dpi)  # Adjust width by y-label width
            # fig.set_size_inches(fig_width, 8)  # Update figure size

            
            # plt.legend(fontsize=30, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 4)   
            plt.xlabel('$I$', fontsize=30)
            # ax.set_xticklabels([])
            
           
            # plt.title(f'$\\mu_p={cof}$', fontsize=30)
            # plt.title('Nematic order', fontsize=30)
            filename = f'{key}_cof_{cof}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300)#, bbox_inches='tight')
            plt.close()

    # Plot total_tangential_dissipation / (total_normal_dissipation + total_tangential_dissipation)
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
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=30, loc='upper right', bbox_to_anchor=(1, 1.4), ncols = 2)   
        plt.xlabel('$I$', fontsize=30)
        plt.ylabel("$P_t/P_n$", fontsize=30)  # Y-axis label for the ratio
        plt.title(f'$\\mu_p={cof}$', fontsize=30)
        
        # Save the plot
        filename = f'power_dissipation_ratio_cof_{cof}.png'
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
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

def plot_pdf_thetax(data, fixed_cof, fixed_I, aspect_ratios):
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
                    plt.plot(bin_centers, pdf_values, label=f'$\\alpha={ap}$', color=colors[idx], linewidth=2)

    # Customize the plot
    plt.xlabel('$\\theta_x$ [rad]', fontsize=18)
    plt.ylabel('$\\psi[\\theta_x]$', fontsize=18)
    plt.gca().xaxis.set_major_locator(MultipleLocator(base=np.pi/4))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    plt.gca().tick_params(axis='x', labelsize=14)
    plt.gca().tick_params(axis='y', labelsize=14)

    # plt.title(f'Circular PDF of Angular Distribution ($\\mu_p={fixed_cof}$, $I={fixed_I}$)', fontsize=16)
    plt.legend(fontsize=16, ncol=2)
    # plt.grid(True)
    
    # Save and show the plot
    plt.savefig(output_dir + "/pdf_thetax_cof_" + str(fixed_cof) + "_I_" + str(fixed_I) + '.png', bbox_inches='tight', dpi = 300)
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
                    cbar.set_label(label=label, fontsize=30)

                    # Save the figure
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(f'{output_dir}/{label}_ellipsoid_ap_{ap}_cof_{fixed_cof}_I_{fixed_I}.png',
                                bbox_inches='tight', dpi=300)
                    plt.close()

def plot_dissipation_ratio_regime_diagram(data):
    """
    Plot a diagram with cof on the x axis and I on the y axis 
    Plot the line where P_t/P_n is equal to 0.5
    Since I have simulations only for 
    """
    pass

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
cof = 0.4
I = 0.01

# create_polar_plots_varying_ap(data, fixed_cof=cof, fixed_I=I, histogram_keys=histogram_keys, pdf_keys=pdf_keys, local_histogram_keys=local_histogram_keys)
# plot_fabric_eigenvectors_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios)
bins_local = np.linspace(0, 90, 10)  # Define your bins for local data (10 values)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=lambda x, y: x / y, label='Mobilized Friction', bins_local=bins_local)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=operation, label='Surface Pdf', bins_local=bins_local)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=operation, label='Total_Power_dissipation', bins_local=bins_local)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=operation, label='surface_pdf_normalized_max', bins_local=bins_local)
# plot_ellipsoids_with_combined_data_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios, variable_keys=ellipsoids_keys, operation=operation, label='surface_pdf_normalized_max', bins_local=bins_local)

create_plots(data)

# plot_pdf_thetax(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios)

os.chdir("..")

# print("Plots saved to parametric_plots")
