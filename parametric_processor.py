import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Define the parameters
cofs = [0.0, 0.4, 1.0]
# Is = [0.0316, 0.01, 0.00316, 0.001,0.000316, 0.0001]
Is = [0.1, 0.046, 0.022, 0.01, 0.0046, 0.0022, 0.001] 
aspect_ratios = [0.33, 0.40, 0.50, 0.56, 0.67, 0.83, 1.0,
                  1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

keys_of_interest = ['thetax_median', 'percent_aligned', 'S2', 'Z', 'phi', 'Nx_diff', 'Nz_diff', 'Omega_z', 'p_yy', 'p_xy', 'Dyy', 'total_normal_dissipation', 'total_tangential_dissipation']
#keys_of_interest = ['Omega_z']

# Initialize a data holder
data = {key: [] for key in keys_of_interest}
data['inertialNumber'] = []
data['cof'] = []
data['ap'] = []
data['shear_rate'] = [] 

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
                        data['shear_rate'].append(file_data.get('shear_rate', None))  # Append shear_rate
                        for key in keys_of_interest:
                            data[key].append(file_data.get(key, None))
            else:
                print(f"File {filename} does not exist.")

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
            label = '$\\%_z$'
        elif key == 'thetax_median':
            label = '$\\theta_x$'
        elif key == 'S2':
            label = '$S_2$'
        elif key == 'Z':
            label = '$Z$'
        elif key == 'phi':
            label = '$\\phi$'
        elif key == 'Nx_diff':
            label = '$N_x/P$'
        elif key == 'Nz_diff':
            label = '$N_z/P$'
        elif key == 'Omega_z':
            label = '$2\\langle \\omega_z \\rangle /\\dot{\\gamma}$'
        elif key == 'Dyy':
            label = '$D_{yy}/\\dot{\\gamma}d^2$'
        elif key == 'total_normal_dissipation' or key == 'total_tangential_dissipation':
            pass  # Skip these keys for now
        else: 
              
            label = key

    for cof in cofs:
        plt.figure(figsize=(10, 8))
        for ap, color in zip(aspect_ratios, colors):
            x_values = [inertial_number for inertial_number, aspect_ratio, coef in zip(data['inertialNumber'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and inertial_number > 0]
            y_values = [value for value, aspect_ratio, coef in zip(data[key], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
            if key == 'thetax_median':
                y_values = [value * 180 / 3.20159 for value in y_values] # Convert to degrees
                if ap < 1:
                    y_values = [90+value for value in y_values]
            elif key == 'Nx_diff':
                p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                y_values = [nx / pyy if pyy != 0 else None for nx, pyy in zip(y_values, p_yy_values)]
            elif key == 'Nz_diff':
                p_yy_values = [value for value, aspect_ratio, coef in zip(data['p_yy'], data['ap'], data['cof']) if aspect_ratio == ap and coef == cof and value is not None]
                y_values = [-nz / pyy if pyy != 0 else None for nz, pyy in zip(y_values, p_yy_values)]
            elif key == 'Dyy':
                if ap > 1:
                    d_eq = ap ** (1 / 3)
                else:                     
                    d_eq = ap ** (-2 / 3)
                plt.ylim(0.1, .65)
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
                plt.ylim(-0.1, 1.1)

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

    # Plot total_tangential_dissipation / (total_normal_dissipation + total_tangential_dissipation)
    for cof in cofs:
        plt.figure(figsize=(10, 8))
        for ap, color in zip(aspect_ratios, colors):
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
                plt.plot(x_values, dissipation_ratios, label=f'$\\alpha={ap}$', color=color, linestyle='--', marker='o')

        # Set plot properties
        plt.xscale('log')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.xlabel('$I$', fontsize=20)
        plt.ylabel('Power Dissipation Ratio', fontsize=20)  # Y-axis label for the ratio
        plt.title(f'$\\mu_p={cof}$', fontsize=20)
        
        # Save the plot
        filename = f'power_dissipation_ratio_cof_{cof}.png'
        plt.savefig(os.path.join(output_dir, filename))
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
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(title, va='bottom')

    # Define a color map to differentiate lines
    colormap = plt.cm.viridis
    num_colors = len(histograms)
    colors = [colormap(i / num_colors) for i in range(num_colors)]

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

    plt.legend(loc='upper right', fontsize=12)
    plt.savefig(f'/home/jacopo/Documents/PhD_research/Data_processing_simple_shear/parametric_plots_oblate_prolate/polar_histogram_{title}_lines.png')
    plt.show()

def create_polar_plots_varying_ap(data, fixed_cof, fixed_I, histogram_keys, pdf_keys, local_histogram_keys):
    bins_global = np.linspace(-180, 180, 145)  # Define your bins for global data (144 values)
    bins_local = np.linspace(0, 90, 10)  # Define your bins for local data (10 values)

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

                    if histogram is not None and box_x_length is not None and box_z_length is not None:
                        # Calculate the area A and normalize by p * A
                        #area = box_x_length * box_z_length
                        area = total_area
                        normalized_histogram = histogram / (50 * area)  # p is always 50

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
                    
                    if local_histogram is not None and box_x_length is not None and box_z_length is not None:
                        # Calculate the area A and normalize by p * A
                        area = box_x_length * box_z_length
                        normalized_local_histogram = local_histogram / (50 * area)  # p is always 50

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
    colormap = plt.cm.viridis#tab20
    num_colors = len(aspect_ratios)
    colors = [colormap(i / num_colors) for i in range(num_colors)]

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
                    for i in range(3):
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

    # Add a legend (showing the aspect ratios)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')

    plt.title(f'Fabric Eigenvectors ($\\mu_p={fixed_cof}$, $I={fixed_I}$)')
    plt.savefig(f'/home/jacopo/Documents/PhD_research/Data_processing_simple_shear/parametric_plots_oblate_prolate/fabric_eigenvectors_cof_{fixed_cof}_I_{fixed_I}.png')
    plt.show()

# Force histograms to be normalized by p * A
histogram_keys = [
    'global_normal_force_hist',  
    'global_tangential_force_hist',
    'global_normal_force_hist_cp',  
    'global_tangential_force_hist_cp'
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
    'local_tangential_force_hist_cp'
]

# Create the plots
cof = 1.0
I = 0.00316

# create_polar_plots_varying_ap(data, fixed_cof=cof, fixed_I=I, histogram_keys=histogram_keys, pdf_keys=pdf_keys, local_histogram_keys=local_histogram_keys)
plot_fabric_eigenvectors_ap(data, fixed_cof=cof, fixed_I=I, aspect_ratios=aspect_ratios)


# Create the plots
create_plots(data)
os.chdir("..")

# print("Plots saved to parametric_plots")
