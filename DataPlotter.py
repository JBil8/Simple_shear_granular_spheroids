import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib.cm import viridis, plasma, cividis, inferno
from matplotlib import cm
from matplotlib import cm, colors as mcolors

class DataPlotter:
    def __init__(self, ap, cof, parameter=None, value=None, muw=None, vwall=None, fraction=None, phi=None):   
        self.ap = str(ap) 
        self.cof = str(cof)
        self.parameter = parameter
        self.value = str(value)
        self.muw = muw
        self.vwall = vwall
        self.fraction = fraction
        self.phi = phi
        self.directory = "output_plots_stress_hertz/"
        os.makedirs(self.directory, exist_ok=True)

    def plot_time_variation(self, results_dict, df):
        # Use LaTeX fonts
        rcParams['text.usetex'] = True
        rcParams['font.family'] = 'serif'
        
        strain = df['shear_strain']
        pressure_variables = ['press', 'p_yy', 'p_xy', 'Nx_diff', 'Nz_diff']
        other_variables = ['phi', 'Omega_z', 'inertialNumber', 'msd']
        
        colormap_pressure = plasma(np.linspace(0, 1, len(pressure_variables)))
        colormap_other = viridis(np.linspace(0, 1, len(other_variables)))

        # Plot pressure variables in the same plot
        plt.figure(figsize=(10, 6))
        for i, variable in enumerate(pressure_variables):
            if variable in df.columns and variable in results_dict.keys():
                color = colormap_pressure[i]
                mean = results_dict[variable] * np.ones(len(strain))
                plt.plot(strain, df[variable], label=f'${variable}$ (time variation)', linestyle='-', linewidth=1, color=color)
                plt.plot(strain, mean, label=f'${variable}$ (average)', linestyle='--', linewidth=2, color=color)
        
        plt.xlabel('$\dot{\gamma}t$')
        plt.ylabel('Value')
        plt.title('Pressure Variables Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.directory + 'pressure_variables_alpha_' + self.ap + '_cof_' + self.cof + '_I_' + self.value + '.png')

        # Plot other variables in separate plots
        for i, variable in enumerate(other_variables):
            if variable in df.columns and variable in results_dict.keys():
                if variable == 'Omega_z':
                    results_dict[variable] = results_dict[variable]/(results_dict['max_vx_diff']/(2*results_dict['box_height']))
                    df[variable] = df[variable]/(results_dict['max_vx_diff']/(2*results_dict['box_height']))
                
                plt.figure(figsize=(10, 6))
                color = colormap_other[i]
                mean = results_dict[variable] * np.ones(len(strain))
                plt.plot(strain, df[variable], label=f'${variable}$ (time variation)', linestyle='-', linewidth=1, color=color)
                plt.plot(strain, mean, label=f'${variable}$ (average)', linestyle='--', linewidth=2, color=color)
                
                plt.xlabel('$\dot{\gamma}t$')
                plt.ylabel('Value')
                plt.title(f'{variable} Over Time')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'{self.directory}{variable}_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')

        # Plot msd as function of strain in a single plot with both msdY and msdZ
        if 'msdY' in df.columns or 'msdZ' in df.columns:
            plt.figure(figsize=(10, 6))

            # Plot msdY if available
            if 'msdY' in df.columns:
                plt.plot(strain, df['msdY'], label='msdY (time variation)', linestyle='-', linewidth=1, color='magenta')

            # Plot msdZ if available
            if 'msdZ' in df.columns:
                plt.plot(strain, df['msdZ'], label='msdZ (time variation)', linestyle='-', linewidth=1, color='blue')

            # Labels and Title
            plt.xlabel('$\\dot{\\gamma}t$')
            plt.ylabel('$\\langle y^2 \\rangle$')
            plt.title('MSDY and MSDZ Over Time')
            plt.legend()
            plt.grid(True)
            # plt.show()
            # Save the figure
            plt.savefig(self.directory + 'msdYZ_alpha_' + self.ap + '_cof_' + self.cof + '_I_' + self.value + '.png')
            plt.close()

    def plot_variable(self, bins, averages, std_devs, variable):
        plt.errorbar(averages, bins / np.max(bins) - 0.5, xerr=std_devs, fmt='-o', capsize=5, label=variable)
        plt.ylabel('Y/H (height)')
        plt.xlabel(f'{variable} (averaged)')
        plt.title(f'Average {variable} with Standard Deviation (excluding first 1/11th)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.directory}{variable}_avg_std_dev_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')

    def plot_averages_with_std(self, results_dict):
        velocity_variables = ['vx', 'vy', 'vz']
        stress_variables = ['v_pxx_loc', 'v_pyy_loc', 'v_pzz_loc', 'v_pxy_loc']
        
        # Plot velocity variables in the same plot
        plt.figure(figsize=(10, 6))
        colormap = cividis(np.linspace(0, 1, len(velocity_variables)))
        for i, variable in enumerate(velocity_variables):
            avg_key = f'{variable}_avg'
            std_dev_key = f'{variable}_std_dev'
            if avg_key in results_dict and std_dev_key in results_dict:
                self.plot_variable_on_same_plot(results_dict['bins'], results_dict[avg_key], results_dict[std_dev_key], variable, colormap[i])
        plt.ylabel('Y/H (height)')
        plt.xlabel('Velocity (averaged)')
        plt.title('Velocity Variables with Standard Deviation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.directory}velocity_avg_std_dev_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        plt.close()
        # plt.show()

        # Plot stress variables in the same plot
        plt.figure(figsize=(10, 6))
        colormap = cividis(np.linspace(0, 1, len(stress_variables)))
        for i, variable in enumerate(stress_variables):
            avg_key = f'{variable}_avg'
            std_dev_key = f'{variable}_std_dev'
            if avg_key in results_dict and std_dev_key in results_dict:
                self.plot_variable_on_same_plot(results_dict['bins'], results_dict[avg_key], results_dict[std_dev_key], variable, colormap[i])
        plt.ylabel('Y/H (height)')
        plt.xlabel('Stress (averaged)')
        plt.title('Stress Variables with Standard Deviation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.directory}stress_avg_std_dev_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        # plt.show()
        plt.close()

        # Plot Ncount in a separate plot
        variable = 'Ncount'
        avg_key = f'{variable}_avg'
        std_dev_key = f'{variable}_std_dev'
        if avg_key in results_dict and std_dev_key in results_dict:
            plt.figure(figsize=(10, 6))
            self.plot_variable(results_dict['bins'], results_dict[avg_key], results_dict[std_dev_key], variable)
        
        # Plot density_mass in a separate plot
        variable = 'density_mass'
        avg_key = f'{variable}_avg'
        std_dev_key = f'{variable}_std_dev'
        if avg_key in results_dict and std_dev_key in results_dict:
            plt.figure(figsize=(10, 6))
            self.plot_variable(results_dict['bins'], results_dict[avg_key], results_dict[std_dev_key], variable)

    def plot_variable_on_same_plot(self, bins, averages, std_devs, variable, color):
        plt.errorbar(averages, bins / np.max(bins) - 0.5, xerr=std_devs, fmt='-o', capsize=5, label=variable, color=color)

    def plot_histogram(self, bins, hist, variable, label='$\\theta_x [^\\circ]$'):
        """
        Plot a 2D histogram with the median bin highlighted and a line at angle tan^-1(1/ap).
        
        Args:
        - bins (array): The bin edges for the histogram.
        - hist (array): The histogram values (heights of bins).
        - variable (str): Name of the variable being plotted.
        """
        # Calculate the bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist = np.deg2rad(hist)
        # Calculate the cumulative sum of the histogram
        cumulative_hist = np.cumsum(hist)
        # Find the index where the cumulative sum crosses half of the total count
        total_count = cumulative_hist[-1]
        median_index = np.searchsorted(cumulative_hist, total_count / 2)

        # Get the value corresponding to the median index
        median_value = hist[median_index]

        # Calculate the angle corresponding to tan^-1(1/ap)
        if float(self.ap) > 1:
            theta_ap = np.arctan(1 / float(self.ap))
        else:
            theta_ap = -np.arctan(1 / float(self.ap))
        
        plt.figure(figsize=(10, 5))

        # Plot the histogram
        plt.hist(bin_centers, bins=bins, weights=hist, color='gray', alpha=0.7, edgecolor='k')

        # Highlight the bin with the median probability
        plt.hist([bin_centers[median_index]], bins=bins, weights=[median_value], color='red', alpha=0.9, edgecolor='k')

        if label == '$\\theta_x [^\\circ]$':
            # Draw a vertical line at the angle corresponding to tan^-1(1/ap)
            plt.axvline(x=theta_ap, color='blue', linestyle='--', linewidth=2, label=r'$\theta = \tan^{-1}(1/\alpha)$')

        # Set axis labels and title
        plt.xlabel(label)
        plt.ylabel('Probability')
        plt.title(f'2D Histogram of {variable} with Highlighted Median', fontsize=14)
        plt.legend()
        plt.savefig(f'{self.directory}histogram_{variable}_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        # plt.show()
        plt.close()

    def plot_histogram_ellipsoid_flat(self, hist_flat, num_bins_theta, num_bins_phi, 
                                      title, label):
        """
        Plot an ellipsoid with axis-symmetric stripes based on a flattened 2D histogram of theta and phi angles.

        Args:
        - hist_flat (1D array): The flattened 2D histogram values to map onto the ellipsoid.
        - num_bins_theta (int): Number of bins for the theta angle.
        - num_bins_phi (int): Number of bins for the phi angle
        """
        ap = float(self.ap)
        # Reshape the flattened histogram back to 2D
        hist_local = hist_flat.reshape((num_bins_theta, num_bins_phi))

        # Use the bin edges for u and v to correspond directly to phi and theta bins
        u = np.linspace(0, 2 * np.pi, num_bins_phi + 1)  # u: azimuthal angle (0 to 2pi)
        v = np.linspace(0, np.pi, num_bins_theta + 1)  # v: polar angle (0 to pi)

        # Parametric equations for the ellipsoid
        x = np.outer(np.sin(v), np.cos(u))
        y = np.outer(np.sin(v), np.sin(u))
        z = ap * np.outer(np.cos(v), np.ones_like(u))

        # Interpolate the colors from the 2D histogram to the ellipsoid surface
        # Use bin centers for u and v and directly map the histogram values
        colors = np.zeros((len(v), len(u)))

        # Map the colors directly based on the reshaped histogram data
        for i in range(num_bins_theta):
            for j in range(num_bins_phi):
                colors[i, j] = hist_local[i, j]

        # Normalize the colors for plotting
        colors = colors / np.max(colors)

        # Plotting the ellipsoid with the histogram-based colormap
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Use a colormap to plot the surface
        ax.plot_surface(x, y, z, facecolors=plt.cm.inferno(colors), rstride=1, cstride=1, antialiased=True, alpha=1.0)

        # Add some labels and set the aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')
        #ax.axis('off')

        # Adjust the viewing angle depending on the ellipsoid type
        ax.view_init(elev=-26, azim=-6, roll=30)

        # Add a colorbar
        mappable = cm.ScalarMappable(cmap=cm.inferno)
        mappable.set_array(hist_flat)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label=label, fontsize=20)

        # Set the title
        plt.title(title)
        plt.savefig(f'{self.directory}ellipsoid_{title}_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        # plt.show()
        plt.close()

    def plot_histogram_ellipsoid(self, hist_local, bins_local, title, label, colormap='YlGnBu'):
        """
        Plot an ellipsoid with axis-symmetric stripes based on the histogram of angles.

        Args:
        - hist_local (array): The histogram values to map onto the ellipsoid.
        - bins_local (array): The bin edges for the angles.
        - title (str): Title for the plot.
        - label (str): Label for the color bar.
        - colormap (str): The colormap to use for the plot.
        """
        ap = float(self.ap)
        
        # Create a grid of u (azimuthal angle) and v (polar angle)
        u = np.linspace(0, 2 * np.pi, 20)  # azimuthal angle
        v = np.linspace(0, np.pi, 2 * len(bins_local)-1)  # use bins_local length for polar angle divisions

        # Parametric equations for the ellipsoid
        
        x = np.outer(np.sin(v), np.cos(u))
        y = np.outer(np.sin(v), np.sin(u))
        z = ap * np.outer(np.cos(v), np.ones_like(u))
      
        # Calculate the bin centers to map the colors
        bin_centers = (bins_local[:-1] + bins_local[1:]) / 2
        # Create a color array where each stripe corresponds to a histogram bin
        colors = np.zeros_like(z)

        # Define a normalization object using the original range of `hist_local`
        norm = mcolors.Normalize(vmin=np.min(hist_local), vmax=np.max(hist_local))
        # norm = mcolors.Normalize(vmin=0, vmax=2)
        # Map the histogram values to the stripes on the ellipsoid surface
        for i in range(len(bin_centers)):
            # Upper hemisphere (v from 0 to 90 degrees)
            indices_upper = (v >= np.radians(bins_local[i])) & (v < np.radians(bins_local[i + 1]))
            colors[indices_upper, :] = hist_local[i]

            # Mirror to the lower hemisphere (v from 90 to 180 degrees)
            indices_lower = (v >= np.radians(180 - bins_local[i + 1])) & (v < np.radians(180 - bins_local[i]))
            colors[indices_lower, :] = hist_local[i]

        # Plotting the ellipsoid with the histogram-based colormap
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Use a colormap to plot the surface, passing in the normalized color array
        cmap = cm.get_cmap(colormap)
        ax.plot_surface(x, y, z, facecolors=cmap(norm(colors)), rstride=1, cstride=1, antialiased=True, alpha=1.0)

        # Add some labels and set the aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Adjust the view based on ellipsoid type
        ax.view_init(elev=-26, azim=-6, roll=30)
        
        # Add a colorbar with the same normalization
        mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(hist_local)  # Attach data for color bar
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label=label, fontsize=20)
        plt.savefig(f'{self.directory}ellipsoid_{title}_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        # plt.show()
        plt.close()
    
    def plot_pdf(self, data, n_bins, variable, label='$\\theta_x [^\\circ]$', median_value = None):
        """
        Plot a 2D pdf with the median bin highlighted and a line at angle tan^-1(1/ap).
        Args:
        - bins (array): The bin edges for the histogram.
        - hist (array): The histogram values (heights of bins).
        - variable (str): Name of the variable being plotted.
        """
        data = np.ravel(data)

        
        hist, bins = np.histogram(data, bins=n_bins, density=True)  # Normalize to density
        
        # Calculate the angle corresponding to tan^-1(1/ap)
        if float(self.ap) > 1:
            theta_ap = np.arctan(1 / float(self.ap))
        else:
            theta_ap = -np.arctan(1 / float(self.ap))
        
        plt.figure(figsize=(10, 5))

        # Plot the histogram
        plt.hist(data, bins=bins, density=True, alpha=0.7, color='gray', edgecolor='k', label='Orientation Data')
        # Highlight the bin with the median probability
        plt.axvline(x=median_value, color='red', linestyle='--', linewidth=2, label=f'$\\theta_x^m$ = {median_value:.2f} rad')

        if label == '$\\theta_x [^\\circ]$':
            # Draw a vertical line at the angle corresponding to tan^-1(1/ap)
            plt.axvline(x=theta_ap, color='blue', linestyle='--', linewidth=2, label=r'$\theta = \tan^{-1}(1/\alpha)$')

        # Set axis labels and title
        plt.xlabel(label)
        plt.ylabel('Pdf')
        plt.title(f'2D pdf of {variable} with Highlighted Median', fontsize=14)
        plt.legend()
        plt.savefig(f'{self.directory}histogram_{variable}_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        # plt.show()
        plt.close()


    def plot_bar_histogram(self, bins, hist, variable):
        # Normalize the histogram values for the colormap
        norm = plt.Normalize(hist.min(), hist.max())

        plt.figure(figsize=(10, 2))

        # Use imshow to create a horizontal bar of color
        # Reshape the histogram values into a 2D array where each row represents the color for each bin
        data = np.array([hist])
        
        # Display the color bar using imshow
        plt.imshow(data, aspect='equal', cmap='plasma', extent=(bins[0], bins[-1], 0, 1))
        
        # Set axis labels and title
        plt.xlabel('Bins')
        plt.ylabel('$\\theta_x$', fontsize=12)
        #plt.title(f'1D Color Bar Histogram of {variable}', fontsize=14)

        # Hide y-axis since it's just a color bar
        plt.yticks([])
        plt.savefig(f'{self.directory}histogram_{variable}_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        # plt.show()
        plt.close()

    def plot_polar_histogram(self, bins, histogram, title, symmetry=False):
        """
        Plot a polar histogram with an option for symmetry.
        
        Args:
        - bins (array): The bin edges for the histogram.
        - histogram (array): The histogram values (heights of bins).
        - title (str): Title for the plot.
        - symmetry (bool): Flag to indicate if the histogram should be symmetric.
        """
        if symmetry:
            # Assuming the bins and histogram are for 0 to 90 degrees
            assert len(histogram) == len(bins) - 1, "The length of histogram should match the number of bins - 1."

            # Reverse the histogram for the 90 to 180 degrees
            histogram_mirror_90_180 = histogram[::-1]

            # Combine 0-90, 90-180, 180-270 (same as 90-0), and 270-360 (same as 0-90)
            extended_histogram = np.concatenate((histogram, histogram_mirror_90_180, histogram, histogram_mirror_90_180))
            
            # Adjust bins to match the extended histogram
            bin_width = bins[1] - bins[0]
            extended_bins = np.arange(0, 360 + bin_width, bin_width)
        else:
            # Use original histogram and bins for non-symmetric data
            extended_histogram = histogram
            extended_bins = bins

        # Calculate the bin centers for the extended bins
        bin_centers = (extended_bins[:-1] + extended_bins[1:]) / 2
        angles_rad = np.radians(bin_centers)  # Convert bin centers to radians

        # Create a polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_title(title, va='bottom')

        # Ensure the number of widths matches the number of bars
        bar_widths = np.diff(np.radians(extended_bins))

        # Plot data on the polar plot
        ax.bar(angles_rad, extended_histogram, width=bar_widths, bottom=0, color='b', alpha=0.6, edgecolor='k')

        # Set radius limit to the maximum histogram value
        ax.set_ylim(0, np.max(extended_histogram) * 1.1)  # 1.1 for a little extra space above the tallest bar

        plt.savefig(f'{self.directory}polar_histogram_{title}_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        # Display the plot
        # plt.show()
        plt.close()

    def plot_eulerian_velocities(self, data):
        #plot eulerian velocities at n_plots time steps in time
        n_plots = 10
        n_time_steps = data['theta_x'].shape[0]
        time_interval = int(n_time_steps/n_plots)
        y = np.linspace(0, 1, 10)
        fig2 = plt.figure(figsize=(10, 20))
        for i in range(n_plots):
            #ax = fig2.add_subplot(2, int(n_plots/2), i+1)
            color = plt.cm.viridis(i / n_plots)  # color will now be an RGBA tuple
            #plot the eulerian velocity at the given time steps skipping the first 2 strains
            if i != 0:
                plt.plot(data['eulerian_vx'][i*time_interval,:], y, label = f'strain  = {i*time_interval*20/n_time_steps+2:.2f}', color=color)
        plt.xlabel('Vx/V')
        plt.ylabel('y/H')
        plt.legend()
        fig2.suptitle('ap = ' + self.ap + ', cof = ' + self.cof + ', ' + self.parameter +  '=' + self.value)
        fig2.savefig(self.directory + 'simple_shear_ap' + self.ap + '_cof_' + self.cof + '_' + self.parameter + '_' + self.value + 'eulerian.png')
        plt.clf()
        #plt.show()
        
    def plot_force_distribution(self, force_normal_distribution, force_tangential_distribution):
    #force distrubution    
        plt.figure()
        plt.subplot(2,1,1)
        plt.title('ap = ' + self.ap + ', cof = ' + self.cof + ', ' + self.parameter +  '=' + self.value)
        plt.plot(force_normal_distribution[1][:-1], force_normal_distribution[0])
        plt.xlabel('F_n')
        plt.ylabel('P(F_n)')
        plt.subplot(2,1,2)
        plt.plot(force_tangential_distribution[1][:-1], force_tangential_distribution[0])
        plt.xlabel('F_t')
        plt.ylabel('P(F_t)')
        plt.savefig(self.directory + 'simple_shear_ap' + self.ap + '_cof_' + self.cof + '_' + self.parameter + '_' + self.value + 'force_distribution.png')
        plt.show()

    def plot_ellipsoids(self, step, data):
        #plot the ellipsoids in 3d
        
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(data['trackedGrainsShapeX'][0])):
            center = data['trackedGrainsPosition'][step, i, :]
            radii = [data['trackedGrainsShapeX'][step, i],
                     data['trackedGrainsShapeX'][step, i],
                     data['trackedGrainsShapeZ'][step, i]]
            rotation_matrix = data['trackedGrainsOrientation'][step, i, :, :]
            self.plot_single_ellipsoid(ax, center, radii, rotation_matrix)
        
        contact_points = data['trackedGrainsContactData'][step][:, 1:4]
        #self.scatter_contact_point_3D(ax, contact_points)

        ax.set_xlabel('X ')
        ax.set_ylabel('Y ')
        ax.set_zlabel('Z ')
        ax.set_title('ap = ' + self.ap + ', cof = ' + self.cof + ', ' + self.parameter +  '=' + self.value + ', step='+str(step))
        plt.savefig(self.directory + 'ellipsoids'+str(step)+'.png')
        plt.show()
        plt.close()

    def plot_single_ellipsoid(self, ax, center, radii, rotation_matrix, color='b', alpha=0.1):
        """
        Plot an ellipsoid in 3D.

        Parameters:
        - ax: Axes3D object (matplotlib)
        - center: Center of the ellipsoid (tuple or array)
        - radii: Semi-axes lengths (tuple or array)
        - rotation_matrix: Rotation matrix for orientation (2D array)
        - color: Color of the ellipsoid (default: 'b')
        - alpha: Transparency of the ellipsoid (default: 0.1)
        """
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # Parametric equations of the ellipsoid
        x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))

        # Rotate the ellipsoid based on the rotation matrix
        for i in range(len(x)):
            for j in range(len(x[i])):
                [x[i, j], y[i, j], z[i, j]] = rotation_matrix@[x[i, j], y[i, j], z[i, j]]

        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)
        # Set axis equal
        ax.set_box_aspect([np.ptp(coord) for coord in [x, y, z]])

    def plot_space_averages_all_cells(self, value, quantity= 'Velocity', component = None, symbol = '', nx_divisions = 40, ny_divisions=6):
            """
            Function to plot the space averages of the velocities in eulerian cells
            """
    
            if component == 0:
                axis_name = 'x'
            elif component == 1:
                axis_name = 'y'
            elif component == 2:
                axis_name = 'z'
            elif component == 3:
                axis_name = 'xy'
            elif component == 4:
                axis_name = 'xz'    
            elif component == 5:
                axis_name = 'yz'


            directory_name = (f"eulerian_ap_{self.ap}_mup_{self.cof}_muw_{self.muw}_"
                f"vw_{self.vwall}_phi_{self.phi}_frac_{self.fraction}_"
                f"nx_{nx_divisions}_ny_{ny_divisions}")

            os.makedirs(directory_name, exist_ok=True)
            os.chdir(directory_name)

            plt.figure(figsize=(20, 12))
            plt.xlabel('x/h', fontsize=36)
            plt.ylabel('y/H', fontsize=36)

            if component is None:
                plt.imshow(value.T, cmap='plasma', origin='lower', extent=[0, nx_divisions, 0, ny_divisions])            
            else:
                plt.imshow(value[:,:,component].T, cmap='plasma', origin='lower', extent=[0, nx_divisions, 0, ny_divisions])

            cbar = plt.colorbar(label=symbol, orientation='horizontal')
            cbar.set_label(label=symbol, fontsize=36)  # Set the label and increase the font size
            cbar.ax.tick_params(labelsize=36)  # Adjust the font size of the color bar tick labels

            plt.title(quantity, fontsize=20)
            #plt.xticks(np.linspace(0, nx_divisions, num=20), np.linspace(-self.box_x/2/self.h, self.box_x/2/self.h, num=20), fontsize=14)
            # Generating tick positions
            tick_positions = np.linspace(0, nx_divisions, num=15)

            # Generating tick labels and formatting them to display only the first couple of digits
            tick_labels = np.linspace(-self.box_x/2/self.h, self.box_x/2/self.h, num=15)
            formatted_tick_labels = [f"{label:.2f}" for label in tick_labels]

            # Setting the ticks with the formatted labels
            plt.xticks(tick_positions, formatted_tick_labels, fontsize=14)
            plt.yticks(np.linspace(0, ny_divisions, num=5), np.linspace(0, 1, num=5), fontsize=14)
            plt.savefig("eulerian_" + quantity.replace("/", "_") + ".png")            
            plt.close()
            os.chdir("..")

    def plot_time_series_forces(self, time, fx, fy, fz):
        """
        Function to plot the time series of the given quantity
        """
        plt.figure(figsize=(20, 12))
        plt.plot(time, fx, label='Fx')
        plt.plot(time, fy, label='Fy')
        plt.plot(time, fz, label='Fz')
        plt.xlabel('Time [s]')
        plt.ylabel('Force [N]')
        plt.legend()
        plt.title('fraction = ' + str(self.fraction) + ', ap = ' + str(self.ap) + ', cof = ' + str(self.cof) + ', muw = ' + str(self.muw) + ', vwall = ' + str(self.vwall))
        plt.savefig("time_series_force_" + ', ap = ' + str(self.ap) + ', cof = ' + str(self.cof) + ', muw = ' + str(self.muw) + ', vwall = ' + str(self.vwall) + ".png")
        plt.show()

    def plot_time_series_energy(self, time, tke, rke):
        """
        Function to plot the time series of the given quantity
        """
        plt.figure(figsize=(20, 12))
        plt.plot(time, tke, label='TKE')
        plt.plot(time, rke, label='RKE')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.legend()
        plt.title('fraction = ' + str(self.fraction) + ', ap = ' + str(self.ap) + ', cof = ' + str(self.cof) + ', muw = ' + str(self.muw) + ', vwall = ' + str(self.vwall))
        plt.savefig("time_series_energy_" + ', ap = ' + str(self.ap) + ', cof = ' + str(self.cof) + ', muw = ' + str(self.muw) + ', vwall = ' + str(self.vwall) + ".png")
        plt.show()

    # def scatter_contact_point_3D(self, ax, points):
    #     """
    #     Plot a scatter of points in 3D.

    #     Parameters:
    #     - ax: Axes3D object (matplotlib)
    #     - points: 3D points (tuple or array)
    #     """
    #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4)