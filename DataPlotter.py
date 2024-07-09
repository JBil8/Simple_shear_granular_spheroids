import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib.cm import viridis, plasma, cividis


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
        
        plt.xlabel('Strain')
        plt.ylabel('Value')
        plt.title('Pressure Variables Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('output_plots/pressure_variables_alpha_' + self.ap + '_cof_' + self.cof + '_I_' + self.value + '.png')

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
                
                plt.xlabel('Strain')
                plt.ylabel('Value')
                plt.title(f'{variable} Over Time')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'output_plots/{variable}_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')

        # Plot msd as function of strain in a separate plot
        if 'msd' in df.columns:
            plt.figure(figsize=(10, 6))
            color = 'magenta'
            plt.plot(strain, df['msd'], label='msd (time variation)', linestyle='-', linewidth=1, color=color)
            
            plt.xlabel('Strain')
            plt.ylabel('msd')
            plt.title('MSD Over Time')
            plt.legend()
            plt.grid(True)
            plt.savefig('output_plots/msd_alpha_' + self.ap + '_cof_' + self.cof + '_I_' + self.value + '.png')

    def plot_variable(self, bins, averages, std_devs, variable):
        plt.errorbar(averages, bins / np.max(bins) - 0.5, xerr=std_devs, fmt='-o', capsize=5, label=variable)
        plt.ylabel('Y/H (height)')
        plt.xlabel(f'{variable} (averaged)')
        plt.title(f'Average {variable} with Standard Deviation (excluding first 1/11th)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output_plots/{variable}_avg_std_dev_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')

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
        plt.xlabel('Y/H (height)')
        plt.ylabel('Velocity (averaged)')
        plt.title('Velocity Variables with Standard Deviation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output_plots/velocity_avg_std_dev_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        plt.show()

        # Plot stress variables in the same plot
        plt.figure(figsize=(10, 6))
        colormap = cividis(np.linspace(0, 1, len(stress_variables)))
        for i, variable in enumerate(stress_variables):
            avg_key = f'{variable}_avg'
            std_dev_key = f'{variable}_std_dev'
            if avg_key in results_dict and std_dev_key in results_dict:
                self.plot_variable_on_same_plot(results_dict['bins'], results_dict[avg_key], results_dict[std_dev_key], variable, colormap[i])
        plt.xlabel('Y/H (height)')
        plt.ylabel('Stress (averaged)')
        plt.title('Stress Variables with Standard Deviation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output_plots/stress_avg_std_dev_alpha_{self.ap}_cof_{self.cof}_I_{self.value}.png')
        plt.show()

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
        fig2.savefig('output_plots/simple_shear_ap' + self.ap + '_cof_' + self.cof + '_' + self.parameter + '_' + self.value + 'eulerian.png')
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
        plt.savefig('output_plots/simple_shear_ap' + self.ap + '_cof_' + self.cof + '_' + self.parameter + '_' + self.value + 'force_distribution.png')
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
        plt.savefig('output_plots/ellipsoids'+str(step)+'.png')
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
            plt.xlabel('x/h', fontsize=18)
            plt.ylabel('y/H', fontsize=18)

            if component is None:
                plt.imshow(value.T, cmap='plasma', origin='lower', extent=[0, nx_divisions, 0, ny_divisions])            
            else:
                plt.imshow(value[:,:,component].T, cmap='plasma', origin='lower', extent=[0, nx_divisions, 0, ny_divisions])

            cbar = plt.colorbar(label=symbol, orientation='horizontal')
            cbar.set_label(label=symbol, fontsize=18)  # Set the label and increase the font size
            cbar.ax.tick_params(labelsize=18)  # Adjust the font size of the color bar tick labels

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