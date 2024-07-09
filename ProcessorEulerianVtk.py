import multiprocessing
import numpy as np
import vtk
import math
from DataProcessor import DataProcessor

class ProcessorEulerianVtk(DataProcessor):
    def __init__(self, data):
        super().__init__(data)
        self.n_sim = self.data_reader.n_sim
        self.n_central_atoms = self.data_reader.n_central_atoms
        self.v0 = self.data_reader.v0
        self.y0 = self.data_reader.y0
        self.directory = self.data_reader.directory
        self.file_list = self.data_reader.file_list
        self.n_wall_atoms = self.data_reader.n_wall_atoms
        self.n_central_atoms = self.data_reader.n_central_atoms
        self.box_x = self.data_reader.box_x
        self.box_y = self.data_reader.box_y
        self.box_z = self.data_reader.box_z
        self.lower_bound_y = self.data_reader.lower_bound_y
        self.polydata = None
        self.polydatapoints = None
        self.ids = None

    def process_data(self, num_processes):
        """
        Processing the data in parallel, returning the averages        
        """
        self.num_processes = num_processes
        with multiprocessing.Pool(self.num_processes) as pool:
            print("Started multiprocessing")
            results = pool.map(self.process_single_step,
                               [step for step in range(self.n_sim)])
        # Extract the averages from the results
        averages = np.array(results)
        return averages

    # def process_single_step(self, step):
    #     """Processing on the data for one single step
    #     Calling the other methods for the processing
    #     Results stored in a dictionary"""

    #     reader = vtk.vtkPolyDataReader()
    #     reader.SetFileName(self.directory+self.file_list[step])
    #     reader.Update()
    #     self.polydata = reader.GetOutput()
    #     self.polydatapoints = self.polydata.GetPointData()
    #     self.get_ids()
    #     self.get_data()
    #     self.compute_space_averages()
    #     #self.compute_box_height() 
    #     self.compute_alignment()
    #     self.compute_mean_square_displacement()
    #     eulerian_velocities = self.eulerian_velocity(10)

    #     trackedGrainsPosition, trackedGrainsOrientation, trackedGrainsShapeX, trackedGrainsShapeZ = self.store_single_particle_data_for_contact(10)

    #     avg_dict = {'v': self.velocities_space_average,
    #                 "v_x": self.vx_space_average,
    #                 "omega": self.omegas_space_average,
    #                 "omega_z": self.omegaz_space_average,
    #                 "F_x": self.shearing_force,
    #                 "theta_x": self.alignment_space_average,
    #                 "theta_z": self.alignment_out_of_flow,
    #                 "percent_aligned": self.percent_aligned,
    #                 "S2": self.S2_space_average,
    #                 "box_height": self.box_height,
    #                 "autocorrelation_v": self.compute_autocorrelation_vel(),
    #                 "msd": self.mean_square_displacement,
    #                 "mu_effective": self.effective_friction,
    #                 "eulerian_vx": eulerian_velocities, 
    #                 "trackedGrainsPosition": trackedGrainsPosition,
    #                 "trackedGrainsOrientation": trackedGrainsOrientation,
    #                 "trackedGrainsShapeX": trackedGrainsShapeX,
    #                 "trackedGrainsShapeZ": trackedGrainsShapeZ}
        
    #     return avg_dict
    def process_single_step(self, step, nx_divisions=40, ny_divisions=6, dx=0.1, dy=0.1):
        """Processing on the data for one single step
        Calling the other methods for the processing
        Results stored in a dictionary"""
        self.step = step
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.directory+self.file_list[step])
        reader.Update()
        self.polydata = reader.GetOutput()
        self.polydatapoints = self.polydata.GetPointData()
        
        self.grid_data = [np.array([], dtype=int) for _ in range(nx_divisions * ny_divisions)]
        self.nx_divisions = nx_divisions
        self.ny_divisions = ny_divisions
        self.cell_volume = dx*dy*self.box_z

        self.get_ids_particles_in_grid()
        self.compute_space_averages_per_cell()
        self.find_max_min_velocity()
        self.find_max_min_angle_x()
        self.find_max_min_force()
        #self.plot_space_averages_all_cells(step, self.velocities_space_average, 'Velocity', 0)
        grid_dict = {'v': self.velocities_space_average,
                    "F": self.forces_space_average, 
                    "phi": self.local_phi, 
                    "theta_x": self.theta_x,
                    "theta_z": self.theta_z, 
                    "stress": self.stress_space_average}
        
        max_values = {'max_vx': self.max_vx,
                    'max_vy': self.max_vy,
                    'max_vz': self.max_vz,
                    'max_theta_x': self.max_theta_x,
                    'max_fx': self.max_fx,
                    'max_fy': self.max_fy,
                    'max_fz': self.max_fz}
        
        min_values = {'min_vx': self.min_vx,
                    'min_vy': self.min_vy,
                    'min_vz': self.min_vz,
                    'min_theta_x': self.min_theta_x,
                    'min_fx': self.min_fx,
                    'min_fy': self.min_fy,
                    'min_fz': self.min_fz}
        

        return grid_dict, max_values, min_values

    def find_max_min_velocity(self):
        """
        Find the maximum and minimum velocity
        """
        self.max_vx = np.max(self.velocities_space_average[:, :, 0])
        self.min_vx = np.min(self.velocities_space_average[:, :, 0])
        self.max_vy = np.max(self.velocities_space_average[:, :, 1])
        self.min_vy = np.min(self.velocities_space_average[:, :, 1])
        self.max_vz = np.max(self.velocities_space_average[:, :, 2])
        self.min_vz = np.min(self.velocities_space_average[:, :, 2])

    def find_max_min_angle_x(self):
        """
        Find the maximum and minimum angle in x
        """
        self.max_theta_x = np.max(self.theta_x)
        self.min_theta_x = np.min(self.theta_x)

    def find_max_min_force(self):
        """
        Find the maximum and minimum force
        """
        self.max_fx = np.max(self.forces_space_average[:, :, 0])
        self.min_fx = np.min(self.forces_space_average[:, :, 0])
        self.max_fy = np.max(self.forces_space_average[:, :, 1])
        self.min_fy = np.min(self.forces_space_average[:, :, 1])
        self.max_fz = np.max(self.forces_space_average[:, :, 2])
        self.min_fz = np.min(self.forces_space_average[:, :, 2])

    def get_ids_particles_in_grid(self):
        """
        For each particle, find the grid cell it belongs to and store info in the grid_data
        Grid_data is a list of arrays of ids, where ids here is simply the location in the data vector
        where information about a particle is stored 
        """
        self.coor = np.array(self.polydata.GetPoints().GetData())[self.n_wall_atoms:,:]
        self.velocities = np.array(self.polydatapoints.GetArray("v"))[self.n_wall_atoms:, :]
        self.forces_particles = np.array(self.polydatapoints.GetArray("f"))[self.n_wall_atoms:, :]
        self.shapex = np.array(self.polydatapoints.GetArray("shapex"))[self.n_wall_atoms:] #shape x
        self.shapey = np.array(self.polydatapoints.GetArray("shapey"))[self.n_wall_atoms:]
        self.shapez = np.array(self.polydatapoints.GetArray("shapez"))[self.n_wall_atoms:]
        self.volume = 4/3*np.pi*self.shapex*self.shapey*self.shapez
        self.orientations = np.array(self.polydatapoints.GetArray("TENSOR"))[self.ids, :][self.n_wall_atoms:, :].reshape(self.n_central_atoms,3,3)
        self.stress = np.concatenate((np.array(self.polydatapoints.GetArray("c_stressAtom[1-3]")), np.array(self.polydatapoints.GetArray("c_stressAtom[4-6]"))), axis=1)

        self.compute_alignment()

        scaled_factor_x = self.box_x/(self.nx_divisions)
        scaled_coor_x = np.floor((self.coor[:,0]+self.box_x/2)/scaled_factor_x)
        scaled_factor_y = self.box_y/(self.ny_divisions)
        scaled_coor_y = np.floor((self.coor[:,1]-self.lower_bound_y)/scaled_factor_y)
        # for each particle, find the grid cell it belongs to
        for j in range(self.n_central_atoms):
            i = int(scaled_coor_x[j])
            k = int(scaled_coor_y[j])
            if i < self.nx_divisions and k < self.ny_divisions:
                self.grid_data[i*self.ny_divisions+k] = np.append(self.grid_data[i*self.ny_divisions+k], j)
    
    def compute_space_averages_per_cell(self):
        """
        compute the space averages of the velocities, angular velocities and forces per cell
        Use the ids stored in grid_data
        """
        self.velocities_space_average = np.zeros((self.nx_divisions, self.ny_divisions, 3))
        self.forces_space_average = np.zeros((self.nx_divisions, self.ny_divisions, 3))
        #self.alignment_in_flow = np.zeros((self.nx_divisions, self.ny_divisions))
        self.local_phi = np.zeros((self.nx_divisions, self.ny_divisions))
        self.theta_x = np.zeros((self.nx_divisions, self.ny_divisions))
        self.theta_z = np.zeros((self.nx_divisions, self.ny_divisions))
        self.stress_space_average = np.zeros((self.nx_divisions, self.ny_divisions, 6))
        for i in range(self.nx_divisions):
            for k in range(self.ny_divisions):
                #check cell is not empty before computing the mean
                if len(self.grid_data[i*self.ny_divisions+k]) > 0:
                    self.velocities_space_average[i, k] = np.mean(self.velocities[self.grid_data[i*self.ny_divisions+k], :], axis=0)
                    self.forces_space_average[i, k] = np.mean(self.forces_particles[self.grid_data[i*self.ny_divisions+k], :], axis=0)
                    self.local_phi[i, k] = np.sum(self.volume[self.grid_data[i*self.ny_divisions+k]])/(self.cell_volume)
                    self.theta_x[i, k] = np.mean(self.flow_angle[self.grid_data[i*self.ny_divisions+k]])
                    self.theta_z[i, k] = np.mean(self.out_flow_angle[self.grid_data[i*self.ny_divisions+k]])
                    self.stress_space_average[i, k] =  np.mean(self.stress[self.grid_data[i*self.ny_divisions+k], :], axis=0)
                else:
                    self.velocities_space_average[i, k] = np.zeros(3)
                    self.forces_space_average[i, k] = np.zeros(3)
                    self.local_phi[i, k] = 0
                    self.theta_x[i, k] = 0
                    self.theta_z[i, k] = 0
                    self.stress_space_average[i, k] = np.zeros(6)


    def plot_space_averages_all_cells(self, step, value, quantity= 'Velocity', component = 0, vmin=-0.2, vmax=0.8):
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
            axis_name = 'yz'
        elif component == 5:
            axis_name = 'xz'
    
        from matplotlib import pyplot as plt
        
        # Plot x component of the velocity
        plt.figure()
        plt.imshow(value[:,:,component].T, cmap='coolwarm', vmin=-0.2, vmax=0.8, extent=[0, self.nx_divisions, 0, self.ny_divisions], origin='lower')
        plt.colorbar(label='$'+ quantity[0]+ '_' + axis_name + '$')  # Set ticks for colorbar

        plt.title('Eulerian ' + axis_name+ ' ' + quantity)
        plt.xlabel('Grid cell x')
        plt.ylabel('Grid cell y')
        #plt.show()

        # save figure to create gif later include the name of the timestep
        plt.savefig(f"gif/eulerian_x_velocity_{step}.png")
        plt.close()

    def save_gif_from_plots(self):
        """
        Save gif from plots generated by the vtk processor
        """
        import imageio
        import os
        images = []
        os.chdir('gif')
        for filename in os.listdir('.'):
            
            if filename.endswith('.png'):
                images.append(imageio.imread(filename))
        imageio.mimsave('eulerian_velocity.gif', images)

    def get_ids(self):
        """
        read and sort the identifiers. Particles making the walls have ids with lower values
        """
        ids = np.array(self.polydatapoints.GetArray("id"))
        self.ids = np.argsort(ids) #ids change at every time step so we need to rearrange them

    def get_data(self):
        """
        Store data from vtk into numpy arrays
        """
        self.coor = np.array(self.polydata.GetPoints().GetData())[self.ids][self.n_wall_atoms:,:]
        self.velocities = np.array(self.polydatapoints.GetArray("v"))[self.ids, :][self.n_wall_atoms:, :]
        self.forces_particles = np.array(self.polydatapoints.GetArray("f"))[self.ids, :][self.n_wall_atoms:, :]
        wall_coordinates = np.array(self.polydata.GetPoints().GetData())[self.ids][:self.n_wall_atoms, :]
        index_min_ycoord = np.argmin(wall_coordinates[:,1])
        indexes_bottom_wall = np.where(np.isclose(wall_coordinates[:,1], wall_coordinates[index_min_ycoord,1], atol=2e-3))[0]
        coordinates_bottom_wall = wall_coordinates[indexes_bottom_wall, :]
        self.forces_walls = np.array(self.polydatapoints.GetArray("f"))[self.ids, :][indexes_bottom_wall, :]
        # coordinates_bottom_wall = np.array(self.polydata.GetPoints().GetData())[self.ids][int(self.n_wall_atoms/2):self.n_wall_atoms, :]
        # if not np.allclose(coordinates_bottom_wall[:,1], np.min(coordinates_bottom_wall[1,1]), atol=1e-5):      
        #     self.forces_walls = np.array(self.polydatapoints.GetArray("f"))[self.ids, :][int(self.n_wall_atoms/2+1):self.n_wall_atoms-1, :]
        # else:
        #     self.forces_walls = np.array(self.polydatapoints.GetArray("f"))[self.ids, :][int(self.n_wall_atoms/2):self.n_wall_atoms, :]
        self.omegas = np.array(self.polydatapoints.GetArray("omega"))[self.ids, :][self.n_wall_atoms:, :]
        self.torques = np.array(self.polydatapoints.GetArray("tq"))[self.ids, :][self.n_wall_atoms:, :]
        self.orientations = np.array(self.polydatapoints.GetArray("TENSOR"))[self.ids, :][self.n_wall_atoms:, :].reshape(self.n_central_atoms,3,3)
        self.stress = np.concatenate((np.array(self.polydatapoints.GetArray("c_stressAtom[1-3]")), np.array(self.polydatapoints.GetArray("c_stressAtom[4-6]"))), axis=1)

    def compute_autocorrelation_vel(self):
        """
        Compute the autocorrelation of the velocities with respect to the initial velocities -> to change to start at steady state
        """
        autocorrelation_vel = np.mean(np.sum((self.v0-np.mean(self.v0))*(self.velocities-np.mean(self.velocities)), axis=1))
        return autocorrelation_vel

    def compute_box_height(self):
        """
        Compute the height of the box from the maximum y coordinate of the particles
        """
        self.box_height = np.max(self.coor[:,1])-np.min(self.coor[:,1])
        

    def compute_mean_square_displacement(self):
        """
        Compute the mean square displacement of the particles in the direction perpendicular to the flow to the flow
        """
        #compute the mean square displacement of the particles
        self.mean_square_displacement = np.mean((self.coor[:,1]-self.y0)**2)

    def compute_alignment(self, store_heads = None):
        """
        Compute the alignment of the particles with respect to the flow direction (angle theta in previous papers)
        I am assuming there is no alignment in the z direction (out of plane)
        Then I compute the nematic order parameter S2 along that direction
        """
        starting_vector = np.array([0,0,1])
        flow_angle = np.zeros(self.n_central_atoms)
        out_flow_angle = np.zeros(self.n_central_atoms)
        S2 = np.zeros(self.n_central_atoms)
        if store_heads is not None:
            arrow_heads = np.zeros((self.n_central_atoms, 3)) #for plotting purposes

        for j in range(self.n_central_atoms):
            rot = self.orientations[j]
            if not np.isclose(rot@rot.T, np.diag(np.ones(3))).all():
                raise SystemExit('Error: One of the points did not store a rotation matrix')
            #project the vector on the plane x and y -> just take the first two components
            main_axis_ellipsoid = rot@starting_vector # longest axis is always stored as the third column
            flow_angle[j] = np.arctan2(main_axis_ellipsoid[1], main_axis_ellipsoid[0])
            #arctan2(a, b) computes the arctan of a/b in the correct quadrant
            out_flow_angle[j] = np.arctan2(main_axis_ellipsoid[2], main_axis_ellipsoid[0])
        
        #correcting the angles to be between -pi/2 and pi/2 considering symmetry of ellipsoid

        flow_angle = np.where(flow_angle > np.pi/2 , flow_angle - np.pi,
                      np.where(flow_angle < -np.pi/2, flow_angle + np.pi,
                                flow_angle))

        out_flow_angle = np.where(out_flow_angle > np.pi/2 , out_flow_angle - np.pi,
                      np.where(out_flow_angle < -np.pi/2, out_flow_angle + np.pi,
                                out_flow_angle))
        
        # compute the numbr of particle not aligned with the flow direction with the out_flow_angle between -pi/4 and pi/4
        n_not_aligned = np.sum(np.logical_or(out_flow_angle < -np.pi/4, out_flow_angle > np.pi/4))
        self.percent_aligned = 1-n_not_aligned/self.n_central_atoms

        # from matplotlib import pyplot as plt
        # plt.figure()
        # counts, bins = np.histogram(flow_angle, bins=100, density=True)
        # plt.stairs(counts, bins)
        # plt.xlabel('theta')
        # plt.ylabel('P(theta)')
        # plt.show()
        self.flow_angle = np.rad2deg(flow_angle)
        self.out_flow_angle = np.rad2deg(out_flow_angle)

        self.alignment_out_of_flow = np.mean(out_flow_angle) #maybe also compute time average
        self.alignment_space_average = np.mean(flow_angle)
        
        # compute nematic order parameter
        S2 = 0
        nematic_vector = np.array([np.cos(self.alignment_space_average), np.sin(self.alignment_space_average), 0])
        for j in range(self.n_central_atoms):
            rot = self.orientations[j]
            grain_vector = rot@starting_vector
            #absolute value needed for the symmetry of the ellipsoid
            cos_theta = np.abs(np.dot(grain_vector, nematic_vector))
            S2 = S2 + (3*cos_theta**2-1)/2


        self.S2_space_average = S2/self.n_central_atoms #nematic order parameter in 3 D
        # #2d alternative as in the paper
        # nematic_vector = np.array([np.cos(self.alignment_space_average), np.sin(self.alignment_space_average)])
        # Qx = np.zeros(self.n_central_atoms)
        # Qy = np.zeros(self.n_central_atoms)
        # for j in range(self.n_central_atoms):
        #     rot = self.orientations[j]
        #     grain_vector = rot@starting_vector
        #     grain_vectorxy_plane = np.array([grain_vector[0], grain_vector[1]])/np.linalg.norm(grain_vector[:2])
        #     cos_theta = np.dot(grain_vectorxy_plane, nematic_vector)
        #     theta = np.arccos(cos_theta)
        #     Qx[j] = np.cos(2*theta)
        #     Qy[j] = np.sin(2*theta)
            
        # self.S2_space_average = np.sqrt(np.mean(Qx)**2+np.mean(Qy)**2) #nematic order parameter in 2 D
        
    def eulerian_velocity(self, n_intervals):
        """
        Compute velocities in eulerian coordinates
        """
        # Layers for velocities eulerian
        velocities_per_layer = np.zeros(n_intervals)
        #finding domain dimensions
        h_max = max(self.coor[:,1]) #maximum height
        h_min = min(self.coor[:,1]) #minimum height
        scaled_factor = (h_max-h_min)/(n_intervals-1)
        scaled_coor = np.floor((self.coor[:,1]-h_min)/scaled_factor)
        for j in range(n_intervals):
            bin_idxs = np.where(scaled_coor==j)[0] #np.where returns a tuple with indexes in the first component
            velocities_per_layer[j] = np.mean(self.velocities[bin_idxs,0])
        return velocities_per_layer

    #compute single particles space trajectory
    def compute_single_particle_trajectory(self, step, n_sampled_particles):
        #find the particle index
        sampledIDxs = np.linspace(0, self.n_central_atoms-1, n_sampled_particles).astype('int')
        trackedGrains = np.zeros((n_sampled_particles, 3))
    
        #find the coordinates of the particles
        trackedGrains = self.coor[sampledIDxs, :]
        
        return trackedGrains
    
    def store_single_particle_data_for_contact(self, n_sampled_particles):
        """
        Store the data for the particles position and orientation and shapex and shapez
        Only a subset of the particles is stored
        """
        #find the particle index
        sampledIDxs = np.linspace(0, self.n_central_atoms-1, n_sampled_particles).astype('int')
        
        #find the coordinates of the particles

        trackedGrainsPosition = self.coor[sampledIDxs, :]
        trackedGrainsOrientation = self.orientations[sampledIDxs, :, :]
        trackedGrainsShapeX = np.array(self.polydatapoints.GetArray("shapex"))[self.ids][self.n_wall_atoms:]
        trackedGrainsShapeX = trackedGrainsShapeX[sampledIDxs]
        trackedGrainsShapeZ = np.array(self.polydatapoints.GetArray("shapez"))[self.ids][self.n_wall_atoms:]
        trackedGrainsShapeZ = trackedGrainsShapeZ[sampledIDxs]

        return trackedGrainsPosition, trackedGrainsOrientation, trackedGrainsShapeX, trackedGrainsShapeZ

    def optional_angle_histogram(flow_angle, out_flow_angle):
        """
        3D plot of the angular distribution of the particles
        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(flow_angle, out_flow_angle, np.zeros(len(flow_angle)), marker='.')
        ax.set_xlabel('theta')
        ax.set_ylabel('phi')
        plt.show()