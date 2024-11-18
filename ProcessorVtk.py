import multiprocessing
import numpy as np
import vtk
from DataProcessor import DataProcessor

class ProcessorVtk(DataProcessor):
    def __init__(self, data):
        super().__init__(data)
        self.n_sim = self.data_reader.n_sim
        self.n_central_atoms = self.data_reader.n_central_atoms
        # self.v0 = self.data_reader.v0
        # self.y0 = self.data_reader.y0
        self.directory = self.data_reader.directory
        self.file_list = self.data_reader.file_list
        self.n_wall_atoms = self.data_reader.n_wall_atoms
        self.n_central_atoms = self.data_reader.n_central_atoms
        self.polydata = None
        self.polydatapoints = None
        self.ids = None
        self.debug = False
        self.ap = self.data_reader.ap

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

    def process_single_step(self, step, dt):
        """Processing on the data for one single step
        Calling the other methods for the processing
        Results stored in a dictionary"""

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.directory+self.file_list[step])
        reader.Update()
        self.polydata = reader.GetOutput()
        self.polydatapoints = self.polydata.GetPointData()
        self.get_ids()
        self.get_data()
        #self.compute_box_height() 
        self.compute_alignment()
        omega_average, omega_fluctuations = self.compute_angular_velocity_fluctuations()
        vx_fluctuations, vy_fluctuations, vz_fluctuations = self.compute_velocity_fluctuations()    
        #self.compute_mean_square_displacement()
        #eulerian_velocities = self.eulerian_velocity(10)
        #trackedGrainsPosition, trackedGrainsOrientation, trackedGrainsShapeX, trackedGrainsShapeZ = self.store_single_particle_data_for_contact(10)

        avg_dict = {"thetax": self.flow_angles,
                    "thetaz": self.out_flow_angles,
                    "percent_aligned": self.percent_aligned,
                    "S2": self.S2,
                    "Omega_x": omega_average[0],
                    "Omega_y": omega_average[1],
                    "Omega_z": omega_average[2],
                    "omega_fluctuations": self.compute_angular_velocity_fluctuations(),
                    "vx_fluctuations": vx_fluctuations,
                    "vy_fluctuations": vy_fluctuations,
                    "vz_fluctuations": vz_fluctuations,
                }
        return avg_dict

    def compute_particle_mass(self, density =1000):
        """
        Compute the mass of the particles
        """
        #mass of the particles
        volume = 4/3*np.pi*self.shape_x*self.shape_x*self.shape_z
       
        return density*volume
    
    def compute_particle_inertia(self, mass):
        """
        Compute the inertia tensor of the particles in the principal axis system
        """
        inertia_tensor = np.zeros((self.n_central_atoms, 3, 3))
        inertia_tensor[:, 0, 0] = 0.2*mass*(self.shape_x**2+self.shape_z**2)
        inertia_tensor[:, 1, 1] = 0.2*mass*(self.shape_x**2+self.shape_z**2)
        inertia_tensor[:, 2, 2] = 0.4*mass*self.shape_x**2

        return inertia_tensor

    def pass_particle_data(self):
        mass = self.compute_particle_mass()
        # inertia_tensor = self.compute_particle_inertia(mass)
        return self.coor, self.orientations, self.shape_x, self.shape_z, self.velocities, self.omegas, self.forces_particles, mass#, inertia_tensor

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
        self.omegas = np.array(self.polydatapoints.GetArray("omega"))[self.ids, :][self.n_wall_atoms:, :]
        #self.torques = np.array(self.polydatapoints.GetArray("tq"))[self.ids, :][self.n_wall_atoms:, :]
        self.orientations = np.array(self.polydatapoints.GetArray("TENSOR"))[self.ids, :][self.n_wall_atoms:, :].reshape(self.n_central_atoms,3,3)
        #self.stress = np.concatenate((np.array(self.polydatapoints.GetArray("c_stressAtom[1-3]")), np.array(self.polydatapoints.GetArray("c_stressAtom[4-6]"))), axis=1)
        self.thetax = np.array(self.polydatapoints.GetArray("c_thetaX"))[self.ids][self.n_wall_atoms:]
        self.shape_x = np.array(self.polydatapoints.GetArray("shapex"))[self.ids][self.n_wall_atoms:]
        self.shape_z = np.array(self.polydatapoints.GetArray("shapez"))[self.ids][self.n_wall_atoms:]

    def compute_velocity_fluctuations(self):
        """
        Compute the velocity fluctuations with respect to the average velocity of the particles at certain values of y.
        This function calculates fluctuations in the x-direction by accounting for spatial variations in average velocity,
        while treating the y and z directions as having uniform zero average velocities.
        """
        import numpy as np

        # Number of layers in the y-direction
        n_layers = 15

        # Determine the range and scaling factor for the layers
        y_positions = self.coor[:, 1]
        h_min = np.min(y_positions)
        h_max = np.max(y_positions)
        scaled_factor = (h_max - h_min) / (n_layers - 1)

        # Assign each particle to a layer based on its y-coordinate
        layer_indices = np.floor((y_positions - h_min) / scaled_factor).astype(int)
        # Ensure layer indices are within valid range [0, n_layers - 1]
        layer_indices = np.clip(layer_indices, 0, n_layers - 1)

        # Total number of particles in each layer
        counts = np.bincount(layer_indices, minlength=n_layers)

        # Handle potential division by zero for empty layers
        nonzero_mask = counts > 0

        # Extract velocity components
        velocities = self.velocities
        vx = velocities[:, 0]
        vy = velocities[:, 1]
        vz = velocities[:, 2]

        # Compute the average velocity in x-direction for each layer
        sum_vx_per_layer = np.bincount(layer_indices, weights=vx, minlength=n_layers)
        avg_vx_per_layer = np.zeros(n_layers)
        avg_vx_per_layer[nonzero_mask] = sum_vx_per_layer[nonzero_mask] / counts[nonzero_mask]

        # Assign the average x-velocity to each particle based on its layer
        avg_vx_particle = avg_vx_per_layer[layer_indices]

        # Calculate fluctuations in the x-direction
        delta_vx = vx - avg_vx_particle
        delta_vx_squared = delta_vx ** 2
        sum_delta_vx_squared_per_layer = np.bincount(
            layer_indices, weights=delta_vx_squared, minlength=n_layers
        )
        velocity_fluctuations_x = np.zeros(n_layers)
        velocity_fluctuations_x[nonzero_mask] = (
            sum_delta_vx_squared_per_layer[nonzero_mask] / counts[nonzero_mask]
        )

       # reduce all velocity fluctuations by averaging over all layers
        velocity_fluctuations_x = np.sqrt(np.mean(velocity_fluctuations_x))
        velocity_fluctuations_y = np.sqrt(np.mean(vy**2) - np.mean(vy)**2)
        velocity_fluctuations_z = np.sqrt(np.mean(vz**2) - np.mean(vz)**2)
        
        return velocity_fluctuations_x, velocity_fluctuations_y, velocity_fluctuations_z

    def compue_angular_velocity_fluctuations(self):
        """
        Compute the angular velocity fluctuations 
        """
        omega_average = np.mean(self.omegas, axis=0)
        omega_fluctuations = np.sqrt(np.mean(self.omegas**2, axis=0) - omega_average**2)
        return omega_fluctuations

    def compute_autocorrelation_vel(self):
        """
        Compute the autocorrelation of the velocities with respect to the initial velocities -> to change to start at steady state
        """
        autocorrelation_vel = np.mean(np.sum((self.v0-np.mean(self.v0))*(self.velocities-np.mean(self.velocities)), axis=1))
        return autocorrelation_vel

    def compute_space_averages(self):
        """
        Compute the space averages of the velocities, angular velocities and forces
        """
        self.velocities_space_average = np.mean(self.velocities, axis=0)
        self.vx_space_average = np.mean(self.velocities[:,0])
        self.omegas_space_average = np.mean(self.omegas, axis=0)
        # self.effective_friction = self.shearing_force/self.vertical_force
        # self.thetax_space_average = np.mean(self.thetax)

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
        starting_vector = np.array([0,0,1]) # always axis of symmetry on z
        
        if self.debug:
            for j in range(self.n_central_atoms):
                rot = self.orientations[j]
                if not np.isclose(rot@rot.T, np.diag(np.ones(3))).all():
                    raise SystemExit('Error: One of the points did not store a rotation matrix')
            # Project the vector on the plane x and y -> take the first two components
        # Compute the grain vectors by applying the rotation matrices to the starting vector
        grain_vectors = np.einsum('ijk,k->ij', self.orientations, starting_vector)    
        # Calculate flow angles
        flow_angles = np.arctan2(grain_vectors[:, 1], grain_vectors[:, 0])
        out_flow_angles = np.arctan2(grain_vectors[:, 2], grain_vectors[:, 0])
    
        
        # Correct angles to be between -pi/2 and pi/2
        self.flow_angles = np.where(flow_angles > np.pi/2, flow_angles - np.pi, 
                    np.where(flow_angles < -np.pi/2, flow_angles + np.pi, flow_angles))
        self.out_flow_angles = np.where(out_flow_angles > np.pi/2, out_flow_angles - np.pi, 
                        np.where(out_flow_angles < -np.pi/2, out_flow_angles + np.pi, out_flow_angles))
        
        # Compute the number of particles not aligned with the flow direction
        not_aligned_mask = np.logical_or(self.out_flow_angles < -np.pi/4, self.out_flow_angles > np.pi/4)
        n_not_aligned = np.sum(not_aligned_mask)
        self.percent_aligned = 1 - n_not_aligned / self.n_central_atoms
        
        # Compute the mean angles
        self.alignment_out_of_flow = np.mean(self.out_flow_angles)
        self.alignment_space_average = np.mean(self.flow_angles)
        
        #bin the angles over 144 bins
        # self.hist_thetax, _ = np.histogram(flow_angles, bins=180, range=(-np.pi/2, np.pi/2))
        # self.hist_thetaz, _ = np.histogram(out_flow_angles, bins=180, range=(-np.pi/2, np.pi/2))
   
        # Compute the nematic matrices using the outer product
        nematic_matrices = np.einsum('ij,ik->ijk', grain_vectors, grain_vectors)
        
        # Sum the nematic matrices and subtract identity matrix
        S2_sum = np.sum(3/2 * (nematic_matrices - np.eye(3)/3), axis=0)
        
        # Compute the space-averaged nematic order parameter
        S2_space_average = S2_sum / self.n_central_atoms
        
        # First eigenvalue of the nematic matrix
        self.S2 = np.max(np.linalg.eigvals(S2_space_average))

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
    def compute_single_particle_trajectory(self, step, n_sampled_particles=10):
        #find the particle index
        sampledIDxs = np.linspace(0, self.n_central_atoms-1, n_sampled_particles).astype('int')
        trackedGrains = np.zeros((n_sampled_particles, 3))
    
        #find the coordinates of the particles
        trackedGrains = self.coor[sampledIDxs, :]

        #find the orientation of the axis of symmetry
        if self.ap >1:
            trackedGrainsAxis = np.array([0, 0, 1])
        else:
            trackedGrainsAxis = np.array([1, 0, 0])

        tracked_axis = self.orientations[sampledIDxs, :, :]@trackedGrainsAxis
        
        return trackedGrains, tracked_axis
    
    def compute_angular_velocity_fluctuations(self):
        """
        Compute the angular velocity fluctuations 
        """
        omega_average = np.mean(self.omegas, axis=0)
        omega_fluctuations = np.mean((self.omegas-omega_average)**2, axis=0)
        # print(omega_average, omega_fluctuations)
        return omega_average, omega_fluctuations
    
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