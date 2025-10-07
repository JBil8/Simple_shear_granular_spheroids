import multiprocessing
import numpy as np
import vtk
from DataProcessor import DataProcessor


class ProcessorVtk(DataProcessor):
    def __init__(self, data):
        super().__init__(data)
        self.n_sim = self.data_reader.n_sim
        self.n_central_atoms = self.data_reader.n_central_atoms
        self.theta0 = self.data_reader.theta0
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

    def process_single_step(self, step, box_lengths):
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
        self.compute_alignment()
        vx_fluctuations, vy_fluctuations, vz_fluctuations = self.compute_velocity_fluctuations(
            box_lengths[1])  # averaged over all particles
        omega_average, omega_fluctuations = self.compute_angular_velocity_fluctuations()
        particles_mass = self.compute_particle_mass()
        particles_inertia = self.compute_particle_inertia(particles_mass)
        tke, rke = self.compute_fluctuating_kinetic_energy(
            particles_mass, particles_inertia, self.particle_fluctuating_velocity, self.particle_fluctuating_omega)

        kinetic_stress = self.compute_kinetic_component_stress(
            particles_mass, self.particle_fluctuating_velocity, np.prod(box_lengths[:3]))
        c_r_values, c_delta_vy = self.compute_spatial_autocorrelation(
            self.delta_vy, box_lengths)
        _, c_delta_omega_z = self.compute_spatial_autocorrelation(
            self.particle_fluctuating_omega[:, 2], box_lengths)

        c_y_values, c_density_y = self.compute_density_correlation_y(
            self.coor[:, 1], box_lengths[:3], n_bins=100)

        avg_dict = {"thetax": self.flow_angles,
                    "thetaz": self.out_flow_angles,
                    "percent_aligned": self.percent_aligned,
                    "S2": self.S2,
                    "Omega_x": omega_average[0],
                    "Omega_y": omega_average[1],
                    "Omega_z": omega_average[2],
                    "tke": tke,
                    "rke": rke,
                    "omega_fluctuations": omega_fluctuations,
                    "vx_fluctuations": vx_fluctuations,
                    "vy_fluctuations": vy_fluctuations,
                    "vz_fluctuations": vz_fluctuations,
                    "vy_velocity": self.delta_vy,
                    "omegaz_velocity": self.particle_fluctuating_omega[:, 2],
                    "kinetic_stress": kinetic_stress,
                    "measured_shear_rate": self.measured_shear_rate,
                    "c_y_values": c_y_values,
                    "c_density_y": c_density_y,
                    "c_delta_vy": c_delta_vy,
                    "c_r_values": c_r_values,
                    "c_delta_omega_z": c_delta_omega_z
                    }
        return avg_dict

    def compute_particle_mass(self, density=1000):
        """
        Compute the mass of the particles
        """
        return density*4/3*np.pi*self.shape_x*self.shape_x*self.shape_z

    def compute_particle_inertia(self, mass):
        """
        Compute the inertia tensor of the particles in the principal axis system
        """
        shape_x2 = self.shape_x ** 2
        shape_z2 = self.shape_z ** 2
        factor1 = 0.2 * mass * (shape_x2 + shape_z2)
        factor2 = 0.4 * mass * shape_x2

        inertia_tensor = np.zeros((self.n_central_atoms, 3, 3))
        inertia_tensor[:, 0, 0] = factor1
        inertia_tensor[:, 1, 1] = factor1
        inertia_tensor[:, 2, 2] = factor2

        return inertia_tensor

    def local_to_global_tensor(self, tensor, orientation):
        """
        Transform the tensor from local (ellispoid frame) to global
        """
        T_prime = np.einsum('nik,njl,nkl->nij',
                            orientation, orientation, tensor)
        return T_prime

    def compute_fluctuating_kinetic_energy(self, mass, inertia, v_fluctuations, omega_fluctuations):
        """
        Compute the translational and rotational component of the flutuating kinetic energy, 
        """

        tke = 0.5 * \
            np.einsum('ij,ij', mass[:, np.newaxis] *
                      v_fluctuations, v_fluctuations)

        global_inertia = self.local_to_global_tensor(
            inertia, self.orientations)
        rke = 0.5 * np.einsum('ij,ijk,ik->', omega_fluctuations,
                              global_inertia, omega_fluctuations)
        return tke, rke

    def pass_particle_data(self):
        mass = self.compute_particle_mass()
        return self.coor, self.orientations, self.shape_x, self.shape_z, self.velocities, self.omegas, self.forces_particles, mass  # , inertia_tensor

    def get_ids(self):
        """
        read and sort the identifiers. Particles making the walls have ids with lower values
        """
        ids = np.array(self.polydatapoints.GetArray("id"))
        # ids change at every time step so we need to rearrange them
        self.ids = np.argsort(ids)

    def get_data(self):
        """
        Store data from vtk into numpy arrays
        """
        self.coor = np.array(self.polydata.GetPoints().GetData())[
            self.ids][self.n_wall_atoms:, :]
        self.velocities = np.array(self.polydatapoints.GetArray("v"))[
            self.ids, :][self.n_wall_atoms:, :]
        self.forces_particles = np.array(self.polydatapoints.GetArray("f"))[
            self.ids, :][self.n_wall_atoms:, :]
        self.omegas = np.array(self.polydatapoints.GetArray("omega"))[
            self.ids, :][self.n_wall_atoms:, :]
        # self.torques = np.array(self.polydatapoints.GetArray("tq"))[self.ids, :][self.n_wall_atoms:, :]
        self.orientations = np.array(self.polydatapoints.GetArray("TENSOR"))[
            self.ids, :][self.n_wall_atoms:, :].reshape(self.n_central_atoms, 3, 3)
        # self.stress = np.concatenate((np.array(self.polydatapoints.GetArray("c_stressAtom[1-3]")), np.array(self.polydatapoints.GetArray("c_stressAtom[4-6]"))), axis=1)
        self.shape_x = np.array(self.polydatapoints.GetArray("shapex"))[
            self.ids][self.n_wall_atoms:]
        self.shape_z = np.array(self.polydatapoints.GetArray("shapez"))[
            self.ids][self.n_wall_atoms:]

    def compute_kinetic_component_stress(self, mass, particle_fluctuating_velocity, box_volume):
        """
        Compute the kinetic component of the stress tensor
        """
        momentum = mass[:, np.newaxis] * particle_fluctuating_velocity
        kinetic_stress = np.einsum(
            'ij,ik->jk', momentum, particle_fluctuating_velocity)/box_volume
        return kinetic_stress

    def compute_velocity_fluctuations(self, box_height):
        """
        Compute the velocity fluctuations with respect to the average velocity of the particles at certain values of y.
        This function calculates fluctuations in the x-direction by accounting for spatial variations in average velocity,
        while treating the y and z directions as having uniform zero average velocities.
        """

        # Number of layers in the y-direction
        n_layers = 20

        # Determine the range and scaling factor for the layers
        y_positions = self.coor[:, 1]
        scaled_factor = box_height / (n_layers - 1)
        h_min = -box_height/2  # minimum y coordinate of the box

        # Assign each particle to a layer based on its y-coordinate
        layer_indices = np.floor(
            (y_positions - h_min) / scaled_factor).astype(int)
        # Ensure layer indices are within valid range [0, n_layers - 1]
        layer_indices = np.clip(layer_indices, 0, n_layers - 1)

        # Total number of particles in each layer
        counts = np.bincount(layer_indices, minlength=n_layers)

        # Handle potential division by zero for empty layers
        nonzero_mask = counts > 0

        # Extract velocity components
        velocities = self.velocities
        vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]

        # Compute the average velocity in x-direction for each layer
        sum_vx_per_layer = np.bincount(
            layer_indices, weights=vx, minlength=n_layers)
        avg_vx_per_layer = np.zeros(n_layers)
        avg_vx_per_layer[nonzero_mask] = sum_vx_per_layer[nonzero_mask] / \
            counts[nonzero_mask]

        # compute average shear rate by measuring difference of vx between two consecutive bins over the height of the bins
        shear_rate = np.diff(avg_vx_per_layer)/scaled_factor
        self.measured_shear_rate = np.mean(shear_rate)

        # Assign the average x-velocity to each particle based on its layer
        avg_vx_particle = avg_vx_per_layer[layer_indices]

        # Calculate fluctuations in the x-direction
        delta_vx = vx - avg_vx_particle
        delta_vx_squared = delta_vx ** 2
        sum_delta_vx_squared_per_layer = np.bincount(
            layer_indices, weights=delta_vx_squared, minlength=n_layers)
        velocity_fluctuations_x = np.zeros(n_layers)
        velocity_fluctuations_x[nonzero_mask] = (
            sum_delta_vx_squared_per_layer[nonzero_mask] / counts[nonzero_mask])

        # reduce all velocity fluctuations by averaging over all layers
        mean_vy, mean_vz = np.mean(vy), np.mean(vz)
        velocity_fluctuations_x = np.sqrt(np.mean(velocity_fluctuations_x))
        velocity_fluctuations_y = np.sqrt(np.mean(vy**2) - mean_vy**2)
        velocity_fluctuations_z = np.sqrt(np.mean(vz**2) - mean_vz**2)

        self.delta_vy = vy - mean_vy
        delta_vz = vz - mean_vz

        self.particle_fluctuating_velocity = np.column_stack(
            (delta_vx, self.delta_vy, delta_vz))

        return velocity_fluctuations_x, velocity_fluctuations_y, velocity_fluctuations_z

    def compute_angular_velocity_fluctuations(self):
        """
        Compute the angular velocity fluctuations 
        """
        omega_average = np.mean(self.omegas, axis=0)
        self.particle_fluctuating_omega = self.omegas - omega_average
        omega_fluctuations = np.sqrt(
            np.mean(self.omegas**2, axis=0) - omega_average**2)
        return omega_average, omega_fluctuations

    def compute_spatial_autocorrelation(self, grain_properties, box_lengths, n_bins=19):
        """
        Compute the spatial autocorrelation of a grain property using pairwise distances 

        Parameters:
            pairwise_distances (numpy.ndarray): Pairwise distances between grains (N_pairs,).
            grain_properties (numpy.ndarray): Grain properties (N,).
            box_lengths (numpy.ndarray): Box lengths [Lx, Ly, Lz].
            n_bins (int, optional): Number of bins for distance histogram. Default is 100.

        Returns:
            distances (numpy.ndarray): Midpoint of distance bins.
            autocorrelation (numpy.ndarray): Spatial autocorrelation values.
        """
        pairwise_distances = self.compute_pairwise_distances_triclinic(
            box_lengths)

        # Limit the maximum distance to min(box_lengths) / 2
        max_distance = 0.5 * np.min(box_lengths[:3])

        # Create bins for distances
        bins = np.linspace(0, max_distance, n_bins + 1)

        # remove bins at less than minimum size of the particles distance
        min_distance = 2*np.min(np.concatenate((self.shape_x, self.shape_z)))

        bins = bins[bins > min_distance]
        n_bins = len(bins) - 1
        bin_midpoints = 0.5 * (bins[:-1] + bins[1:])

        # Initialize arrays to accumulate correlations and counts
        correlation = np.zeros(n_bins, dtype=np.float64)
        counts = np.zeros(n_bins, dtype=np.int32)

        # Compute pairwise property products
        property_products = np.multiply.outer(
            grain_properties, grain_properties)
        pairwise_distances = pairwise_distances.ravel()
        property_products = property_products.ravel()

        # Digitize distances into bins
        # Convert to 0-based index
        bin_indices = np.digitize(pairwise_distances, bins) - 1

        # Mask for valid bins
        valid_mask = (bin_indices >= 0) & (bin_indices < n_bins)

        # Use np.bincount for faster accumulation instead of np.add.at
        correlation[:n_bins] += np.bincount(bin_indices[valid_mask],
                                            weights=property_products[valid_mask], minlength=n_bins)
        counts[:n_bins] += np.bincount(bin_indices[valid_mask],
                                       minlength=n_bins)

        # Normalize correlations by counts
        autocorrelation = correlation / \
            np.maximum(counts, 1)  # Avoid division by zero

        # Compute the average of the squared property
        property_squared = np.mean(grain_properties**2)

        # Add at the first bin the average of the squared property and distance 0 by increasing the length of the arrays
        autocorrelation = np.insert(autocorrelation, 0, property_squared)
        bin_midpoints = np.insert(bin_midpoints, 0, 0)

        # Normalize by the average squared property
        autocorrelation = autocorrelation / property_squared

        return bin_midpoints, autocorrelation

    def compute_density_correlation_y(self, y_coords, box_lengths, n_bins=40):
        """
        Computes the 1D density-density autocorrelation function C(y) along the y-axis.

        This function calculates the correlation function defined as in MArschall et al 2019 PRL to detect smectic (layered) order in a particle system.
        The calculation is performed efficiently using the Fast Fourier Transform (FFT)

        Parameters:
            y_coords (numpy.ndarray): 
                A 1D array containing the y-coordinates of all particles in a single frame.
            box_lengths (list or tuple or numpy.ndarray): 
                The dimensions of the simulation box [Lx, Ly, Lz].
            n_bins (int, optional): 
                The number of bins to use for creating the density profile along the y-axis.
                A larger number gives better resolution but can be noisier.
                Defaults to 200.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]:
                A tuple containing:
                - y_values (numpy.ndarray): The distance values (bin centers) for the y-axis.
                - C_y (numpy.ndarray): The computed correlation function C(y).
        """
        # --- 1. Setup and System Parameters ---
        Lx, Ly, Lz = box_lengths
        n_particles = len(y_coords)
        box_volume = Lx * Ly * Lz

        if n_particles == 0:
            y_values = np.linspace(0, Ly / 2, n_bins // 2)
            return y_values, np.zeros_like(y_values)

        # --- 2. Create the Binned Density Profile n(y) ---
        counts, bin_edges = np.histogram(
            y_coords, bins=n_bins, range=(-Ly/2, Ly/2))

        # Calculate the volume of each slab-like bin
        delta_y = Ly / n_bins
        bin_volume = delta_y * Lx * Lz

        # Convert counts to number density for each bin
        n_binned = counts / bin_volume

        # --- 3. Compute Correlation C(y) using FFT ---
        n_avg = n_particles / box_volume

        # Calculate the density fluctuations in each bin (delta_n = n(y) - <n>)
        delta_n = n_binned - n_avg

        # Compute the Power Spectral Density using FFT.
        F_delta_n = np.fft.fft(delta_n)

        # Step 3b: Power spectrum (magnitude squared of the FFT result)
        power_spectrum = np.abs(F_delta_n)**2

        # Step 3c: Inverse FFT of the power spectrum gives the unnormalized autocorrelation
        # The result should be real; the imaginary part is due to numerical noise.
        autocorr_unnormalized = np.fft.ifft(power_spectrum).real

        # --- 4. Normalize the Correlation Function ---
        prefactor = (Lx * Lz) / (n_avg * Ly)
        C_y = prefactor * autocorr_unnormalized * delta_y

        # --- 5. Prepare Final Output ---
        # The correlation function is symmetric, so we only need the first half.
        y_values = np.arange(n_bins // 2) * delta_y
        C_y_half = C_y[:n_bins // 2]

        return y_values, C_y_half

    def compute_pairwise_distances_triclinic(self, box_lengths):
        """
        Compute pairwise distances for triclinic domains with periodic boundary conditions.

        Parameters:
            positions (numpy.ndarray): Positions of shape (N, 3).
            box_matrix (numpy.ndarray): 3x3 matrix defining the triclinic box.

        Returns:
            distances (numpy.ndarray): Pairwise distance matrix of shape (N, N).
        """
        positions = self.coor
        box_matrix = np.diag(box_lengths[:3]) + np.array(
            [[0, box_lengths[3], 0], [0, 0, 0], [0, 0, 0]])  # Triclinic box matrix
        # Inverse for fractional coordinates
        inv_box_matrix = np.linalg.inv(box_matrix)
        diff = positions[:, np.newaxis, :] - \
            positions[np.newaxis, :, :]  # Shape: (N, N, 3)

        # Convert to fractional coordinates and apply periodic wrapping
        fractional_diff = diff@inv_box_matrix.T
        fractional_diff -= np.round(fractional_diff)
        diff = fractional_diff@box_matrix.T
        # Efficient L2 norm
        distances = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        return distances

    def compute_space_averages(self):
        """
        Compute the space averages of the velocities, angular velocities and forces
        """
        self.velocities_space_average = np.mean(self.velocities, axis=0)
        self.vx_space_average = np.mean(self.velocities[:, 0])
        self.omegas_space_average = np.mean(self.omegas, axis=0)

    def compute_box_height(self):
        """
        Compute the height of the box from the maximum y coordinate of the particles
        """
        self.box_height = np.max(self.coor[:, 1])-np.min(self.coor[:, 1])

    def compute_mean_square_displacement(self):
        """
        Compute the mean square displacement of the particles in the direction perpendicular to the flow to the flow
        """
        # compute the mean square displacement of the particles
        self.mean_square_displacement = np.mean((self.coor[:, 1]-self.y0)**2)

    def compute_alignment(self, store_heads=None):
        """
        Compute the alignment of the particles with respect to the flow direction (angle theta in previous papers)
        Then I compute the nematic order parameter S2 along that direction
        """
        starting_vector = np.array([0, 0, 1])  # always axis of symmetry on z

        if self.debug:
            for j in range(self.n_central_atoms):
                rot = self.orientations[j]
                if not np.isclose(rot@rot.T, np.diag(np.ones(3))).all():
                    raise SystemExit(
                        'Error: One of the points did not store a rotation matrix')
            # Project the vector on the plane x and y -> take the first two components
        # Compute the grain vectors by applying the rotation matrices to the starting vector
        grain_vectors = np.einsum(
            'ijk,k->ij', self.orientations, starting_vector)
        # Calculate flow angles
        flow_angles = np.arctan2(grain_vectors[:, 1], grain_vectors[:, 0])
        # out_flow_angles = np.arctan2(grain_vectors[:, 2], grain_vectors[:, 0])
        out_flow_angles = np.acos(np.abs(grain_vectors[:, 2]))

        # Correct angles to be between -pi/2 and pi/2
        self.flow_angles = np.where(flow_angles > np.pi/2, flow_angles - np.pi,
                                    np.where(flow_angles < -np.pi/2, flow_angles + np.pi, flow_angles))

        # - np.pi/2 # to have the angles between -pi/2 and pi/2
        self.out_flow_angles = out_flow_angles

        # Compute the number of particles not aligned with the flow direction
        not_aligned_mask = np.logical_or(
            self.out_flow_angles < -np.pi/4, self.out_flow_angles > np.pi/4)
        n_not_aligned = np.sum(not_aligned_mask)
        self.percent_aligned = 1 - n_not_aligned / self.n_central_atoms

        # Compute the mean angles
        self.alignment_out_of_flow = np.mean(self.out_flow_angles)
        self.alignment_space_average = np.mean(self.flow_angles)

        # Compute the nematic matrices using the outer product
        nematic_matrices = np.einsum(
            'ij,ik->ijk', grain_vectors, grain_vectors)

        # Sum the nematic matrices and subtract identity matrix and average over the number of particles
        S2_space_average = np.sum(
            3/2 * (nematic_matrices - np.eye(3)/3), axis=0) / self.n_central_atoms

        # First eigenvalue of the nematic matrix
        self.S2 = np.max(np.linalg.eigvals(S2_space_average))

    def eulerian_velocity(self, n_intervals):
        """
        Compute velocities in eulerian coordinates
        """
        # Layers for velocities eulerian
        velocities_per_layer = np.zeros(n_intervals)
        # finding domain dimensions
        h_max = max(self.coor[:, 1])  # maximum height
        h_min = min(self.coor[:, 1])  # minimum height
        scaled_factor = (h_max-h_min)/(n_intervals-1)
        scaled_coor = np.floor((self.coor[:, 1]-h_min)/scaled_factor)
        for j in range(n_intervals):
            # np.where returns a tuple with indexes in the first component
            bin_idxs = np.where(scaled_coor == j)[0]
            velocities_per_layer[j] = np.mean(self.velocities[bin_idxs, 0])
        return velocities_per_layer

    # compute single particles space trajectory
    def compute_single_particle_trajectory(self, step, n_sampled_particles=10):

        sampledIDxs = np.linspace(
            0, self.n_central_atoms-1, n_sampled_particles).astype('int')
        trackedGrains = np.zeros((n_sampled_particles, 3))

        trackedGrains = self.coor[sampledIDxs, :]
        if self.ap > 1:
            trackedGrainsAxis = np.array([0, 0, 1])
        else:
            trackedGrainsAxis = np.array([1, 0, 0])

        tracked_axis = self.orientations[sampledIDxs, :, :]@trackedGrainsAxis

        return trackedGrains, tracked_axis

    def store_single_particle_data_for_contact(self, n_sampled_particles):
        """
        Store the data for the particles position and orientation and shapex and shapez
        Only a subset of the particles is stored
        """
        # find the particle index
        sampledIDxs = np.linspace(
            0, self.n_central_atoms-1, n_sampled_particles).astype('int')

        trackedGrainsPosition = self.coor[sampledIDxs, :]
        trackedGrainsOrientation = self.orientations[sampledIDxs, :, :]
        trackedGrainsShapeX = np.array(self.polydatapoints.GetArray("shapex"))[
            self.ids][self.n_wall_atoms:]
        trackedGrainsShapeX = trackedGrainsShapeX[sampledIDxs]
        trackedGrainsShapeZ = np.array(self.polydatapoints.GetArray("shapez"))[
            self.ids][self.n_wall_atoms:]
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
        ax.scatter(flow_angle, out_flow_angle,
                   np.zeros(len(flow_angle)), marker='.')
        ax.set_xlabel('theta')
        ax.set_ylabel('phi')
        plt.show()
