import multiprocessing
import numpy as np
from DataProcessor import DataProcessor

class ProcessorDump(DataProcessor):
    def __init__(self, data, n_wall_atoms, n_central_atoms):
        super().__init__(data)
        self.n_sim = self.data_reader.n_sim
        self.directory = self.data_reader.directory
        self.file_list = self.data_reader.file_list
        self.cof = self.data_reader.cof
        self.n_wall_atoms = n_wall_atoms
        self.n_central_atoms = n_central_atoms
        self.force = None
        self.point = None
        self.force_tangential = None
        self.shear = None
        self.area = None
        self.delta = None
    
    def process_single_step(self, step, coor, orientation, shapex, shapez, vel, omega):
        data = np.loadtxt(self.directory + self.file_list[step], skiprows=9)
        self.assign_data(data)
        
        # Extract matching contact data for the central particles
        (centers1, centers2, orientations1, orientations2,
            shapex1, shapex2, shapez1, shapez2,
            vel1, vel2, omega1, omega2, 
            equivalent_mass, equivalent_radius
        ) = self.match_contact_data_with_particles(
            coor, orientation, shapex, shapez, vel, omega)

        relative_velocities = self.compute_relative_velocity(vel1, vel2, omega1, omega2, self.point, centers1)
        damping_coeff = self.compute_damping_coefficient(equivalent_mass, equivalent_radius, self.overlap1+self.overlap2)

        # Process contacts for each pair of ellipsoids (first particle)
        (normal_hist_cont_point_global1, tangential_hist_cont_point_global1, counts_cont_point_global1, 
            normal_hist_cont_point_local1, tangential_hist_cont_point_local1,  counts_cont_point_local1, 
            normal_hist_global1, normal_count_global1,
            tangential_hist_global1, tangential_count_global1,
            power_dissipation_normal1, power_dissipation_tangential1, bin_counts_power1
        ) = self.process_contacts(centers1, orientations1, shapex1, shapez1,
            self.force, relative_velocities, damping_coeff)

        # Process contacts for each pair of ellipsoids (second particle)
        (normal_hist_cont_point_global2, tangential_hist_cont_point_global2, counts_cont_point_global2, 
            normal_hist_cont_point_local2, tangential_hist_cont_point_local2, counts_cont_point_local2, 
            normal_hist_global2, normal_count_global2,
            tangential_hist_global2, tangential_count_global2,
            power_dissipation_normal2, power_dissipation_tangential2, bin_counts_power2
        ) = self.process_contacts(centers2, orientations2, shapex2, shapez2,
            -self.force, relative_velocities, damping_coeff)

        # Add the histograms and counts
        normal_hist_cont_point_global = normal_hist_cont_point_global1 + normal_hist_cont_point_global2
        tangential_hist_cont_point_global = tangential_hist_cont_point_global1 + tangential_hist_cont_point_global2
        
        counts_cont_point_global = counts_cont_point_global1 + counts_cont_point_global2

        normal_hist_cont_point_local = normal_hist_cont_point_local1 + normal_hist_cont_point_local2
    
        tangential_hist_cont_point_local = tangential_hist_cont_point_local1 + tangential_hist_cont_point_local2
        counts_cont_point_local = counts_cont_point_local1 + counts_cont_point_local2

        normal_hist_global = normal_hist_global1 + normal_hist_global2
        tangential_hist_global = tangential_hist_global1 + tangential_hist_global2
        normal_count_global = normal_count_global1 + normal_count_global2
        tangential_count_global = tangential_count_global1 + tangential_count_global2

        power_dissipation_normal = power_dissipation_normal1 + power_dissipation_normal2
        power_dissipation_tangential = power_dissipation_tangential1 + power_dissipation_tangential2
        bin_counts_power = bin_counts_power1 + bin_counts_power2

        
        avg_dict = {
            'global_normal_force_hist_cp': normal_hist_cont_point_global,
            'global_tangential_force_hist_cp': tangential_hist_cont_point_global,
            'local_normal_force_hist_cp': normal_hist_cont_point_local,
            'local_tangential_force_hist_cp': tangential_hist_cont_point_local,
            'global_normal_force_hist': normal_hist_global,
            'global_tangential_force_hist': tangential_hist_global,
            'contacts_hist_cont_point_global': counts_cont_point_global,
            'contacts_hist_cont_point_local': counts_cont_point_local,
            'contacts_hist_global_normal': normal_count_global,
            'contacts_hist_global_tangential': tangential_count_global,
            'power_dissipation_normal': power_dissipation_normal,
            'power_dissipation_tangential': power_dissipation_tangential,
            'bin_counts_power': bin_counts_power,
            'Z': self.contact_number
        }
        return avg_dict
    
    def assign_data(self,data):
        # Identify rows where periodic is not equal to 1
        self.contact_number = 2*len(data)/self.n_central_atoms
        
        # Filter the data based on valid indices
        self.id1 = data[:, 0].astype(int) - 1
        self.id2 = data[:, 1].astype(int) - 1
        # self.periodic = data[:, 2]  # If you need to keep the filtered periodic values
        self.force = data[:, 3:6]
        self.point = data[:, 8:11]
        self.overlap1 = data[:, 11]
        self.overlap2 = data[:, 12] 

    def match_contact_data_with_particles(self, coor, orientation, shapex, shapez, vel, omega):
        centers1 = coor[self.id1]
        centers2 = coor[self.id2]
        orientations1 = orientation[self.id1]
        orientations2 = orientation[self.id2]
        shapex1 = shapex[self.id1]
        shapex2 = shapex[self.id2]
        shapez1 = shapez[self.id1]
        shapez2 = shapez[self.id2]
        vel1 = vel[self.id1]
        vel2 = vel[self.id2]
        omega1 = omega[self.id1]
        omega2 = omega[self.id2]
        mass1 = self.compute_mass(shapex1, shapez1)
        mass2 = self.compute_mass(shapex2, shapez2)
        gauss_curv1 = self.compute_gaussian_curvature(shapex1, shapez1, self.point, centers1, orientations1)
        gauss_curv2 = self.compute_gaussian_curvature(shapex2, shapez2, self.point, centers2, orientations2)
        equivalent_mass = mass1 * mass2 / (mass1 + mass2)
        equivalent_radius = 1 / (gauss_curv1 + gauss_curv2)
        return (centers1, centers2, orientations1, orientations2,
                 shapex1, shapex2, shapez1, shapez2, vel1, vel2, omega1, omega2,
                equivalent_mass, equivalent_radius)

    def compute_mass(self, shapex, shapez, rho=1000, is_prolate=True):
        if is_prolate:
            volume = 4/3 * np.pi * shapex**2 * shapez
        else:
            volume = 4/3 * np.pi * shapex * shapez**2
        return rho*volume

    def compute_gaussian_curvature(self, shapex, shapez, contact_points, centers, orientations, is_prolate=True):
        """
        Compute the Gaussian curvature coefficient (1/length=sqrt(K_gauss)) at the contact point on the ellipsoid.

        Parameters:
        - shapex: The semi-axis length in the x direction.
        - shapey: The semi-axis length in the y direction.
        - shapez: The semi-axis length in the z direction.
        - contact_points: The positions of the contact points in the global frame.
        - centers: The centers of the ellipsoids in the global frame.
        - orientations: The orientation matrices of the ellipsoids.

        Returns:
        - curvatures: The Gaussian curvature at each contact point.
        """
        # Translate the contact points to the local frame of the ellipsoids
        local_contact_points = np.einsum('ijk,ik->ij', orientations.transpose(0, 2, 1), contact_points - centers)

        x = local_contact_points[:, 0]
        y = local_contact_points[:, 1]
        z = local_contact_points[:, 2]

        a = shapex
        if is_prolate:
            b = shapex
        else:
            b = shapez
        c = shapez

        numerator = a**2*b**6*c**6
        denominator = ((c*b)**4 + c**4*(a**2-b**2)*y**2 + b**4*(a**2-c**2)*z**2)**2

        curvature = np.sqrt(numerator / denominator )#this is the gaussian curvature coefficient (1/length)

        return curvature

    def compute_damping_coefficient(self, equivalent_mass, radius_curvature_avg, overlap, restitution_coeff=0.1):
        """
        Compute the damping coefficient for a Hertzian contact model.

        Parameters:
        - equivalent_mass: The equivalent mass of the two particles.
        - radius_curvature_avg: The average radius of curvature at the contact point.
        - overlap: The overlap distance at the contact point.
        - restitution_coeff: The coefficient of restitution.

        Returns:
        - damping_coefficient: The damping coefficient for the contact.
        """
        Young_modulus = 5.0e6  # Young's modulus of 5 MPa
        Poisson_ratio = 0.3
        effective_Young_modulus = Young_modulus / (1 - Poisson_ratio**2)/2
        beta = np.log(restitution_coeff)/np.sqrt(np.pi**2 + (np.log(restitution_coeff))**2)
        # Damping coefficient
        damping_coefficients = 2*np.sqrt(5/6)*beta*np.sqrt(2*equivalent_mass*effective_Young_modulus*np.sqrt(radius_curvature_avg*overlap))
        return damping_coefficients

    def compute_relative_velocity(self, vel1, vel2, omega1, omega2, contact_points, centers):
        """
        Compute the relative velocity at the contact points using vectorized operations.
        """
        r1_to_contact = contact_points - centers
        r2_to_contact = contact_points - centers
        v1 = vel1 + np.cross(omega1, r1_to_contact)
        v2 = vel2 + np.cross(omega2, r2_to_contact)

        # Compute relative velocities
        return v1 - v2

    def process_contacts(self, centers1, orientations1, shapex1, shapez1, forces, relative_velocities, damping_coeff):
        """
        Compute and bin the forces for contacts between two sets of ellipsoids.
        """
        # Compute normal vectors for the first ellipsoid (corresponding to id1 or id2)
        global_normals, local_normals = self.compute_normals(centers1, self.point, orientations1, shapex1, shapez1)

        # Compute the distance between the centers and contact points
        distances = np.linalg.norm(self.point - centers1, axis=1)
        
        # Compute the maximum shape parameter (shapex or shapez) for each contact
        max_shape = np.maximum(shapex1, shapez1)
        
        # Create a mask to filter out contacts where the distance is greater than max_shape (periodic boundary)
        valid_contact_mask = distances <= max_shape
        
        # Filter forces, normals, and other relevant data using the mask
        valid_forces = forces[valid_contact_mask]
        valid_global_normals = global_normals[valid_contact_mask]
        valid_centers1 = centers1[valid_contact_mask]
        valid_contact_points = self.point[valid_contact_mask]
        valid_orientations = orientations1[valid_contact_mask]
        valid_relative_velocities = relative_velocities[valid_contact_mask]
        valid_damping_coeff = damping_coeff[valid_contact_mask]
        # Project forces onto normals and tangents using vectorized operations
        normal_projection = np.einsum('ij,ij->i', valid_forces, valid_global_normals)[:, np.newaxis]
        normal_forces_global = normal_projection * valid_global_normals
        tangential_forces_global = valid_forces - normal_forces_global

        # # Calculate the magnitudes of the tangential forces
        # tangential_force_magnitudes = np.linalg.norm(tangential_forces_global, axis=1)

        # # Set a tolerance to determine when forces are considered parallel
        # tolerance = 0.1
        # non_parallel_indices = np.where(tangential_force_magnitudes > tolerance)[0]

        # # Report the indices and corresponding tangential force magnitudes
        # if len(non_parallel_indices) > 0:
        #     for index in non_parallel_indices:
        #         original_force_magnitude = np.linalg.norm(valid_forces[index])
        #         print(f"Index {index}: Tangential force magnitude = {tangential_force_magnitudes[index]:.6f}, Original force = {original_force_magnitude:.6f}")
        #         print(f"center1: {valid_centers1[index]}, point: {valid_contact_points[index]}, radius: {shapex1[index]}")
            
        # Bin forces using vectorized binning
        (normal_hist_cont_point_global, tangential_hist_cont_point_global,
          counts_cont_point_global) = self.bin_forces_by_xy_angle_contact_point_global(
            valid_contact_points, valid_centers1, normal_forces_global, tangential_forces_global)
        
        (normal_hist_cont_point_local, tangential_hist_cont_point_local,
          counts_cont_point_local) = self.bin_forces_by_ellipsoid_angle(
            valid_contact_points, valid_centers1, normal_forces_global, tangential_forces_global, valid_orientations)
        
        normal_hist_global, counts_normal_hist_global = self.accumulate_force_histogram(normal_forces_global)
        tangential_hist_global, counts_tangential_hist_global = self.accumulate_force_histogram(tangential_forces_global)

        
        (power_dissipation_normal, power_dissipation_tangential,
          bin_counts) = self.compute_and_bin_dissipation(valid_contact_points, valid_centers1, valid_global_normals, 
            tangential_forces_global, normal_forces_global, valid_orientations, valid_relative_velocities, valid_damping_coeff)

        return (normal_hist_cont_point_global, tangential_hist_cont_point_global, counts_cont_point_global, 
                normal_hist_cont_point_local, tangential_hist_cont_point_local, counts_cont_point_local,
                normal_hist_global, counts_normal_hist_global, tangential_hist_global, counts_tangential_hist_global,
                power_dissipation_normal, power_dissipation_tangential, bin_counts)

    def compute_normals(self, centers, contact_points, rotations, shapex, shapez):
        """
        Compute global and local normals for contacts on ellipsoids using vectorized operations.
        """
        local_contact_points = np.einsum('ijk,ik->ij', rotations.transpose(0, 2, 1), contact_points - centers)

        # Compute local normals with vectorized operations
        a_squared = shapex ** 2
        c_squared = shapez ** 2
        local_normals = np.zeros_like(local_contact_points)
        local_normals[:, 0] = local_contact_points[:, 0] / a_squared
        local_normals[:, 1] = local_contact_points[:, 1] / a_squared
        local_normals[:, 2] = local_contact_points[:, 2] / c_squared
        local_normals /= np.linalg.norm(local_normals, axis=1)[:, np.newaxis]

        # Transform to global normals
        global_normals = np.einsum('ijk,ik->ij', rotations, local_normals)

        # Vector from contact points to centers
        vectors_to_centers = centers - contact_points

        # vectors_to_centers /= np.linalg.norm(vectors_to_centers, axis=1)[:, np.newaxis]

        # # Calculate the cross product to check if vectors are parallel
        # cross_products = np.cross(global_normals, vectors_to_centers)

        # # Calculate norms of the cross products
        # cross_product_norms = np.linalg.norm(cross_products, axis=1)

        # # Check for non-parallel vectors (cross product norm > small tolerance)
        # tolerance = 1e-6
        # non_parallel_indices = np.where(cross_product_norms > tolerance)[0]

        # print(np.linalg.norm(vectors_to_centers, axis=1)>10)

        # # Raise a warning for the selected indices
        # if len(non_parallel_indices) > 0:
        #     for index in non_parallel_indices:
        #         print(f"Warning: Local normal and vector to center are not parallel at index {index}")
        
        return global_normals, local_normals

    def bin_forces_by_xy_angle_contact_point_global(self, contact_points, ellipsoid_centers, normal_forces, tangential_forces, num_bins=72):
        """
        Bin the normal and tangential forces based on angles in the XY plane using vectorized operations.
        """
        center_to_contact_vector_global = contact_points[:, :2] - ellipsoid_centers[:, :2]
        angles_rad = np.arctan2(center_to_contact_vector_global[:, 1], center_to_contact_vector_global[:, 0])
        angles_deg = np.degrees(angles_rad)
        angles_deg = np.mod(angles_deg + 360, 360)

        bins = np.linspace(0, 360, num_bins + 1)
        bin_indices = np.digitize(angles_deg, bins) - 1

        # Clip bin indices to ensure they are within the valid range
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Use np.add.at for vectorized binning
        normal_hist = np.zeros(num_bins)
        tangential_hist = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)  # Count of forces in each bin
        np.add.at(normal_hist, bin_indices, np.linalg.norm(normal_forces, axis=1))
        np.add.at(tangential_hist, bin_indices, np.linalg.norm(tangential_forces, axis=1))
        np.add.at(bin_counts, bin_indices, 1)

        return normal_hist, tangential_hist, bin_counts

    def bin_forces_by_ellipsoid_angle(self, contact_points, ellipsoid_centers, normal_forces, tangential_forces, orientations, num_bins=36):
        """
        Bin the normal and tangential forces based on angles with respect to the ellipsoid's principal axis using vectorized operations.
        """
        center_to_contact_vector_global = contact_points - ellipsoid_centers

        center_to_contact_vector_local = np.einsum('ijk,ik->ij', orientations.transpose(0, 2, 1), center_to_contact_vector_global) # Rotate to local coordinates
        angles_rad = np.arccos(center_to_contact_vector_local[:, 2] / np.linalg.norm(center_to_contact_vector_local, axis=1))
        angles_deg = np.degrees(angles_rad)

        # Use symmetry to map angles into the [0, 90] range with vectorized operations
        angles_deg = np.abs(angles_deg)
        angles_deg = np.where(angles_deg > 90, 180 - angles_deg, angles_deg)

        bins = np.linspace(0, 90, num_bins + 1)
        bin_indices = np.digitize(angles_deg, bins) - 1

        # Clip bin indices to ensure they are within the valid range
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Use np.add.at for vectorized binning
        normal_hist = np.zeros(num_bins)
        tangential_hist = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)  # Count of forces in each bin
        np.add.at(normal_hist, bin_indices, np.linalg.norm(normal_forces, axis=1))
        np.add.at(tangential_hist, bin_indices, np.linalg.norm(tangential_forces, axis=1))
        np.add.at(bin_counts, bin_indices, 1)

        return normal_hist, tangential_hist, bin_counts

    def accumulate_force_histogram(self, forces, num_bins=72):
        """
        Accumulate the magnitudes of forces into bins based on their angles.

        Parameters:
        - forces: An array of 2D force vectors (numpy array) of shape (n, 2).
        - num_bins: The number of bins to divide the 360-degree circle into.

        Returns:
        - hist: A histogram of force magnitudes accumulated in bins.
        """
        # Calculate the magnitudes of the forces
        magnitudes = np.linalg.norm(forces, axis=1)

        # Calculate the angles of the forces in degrees
        angles_rad = np.arctan2(forces[:, 1], forces[:, 0])
        angles_deg = np.degrees(angles_rad) % 360

        # Calculate the bin indices for each angle
        bins = np.linspace(0, 360, num_bins + 1)
        bin_indices = np.digitize(angles_deg, bins) - 1

        # Ensure bin indices are within the valid range
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Accumulate magnitudes into histogram bins
        hist = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)  # Count of forces in each bin

        np.add.at(hist, bin_indices, magnitudes)
        np.add.at(bin_counts, bin_indices, 1)

        return hist, bin_counts
    
    def compute_and_bin_dissipation(self, contact_points, ellipsoid_centers_1, normals, 
                                    tangential_forces, normal_forces_global, orientations_1, 
                                    relative_velocities, damping_coeff, num_bins=36):
        """
        Compute the power dissipation at contact points and bin it based on angles with respect to the ellipsoid's principal axis.
        
        Parameters:
        - contact_points: Positions of contact points.
        - ellipsoid_centers_1: Positions of ellipsoid centers for the first particle.
        - normals: Normal vectors at contact points.
        - tangential_forces: Tangential forces at contact points.
        - orientations_1: Orientation matrices for the first particle.
        - relative_velocities: Relative velocities at contact points.
        - damping_coeff: Damping coefficient for the contact.
        - num_bins: Number of bins to divide the 90-degree range into.

        Returns:
        - normal_dissipation_hist: Binned power dissipation in the normal direction.
        - tangential_dissipation_hist: Binned power dissipation in the tangential direction.
        - bin_counts: Count of forces in each bin.
        """
 
        # Normalize the normal vectors
        normal_directions = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        # Compute normal and tangential components of relative velocities
        normal_velocities = np.einsum('ij,ij->i', relative_velocities, normal_directions)[:, np.newaxis]*normal_directions
        tangential_velocities = relative_velocities - normal_velocities

        # Compute the dissipation in both directions and divide by 2 to get power on single ellipsoid
        power_dissipation_normal = np.abs(damping_coeff * np.linalg.norm(normal_velocities, axis=1)**2)/2
        power_dissipation_tangential =  np.abs(np.einsum('ij,ij->i', tangential_forces, tangential_velocities))/2 # no need for friction coefficient as it is already included in the tangential forces

        # #ratio of damping to elastic normal force
        # damping_force = damping_coeff * np.linalg.norm(normal_velocities, axis=1)
        # total_normal_force = np.linalg.norm(normal_forces_global, axis=1)
        # damping_ratio = damping_force / total_normal_force
   
        # Binning process as in bin_forces_by_ellipsoid_angle
        center_to_contact_vector_global = contact_points - ellipsoid_centers_1
        center_to_contact_vector_local = np.einsum('ijk,ik->ij', orientations_1.transpose(0, 2, 1), center_to_contact_vector_global)  # Rotate to local coordinates
        angles_rad = np.arccos(center_to_contact_vector_local[:, 2] / np.linalg.norm(center_to_contact_vector_local, axis=1))
        angles_deg = np.degrees(angles_rad)

        # Use symmetry to map angles into the [0, 90] range with vectorized operations
        angles_deg = np.abs(angles_deg)
        angles_deg = np.where(angles_deg > 90, 180 - angles_deg, angles_deg)

        bins = np.linspace(0, 90, num_bins + 1)
        bin_indices = np.digitize(angles_deg, bins) - 1

        # Clip bin indices to ensure they are within the valid range
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Use np.add.at for vectorized binning
        normal_dissipation_hist = np.zeros(num_bins)
        tangential_dissipation_hist = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)  # Count of forces in each bin
        
        np.add.at(normal_dissipation_hist, bin_indices, power_dissipation_normal)
        np.add.at(tangential_dissipation_hist, bin_indices, power_dissipation_tangential)
        np.add.at(bin_counts, bin_indices, 1)

        return normal_dissipation_hist, tangential_dissipation_hist, bin_counts

    def compute_number_of_rattlers(self):
        """Compute the number of rattlers"""
        self.rattlers = 0
        for i in range(self.n_central_atoms):
            if (np.sum(self.id1 == i) + np.sum(self.id2 == i)) < 3:
                self.rattlers += 1
        return self.rattlers

    def force_single_step(self, step):
        """Processing on the data for one single step
        Calling the other methods for the processing"""
        data = np.loadtxt(self.directory+self.file_list[step], skiprows=9)
        #excluding contacts between wall particles
        first_col_check = (data[:,6] > self.n_wall_atoms)
        second_col_check = (data[:,7] > self.n_wall_atoms)
        not_wall_contacts = np.logical_and(first_col_check, second_col_check)
        data = data[not_wall_contacts]  
        self.force = data[:, 9:12] #contact_force
        self.force_tangential = data[:, 12:15] #contact_tangential_force
        #compute intesity of the normal force
        force_normal = np.linalg.norm(self.force-self.force_tangential, axis=1)
        force_dict = {'force_normal': force_normal,
                    'force_tangential': np.linalg.norm(self.force_tangential, axis=1)}
        return force_dict

    def compute_force_distribution(self):
        """Compute the force distribution"""
        self.force_distribution_x = np.histogram(np.abs(self.force[:, 0]), bins=100)
        self.force_distribution_y = np.histogram(np.abs(self.force[:, 1]), bins=100)
        self.force_distribution_z = np.histogram(np.abs(self.force[:, 2]), bins=100)
        self.force_distribution = np.histogram(np.linalg.norm(self.force, axis=1), bins=100)

    def compute_force_tangential_distribution(self):
        """Compute the tangential force distribution"""
        self.force_tangential_distribution_x = np.histogram(self.force_tangential[:, 0], bins=100)
        self.force_tangential_distribution_y = np.histogram(self.force_tangential[:, 1], bins=100)
        self.force_tangential_distribution_z = np.histogram(self.force_tangential[:, 2], bins=100)
        self.force_tangential_distribution = np.histogram(np.linalg.norm(self.force_tangential, axis=1), bins=100)

    def compute_space_averages(self):
        """Compute the space averages"""
        force_normal_magnitude = np.linalg.norm(self.force_normal, axis=1)
        force_tangential_magnitude = np.linalg.norm(self.force_tangential, axis=1)
        self.force_normal_space_average = np.mean(force_normal_magnitude)
        self.force_tangential_space_average = np.mean(force_tangential_magnitude)
        #self.shear_space_average = np.mean(self.shear, axis=0)

        try :
            self.contact_angles
            self.contact_angles_space_average = np.mean(self.contact_angles)
        except AttributeError:
            raise AttributeError("The contact angles have not been computed yet.")
        
    def store_data_tracked_grains(self, data, n_sampled_particles):
        """
        Store the data for a subset of the particles ids
        """
        #find the particle index
        sampledIDxs = np.linspace(0, self.n_central_atoms-1, n_sampled_particles).astype('int')
        avg_max_contact_per_particle = 10
        data_sampled = np.zeros((avg_max_contact_per_particle*n_sampled_particles, 10))
        # find data corresponding to the sampled particles
        # first column force keeps the sign
        count = 0
        for i in range(n_sampled_particles):
            # initilize a list for the information to store for each particle
            
            # select the rows corresponding to the particle
            rows_plus = data[(data[:, 6] == sampledIDxs[i])]
            rows_minus = data[(data[:, 7] == sampledIDxs[i])]

            for j in range(rows_plus.shape[0]):

                data_sampled[count, 0] = i #index of the particle the contact referes to in the zeroth column
                data_sampled[count, 1:4] = rows_plus[j, 19:22] #contact point
                data_sampled[count, 4:7] = rows_plus[j, 9:12] #contact force
                data_sampled[count, 7:10] = rows_plus[j , 12:15] #contact tangential force
                count += 1  

            for j in range(rows_minus.shape[0]):
                data_sampled[count, 0] = i
                data_sampled[count, 1:4] = rows_minus[j, 19:22] #contact point
                data_sampled[count, 4:7] -= rows_minus[j, 9:12]
                data_sampled[count, 7:10] -= rows_minus[j, 12:15]
                #store the data   
                count += 1
        #remove the rows with all zeros which remained empty
        data_sampled = data_sampled[~np.all(data_sampled == 0, axis=1)]
        return data_sampled
    
    def plot_force_chain(self, step, phi):
        """Plot the force chain"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        data = np.loadtxt(self.directory+self.file_list[step], skiprows=9)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        force_intensity = []
        for i in range(len(data)):
            if data[i, 6] > self.n_wall_atoms and data[i, 7] > self.n_wall_atoms:
                if np.linalg.norm(data[i, 0:3]-data[i, 3:6]) < 0.15:
                    force_intensity.append(np.linalg.norm(data[i, 9:12]))

        count = 0
        max_force = max(force_intensity)
        for i in range(len(data)):
            if data[i, 6] > self.n_wall_atoms and data[i, 7] > self.n_wall_atoms:
                # if distance between particles is smaller than certain value
                if np.linalg.norm(data[i, 0:3]-data[i, 3:6]) < 0.15:
                    ax.plot([data[i, 0], data[i,3]], [data[i, 1], data[i,4]], [data[i, 2], data[i,5]], linestyle='-', color='k', linewidth=2*force_intensity[count]/max_force)
                    count += 1
        # Set equal axis scaling
        ax.set_box_aspect([np.ptp(coord) for coord in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])

        # set view to xy plane
        ax.view_init(90, -90)

        # put only a few values per axis and big font
        #ax.set_xticks([])
        #ax.set_yticks([])
        ax.set_zticks([])     
        
        ax.set_xlabel('X ')
        ax.set_ylabel('Y ')
        ax.set_zlabel('Z ')
        ax.set_title('Force chains, ap='+str(self.data_reader.ap)+', cof='+str(self.data_reader.cof)+', phi='+str(phi)+', step='+str(step))

        plt.savefig('output_plots/force_chain_ap_'+str(self.data_reader.ap)+', cof_'+str(self.data_reader.cof)+'phi='+str(phi)+'step='+ str(step)+ '.png')
        plt.show()
        plt.close()