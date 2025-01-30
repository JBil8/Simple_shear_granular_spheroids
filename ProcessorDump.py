import multiprocessing
import numpy as np
from DataProcessor import DataProcessor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class ProcessorDump(DataProcessor):
    def __init__(self, data, n_wall_atoms, n_central_atoms):
        super().__init__(data)
        self.n_sim = self.data_reader.n_sim
        self.directory = self.data_reader.directory
        self.file_list = self.data_reader.file_list
        self.I = self.data_reader.I
        self.cof = self.data_reader.cof
        self.ap = self.data_reader.ap
        self.n_wall_atoms = n_wall_atoms
        self.n_central_atoms = n_central_atoms
        self.force = None
        self.point = None
        self.force_tangential = None
        self.shear = None
        self.area = None
        self.delta = None
    
    def process_single_step(self, step, coor, orientation, shapex, shapez, 
                            vel, omega, box_lengths, shear_rate, dt, forces_particles, mass_particles):
        data = np.loadtxt(self.directory + self.file_list[step], skiprows=9)
        self.assign_data(data)
        # Extract matching contact data for the central particles
        (centers1, centers2, orientations1, orientations2,
            shapex1, shapex2, shapez1, shapez2,
            vel1, vel2, omega1, omega2, 
            equivalent_mass,
            mass1, mass2, inertia1, inertia2
        ) = self.match_contact_data_with_particles(
            coor, orientation, shapex, shapez, vel, omega, box_lengths, shear_rate)
        
        rattlers = self.compute_contact_without_rattlers()
        global_normals = self.compute_normals(centers1, self.point, orientations1, shapex1, shapez1, self.force, self.shear) 
        fabric = self.compute_fabric_tensor(global_normals)

        # self.check_plot_global_forces(self.force)

        cp1 = self.point+global_normals*self.overlap1[:, np.newaxis]
        cp2 = self.point-global_normals*self.overlap2[:, np.newaxis]
        curv1 = self.compute_gaussian_curvature(shapex1, shapez1, cp1, centers1, orientations1)
        curv2 = self.compute_gaussian_curvature(shapex2, shapez2, cp2, centers2, orientations2)
        # curv1 = self.compute_volume_equivalent_curvature(shapex1, shapez1)
        # curv2 = self.compute_volume_equivalent_curvature(shapex2, shapez2)

        equivalent_radius = 1 / (curv1 + curv2)
        total_overlap = self.overlap1 + self.overlap2
        forces_contacts_computed, particles_torques = self.compute_particle_force_and_torque(centers1, centers2)

        vel_correction1 = self.integrate_half_linear(mass1, forces_contacts_computed[self.id1], dt)
        vel_correction2 = self.integrate_half_linear(mass2, forces_contacts_computed[self.id2], dt)
        angular_correction1 = self.integrate_half_angular(inertia1, particles_torques[self.id1], orientations1, omega1, dt)
        angular_correction2 = self.integrate_half_angular(inertia2, particles_torques[self.id2], orientations2, omega2, dt)
        # vel_correction1 = vel_correction2 = angular_correction1 = angular_correction2 = np.zeros_like(vel1)
        relative_velocities = self.compute_relative_velocity(vel1, vel2, omega1, omega2, self.point, self.point, centers1, centers2, 
                                                             vel_correction1, vel_correction2, angular_correction1, angular_correction2) #relative velocity is computed at the contact point

        damping_coeff = self.compute_damping_coefficient(equivalent_mass, equivalent_radius, total_overlap)
        normal_hertz = self.compute_normal_hertzian_force(equivalent_radius, total_overlap)
        tangential_hertz_stiffness = self.compute_tangential_hertzian_stiffness(equivalent_radius, total_overlap)

        # self.plot_force_chains(centers1, centers2, step)
        # optional, compute the stress tensor
        box_volume = box_lengths[0] * box_lengths[1] * box_lengths[2] #volume of tricilinic box
        # Process contacts for each pair of ellipsoids (first particle)
        (normal_hist_cont_point_global1, tangential_hist_cont_point_global1, counts_cont_point_global1, 
            normal_hist_cont_point_local1, tangential_hist_cont_point_local1,  counts_cont_point_local1, 
            normal_hist_global1, normal_count_global1,
            tangential_hist_global1, tangential_count_global1,
            power_dissipation_normal1, power_dissipation_tangential1, bin_counts_power1,
            normal_hist_mixed1, tangential_hist_mixed1, counts_mixed1, computed_force
        ) = self.process_contacts(centers1, orientations1,
            self.force, relative_velocities, equivalent_radius, normal_hertz, 
            damping_coeff, global_normals,tangential_hertz_stiffness, self.shear, dt)

        # Process contacts for each pair of ellipsoids (second particle)
        (normal_hist_cont_point_global2, tangential_hist_cont_point_global2, counts_cont_point_global2, 
            normal_hist_cont_point_local2, tangential_hist_cont_point_local2, counts_cont_point_local2, 
            normal_hist_global2, normal_count_global2,
            tangential_hist_global2, tangential_count_global2,
            power_dissipation_normal2, power_dissipation_tangential2, bin_counts_power2,
            normal_hist_mixed2, tangential_hist_mixed2, counts_mixed2, _
        ) = self.process_contacts(centers2, orientations2,
            -self.force, -relative_velocities, equivalent_radius, normal_hertz, 
            damping_coeff, -global_normals, -tangential_hertz_stiffness, self.shear, dt)
        
        stress = self.compute_stress(centers1, centers2, computed_force, box_volume)
        particle_stress, N1_diff_distr, N2_diff_distr = self.compute_particle_stress(centers1, centers2, computed_force, mass_particles)
        
        # print(f"max N1: {np.max(N1_diff_distr)}, max N2: {np.max(N2_diff_distr)}, min N1: {np.min(N1_diff_distr)}, min N2: {np.min(N2_diff_distr)}")
        # print(f"mean N1: {np.mean(N1_diff_distr)}, mean N2: {np.mean(N2_diff_distr)}")
        # print(f"max N1 no rattlers: {np.max(N1_diff_distr[~rattlers])}") 
        # Add the histograms and counts
        (normal_hist_cont_point_global, 
            tangential_hist_cont_point_global, 
            counts_cont_point_global,
            normal_hist_cont_point_local,
            tangential_hist_cont_point_local,
            counts_cont_point_local,
            normal_hist_global,
            tangential_hist_global,
            normal_count_global,
            tangential_count_global,
            power_dissipation_normal,
            power_dissipation_tangential,
            bin_counts_power, 
            normal_hist_mixed,
            tangential_hist_mixed,
            counts_mixed
        ) = self.sum_histograms(
            (normal_hist_cont_point_global1, normal_hist_cont_point_global2),
            (tangential_hist_cont_point_global1, tangential_hist_cont_point_global2),
            (counts_cont_point_global1, counts_cont_point_global2),
            (normal_hist_cont_point_local1, normal_hist_cont_point_local2),
            (tangential_hist_cont_point_local1, tangential_hist_cont_point_local2),
            (counts_cont_point_local1, counts_cont_point_local2),
            (normal_hist_global1, normal_hist_global2),
            (tangential_hist_global1, tangential_hist_global2),
            (normal_count_global1, normal_count_global2),
            (tangential_count_global1, tangential_count_global2),
            (power_dissipation_normal1, power_dissipation_normal2),
            (power_dissipation_tangential1, power_dissipation_tangential2),
            (bin_counts_power1, bin_counts_power2),
            (normal_hist_mixed1, normal_hist_mixed2),
            (tangential_hist_mixed1, tangential_hist_mixed2),
            (counts_mixed1, counts_mixed2)
        )


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
            'normal_force_hist_mixed': normal_hist_mixed,
            'tangential_force_hist_mixed': tangential_hist_mixed,
            'counts_mixed': counts_mixed,
            'Z': self.contact_number,
            'percent_sliding': self.percent_sliding,
            'stress_contacts': stress, 
            'fabric': fabric,
            # 'N1_diff': N1_diff_distr, 
            # 'N2_diff': N2_diff_distr
            }
        return avg_dict
 
    def process_contacts(self, centers1, orientations, forces, 
                         relative_velocities, equivalent_radius, normal_hertz,
                         damping_coeff, global_normals, tangential_hertz_stiffness, shear, dt):
        """
        Compute and bin the forces for contacts between two sets of ellipsoids.
        """
        # Project forces onto normals and tangents using vectorized operations
        # normal_projection = np.einsum('ij,ij->i', forces, global_normals)[:, np.newaxis]
        # normal_forces_global = normal_projection * global_normals
        # tangential_forces_global = forces - normal_forces_global

        # Compute the norm of tangential and normal forces for each row
        # norm_tangential = np.linalg.norm(tangential_forces_global, axis=1)
        # norm_normal = np.linalg.norm(normal_forces_global, axis=1)

        if shear.any(): # Check if shear is not zero  
            tangential_direction = shear / np.linalg.norm(shear, axis=1)[:, np.newaxis]
            tangential_global_force = np.einsum('ij,ij->i', forces, tangential_direction)[:, np.newaxis] * tangential_direction
        else:
            tangential_global_force = np.zeros_like(forces)
            tangential_direction = np.zeros_like(forces)

        normal_global_force = forces-tangential_global_force
        # compare global normals with computed normals
        #computed_normals = normal_global_force/np.linalg.norm(normal_global_force, axis=1)[:, np.newaxis]
        # diff = np.linalg.norm(global_normals - computed_normals, axis=1)
        # diff_indices = np.where(diff > 0.001)[0]
        # for index in diff_indices:
        #     print(f"Index {index}: Difference = {diff[index]:.6f} simulation: {computed_normals[index]}, computed: {global_normals[index]}")
        global_normals = normal_global_force/np.linalg.norm(normal_global_force, axis=1)[:, np.newaxis]

        normal_force = normal_global_force
        tangential_force = tangential_global_force
        total_force = normal_force + tangential_force

        # Bin forces using vectorized binning
        (normal_hist_cont_point_global, tangential_hist_cont_point_global,
          counts_cont_point_global) = self.bin_forces_by_xy_angle_contact_point_global(
            self.point, centers1, normal_force, tangential_force) #normal_forces_global, tangential_forces_global)
        
        (normal_hist_cont_point_local, tangential_hist_cont_point_local,
          counts_cont_point_local) = self.bin_forces_by_ellipsoid_angle(
            self.point, centers1, normal_force, tangential_force, orientations)
        
        (normal_hist_mixed, tangential_hist_mixed, counts_mixed) = self.bin_forces_by_ellipsoid_angle_mixed(
            self.point, centers1, normal_force, tangential_force, orientations)

        normal_hist_global, counts_normal_hist_global = self.accumulate_force_histogram(normal_force)
        # normal_hist_global, counts_normal_hist_global = self.accumulate_force_histogram(normal_force)
        tangential_hist_global, counts_tangential_hist_global = self.accumulate_force_histogram(tangential_force)
        
        (power_dissipation_normal, power_dissipation_tangential,
          bin_counts) = self.compute_and_bin_dissipation(self.point, centers1, global_normals, 
            tangential_force, normal_force, orientations, relative_velocities, 
            normal_hertz, damping_coeff, tangential_hertz_stiffness, shear, dt)

        return (normal_hist_cont_point_global, tangential_hist_cont_point_global, counts_cont_point_global, 
                normal_hist_cont_point_local, tangential_hist_cont_point_local, counts_cont_point_local,
                normal_hist_global, counts_normal_hist_global, tangential_hist_global, counts_tangential_hist_global,
                power_dissipation_normal, power_dissipation_tangential, bin_counts, normal_hist_mixed, tangential_hist_mixed, counts_mixed, total_force
                )
 
    def sum_histograms(self, *histogram_pairs):
        """
        Sums corresponding pairs of histograms and returns the results.
        
        Parameters:
        - histogram_pairs: Each argument should be a tuple of the form (hist1, hist2).
        
        Returns:
        - A tuple of summed histograms for each provided pair.
        """
        return tuple(hist1 + hist2 for hist1, hist2 in histogram_pairs)

    def bin_1d_histogram(self, angles_deg, physical_quantity, ang_range, num_bins=10):
        """
        Assuming that anglesrange starts from 0 and ends at ang_range, bin the physical_quantity based on the angles using vectorized operations.
        Consider physical_quantity as 3d*Nc or 1d*Nc vectors"""    

        if physical_quantity.ndim == 1:
            magnitudes = physical_quantity  # Simply take absolute value for 1
            #print(f"force_magnitudes: {force_magnitudes[force_magnitudes < 0]}")
        else:
            magnitudes = np.linalg.norm(physical_quantity, axis=1)  # Compute magnitude for 3D physical_quantity
 
        bins = np.linspace(0, ang_range, num_bins + 1)
        bin_indices = np.digitize(angles_deg, bins) - 1

        # Clip bin indices to ensure they are within the valid range
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Use np.add.at for vectorized binning
        hist = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)
        np.add.at(hist, bin_indices, magnitudes)
        np.add.at(bin_counts, bin_indices, 1)

        return hist, bin_counts

    def compute_ellipsoid_angle_local(self, contact_points, ellipsoid_centers, orientations):
        """
        Bin the contact points based on angles with respect to the ellipsoid's principal axis using vectorized operations.
        """
        center_to_contact_vector_global = contact_points - ellipsoid_centers
        center_to_contact_vector_local = np.einsum('ikj,ik->ij', orientations, center_to_contact_vector_global)
        angles_rad = np.arccos(center_to_contact_vector_local[:, 2] / np.linalg.norm(center_to_contact_vector_local, axis=1))
        
        angles_deg = np.degrees(angles_rad)

        # Use symmetry to map angles into the [0, 90] range with vectorized operations
        angles_deg = np.abs(angles_deg)
        angles_deg = np.where(angles_deg > 90, 180 - angles_deg, angles_deg)

        return angles_deg

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
        self.shear = -data[:, 13:16]

    def match_contact_data_with_particles(self, coor, orientation, shapex, shapez, vel, omega, box_lengths, shear_rate):
   
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

        #Compute the maximum shape extent for each pair
        max_shape_extent = np.maximum(np.maximum(shapex1, shapex2), np.maximum(shapez1, shapez2))

        #Check and adjust centers for y-axis if the distance exceeds 2 * max_shape_extent
        distances_y = centers1[:, 1] - centers2[:, 1]
        adjust_y = np.abs(distances_y) > 2 * max_shape_extent
        centers1[adjust_y, 1] -= np.sign(distances_y[adjust_y]) * box_lengths[1]
        vel1[adjust_y, 0] -= np.sign(distances_y[adjust_y]) * shear_rate*box_lengths[1]
        centers1[adjust_y, 0] -= np.sign(distances_y[adjust_y]) * box_lengths[3]

        # Check and adjust centers for x-axis if the distance exceeds 2 * max_shape_extent
        distances_x = centers1[:, 0] - centers2[:, 0]
        adjust_x = np.abs(distances_x) > 2 * max_shape_extent
        centers1[adjust_x, 0] -= np.sign(distances_x[adjust_x]) * box_lengths[0]
        
        # Check and adjust centers for z-axis if the distance exceeds 2 * max_shape_extent
        distances_z = centers1[:, 2] - centers2[:, 2]
        adjust_z = np.abs(distances_z) > 2 * max_shape_extent
        centers1[adjust_z, 2] -= np.sign(distances_z[adjust_z]) * box_lengths[2]

        # Adjust the contact point for y-axis if the distance is greater than the max shape extent
        dist_contact_y = self.point[:, 1] - centers1[:, 1]
        adjust_y = np.abs(dist_contact_y) > 2 * max_shape_extent
        self.point[adjust_y, 1] -= np.sign(dist_contact_y[adjust_y]) * box_lengths[1]
        self.point[adjust_y, 0] -= np.sign(dist_contact_y[adjust_y]) * box_lengths[3]

        # Adjust the contact point for x-axis if the distance is greater than the max shape extent
        dist_contact_x = self.point[:, 0] - centers1[:, 0]
        adjust_x = np.abs(dist_contact_x) > 2 * max_shape_extent
        self.point[adjust_x, 0] -= np.sign(dist_contact_x[adjust_x]) * box_lengths[0]

        # Adjust the contact point for z-axis if the distance is greater than the max shape extent
        dist_contact_z = self.point[:, 2] - centers1[:, 2]
        adjust_z = np.abs(dist_contact_z) > 2 * max_shape_extent
        self.point[adjust_z, 2] -= np.sign(dist_contact_z[adjust_z]) * box_lengths[2]

        #print(f"Number of valid contacts: {np.sum(valid_indices)}, Total contacts: {len(valid_indices)}")

        # # Compute the distances between centers1 and centers2
        # distances = np.linalg.norm(centers1 - centers2, axis=1)
        
        # # Compute the maximum shape extent for each pair
        # max_shape_extent = np.maximum(np.maximum(shapex1, shapex2), np.maximum(shapez1, shapez2))
        
        # # Filter indices where distance is greater than the corresponding maximum shape extent
        # valid_indices = distances <= 2*max_shape_extent

        # # Apply the filter to all the data arrays
        # centers1 = centers1[valid_indices]
        # centers2 = centers2[valid_indices]
        # orientations1 = orientations1[valid_indices]
        # orientations2 = orientations2[valid_indices]
        # shapex1 = shapex1[valid_indices]
        # shapex2 = shapex2[valid_indices]
        # shapez1 = shapez1[valid_indices]
        # shapez2 = shapez2[valid_indices]
        # vel1 = vel1[valid_indices]
        # vel2 = vel2[valid_indices]
        # omega1 = omega1[valid_indices]
        # omega2 = omega2[valid_indices]
        # self.point = self.point[valid_indices]
        # self.force = self.force[valid_indices]
        # self.overlap1 = self.overlap1[valid_indices]
        # self.overlap2 = self.overlap2[valid_indices]
        # self.shear = self.shear[valid_indices]

        # Compute mass, inrtia and equivalent mass
        mass1 = self.compute_mass(shapex1, shapez1)
        mass2 = self.compute_mass(shapex2, shapez2)

        inertia1 = self.compute_inertia(shapex1, shapez1, mass1)
        inertia2 = self.compute_inertia(shapex2, shapez2, mass2)

        equivalent_mass = mass1 * mass2 / (mass1 + mass2)
        
        return (centers1, centers2, orientations1, orientations2,
                shapex1, shapex2, shapez1, shapez2, vel1, vel2, omega1, omega2,
                equivalent_mass, mass1, mass2, inertia1, inertia2)

    def compute_particle_force_and_torque(self, centers1, centers2):

        total_force = np.zeros((self.n_central_atoms, 3))
        total_torque = np.zeros((self.n_central_atoms, 3))
        
        # Vectorized computation of contact vectors for both particles
        contact_vector1 = self.point - centers1  # Vector from center of particle 1 to contact point
        contact_vector2 = self.point - centers2  # Vector from center of particle 2 to contact point

        # Vectorized computation of torques using cross product
        torque1 = np.cross(contact_vector1, self.force)  # Torque on particle 1
        torque2 = np.cross(contact_vector2, -self.force)  # Torque on particle 2

        # Accumulate forces and torques for each particle using np.add.at to handle possible multiple contacts per particle
        np.add.at(total_force, self.id1, self.force)      # Add contact forces to particle 1
        np.add.at(total_torque, self.id1, torque1)        # Add torques to particle 1

        np.add.at(total_force, self.id2, -self.force)     # Subtract contact forces for particle 2
        np.add.at(total_torque, self.id2, torque2)        # Add torques to particle 2

        return total_force, total_torque  

    def compute_particle_dissipation(self, dissipation_normal, dissipation_tangential):
        """
        Compute the total dissipation for each particle using vectorized operations.
        """
        total_dissipation = np.zeros(self.n_central_atoms)
        np.add.at(total_dissipation, self.id1, dissipation_normal)
        np.add.at(total_dissipation, self.id2, dissipation_normal)
        np.add.at(total_dissipation, self.id1, dissipation_tangential)
        np.add.at(total_dissipation, self.id2, dissipation_tangential)

        return total_dissipation

    def integrate_half_linear(self, mass, force, dt):
        """
        Integrate the linear velocity for a half time-step
        Return the velocity correction
        """
        return force * dt / (4* mass[:, np.newaxis]) #half-step integration
    
    def integrate_half_angular(self, inertia, torque, orientation, omega, dt):
        """
        integrate the angular movement with euler method and compute the velocity correction for the contact point
        inertia: The inertia of the particle in the principal axis frame
        """
        local_torque = np.einsum('ikj,ik->ij', orientation, torque)
        omega_local = np.einsum('ikj,ik->ij', orientation, omega)
        gyroscopic_term = np.cross(omega_local, np.einsum('ijk,ik->ij', inertia, omega_local))
        total_local_torque = local_torque - gyroscopic_term 
        alpha = np.zeros_like(omega)
        alpha[:,0] = total_local_torque[:,0]/inertia[:,0,0]
        alpha[:,1] = total_local_torque[:,1]/inertia[:,1,1]
        alpha[:,2] = total_local_torque[:,2]/inertia[:,2,2]
        delta_omega_local = alpha*dt/4
        delta_omega_global = np.einsum('ijk,ik->ij', orientation, delta_omega_local)

        return delta_omega_global

    def compute_normals(self, centers, contact_points, rotations, shapex, shapez, force, shear):
        """
        Compute global and local normals for contacts on ellipsoids using vectorized operations.
        """
        local_contact_points = np.einsum('ikj,ik->ij', rotations, contact_points - centers)

        # Compute local normals with vectorized operations
        a_squared = shapex ** 2
        c_squared = shapez ** 2
        local_normals = np.zeros_like(local_contact_points)
        local_normals[:, 0] = local_contact_points[:, 0] / a_squared
        if self.ap < 1:
            local_normals[:, 1] = local_contact_points[:, 1] / a_squared
        local_normals[:, 2] = local_contact_points[:, 2] / c_squared
        local_normals /= np.linalg.norm(local_normals, axis=1)[:, np.newaxis]

        # Transform to global normals
        global_normals = np.einsum('ijk,ik->ij', rotations, local_normals)
        
        return global_normals

    def compute_mass(self, shapex, shapez, rho=1000):
        volume = 4/3 * np.pi * shapex**2 * shapez
        return rho*volume

    def compute_inertia(self, shapex, shapez, mass):
        """
        Compute the inertia tensor of the particles in the principal axis system
        """
        inertia_tensor = np.zeros((len(shapex), 3, 3))
        inertia_tensor[:, 0, 0] = 0.2*mass*(shapex**2+shapez**2)
        inertia_tensor[:, 1, 1] = 0.2*mass*(shapex**2+shapez**2)
        inertia_tensor[:, 2, 2] = 0.4*mass*shapex**2
        return inertia_tensor

    def compute_gaussian_curvature(self, shapex, shapez, contact_points, centers, orientations):
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
        local_contact_points = np.einsum('ikj,ik->ij', orientations, contact_points - centers)

        x = local_contact_points[:, 0]
        y = local_contact_points[:, 1]
        z = local_contact_points[:, 2]

        a = shapex
        b = shapex
        c = shapez

        curvature_coeff = (a*b*c*(x**2/a**4+y**2/b**4+z**2/c**4))**-1

        return curvature_coeff
    
    def compute_mean_curvature(self, shapex, shapez, contact_points, centers, orientations):
        """
        Compute the Mean curvature coefficient (1/length = H) at the contact point on the ellipsoid.

        Parameters:
        - shapex: The semi-axis length in the x direction.
        - shapey: The semi-axis length in the y direction.
        - shapez: The semi-axis length in the z direction.
        - contact_points: The positions of the contact points in the global frame.
        - centers: The centers of the ellipsoids in the global frame.
        - orientations: The orientation matrices of the ellipsoids.

        Returns:
        - curvatures: The mean curvature at each contact point.
        """
        # Translate the contact points to the local frame of the ellipsoids
        local_contact_points = np.einsum('ikj,ik->ij', orientations, contact_points - centers)

        # Coordinates in the local frame
        x = local_contact_points[:, 0]
        y = local_contact_points[:, 1]
        z = local_contact_points[:, 2]

        # Semi-axes lengths
        a = shapex
        b = shapez
        c = shapez

        gradient = np.zeros((len(x), 3))
        gradient[:, 0] = 2 * x / a**2
        gradient[:, 1] = 2 * y / b**2
        gradient[:, 2] = 2 * z / c**2

        #hessian 
        hessian = np.zeros((len(x), 3, 3))
        hessian[:, 0, 0] = 2 / a**2
        hessian[:, 1, 1] = 2 / b**2
        hessian[:, 2, 2] = 2 / c**2

        num = np.einsum('ij,ij->i', gradient, np.einsum('ijk,ik->ij', hessian, gradient))
        num2 = np.einsum('ij,ij->i', gradient, gradient)*np.einsum('ijj->i', hessian)
        den = 2*np.einsum('ij,ij->i', gradient, gradient)**(3/2)
        # Mean curvature formula
        #H = (a * b * c) * ((x**2/a**4 + y**2/b**4 + z**2/c**4) / (x**2 + y**2 + z**2))
        H = (num-num2)/den
        return np.absolute(H)
    
    def compute_volume_equivalent_curvature(self, shapex, shapez):
        """
        Compute the equivalent radius of the volume of the ellipsoid.
        """
        radius = (shapex**2 * shapez)**(1/3)
        curvature = 1/radius
        return curvature

    def compute_normal_hertzian_force(self, radius_curvature_avg, overlap):
        """
        Compute the normal force for a Hertzian contact model.

        Parameters:
        - radius_curvature_avg: The average radius of curvature at the contact point.
        - overlap: The overlap distance at the contact point.

        Returns:
        - normal_force: The normal force for the contact.
        """
        Young_modulus = 5.0e6
        Poisson_ratio = 0.3
        effective_Young_modulus = Young_modulus / (1 - Poisson_ratio**2)/2
        normal_force = 4/3 * effective_Young_modulus * np.sqrt(radius_curvature_avg* overlap)*overlap #always positive

        return normal_force

    def compute_tangential_hertzian_stiffness(self, radius_curvature_avg, overlap):
        """
        Compute the tanegntial force for a Hertzian contact model.

        Parameters:
        - radius_curvature_avg: The average radius of curvature at the contact point.
        - overlap: The overlap distance at the contact point.
        - shear_amount: The amount of shear at the contact point.
        - friction_coefficient: The coefficient of friction.

        Returns:
        - normal_force: The normal force for the contact.
        """
        Young_modulus = 5.0e6
        Poisson_ratio = 0.3
        effective_shear_modulus = Young_modulus / (4*(2 - Poisson_ratio)*(1+Poisson_ratio))
        shear_stiffness = 8*effective_shear_modulus*np.sqrt(radius_curvature_avg*overlap)
        return shear_stiffness

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

    def compute_relative_velocity(self, vel1, vel2, omega1, omega2, cp1, cp2, centers1, centers2,
                                vel_correction1, vel_correction2, angular_correction1, angular_correction2): 
        """
        Compute the relative velocity at the contact points using vectorized operations.
        """
        r1_to_contact = cp1 - centers1
        r2_to_contact = cp2 - centers2

        #compute half step corrections 
        vel1 += vel_correction1
        vel2 += vel_correction2

        omega1 += angular_correction1
        omega2 += angular_correction2

        v1 = vel1 + np.cross(omega1, r1_to_contact)
        v2 = vel2 + np.cross(omega2, r2_to_contact)

        # Compute relative velocities
        return v1 - v2
   
    def compute_stress(self, centers1, centers2, forces, volume_box):
        """
        Compute the stress with the virial approximation
        """

           # Relative positions and velocities
        relative_positions = centers2 - centers1

        # Kinetic term: sum of outer products of velocity components
        # for the kinetic term the average of the local velocity would be needed
        # Potential term: sum of outer products of force and position
        potential_term = np.einsum('ij,ik->jk', relative_positions, forces)

        # Total virial stress tensor
        stress_tensor = potential_term / volume_box
        return stress_tensor
   
    def bin_forces_by_xy_angle_contact_point_global(self, contact_points, ellipsoid_centers, normal_forces, tangential_forces, num_bins=144):
        """
        Bin the normal and tangential forces based on angles in the XY plane using vectorized operations.
        """
        center_to_contact_vector_global = contact_points[:, :2] - ellipsoid_centers[:, :2]
        angles_rad = np.arctan2(center_to_contact_vector_global[:, 1], center_to_contact_vector_global[:, 0])
        angles_deg = np.degrees(angles_rad)
        angles_deg = np.mod(angles_deg + 360, 360)

        normal_hist, bin_counts = self.bin_1d_histogram(angles_deg, normal_forces, 360, num_bins)
        tangential_hist, _ = self.bin_1d_histogram(angles_deg, tangential_forces, 360, num_bins)

        return normal_hist, tangential_hist, bin_counts

    def bin_forces_by_ellipsoid_angle(self, contact_points, ellipsoid_centers, normal_forces, tangential_forces, orientations, num_bins=10):
        """
        Bin the normal and tangential forces based on angles with respect to the ellipsoid's principal axis using vectorized operations.
        """
        angles_deg = self.compute_ellipsoid_angle_local(contact_points, ellipsoid_centers, orientations)
        normal_hist, bin_counts = self.bin_1d_histogram(angles_deg, normal_forces, 90, num_bins)
        tangential_hist, _ = self.bin_1d_histogram(angles_deg, tangential_forces, 90, num_bins)

        return normal_hist, tangential_hist, bin_counts

    def bin_forces_by_ellipsoid_angle_mixed(self, contact_points, ellipsoid_centers, normal_forces, tangential_forces, orientations, num_bins_theta=10, num_bins_phi=10):
        """
        Bin the normal and tangential forces based on angles with respect to the ellipsoid's principal axis using vectorized operations.
        """
        center_to_contact_vector_global = contact_points - ellipsoid_centers

        center_to_contact_vector_local = np.einsum('ikj,ik->ij', orientations, center_to_contact_vector_global) # Rotate to local coordinates
        angles_rad = np.arccos(center_to_contact_vector_local[:, 2] / np.linalg.norm(center_to_contact_vector_local, axis=1))
        axis = orientations[:, 2, :]/np.linalg.norm(orientations[:, 2, :], axis=1)[:, np.newaxis]       
        angles_deg = np.degrees(angles_rad)

        angle_with_x = np.arccos(axis[:, 0])
        mask_sign = np.where(angle_with_x > np.pi/2, -1, 1)

        # project the x-axis on the plane perpendicular to the symmetry axis to obtain a reference x-axis in the plane
        x_global = np.tile(np.array([1, 0, 0]), (len(orientations), 1))
        x_ref = x_global - np.einsum('ij,ij->i', x_global, axis)[:, np.newaxis]*axis
        x_ref = mask_sign[:, np.newaxis] *x_ref/np.linalg.norm(x_ref, axis=1)[:, np.newaxis]

        # project global contact point vector onto plane perpendicular to symmetry axis
        contact_point_plane = center_to_contact_vector_global - np.einsum('ij,ij->i', center_to_contact_vector_global, axis)[:,np.newaxis]*axis
        # compute the angle between the projected contact point vector and the reference x-axis
        x_components = np.einsum('ij,ij->i', contact_point_plane, x_ref)  # Dot product with x_ref gives x-component
        y_components = np.cross(x_ref, contact_point_plane)[:, 2]  # Cross product gives z-component (the sine part)

        # Now compute the angle phi using np.arctan2
        phi = np.arctan2(y_components, x_components)

        # Convert the angle from [-pi, pi] to [0, 2*pi] range
        phi = np.mod(phi, 2 * np.pi)
        
        bins_theta = np.linspace(0, 180, num_bins_theta + 1)
        bins_phi = np.linspace(0, 2*np.pi, num_bins_phi + 1)

        theta_bin_indices = np.digitize(angles_deg, bins_theta) - 1
        phi_bin_indices = np.digitize(phi, bins_phi) - 1

        theta_bin_indices = np.clip(theta_bin_indices, 0, num_bins_theta - 1)
        phi_bin_indices = np.clip(phi_bin_indices, 0, num_bins_phi - 1)

        # Combine the theta and phi bin indices into a single index
        combined_bin_indices = theta_bin_indices * num_bins_phi + phi_bin_indices
        
         # Initialize the histograms
        num_bins_total = num_bins_theta * num_bins_phi

        normal_hist_flat = np.zeros(num_bins_total)
        tangential_hist_flat = np.zeros(num_bins_total)
        bin_counts_flat = np.zeros(num_bins_total)

        # Use np.add.at for vectorized binning
        np.add.at(normal_hist_flat, combined_bin_indices, np.linalg.norm(normal_forces, axis=1))
        np.add.at(tangential_hist_flat, combined_bin_indices, np.linalg.norm(tangential_forces, axis=1))
        np.add.at(bin_counts_flat, combined_bin_indices, 1)

        # Reshape the flattened histograms back to 2D
        # normal_hist = normal_hist_flat.reshape((num_bins_theta, num_bins_phi))
        # tangential_hist = tangential_hist_flat.reshape((num_bins_theta, num_bins_phi))
        # bin_counts = bin_counts_flat.reshape((num_bins_theta, num_bins_phi))
        return normal_hist_flat, tangential_hist_flat, bin_counts_flat

    def accumulate_force_histogram(self, forces, num_bins=144):
        """
        Accumulate the magnitudes of forces into bins based on their angles.

        Parameters:
        - forces: An array of 2D vectors (numpy array) of shape (n, 2).
        - num_bins: The number of bins to divide the 360-degree circle into.

        Returns:
        - hist: A histogram of force magnitudes accumulated in bins.
        """
        # project the forces onto the XY plane
        forces = forces[:, :2]

        # Calculate the angles of the forces in degrees
        angles_rad = np.arctan2(forces[:, 1], forces[:, 0])
        angles_deg = np.degrees(angles_rad) % 360

        # Bin the angles using the bin_1d_histogram function
        hist, bin_counts = self.bin_1d_histogram(angles_deg, forces, 360, num_bins)

        return hist, bin_counts
    
    def compute_and_bin_dissipation(self, contact_points, ellipsoid_centers_1, normals, 
                                    tangential_forces, normal_forces, orientations_1, 
                                    relative_velocities, normal_hertz, damping_coff,
                                    kt, shear, dt, num_bins=10):
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

        damping_force = np.linalg.norm(normal_forces - normal_hertz[:, np.newaxis]*normals, axis=1)
        normal_vr = np.einsum('ij,ij->i', relative_velocities, normals) # modulus of normal velocity
        #damping_force = damping_coff * normal_vr
        #normal_forces_global = (damping_force+normal_hertz)[:, np.newaxis]*normals
        tangential_vr = relative_velocities - normal_vr[:, np.newaxis]*normals
        shear_elastic_force = kt[:, np.newaxis] * shear
        #compare shear elastic with tangential force
        elastic_force_magnitude = np.linalg.norm(shear_elastic_force, axis=1)
        #print(f"max elastic force: {elastic_force_magnitude.max()}")
        coulomb_limit = self.cof * np.linalg.norm(normal_forces, axis=1)
        #numerator = (elastic_force_magnitude_last**2-elastic_force_magnitude**2)

        sliding_limit = np.where(np.linalg.norm(tangential_forces, axis=1)>= coulomb_limit*0.99)
        # beyond_limit = np.sum(np.where(np.linalg.norm(tangential_forces, axis=1)>coulomb_limit))
        # print(f"Beyond limit: {beyond_limit}")

        # compute the mean tangential force for sliding contacts
        mean_tangential_force = np.mean(np.linalg.norm(tangential_forces[sliding_limit], axis=1))

        friction_dissipation = np.zeros_like(elastic_force_magnitude)
        friction_dissipation[sliding_limit] = 0.5*np.abs(np.einsum('ij,ij->i', tangential_forces[sliding_limit], tangential_vr[sliding_limit]))     
        #friction_dissipation[sliding_limit] = 0.5*numerator[sliding_limit]/(kt[sliding_limit]*dt)
        damping_dissipation = 0.5*np.abs((-normal_vr)*damping_force)
        
        power_dissipation_normal = damping_dissipation
        power_dissipation_tangential = friction_dissipation

        # Binning process 
        angles_deg = self.compute_ellipsoid_angle_local(contact_points, ellipsoid_centers_1, orientations_1)
        normal_dissipation_hist, bin_counts = self.bin_1d_histogram(angles_deg, power_dissipation_normal, 90, num_bins)
        
        tangential_dissipation_hist, _ = self.bin_1d_histogram(angles_deg, power_dissipation_tangential, 90, num_bins)

        self.dissipation_particles = self.compute_particle_dissipation(power_dissipation_normal, power_dissipation_tangential)
        self.percent_sliding = np.sum(np.linalg.norm(tangential_forces, axis=1) >= coulomb_limit * 0.99)/len(kt)
       
        return normal_dissipation_hist, tangential_dissipation_hist, bin_counts

    def compute_fabric_tensor(self, global_normals):
        """
        Compute the fabric tensor using the global normal vectors.
        """
        fabric_tensor = np.einsum('ij,ik->jk', global_normals, global_normals)
        return fabric_tensor/len(global_normals)

    def compute_particle_stress(self, centers1, centers2, forces, particle_mass, density =1000):
        """
        Compute the particle level stress for every particle by considering all contacts and
        their respective contact points
        """
        particle_stress = np.zeros((self.n_central_atoms, 3, 3))
        particle_volume = particle_mass/density
        # Vectorized computation of contact vectors for both particles
        contact_vector1 = self.point - centers1  # Vector from center of particle 1 to contact point
        contact_vector2 = self.point - centers2  # Vector from center of particle 2 to contact point

        # Vectorized computation of virial stress tensor using outer product
        np.add.at(particle_stress, self.id1, np.einsum('ij,ik->ijk', contact_vector1, forces))
        np.add.at(particle_stress, self.id2, np.einsum('ij,ik->ijk', contact_vector2, -forces))

        particle_stress /= particle_volume[:, np.newaxis, np.newaxis]

        # N1 = particle_stress[:, 0, 0]-particle_stress[:, 1, 1]
        # N2 = particle_stress[:, 1, 1]-particle_stress[:, 2, 2]

        # pressure = particle_stress[:, 1, 1]

        # N1 = np.where(~np.isclose(pressure, 0, rtol = 1e-2), N1/pressure, 0)    
        # N2 = np.where(~np.isclose(pressure, 0, rtol = 1e-2), N2/pressure, 0)

        N1 = 0
        N2 = 0
        
        return particle_stress, N1, N2

    def plot_force_chains(self, centers1, centers2, step):
        # Step 1: Calculate the magnitude of the forces
        magnitudes = np.linalg.norm(self.force, axis=1)
        avg_magnitude = np.mean(magnitudes)
        
        # Step 2: Filter the forces whose magnitude is above the average
        above_avg_indices = magnitudes > avg_magnitude
        forces_above_avg = self.force[above_avg_indices]
        points_above_avg = self.point[above_avg_indices]
        centers1_above_avg = centers1[above_avg_indices]
        centers2_above_avg = centers2[above_avg_indices]
        magnitudes_above_avg = magnitudes[above_avg_indices]
        
        # Step 3: Normalize magnitudes for line thickness
        min_thickness = 0.001
        max_thickness = 3
        thickness = np.interp(magnitudes_above_avg, (magnitudes_above_avg.min(), magnitudes_above_avg.max()), (min_thickness, max_thickness))
        
        # Step 4: Calculate the angle between the force vector and Y-axis (in degrees)
        shear_axis = 1/np.sqrt(2)*np.array([-1, 1, 0])
        unit_forces = forces_above_avg / magnitudes_above_avg[:, np.newaxis]  # Normalize forces to unit vectors
        dot_products = np.dot(unit_forces, shear_axis)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0)) * (180.0 / np.pi)  # Convert radians to degrees
        
        angles = np.where(angles > 90, 180 - angles, angles)  # Map angles to [0, 90] range
        # Step 5: Map angles to a colormap (0 degrees -> min, 90 degrees -> max)
        colormap = cm.get_cmap('summer')
        colors = colormap(angles/90)  # Normalize angles between 0 and 90
        
        fig = plt.figure(figsize=(10, 11))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(centers1_above_avg)):
            # Plot line from center1 to contact point
            ax.plot([centers1_above_avg[i][0], points_above_avg[i][0]],
                    [centers1_above_avg[i][2], points_above_avg[i][2]],
                    [centers1_above_avg[i][1], points_above_avg[i][1]],
                    color=colors[i], linewidth=thickness[i])
            
            # Plot line from center2 to contact point
            ax.plot([centers2_above_avg[i][0], points_above_avg[i][0]],
                    [centers2_above_avg[i][2], points_above_avg[i][2]],
                    [centers2_above_avg[i][1], points_above_avg[i][1]],
                    color=colors[i], linewidth=thickness[i])
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        
        # Hide ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Set labels
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Z', fontsize=20)
        ax.set_zlabel('Y', fontsize=20)
        
        # Show color bar for angle mapping
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=-45, vmax=45))
        sm.set_array([])
        bar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.8, aspect=10)
        bar.set_label('Angle with Compression direction (degrees)', fontsize=20)
        azimuth_angle = 360/50*step
        ax.view_init(elev=0, azim=90)
        # Reduce padding around the plot (tight layout)
        plt.tight_layout()
        plt.show()
        plt.savefig('force_chains/ap' + str(self.ap) + '_cof_' + str(self.cof) + '_I_' + str(self.I) + '_step_' + str(step) + '.png')
        plt.close()

    def check_plot_global_forces(self, forces):
        """ Histogram of forces by angle wrt x-axis"""

        # get forces_directions
        directions = forces / np.linalg.norm(forces, axis=1)[:, np.newaxis]  # Normalize forces to unit vectors

        print(directions[:,0]>1)

        forces_angle_with_x = np.arccos(directions[:, 0])  # Angle between force vector and X-axis
        #create the histogram


        plt.figure(figsize=(10, 6))
        plt.hist(forces_angle_with_x, bins=100, color='blue', alpha=0.7, density=True)
        plt.xlabel('Angle with X-axis (radians)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Histogram of Forces by Angle with X-axis', fontsize=16)
        plt.show()

    def compute_contact_without_rattlers(self):
        """ Find rattlers as particles with"""

        contacts_particles = np.zeros(self.n_central_atoms)

        np.add.at(contacts_particles, self.id1, 1)
        np.add.at(contacts_particles, self.id2, 1)

        if self.cof == 0.0:
            min_contacts = 4
        else:
            min_contacts = 3

        rattlers = np.where(contacts_particles < min_contacts)[0]
        return rattlers

    def force_single_step(self, step):
        """Processing on the data for one single step
        Calling the other methods for the processing"""
        data = np.loadtxt(self.directory+self.file_list[step], skiprows=10)
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
                data_sampled[count, 1:4] = rows_plus[j, 37:22] #contact point
                data_sampled[count, 4:7] = rows_plus[j, 9:12] #contact force
                data_sampled[count, 7:10] = rows_plus[j , 12:15] #contact tangential force
                count += 1  

            for j in range(rows_minus.shape[0]):
                data_sampled[count, 0] = i
                data_sampled[count, 1:4] = rows_minus[j, 37:22] #contact point
                data_sampled[count, 4:7] -= rows_minus[j, 9:12]
                data_sampled[count, 7:10] -= rows_minus[j, 12:15]
                #store the data   
                count += 1
        #remove the rows with all zeros which remained empty
        data_sampled = data_sampled[~np.all(data_sampled == 0, axis=1)]
        return data_sampled
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
        ax.view_init(-90, 90)

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