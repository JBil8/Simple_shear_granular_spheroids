import multiprocessing
import numpy as np
from DataProcessor import DataProcessor

class ProcessorDump(DataProcessor):
    def __init__(self, data, n_wall_atoms, n_central_atoms):
        super().__init__(data)
        self.n_sim = self.data_reader.n_sim
        self.directory = self.data_reader.directory
        self.file_list = self.data_reader.file_list
        self.n_wall_atoms = n_wall_atoms
        self.n_central_atoms = n_central_atoms
        self.force = None
        self.point = None
        self.force_tangential = None
        self.shear = None
        self.area = None
        self.delta = None

    def process_data(self, num_processes):
        self.num_processes = num_processes
        with multiprocessing.Pool(self.num_processes) as pool:
            print("Started multiprocessing")
            results = pool.map(self.process_single_step, [step for step in range(self.n_sim)])
        averages = np.array(results)
        return averages
    
    def process_single_step(self, step, coor, orientation, shapex, shapez):
        data = np.loadtxt(self.directory+self.file_list[step], skiprows=9)
        self.assign_data(data)
        
        # Extract matching contact data for the central particles
        centers1, centers2, orientations1, orientations2, shapex1, shapex2, shapez1, shapez2 = self.match_contact_data_with_particles(coor, orientation, shapex, shapez)

        # Process contacts for each pair of ellipsoids
        normal_hist_cont_point_global1, tangential_hist_cont_point_global1, counts_cont_point_global1, normal_hist_cont_point_local1, tangential_hist_cont_point_local1, counts_cont_point_local1, normal_hist_global1, normal_count_global1, tangential_hist_global1, tangential_count_global1 = self.process_contacts(
             centers1, centers2, orientations1, shapex1, shapez1, self.force)

        normal_hist_cont_point_global2, tangential_hist_cont_point_global2, counts_cont_point_global2, normal_hist_cont_point_local2, tangential_hist_cont_point_local2, counts_cont_point_local2, normal_hist_global2, normal_count_global2, tangential_hist_global2, tangential_count_global2 = self.process_contacts(
                centers2, centers1, orientations2, shapex2, shapez2, -self.force)

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

        # # Normalize the histograms by their bin counts
        # normal_hist_cont_point_global = np.divide(normal_hist_cont_point_global, counts_cont_point_global, where=counts_cont_point_global != 0)
        # tangential_hist_cont_point_global = np.divide(tangential_hist_cont_point_global, counts_cont_point_global, where=counts_cont_point_global != 0)
        # normal_hist_cont_point_local = np.divide(normal_hist_cont_point_local, counts_cont_point_local, where=counts_cont_point_local != 0)
        # tangential_hist_cont_point_local = np.divide(tangential_hist_cont_point_local, counts_cont_point_local, where=counts_cont_point_local != 0)
        # normal_hist_global = np.divide(normal_hist_global, normal_count_global, where=normal_count_global != 0)
        # tangential_hist_global = np.divide(tangential_hist_global, tangential_count_global, where=tangential_count_global != 0)

        contact_number = 2 * len(self.id1) / self.n_central_atoms

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
            'Z': contact_number
        }
        return avg_dict
    
    def assign_data(self,data):
        self.id1 = data[:, 0].astype(int) - 1
        self.id2 = data[:, 1].astype(int) - 1
        #self.periodic = data[:, 2] #periodicity flag, needed later
        self.force = data[:, 3:6]
        self.point = data[:, 8:11]

    def match_contact_data_with_particles(self, coor, orientation, shapex, shapez):
        centers1 = coor[self.id1]
        centers2 = coor[self.id2]
        orientations1 = orientation[self.id1]
        orientations2 = orientation[self.id2]
        shapex1 = shapex[self.id1]
        shapex2 = shapex[self.id2]
        shapez1 = shapez[self.id1]
        shapez2 = shapez[self.id2]
        return centers1, centers2, orientations1, orientations2, shapex1, shapex2, shapez1, shapez2

    def process_contacts(self, centers1, centers2, orientations1, shapex1, shapez1, forces):
        """
        Compute and bin the forces for contacts between two sets of ellipsoids.
        """
        # Compute normal vectors for the first ellipsoid (corresponding to id1 or id2)
        global_normals, local_normals = self.compute_normals(centers1, self.point, orientations1, shapex1, shapez1)

        # Project forces onto normals and tangents using vectorized operations
        normal_projection = np.einsum('ij,ij->i', forces, global_normals)[:, np.newaxis]
        normal_forces_global = normal_projection * global_normals
        tangential_forces_global = forces - normal_forces_global

        # Bin forces using vectorized binning
        normal_hist_cont_point_global, tangential_hist_cont_point_global, counts_cont_point_global = self.bin_forces_by_xy_angle_contact_point_global(self.point, centers1, normal_forces_global, tangential_forces_global)
        normal_hist_cont_point_local, tangential_hist_cont_point_local, counts_cont_point_local = self.bin_forces_by_ellipsoid_angle(self.point, centers1, normal_forces_global, tangential_forces_global, orientations1)
        normal_hist_global, counts_normal_hist_global = self.accumulate_force_histogram(normal_forces_global)
        tangential_hist_global, counts_tangential_hist_global = self.accumulate_force_histogram(tangential_forces_global)

        return normal_hist_cont_point_global, tangential_hist_cont_point_global, counts_cont_point_global, normal_hist_cont_point_local, tangential_hist_cont_point_local, counts_cont_point_local, normal_hist_global, counts_normal_hist_global, tangential_hist_global, counts_tangential_hist_global

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