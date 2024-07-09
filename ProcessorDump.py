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
        #self.n_wall_atoms = self.data_reader.n_wall_atoms

    def process_data(self, num_processes):
        self.num_processes = num_processes
        with multiprocessing.Pool(self.num_processes) as pool:
            print("Started multiprocessing")
            results = pool.map(self.process_single_step,
                               [step for step in range(self.n_sim)])
        # Extract the averages from the results
        averages = np.array(results)
        return averages
    
    def process_single_step(self, step, coor, orientation, shapex, shapez):
        """Processing on the data for one single step
        Calling the other methods for the processing"""

        data = np.loadtxt(self.directory+self.file_list[step], skiprows=9)
        # #excluding contacts between wall particles
        # first_col_check = (data[:,6] > self.n_wall_atoms)
        # second_col_check = (data[:,7] > self.n_wall_atoms)
        # not_wall_contacts = np.logical_and(first_col_check, second_col_check)
        # data = data[not_wall_contacts]  
        self.id1 = data[:, 0].astype(int) #id of the first particle
        self.id2 = data[:, 1] #id of the second particle
        self.periodic = data[:, 2] #contact over boundary 0 or 1
        self.force = data[:, 3:6] #contact_force
        self.point = data[:, 8:11] #contact_point        
        self.offset1 = data[:,11] #offset along normal direction for the first particle
        self.offset2 = data[:,12] #offset along normal direction for the second particle
        self.shear = data[:, 13:16] #shear components of the contact force
        
        self.compute_normal_vector_at_contact_and_angle(coor, self.point, orientation, shapex, shapez)
        self.compute_normal_tangential_force(self.global_normals)

        self.compute_space_averages()
        avg_dict = {'force_normal': self.force_normal_space_average,
                    'force_tangential': self.force_tangential_space_average,
                    'contact_angle': self.contact_angles_space_average,
                    'Z': self.contact_number}
        return avg_dict

    def compute_normal_vector_at_contact_and_angle(self, ellipsoid_centers, contact_points, rotation_matrices, shapex, shapez):
        """
        Compute the contact angle on the ellipsoid surface.
        An angle of zero corresponds to contact on the head of the ellipsoid.
        An angle of 90 corresponds to contact on the side of the ellipsoid.
        """
        # Match each contact point with the corresponding particle's properties
        matched_centers = ellipsoid_centers[self.id1]
        matched_rotations = rotation_matrices[self.id1]
        matched_shapex = shapex[self.id1]
        matched_shapez = shapez[self.id1]

        # Translate the contact points to the ellipsoid's local coordinate system
        local_contact_points = np.einsum('ijk,ik->ij', matched_rotations.transpose(0, 2, 1), contact_points - matched_centers)

        # Compute the normal vectors in the local coordinate system
        a_squared = matched_shapex ** 2
        c_squared = matched_shapez ** 2
        local_normals = np.zeros_like(local_contact_points)
        local_normals[:, 0] = local_contact_points[:, 0] / a_squared
        local_normals[:, 1] = local_contact_points[:, 1] / a_squared
        local_normals[:, 2] = local_contact_points[:, 2] / c_squared

        # Normalize the local normal vectors
        local_normals /= np.linalg.norm(local_normals, axis=1)[:, np.newaxis]

        # Transform the normal vectors back to the global coordinate system
        self.global_normals = np.einsum('ijk,ik->ij', matched_rotations, local_normals)

        # Compute the contact angles in degrees
        self.contact_angles = np.arccos(np.abs(self.global_normals[:, 2])) * 180 / np.pi


    def compute_normal_tangential_force(self, global_normal):

        self.force_normal = np.sum(self.force * global_normal, axis=1)[:, np.newaxis] * global_normal
        self.force_tangential = self.force - self.force_normal


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
        self.contact_number = len(self.id1)/self.n_central_atoms*2

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