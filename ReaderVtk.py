import os 
import re
import numpy as np
import vtk 
from DataReader import DataReader
import math 

class ReaderVtk(DataReader):
    def __init__(self, cof, ap, parameter=None, pressure = None, value=None, muw=None, vwall=None, fraction=None, phi = None, I = None):
        super().__init__(cof, ap, parameter, value, pressure, muw, vwall, fraction, phi, I)
        self.n_wall_atoms = None
        self.n_central_atoms = None

    #to call only after the data have been read
    def set_number_wall_atoms(self):
        '''
        For simulations with walls made of grains
        '''
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.directory + self.file_list[0])
        reader.Update()
        polydata = reader.GetOutput()
        coor = np.array(polydata.GetPoints().GetData())
        _, counts = np.unique(coor[:,1], axis=0, return_counts=True)
        self.n_wall_atoms = sum(counts[counts>4])

    def no_wall_atoms(self):
        '''
        For simulations with walls made of stl meshes
        '''
        self.n_wall_atoms = 0
        self.n_central_atoms = self.n_all_atoms

    def get_number_of_atoms(self):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.directory + self.file_list[0])
        reader.Update()
        polydata = reader.GetOutput()
        self.n_all_atoms = polydata.GetNumberOfPoints()

    def get_number_central_atoms(self):
        self.n_central_atoms = self.n_all_atoms - self.n_wall_atoms

    def get_initial_velocities_and_orientations(self):
        reader = vtk.vtkPolyDataReader()
        if self.I >= 0.01:
            index = 400
        else:
            index = 600
        reader.SetFileName(self.directory + self.file_list[index])
        reader.Update()
        polydata = reader.GetOutput()
        polydatapoints = polydata.GetPointData()
        ids = np.array(polydata.GetPointData().GetArray(0))
        self.sorted_idxs = np.argsort(ids)
        self.v0 = np.array(polydata.GetPointData().GetArray(3))[self.sorted_idxs, :][self.n_wall_atoms:, :]
        self.y0 = np.array(polydata.GetPoints().GetData())[self.sorted_idxs, :][self.n_wall_atoms:, 1]
        self.v_shearing = np.array(polydatapoints.GetArray("v"))[self.sorted_idxs, :][self.n_wall_atoms-1, 0]
        orientations = np.array(polydatapoints.GetArray("TENSOR"))[self.sorted_idxs, :].reshape(self.n_central_atoms,3,3)
        starting_vector = np.array([0, 0, 1])
        grain_vectors = np.einsum('ijk,k->ij', orientations, starting_vector)    
        # Calculate flow angles
        self.theta0 = np.arctan2(grain_vectors[:, 1], grain_vectors[:, 0])
        
    def get_particles_volume(self):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.directory + self.file_list[0])
        reader.Update()
        polydata = reader.GetOutput()
        polydatapoints = polydata.GetPointData()
        #get shapex array for all particles
        shapex = np.array(polydatapoints.GetArray("shapex"))[self.n_wall_atoms:]
        shapey = np.array(polydatapoints.GetArray("shapey"))[self.n_wall_atoms:]
        shapez = np.array(polydatapoints.GetArray("shapez"))[self.n_wall_atoms:]
        volume = 4*np.pi/3 * np.sum(shapex * shapey * shapez)
        return volume
    
    def get_xz_surface(self, radius=0.00666):
        """
        Value of radius is the average one.
        So far needs to be know from simulation
        Maybe find alternative way
        """
        x_length = 61*radius
        z_length =8*radius*self.ap
        return x_length*z_length

    def filter_relevant_files(self, prefix='shear_ellipsoids_', box=False):
        file_list = [filename for filename in self.file_list if filename.startswith(prefix) and filename[len(prefix):len(prefix)+1].isdigit() and filename.endswith('.vtk')]
        if box:
            prefix = prefix + 'boundingBox_'
            self.file_list_box = [filename for filename in self.file_list if filename.startswith(prefix) and filename[len(prefix):len(prefix)+1].isdigit() and filename.endswith('.vtk')]
        self.file_list = file_list

    def get_box_dimensions(self):
        reader = vtk.vtkRectilinearGridReader()
        name_first_file = re.sub(r'(\d+)', r'boundingBox_\1', self.file_list[0])
        reader.SetFileName(self.directory + name_first_file)
        reader.Update()
        bounds = reader.GetOutput().GetBounds()
        self.box_x = bounds[1] - bounds[0]
        self.box_y = bounds[3] - bounds[2]
        self.box_z = bounds[5] - bounds[4]
        self.lower_bound_y = bounds[2]

    def make_2D_grid(self, nx_divisions = 40):
        nx_divisions = int(nx_divisions)
        # ny_divisions = math.ceil(nx_divisions*self.box_y/self.box_x) # square cells
        ny_divisions = 6

        #distance of each cell in x and y
        dx = self.box_x/nx_divisions
        dy = self.box_y/ny_divisions
        return ny_divisions, dx, dy

    def read_data(self, global_path, prefix, box=False):

        self.prepare_data(global_path)
        self.filter_relevant_files(prefix, box)
        self.sort_files_by_time(box)
        self.get_number_of_time_steps()
        self.get_number_of_atoms()