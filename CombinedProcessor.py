import multiprocessing
import numpy as np
import vtk


class CombinedProcessor:
    def __init__(self, vtk_processor, dump_processor, file_list_box= None, shear_rate=None, dt=None):
        self.vtk_processor = vtk_processor
        self.dump_processor = dump_processor
        self.n_sim = min(vtk_processor.n_sim, dump_processor.n_sim)  # Assuming both have the same number of steps
        if file_list_box is not None:
            self.file_list_box = file_list_box
            self.directory = vtk_processor.directory
        if shear_rate is not None:
            self.shear_rate = shear_rate
        if dt is not None:
            self.dt = dt

    def process_single_step(self, step):
        # Process the VTK data for the given step
        vtk_result = self.vtk_processor.process_single_step(step)
        coor, orientation, shapex, shapez, vel, omega = self.vtk_processor.pass_particle_data()
        # Get the box dimensions
        box_lengths = self.process_box_data(step)
        
        # Process the dump data for the given step
        dump_result = self.dump_processor.process_single_step(step, coor, orientation, shapex, shapez, vel, omega, box_lengths, self.shear_rate, self.dt)
        # Combine the results as needed
        combined_result = self.combine_results(vtk_result, dump_result, box_lengths)
        
        return combined_result

    def combine_results(self, vtk_result, dump_result, box_lengths):
        # Combine the results of the two dictionaries
        vtk_result["box_x_length"] = box_lengths[0]
        vtk_result["box_y_length"] = box_lengths[1]
        vtk_result["box_z_length"] = box_lengths[2]
        total_dictionary = {**vtk_result, **dump_result}
        return total_dictionary
    
    def process_box_data(self, step):
        # Read the box data for the given step
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(self.directory+ self.file_list_box[step])
        reader.Update()
        box_points = np.array([reader.GetOutput().GetPoints().GetPoint(i) for i in range (reader.GetOutput().GetNumberOfPoints())])
        # Calculate the lengths of each vector (box dimensions)
        x_length = np.linalg.norm(box_points[1][0] - box_points[0][0]) # along x
        y_length = np.linalg.norm(box_points[3][1] - box_points[0][1]) # along y
        z_length = np.linalg.norm(box_points[4][2] - box_points[0][2]) # along z
        delta_xy = box_points[3][0] - box_points[0][0] # delta x in the tilted box

        #print(f"Box dimensions: x = {x_length}, y = {y_length}, z = {z_length}, delta_xy = {delta_xy}")   
        return np.array([x_length, y_length, z_length, delta_xy])