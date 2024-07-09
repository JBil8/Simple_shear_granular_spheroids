import multiprocessing
import numpy as np
import vtk


class CombinedProcessor:
    def __init__(self, vtk_processor, dump_processor):
        self.vtk_processor = vtk_processor
        self.dump_processor = dump_processor
        self.n_sim = min(vtk_processor.n_sim, dump_processor.n_sim)  # Assuming both have the same number of steps

    def process_single_step(self, step):
        # Process the VTK data for the given step
        vtk_result = self.vtk_processor.process_single_step(step)
        coor, orientation, shapex, shapez = self.vtk_processor.pass_coord_orientation_shape()
        # Process the dump data for the given step
        dump_result = self.dump_processor.process_single_step(step, coor, orientation, shapex, shapez)
        # Combine the results as needed
        combined_result = self.combine_results(vtk_result, dump_result)
        
        return combined_result

    def combine_results(self, vtk_result, dump_result):
        # Combine the results of the two dictionaries

        total_dictionary = {**vtk_result, **dump_result}
        return total_dictionary