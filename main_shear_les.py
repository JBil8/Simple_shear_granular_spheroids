import sys
import numpy as np
import pickle
import vtk
import argparse
import matplotlib.pyplot as plt
#from scipy.spatial.transform import Rotation as R
import os
import re
import multiprocessing
import time as tm
from ProcessorVtk import ProcessorVtk   
from ProcessorDump import ProcessorDump
from CombinedProcessor import CombinedProcessor
from DataPlotter import DataPlotter
from ReaderVtk import ReaderVtk
from ReaderDump import ReaderDump
from DataExporter import DataExporter
from ProcessorCsv import ProcessorCsv
from ProcessorDat import ProcessorDat

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process granular simulation.')
    parser.add_argument('-c', '--cof', type=float, help='coefficient of friction particles-particles')
    parser.add_argument('-a', '--ap', type=float, help='aspect ratio')
    parser.add_argument('-v', '--value', type=float, help='packing fraction or Inertial number depensing on the type of simulation')
    parser.add_argument('-p', '--postprocess', action='store_false', help='whether to postprocess the data or simply import the pkl file, default is True', default=True)
    parser.add_argument('-cw', '--cof_wall', action='store_false', help='coefficient of friction particles-walls')
    parser.add_argument('-s', '--pressure', type = int, help='pressure')

    args = parser.parse_args()

    #parsing command line arguments
    cof = args.cof
    ap = args.ap
    param = args.value
    full_postprocess = args.postprocess
    pressure = args.pressure
    num_processes = 32

    if full_postprocess == True:

        global_path = "/scratch/bilotto/simulations_simple_shear_updated_stress/"
        plt.ioff()
        #initialize the vtk reader
        data_read = ReaderVtk(cof, ap, I=param, pressure = pressure)
        data_read.read_data(global_path, 'simple_shear_')
        data_read.no_wall_atoms()
        #data_read.get_initial_velocities()
        df_csv = data_read.read_csv_file()
        df_dat = data_read.read_dat_file()
        particles_volume = data_read.get_particles_volume()
        #intialize the dump reader
        data_dump = ReaderDump(cof, ap, I=param, pressure = pressure)
        data_dump.read_data(global_path, 'simple_shear_contact_data_')
        to_process_vtk = ProcessorVtk(data_read)
        to_process_dump = ProcessorDump(data_dump, data_read.n_wall_atoms, data_read.n_central_atoms)  
        combined_processor = CombinedProcessor(to_process_vtk, to_process_dump)

        shear_one_index = int(to_process_vtk.n_sim/11)

        with multiprocessing.Pool(num_processes) as pool:
            print("Started multiprocessing")
            results = pool.map(combined_processor.process_single_step,
                                [step for step in range(shear_one_index, combined_processor.n_sim)])


            #Extract the averages from the results
            averages = {}
            
            for key in results[0].keys():
                if key == 'trackedGrainsOrientation' or key == 'trackedGrainsPosition':
                # Stack the arrays along a new axis to preserve the structure
                    averages[key] = np.stack([result[key] for result in results], axis=0)
                else:
                    averages[key] = np.array([result[key] for result in results])
                    averages[key] = np.mean(averages[key], axis=0)

        print(averages['contact_angle'])

        csvProcessor = ProcessorCsv(df_csv)
        csvProcessor.exclude_initial_strain_cycle()
        datProcessor = ProcessorDat(df_dat)
        avgdat = datProcessor.compute_averages()
        avgdat = datProcessor.compute_max_vx_diff(avgdat)
        avgcsv = csvProcessor.get_averages()
        averages = {**averages, **avgcsv, **avgdat} #merge the two dictionaries

        #export the data with pickle
        exporter = DataExporter(ap, cof,I=param)
        exporter.export_with_pickle(averages)

        plotter = DataPlotter(ap, cof,value=param)
        plotter.plot_time_variation(averages, df_csv) 
        plotter.plot_averages_with_std(averages)
        #force distrubution
        



        # with multiprocessing.Pool(num_processes) as pool:
        #     print("Started multiprocessing")
        #     results = pool.map(to_process_dump.force_single_step,
        #                         [step for step in range(to_process_dump.n_sim)])
        # # stack all the forces and force tangential to compute the distribution
        # force_normal_stack = np.concatenate([result['force_normal'] for result in results])
        # force_tangential_stack = np.concatenate([result['force_tangential'] for result in results])

        # force_normal_distribution = np.histogram(force_normal_stack, bins=100, density=True)
        # force_tangential_distribution = np.histogram(force_tangential_stack, bins=100, density=True)

        # plotter.plot_force_distribution(force_normal_distribution, force_tangential_distribution)

        # #export force distribution data
        # exporter.export_force_distribution(force_normal_distribution, force_tangential_distribution)

        #analysis of the tracked grains contact data

        #plot the ellipsoids in 3d
        
        #plotter.plot_ellipsoids(0, averages)

        plt.ion()

    else:
        #import the data with pickle
        importer = DataExporter(ap, cof, simulation_type ,param)
        averages, averages_dump = importer.import_with_pickle()

        plotter = DataPlotter(ap, cof, simulation_type ,param)
        #plotter.plot_data(averages, averages_dump)
        #plotter.plot_eulerian_velocities(averages)

        print(averages_dump['trackedGrainsContactData'][0][:2, :])
        # plot ellipsois in 3d
        plotter.plot_ellipsoids(500, averages, averages_dump)