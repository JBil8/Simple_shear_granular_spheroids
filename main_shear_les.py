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


def compute_histogram_sum(results, key, num_bins):
    """Compute the sum of histograms for a given key."""
    hist_sum = np.zeros(num_bins)
    for result in results:
        hist_sum += result[key]
    return hist_sum

def compute_histogram_avg(hist_sum, num_results):
    """Compute the average of histograms."""
    return hist_sum / num_results

def compute_pdf(hist_avg):
    """Compute the probability density function (PDF) of a histogram."""
    return hist_avg / np.sum(hist_avg)

def process_results(results):
    """Process the results to extract and average histograms."""
    num_results = len(results)
    
    # Define the number of bins for each histogram type
    bins_config = {
        'hist_thetax': 36,
        'hist_thetaz': 36,
        'global_normal_force_hist': 36,
        'global_tangential_force_hist': 36,
        'local_normal_force_hist_cp': 18,
        'local_tangential_force_hist_cp': 18,
        'global_normal_force_hist_cp': 36,
        'global_tangential_force_hist_cp': 36,
    }
    
    # Initialize sums for each histogram type
    histogram_sums = {key: np.zeros(bins) for key, bins in bins_config.items()}

    # Extract averages and sum histograms
    averages = {}
    for key in results[0].keys():
        if key in histogram_sums:
            histogram_sums[key] = compute_histogram_sum(results, key, bins_config[key])
        elif key in ['trackedGrainsOrientation', 'trackedGrainsPosition']:
            averages[key] = np.stack([result[key] for result in results], axis=0)
        else:
            averages[key] = np.mean([result[key] for result in results], axis=0)

    # Calculate averages and PDFs for histograms
    histograms_avg = {key: compute_histogram_avg(hist_sum, num_results) for key, hist_sum in histogram_sums.items()}
    pdfs = {key: compute_pdf(hist_avg) for key, hist_avg in histograms_avg.items()}

    return averages, histograms_avg, pdfs

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
    num_processes = 8

    if full_postprocess == True:

        global_path = "/home/jacopo/Documents/PhD_research/Liggghts_simulations/cluster_simulations/"
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

        #shear_one_index = int(to_process_vtk.n_sim/11)
        shear_one_index = 1000

        with multiprocessing.Pool(num_processes) as pool:
            print("Started multiprocessing")
            results = pool.map(combined_processor.process_single_step,
                                [step for step in range(shear_one_index, combined_processor.n_sim)])

        averages, histograms_avg, pdfs = process_results(results)

        # Access the specific PDFs or histograms as needed
        pdf_thetax = pdfs['hist_thetax']
        pdf_thetaz = pdfs['hist_thetaz']
        hist_global_normal_avg = histograms_avg['global_normal_force_hist']
        hist_global_tangential_avg = histograms_avg['global_tangential_force_hist']
        hist_local_normal_avg = histograms_avg['local_normal_force_hist_cp']
        hist_local_tangential_avg = histograms_avg['local_tangential_force_hist_cp']
        hist_global_normal_cp_avg = histograms_avg['global_normal_force_hist_cp']
        hist_global_tangential_cp_avg = histograms_avg['global_tangential_force_hist_cp']

        # Define bins for plotting or further analysis
        bins_orientation = np.linspace(-90, 90, 37)
        bins_global = np.linspace(-180, 180, 37)
        bins_local = np.linspace(0, 90, 19)

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
        # plotter.plot_time_variation(averages, df_csv) 
        # plotter.plot_averages_with_std(averages)
        #plotter.plot_histogram(bins_orientation, pdf_thetax, "$\\theta_x")
        #plotter.plot_histogram(bins_orientation, pdf_thetaz, "$\\theta_z")
        plotter.plot_polar_histogram(bins_global, hist_global_normal_avg, "Global force normal", periodicity=False)
        plotter.plot_polar_histogram(bins_local, hist_local_normal_avg, "Local force normal contact point", periodicity=True)
        plotter.plot_polar_histogram(bins_global, hist_global_tangential_avg, "Global force tangential", periodicity=False)
        plotter.plot_polar_histogram(bins_local, hist_local_tangential_avg, "Local force tangential contact point", periodicity=True)
        plotter.plot_polar_histogram(bins_global, hist_global_normal_cp_avg, "Global force normal contact point", periodicity=False)
        plotter.plot_polar_histogram(bins_global, hist_global_tangential_cp_avg, "Global force tangential contact point", periodicity=False)

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