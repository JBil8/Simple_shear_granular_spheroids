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
    """Compute the sum of histograms for a given key in a vectorized manner."""
    # Stack all histograms into a 2D array
    histograms = np.array([result[key] for result in results])
    
    # Sum the histograms along the first axis
    hist_sum = np.sum(histograms, axis=0)
    
    return hist_sum

def compute_histogram_avg(hist_sum, num_results):
    """Compute the average histogram."""
    return np.divide(hist_sum, num_results, where=num_results != 0)

def compute_weighted_average_hist(hist_values, hist_counts):
    return np.divide(hist_values, hist_counts , where=hist_counts != 0)

def compute_pdf(hist_sum):
    """Compute the PDF for a histogram."""
    total_count = np.sum(hist_sum)
    return hist_sum / total_count if total_count != 0 else hist_sum

def process_results(results):
    """Process the results to extract and average histograms."""
    num_results = len(results)
    
    # Define the number of bins for each histogram type
    bins_config = {
        'hist_thetax': 72,
        'hist_thetaz': 72,
        'global_normal_force_hist': 72,
        'global_tangential_force_hist': 72,
        'local_normal_force_hist_cp': 36,
        'local_tangential_force_hist_cp': 36,
        'global_normal_force_hist_cp': 72,
        'global_tangential_force_hist_cp': 72,
        'contacts_hist_cont_point_global': 72,
        'contacts_hist_cont_point_local': 36,
        'contacts_hist_global_normal': 72,
        'contacts_hist_global_tangential': 72,
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
    
    # Calculate simple averages for hist_thetax and hist_thetaz
    histograms_avg = {key: compute_histogram_avg(hist_sum, num_results) 
                      for key, hist_sum in histogram_sums.items() 
                      if key in ['hist_thetax', 'hist_thetaz']}
    
    # Compute the weighted average histograms
    histograms_weighted_avg = {}
    histograms_weighted_avg['global_normal_force_hist'] = compute_weighted_average_hist(
        histogram_sums['global_normal_force_hist'], histogram_sums['contacts_hist_global_normal'])
    histograms_weighted_avg['global_tangential_force_hist'] = compute_weighted_average_hist(
        histogram_sums['global_tangential_force_hist'], histogram_sums['contacts_hist_global_tangential'])
    
    histograms_weighted_avg['local_normal_force_hist_cp'] = compute_weighted_average_hist(
        histogram_sums['local_normal_force_hist_cp'], histogram_sums['contacts_hist_cont_point_local'])
    histograms_weighted_avg['local_tangential_force_hist_cp'] = compute_weighted_average_hist(
        histogram_sums['local_tangential_force_hist_cp'], histogram_sums['contacts_hist_cont_point_local'])
    
    histograms_weighted_avg['global_normal_force_hist_cp'] = compute_weighted_average_hist(
        histogram_sums['global_normal_force_hist_cp'], histogram_sums['contacts_hist_cont_point_global'])
    histograms_weighted_avg['global_tangential_force_hist_cp'] = compute_weighted_average_hist(
        histogram_sums['global_tangential_force_hist_cp'], histogram_sums['contacts_hist_cont_point_global'])
    
    # Compute PDFs for contact distributions
    pdfs = {}
    pdfs['hist_thetax'] = compute_pdf(histogram_sums['hist_thetax'])
    pdfs['hist_thetaz'] = compute_pdf(histogram_sums['hist_thetaz'])
    pdfs['contacts_hist_global_normal'] = compute_pdf(histogram_sums['contacts_hist_global_normal'])
    pdfs['contacts_hist_global_tangential'] = compute_pdf(histogram_sums['contacts_hist_global_tangential'])
    pdfs['contacts_hist_cont_point_global'] = compute_pdf(histogram_sums['contacts_hist_cont_point_global'])
    pdfs['contacts_hist_cont_point_local'] = compute_pdf(histogram_sums['contacts_hist_cont_point_local'])

    
    return averages, histograms_avg, histograms_weighted_avg, pdfs

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

        shear_one_index = int(to_process_vtk.n_sim/11)
        #shear_one_index = 1000

        with multiprocessing.Pool(num_processes) as pool:
            print("Started multiprocessing")
            results = pool.map(combined_processor.process_single_step,
                                [step for step in range(shear_one_index, combined_processor.n_sim)])

        averages, histograms_avg, hist_weigh_avg, pdfs = process_results(results)

        # Access the specific PDFs or histograms as needed
        pdf_thetax = pdfs['hist_thetax']
        pdf_thetaz = pdfs['hist_thetaz']
        hist_global_normal_avg = hist_weigh_avg['global_normal_force_hist']
        hist_global_tangential_avg = hist_weigh_avg['global_tangential_force_hist']
        hist_local_normal_avg = hist_weigh_avg['local_normal_force_hist_cp']
        hist_local_tangential_avg = hist_weigh_avg['local_tangential_force_hist_cp']
        hist_global_normal_cp_avg = hist_weigh_avg['global_normal_force_hist_cp']
        hist_global_tangential_cp_avg = hist_weigh_avg['global_tangential_force_hist_cp']

        # Define bins for plotting or further analysis
        bins_orientation = np.linspace(-90, 90, 73)
        bins_global = np.linspace(-180, 180, 73)
        bins_local = np.linspace(0, 90, 37)

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
        plotter.plot_histogram(bins_orientation, pdf_thetax, "$\\theta_x",  label = '$\\theta_x [^\circ]$')
        plotter.plot_histogram(bins_orientation, pdf_thetaz, "$\\theta_z",  label = '$\\theta_z [^\circ]$')
        plotter.plot_polar_histogram(bins_global, hist_global_normal_avg, "Global force normal", symmetry=False)
        plotter.plot_polar_histogram(bins_local, hist_local_normal_avg, "Local force normal contact point", symmetry=True)
        plotter.plot_polar_histogram(bins_global, hist_global_tangential_avg, "Global force tangential", symmetry=False)
        plotter.plot_polar_histogram(bins_local, hist_local_tangential_avg, "Local force tangential contact point", symmetry=True)
        plotter.plot_polar_histogram(bins_global, hist_global_normal_cp_avg, "Global force normal contact point", symmetry=False)
        plotter.plot_polar_histogram(bins_global, hist_global_tangential_cp_avg, "Global force tangential contact point", symmetry=False)
        plotter.plot_polar_histogram(bins_global, pdfs['contacts_hist_global_normal'], "Global normal direction density", symmetry=False)
        plotter.plot_polar_histogram(bins_global, pdfs['contacts_hist_global_tangential'], "Global tangential direction density", symmetry=False)
        plotter.plot_polar_histogram(bins_global, pdfs['contacts_hist_cont_point_global'], "Global contact point density", symmetry=False)
        plotter.plot_polar_histogram(bins_local, pdfs['contacts_hist_cont_point_local'], "Local contact point density", symmetry=True)
        
        
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