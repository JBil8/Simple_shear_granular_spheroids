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
from histogram_utils import*


def time_step_used(shear_rate, ap, factor, is_prolate, Young=5e6, rho=1000, nu=0.3, small_axis = 1.0):
    """compute the dem stable time step with hertz contact time"""
    long_axis = ap*small_axis
    if is_prolate:
        avg_diameter = (2*small_axis+long_axis)/3*2
    else:
        avg_diameter = (small_axis+2*long_axis)/3*2
    
    v_collision = 2*shear_rate*avg_diameter
    hertz_factor = 2.943*(5*np.sqrt(2)*3.14*rho*(1-nu**2)/(4*Young))**(2/5)
    dt = factor* hertz_factor*small_axis/(v_collision)**(1/5)
    print(dt)
    return dt

def process_results(results):
    """Process the results to extract and average histograms."""
    num_results = len(results)
    
    # Define the number of bins for each histogram type
    bins_config = {
        'hist_thetax': 144,
        'hist_thetaz': 144,
        'global_normal_force_hist': 144,
        'global_tangential_force_hist': 144,
        'local_normal_force_hist_cp': 36,
        'local_tangential_force_hist_cp': 36,
        'global_normal_force_hist_cp': 144,
        'global_tangential_force_hist_cp': 144,
        'contacts_hist_cont_point_global': 144,
        'contacts_hist_cont_point_local': 36,
        'contacts_hist_global_normal': 144,
        'contacts_hist_global_tangential': 144,
        'power_dissipation_normal': 36,
        'power_dissipation_tangential': 36,
        'bin_counts_power': 36,
        'normal_force_hist_mixed': 36**2,
        'tangential_force_hist_mixed': 36**2,
        'counts_mixed': 36**2
    }
    
    # Initialize sums for each histogram type
    histogram_sums = {key: np.zeros(bins) for key, bins in bins_config.items()}

    # Extract averages and sum histograms
    averages = {}
    for key in results[0].keys():
        if key in histogram_sums:
            histogram_sums[key] = compute_histogram_sum(results, key)
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
    
    histograms_weighted_avg['power_dissipation_normal'] = compute_weighted_average_hist(
        histogram_sums['power_dissipation_normal'], n_sim)
    histograms_weighted_avg['power_dissipation_tangential'] = compute_weighted_average_hist(
        histogram_sums['power_dissipation_tangential'], n_sim)

    histograms_weighted_avg['normal_force_hist_mixed'] = compute_weighted_average_hist(
        histogram_sums['normal_force_hist_mixed'], histogram_sums['counts_mixed'])
    
    histograms_weighted_avg['tangential_force_hist_mixed'] = compute_weighted_average_hist(
        histogram_sums['tangential_force_hist_mixed'], histogram_sums['counts_mixed'])

    # Compute PDFs for contact distributions
    pdfs = {}
    pdfs['hist_thetax'] = compute_pdf(histogram_sums['hist_thetax'])
    pdfs['hist_thetaz'] = compute_pdf(histogram_sums['hist_thetaz'])
    pdfs['contacts_hist_global_normal'] = compute_pdf(histogram_sums['contacts_hist_global_normal'])
    pdfs['contacts_hist_global_tangential'] = compute_pdf(histogram_sums['contacts_hist_global_tangential'])
    pdfs['contacts_hist_cont_point_global'] = compute_pdf(histogram_sums['contacts_hist_cont_point_global'], max_angle=180)
    pdfs['contacts_hist_cont_point_local'] = compute_pdf_on_ellipsoid(histogram_sums['contacts_hist_cont_point_local'])
    pdfs['bin_counts_power'] = compute_pdf_on_ellipsoid(histogram_sums['bin_counts_power'])

    
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
    if ap >=1.0:
        is_prolate = True
    else:
        is_prolate = False    
    param = args.value
    full_postprocess = args.postprocess
    pressure = args.pressure
    num_processes = 8

    if full_postprocess == True:

        global_path = "/home/jacopo/Documents/PhD_research/Liggghts_simulations/cluster_simulations/"
        plt.ioff()
         #initialize the vtk reader
        data_read = ReaderVtk(cof, ap, I=param, pressure = pressure)
        data_read.read_data(global_path, 'simple_shear_', box=True)
        data_read.no_wall_atoms()
        #data_read.get_initial_velocities()
        df_csv, shear_rate = data_read.read_csv_file()
        dt_hertz = time_step_used(shear_rate, ap, 0.012, is_prolate)
       
        df_dat = data_read.read_dat_file()
        particles_volume = data_read.get_particles_volume()
        #intialize the dump reader
        data_dump = ReaderDump(cof, ap, I=param, pressure = pressure)
        data_dump.read_data(global_path, 'simple_shear_contact_data_')
        to_process_vtk = ProcessorVtk(data_read)
        to_process_dump = ProcessorDump(data_dump, data_read.n_wall_atoms, data_read.n_central_atoms)  
        combined_processor = CombinedProcessor(to_process_vtk, to_process_dump, data_read.file_list_box, shear_rate, dt_hertz)

        #shear_one_index = int(to_process_vtk.n_sim/11)
        shear_one_index = 100
        n_sim = combined_processor.n_sim-shear_one_index
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
        bins_orientation = np.linspace(-90, 90, 145)
        bins_global = np.linspace(-180, 180, 145)
        bins_local = np.linspace(0, 90, 37)

        csvProcessor = ProcessorCsv(df_csv)
        csvProcessor.exclude_initial_strain_cycle()
        datProcessor = ProcessorDat(df_dat)
        avgdat = datProcessor.compute_averages()
        avgdat = datProcessor.compute_max_vx_diff(avgdat)
        avgcsv = csvProcessor.get_averages()
        thetax_median = compute_histogram_median(pdf_thetax)
        thetaz_median = compute_histogram_median(pdf_thetaz)
        averages['thetax_median'] = thetax_median
        averages['thetaz_median'] = thetaz_median

        averages = {**averages, **avgcsv, **avgdat} #merge the two dictionaries

        averages['shear_rate'] = shear_rate #add the shear rate to the averages 

        #export the data with pickle
        exporter = DataExporter(ap, cof,I=param)
        exporter.export_with_pickle(averages)
        
        plotter = DataPlotter(ap, cof,value=param)

        total_average_dissipation_local = np.sum(hist_weigh_avg['power_dissipation_normal']+hist_weigh_avg['power_dissipation_tangential'])
        total_tangential_dissipation = np.sum(hist_weigh_avg['power_dissipation_tangential'])
        actual_dissipation = avgcsv['p_xy']*shear_rate*averages['box_x_length']*averages['box_y_length']*averages['box_z_length']

        # normalized_global_normal = hist_global_normal_avg/np.mean(hist_global_normal_avg)*pdfs['contacts_hist_global_normal']

        # normalized_global_tangential = hist_global_tangential_avg/np.mean(hist_global_tangential_avg)*pdfs['contacts_hist_global_tangential']

        normalized_global_cp_normal = hist_global_normal_cp_avg/np.mean(hist_global_normal_cp_avg)*pdfs['contacts_hist_cont_point_global']
        normalized_global_cp_tangential = hist_global_tangential_cp_avg/np.mean(hist_global_normal_cp_avg)*pdfs['contacts_hist_cont_point_global']

        print(np.sum(pdfs['contacts_hist_cont_point_global']))
        print(np.sum(normalized_global_cp_normal))
        print(np.sum(normalized_global_cp_tangential))

        plotter.plot_polar_histogram(bins_global, normalized_global_cp_normal, "$\zeta_N$", symmetry=False)
                                #label = '$\rho(\lambda)\langle N(\lambda)\\rangle / \langle N \\rangle $')
        plotter.plot_polar_histogram(bins_global, normalized_global_cp_tangential, '$\zeta_T$', symmetry=False)

        measured_shear_rate = avgdat['max_vx_diff']/averages['box_y_length']
        print("Measured shear rate: ", measured_shear_rate)
        print("Shear rate: ", shear_rate)
        print("Total average dissipation: ", total_average_dissipation_local)
        print("Actual dissipation: ", actual_dissipation)
        # print("Total tangential dissipation: ", total_tangential_dissipation)
        #print(avgdat.keys())
        plotter.plot_time_variation(averages, df_csv) 
        plotter.plot_averages_with_std(averages)
        plotter.plot_histogram(bins_orientation, pdf_thetax, "$\theta_x",  label = '$\\theta_x [^\circ]$')
        plotter.plot_histogram(bins_orientation, pdf_thetaz, "$\theta_z",  label = '$\\theta_z [^\circ]$')
        plotter.plot_polar_histogram(bins_global, hist_global_normal_avg, "Global force normal", symmetry=False)
        plotter.plot_polar_histogram(bins_global, hist_global_tangential_avg, "Global force tangential", symmetry=False)
        plotter.plot_polar_histogram(bins_global, hist_global_normal_cp_avg, "Global force normal contact point", symmetry=False)
        plotter.plot_polar_histogram(bins_global, hist_global_tangential_cp_avg, "Global force tangential contact point", symmetry=False)
        plotter.plot_polar_histogram(bins_global, pdfs['contacts_hist_global_normal'], "Global normal direction density", symmetry=False)
        plotter.plot_polar_histogram(bins_global, pdfs['contacts_hist_global_tangential'], "Global tangential direction density", symmetry=False)
        plotter.plot_polar_histogram(bins_global, pdfs['contacts_hist_cont_point_global'], "Global contact point density", symmetry=False)
        plotter.plot_polar_histogram(bins_local, pdfs['contacts_hist_cont_point_local'], "Local contact point density", symmetry=True)
        plotter.plot_polar_histogram(bins_local, hist_local_normal_avg, "Local force normal contact point", symmetry=True)
        plotter.plot_polar_histogram(bins_local, hist_local_tangential_avg, "Local force tangential contact point", symmetry=True)
        plotter.plot_histogram_ellipsoid(pdfs['contacts_hist_cont_point_local'], bins_local, "Local contact point density",'$pdf$' , is_prolate = is_prolate)
        plotter.plot_histogram_ellipsoid(hist_local_normal_avg, bins_local, "Local force normal contact point",'$F_n [N] $', is_prolate = is_prolate)
        plotter.plot_histogram_ellipsoid(hist_local_tangential_avg, bins_local, "Local force tangential contact point",'$F_t [N] $', is_prolate = is_prolate)
        plotter.plot_histogram_ellipsoid(hist_local_tangential_avg/hist_local_normal_avg, bins_local, "Ratio tangential to normal force",'$F_t/F_N$', is_prolate = is_prolate)
        plotter.plot_histogram_ellipsoid_flat(hist_weigh_avg['normal_force_hist_mixed'] , 36,36, "Normal force mixed", '$F_n [N]$', is_prolate = is_prolate)
        plotter.plot_histogram_ellipsoid_flat(hist_weigh_avg['tangential_force_hist_mixed'] , 36,36, "Tangential force mixed", '$F_t [N]$', is_prolate = is_prolate)
        plotter.plot_histogram_ellipsoid(pdfs['bin_counts_power'], bins_local, "Power dissipation", '$p(contacts)$', is_prolate = is_prolate)
        area_normalized_power = rescale_histogram_ellipsoid(hist_weigh_avg['power_dissipation_normal'], ap)
        plotter.plot_histogram_ellipsoid(area_normalized_power, bins_local, "Power dissipation normal (area)",'$P_n A_{loc}/A_{tot} [W]$', is_prolate = is_prolate)
        plotter.plot_histogram_ellipsoid(hist_weigh_avg['power_dissipation_normal'], bins_local, "Power dissipation normal",'$P_n [W]$', is_prolate = is_prolate)
        plotter.plot_histogram_ellipsoid(hist_weigh_avg['power_dissipation_tangential'], bins_local, "Power dissipation tangential",'$P_t [W]$', is_prolate = is_prolate)
        plotter.plot_histogram_ellipsoid(hist_weigh_avg['power_dissipation_tangential']/hist_weigh_avg['power_dissipation_normal'], bins_local, "Power dissipation tangential to normal ratio", '$P_t/P_n$', is_prolate = is_prolate)
        total_dissipartion_histogram = hist_weigh_avg['power_dissipation_normal']+hist_weigh_avg['power_dissipation_tangential']
        plotter.plot_histogram_ellipsoid(total_dissipartion_histogram/total_average_dissipation_local, bins_local, "Total power dissipation",'$P/P_{tot}$', is_prolate = is_prolate)
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