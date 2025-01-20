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

def time_step_used(shear_rate, ap, factor, Young=5e6, rho=1000, nu=0.3, small_axis=1.0, shear_mod=1e6):
    """
    Compute the DEM stable time step using Hertz and Rayleigh contact time calculations.
    
    Parameters:
        shear_rate (float): Shear rate.
        ap (float): Aspect ratio (ratio of long to small axis for prolate shapes).
        factor (float): Factor to scale the computed time step.
        Young (float): Young's modulus of the material (default is 5e6).
        rho (float): Density of the material (default is 1000).
        nu (float): Poisson's ratio (default is 0.3).
        small_axis (float): Length of the smallest axis (default is 1.0).
        shear_mod (float): Shear modulus of the material (default is 1e6).

    Returns:
        float: The minimum time step for the simulation.
    """
    # Define the long axis and average diameter based on the particle shape
    long_axis = ap * small_axis
    if ap>1:
        avg_diameter = (2 * small_axis + long_axis) / 3 * 2
    else:
        avg_diameter = (small_axis + 2 * long_axis) / 3 * 2

    # Calculate collision velocity
    v_collision = 2 * shear_rate * avg_diameter

    # Compute hertz and rayleigh time steps
    hertz_factor = 2.943 * (5 * np.sqrt(2) * np.pi * rho * (1 - nu**2) / (4 * Young))**(2 / 5)
    min_dimen = min(small_axis, long_axis)
    dt_hertz = 2 * factor * hertz_factor * min_dimen / (v_collision)**(1 / 5)
    dt_rayleigh = 2 * factor * np.pi * min_dimen * np.sqrt(rho / shear_mod) * (1 / (0.8766 + 0.1631 * nu))

    # Select the smaller of the Hertz and Rayleigh time steps
    dt = min(dt_hertz, dt_rayleigh)

    return dt

def process_results(results):
    """Process the results to extract and average histograms."""
    num_results = len(results)
    
    # Define the number of bins for each histogram type
    bins_config = {
        'global_normal_force_hist': 144,
        'global_tangential_force_hist': 144,
        'local_normal_force_hist_cp': 10,
        'local_tangential_force_hist_cp': 10,
        'global_normal_force_hist_cp': 144,
        'global_tangential_force_hist_cp': 144,
        'contacts_hist_cont_point_global': 144,
        'contacts_hist_cont_point_local': 10,
        'contacts_hist_global_normal': 144,
        'contacts_hist_global_tangential': 144,
        'power_dissipation_normal': 10,
        'power_dissipation_tangential': 10,
        'bin_counts_power': 10,
        'normal_force_hist_mixed': 10**2,
        'tangential_force_hist_mixed': 10**2,
        'counts_mixed': 10**2
    }
    
    # Initialize sums for each histogram type
    histogram_sums = {key: np.zeros(bins) for key, bins in bins_config.items()}

    # Extract averages and sum histograms
    averages = {}
    distributions = {}

    for key in results[0].keys():
        if key in histogram_sums:
            histogram_sums[key] = compute_histogram_sum(results, key)
        elif key in ['trackedGrainsOrientation', 'trackedGrainsPosition', 'thetax', 'thetaz']:
            distributions[key] = np.concatenate([result[key] for result in results])
        elif key in ['vy_velocity']:
            distributions[key] = np.stack([result[key] for result in results], axis=1)
        else:
            averages[key] = np.mean([result[key] for result in results], axis=0)
    
    distributions['thetax_particles'] = np.stack([result['thetax'] for result in results], axis=1)
    
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
    pdfs['contacts_hist_global_normal'] = compute_pdf(histogram_sums['contacts_hist_global_normal'], 360/144)
    pdfs['contacts_hist_global_tangential'] = compute_pdf(histogram_sums['contacts_hist_global_tangential'], 360/144)
    pdfs['contacts_hist_cont_point_global'] = compute_pdf(histogram_sums['contacts_hist_cont_point_global'], 360/144)
    pdfs['contacts_hist_cont_point_local'] = compute_pdf_on_ellipsoid(histogram_sums['contacts_hist_cont_point_local'], area_adjustments_ellipsoid)
    pdfs['bin_counts_power'] = compute_pdf_on_ellipsoid(histogram_sums['bin_counts_power'], area_adjustments_ellipsoid)

    return averages, histograms_weighted_avg, pdfs, distributions

def compute_autocorrelation_function(vy):
    """
    Compute the autocorrelation function as a function of strain for the y-velocity component.
    
    Parameters:
        vy (numpy.ndarray): A 2D array of shape (n_particles, n_strains) representing the y-velocity fluctuations.
    
    Returns:
        numpy.ndarray: The normalized autocorrelation function as a function of strain (averaged over particles).
    """
    # Extract dimensions
    n_particles, n_strains = vy.shape
    
    # Compute the FFT along the strain axis (axis=1)
    fft_vy = np.fft.fft(vy, n=2*n_strains, axis=1)  # Zero-padding to 2*n_strains for circular convolution
    power_spectrum = fft_vy * np.conjugate(fft_vy)  # Compute the power spectrum
    
    # Compute the inverse FFT to get the autocorrelation along the strain axis
    autocorr = np.fft.ifft(power_spectrum, axis=1).real[:, :n_strains]  # First n_strains terms only
    
    # Average across particles to get the final autocorrelation function
    avg_autocorr = np.mean(autocorr, axis=0)

    # Normalize the autocorrelation function
    avg_autocorr /= np.mean(vy**2)
    
    return avg_autocorr

def compute_rotational_diffusion(thetax, n_sim):
    """
    Compute the rotational diffusion coefficient from the angular displacement.
    
    Parameters:
        thetax (numpy.ndarray): A 2D array of shape (n_particles, n_strains) representing the angular displacement.
        
        Returns:
            scalar value of the rotational diffusion coefficient measured as the best fit of the mean square angular displacement
            thetax values are capped in the range [-pi/2, pi/2] therefore we adjust the
            thetax if there is a jump of more than pi/2 by unwrapping the actual angulr value
    """
    n_particles, n_strains = thetax.shape
    
    # update thetax values if there is a jump of more than pi/2 in numpy efficient way
    unwarpped_thetax = np.unwrap(thetax, period= np.pi, axis=1)

    # Compute the mean square angular displacement
    msd = np.mean(unwarpped_thetax**2, axis=0)

    # Fit the mean square angular displacement to a linear function
    final_strain = 16
    intial_strain = 4
    total_strain = final_strain - intial_strain
    strain = np.arange(n_strains)*total_strain/n_strains
    fit = np.polyfit(strain, msd, 1)
    print("Angular diffusion coefficient: ", fit[0]/2)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process granular simulation.')
    parser.add_argument('-c', '--cof', type=float, help='coefficient of friction particles-particles')
    parser.add_argument('-a', '--ap', type=float, help='aspect ratio')
    parser.add_argument('-v', '--value', type=float, help='packing fraction or Inertial number depensing on the type of simulation')
    parser.add_argument('-p', '--postprocess', action='store_false', help='whether to postprocess the data or simply import the pkl file, default is True', default=True)
    parser.add_argument('-cw', '--cof_wall', action='store_false', help='coefficient of friction particles-walls')
    parser.add_argument('-s', '--pressure', type = int, help='pressure')
    parser.add_argument('-np', '--num_processes', type = int, help='number of processes to use in parallel', default=12)
    args = parser.parse_args()

    #parsing command line arguments
    cof = args.cof
    ap = args.ap
    param = args.value
    full_postprocess = args.postprocess
    pressure = args.pressure
    num_processes = args.num_processes

    if full_postprocess == True:

        global_path = "/home/jacopo/Documents/phd_research/Liggghts_simulations/cluster_simulations/"
        # global_path = "/scratch/bilotto/simulations_simple_shear_hertz_dt_0.15/"
        # global_path = "/work/lsms/jbilotto/simulations_simple_shear_hertz_dt_0.15/"
        # global_path = "/scratch/bilotto/simulations_simple_shear_hertz_dt_0.08/"
        # global_path = "/scratch/bilotto/simulations_simple_shear_hertz_cof_0.01/"
        # global_path = "/scratch/bilotto/simulations_simple_shear_hertz_vy_0.00001/"
        
        plt.ioff()
         #initialize the vtk reader
        data_read = ReaderVtk(cof, ap, I=param, pressure = pressure)
        data_read.read_data(global_path, 'simple_shear_', box=True)
        data_read.no_wall_atoms()
        data_read.get_initial_velocities_and_orientations()
        df_csv, shear_rate = data_read.read_csv_file()
        dt_hertz = time_step_used(shear_rate, ap, 1.5*0.08)
       
        df_dat = data_read.read_dat_file()
        particles_volume = data_read.get_particles_volume()
        #intialize the dump reader
        data_dump = ReaderDump(cof, ap, I=param, pressure = pressure)
        data_dump.read_data(global_path, 'simple_shear_contact_data_')
        to_process_vtk = ProcessorVtk(data_read)
        to_process_dump = ProcessorDump(data_dump, data_read.n_wall_atoms, data_read.n_central_atoms)  
        combined_processor = CombinedProcessor(to_process_vtk, to_process_dump, data_read.file_list_box, shear_rate, dt_hertz)

        if param >=0.01:
            shear_one_index = 600
        else:
            shear_one_index = 400
        print("Shear one index: ", shear_one_index)
        n_sim = combined_processor.n_sim-shear_one_index
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(combined_processor.process_single_step,
                                [step for step in range(shear_one_index, combined_processor.n_sim)])

        area_adjustments_ellipsoid, total_area = area_adjustment_ellipsoid(10, ap)
        averages, hist_weigh_avg, pdfs, distributions = process_results(results)

        D_rot = compute_rotational_diffusion(distributions['thetax_particles'], n_sim)
        auto_corr = compute_autocorrelation_function(distributions['vy_velocity'])

        final_strain = (combined_processor.n_sim-shear_one_index)/100
        strain = np.linspace(0, final_strain, auto_corr.size)

        #plot the autocorrelation function
        plt.figure()
        plt.loglog(strain, auto_corr)
        plt.xlabel('$\\gamma$')
        plt.ylabel('$\\tilde{{C}}_{v}(\\gamma)$')
        plt.xlim([0, 0.1])
        plt.savefig('autocorrelation_vy_strain.png')
        plt.close()

        # plot spatial autocorrelation
        plt.figure()
        plt.plot(averages["c_r_values"], averages["c_delta_vy"])
        plt.xlabel('$r$')
        plt.ylabel('$\\tilde{C}_{\\delta v_y}(r)$')
        plt.savefig('spatial_autocorrelation_vy.png')
        plt.close()

                   # Access the specific PDFs or histograms as needed
        hist_global_normal_avg = hist_weigh_avg['global_normal_force_hist']
        hist_global_tangential_avg = hist_weigh_avg['global_tangential_force_hist']
        hist_local_normal_avg = hist_weigh_avg['local_normal_force_hist_cp']
        hist_local_tangential_avg = hist_weigh_avg['local_tangential_force_hist_cp']
        hist_global_normal_cp_avg = hist_weigh_avg['global_normal_force_hist_cp']
        hist_global_tangential_cp_avg = hist_weigh_avg['global_tangential_force_hist_cp']

        # Define bins for plotting or further analysis
        bins_orientation = np.linspace(-np.pi/2, np.pi/2, 145)
        bins_global = np.linspace(-180, 180, 145)
        bins_local = np.linspace(0, 90, 11)

        csvProcessor = ProcessorCsv(df_csv)
        csvProcessor.exclude_initial_strain_cycle(param)
        avgcsv = csvProcessor.get_averages(shear_rate)
        datProcessor = ProcessorDat(df_dat)
        avgdat = datProcessor.compute_averages(shear_one_index/combined_processor.n_sim)
        avgdat = datProcessor.compute_max_vx_diff(avgdat) 
        averages['muI_dissipation'] = csvProcessor.compute_dissipation_mu_I_average(shear_rate, particles_volume)
        n_bins_orientation = 180
        thetax_mean = compute_circular_mean(distributions['thetax'], n_bins_orientation)
        thetaz_mean = compute_circular_mean(distributions['thetaz'], n_bins_orientation)
        averages['thetax_mean'] = thetax_mean
        averages['thetaz_mean'] = thetaz_mean
        averages['shear_rate'] = shear_rate
        averages['area_adjustment_ellipsoid'] = area_adjustments_ellipsoid
        averages['total_area'] = total_area
        averages['total_normal_dissipation'] = np.sum(hist_weigh_avg['power_dissipation_normal'])
        averages['total_tangential_dissipation'] = np.sum(hist_weigh_avg['power_dissipation_tangential'])
        averages['pdf_thetax'] = compute_pdf_orientation(distributions['thetax'], n_bins_orientation)
        averages['pdf_thetaz'] = compute_pdf_orientation(distributions['thetaz'], n_bins_orientation)
        averages['auto_corr'] = auto_corr
        averages['strain'] = strain
        averages['D_rot'] = D_rot
        total_average_dissipation_local = np.sum(hist_weigh_avg['power_dissipation_normal']+hist_weigh_avg['power_dissipation_tangential'])
        ratio_computed_mu_I_dissipation = total_average_dissipation_local/averages['muI_dissipation']
        averages['ratio_diss_measurement'] = ratio_computed_mu_I_dissipation
        print(averages)
        averages = {**averages, **avgcsv, **avgdat, **pdfs, **hist_weigh_avg}


        print(f"Average stress measured from contacts {averages['stress_contacts']}")
        print(f"Average stress measured from pressure {averages['p_yy'], averages['p_xy']}")
        #export the data with pickle
        exporter = DataExporter(ap, cof,I=param)
        exporter.export_with_pickle(averages)
        
        # plotter = DataPlotter(ap, cof,value=param)
    
        # # # print("Total tangential dissipation: ", averages['total_tangential_dissipation'])
        # mean_global_normal = np.sum(hist_global_normal_avg*pdfs['contacts_hist_global_normal'])
        # xi_N_global = hist_global_normal_avg/mean_global_normal*pdfs['contacts_hist_global_normal']
        # xi_T_global = hist_global_tangential_avg/mean_global_normal*pdfs['contacts_hist_global_tangential']

        # normalized_global_cp_normal = hist_global_normal_cp_avg*pdfs['contacts_hist_cont_point_global']
        # normalized_global_cp_tangential = hist_global_tangential_cp_avg/np.mean(hist_global_normal_cp_avg)*pdfs['contacts_hist_cont_point_global']

        # zeta_N_global = hist_global_normal_avg*pdfs['contacts_hist_global_normal']
        # zeta_T_tangential = hist_global_tangential_avg*pdfs['contacts_hist_global_tangential']

        # plotter.plot_polar_histogram(bins_global, xi_N_global, "$\\xi_N$", symmetry=False)
        # plotter.plot_polar_histogram(bins_global, xi_T_global, "$\\xi_T$", symmetry=False)
                                
        # plotter.plot_polar_histogram(bins_global, zeta_N_global, "$\\zeta_N$", symmetry=False)
        #                         #label = '$\rho(\lambda)\langle N(\lambda)\\rangle / \langle N \\rangle $')
        # plotter.plot_polar_histogram(bins_global, zeta_T_tangential, '$\\zeta_T$', symmetry=False)

        # print("Sum of xi_N: ", np.sum(xi_N_global))    
        # print("Sum of xi_T: ", np.sum(xi_T_global))

        # plotter.plot_time_variation(averages, df_csv) 
        # plotter.plot_averages_with_std(averages)
        # plotter.plot_pdf(distributions['thetax'], n_bins_orientation, "$\\theta_x$",  label = '$\\theta_x [^\\circ]$', median_value = thetax_mean)
        # plotter.plot_pdf(distributions['thetaz'], n_bins_orientation, "$\\theta_z$",  label = '$\\theta_z [^\\circ]$', median_value = thetaz_mean)
        # plotter.plot_polar_histogram(bins_global, hist_global_normal_avg, "Global force normal", symmetry=False)
        # plotter.plot_polar_histogram(bins_global, hist_global_tangential_avg, "Global force tangential", symmetry=False)
        # plotter.plot_polar_histogram(bins_global, hist_global_normal_cp_avg, "Global force normal contact point", symmetry=False)
        # plotter.plot_polar_histogram(bins_global, hist_global_tangential_cp_avg, "Global force tangential contact point", symmetry=False)
        # plotter.plot_polar_histogram(bins_global, pdfs['contacts_hist_global_normal'], "Global normal direction density", symmetry=False)
        # plotter.plot_polar_histogram(bins_global, pdfs['contacts_hist_global_tangential'], "Global tangential direction density", symmetry=False)
        # plotter.plot_polar_histogram(bins_global, pdfs['contacts_hist_cont_point_global'], "Global contact point density", symmetry=False)
        # plotter.plot_polar_histogram(bins_local, pdfs['contacts_hist_cont_point_local'], "Local contact point density", symmetry=True)
        # plotter.plot_polar_histogram(bins_local, hist_local_normal_avg, "Local force normal contact point", symmetry=True)
        # plotter.plot_polar_histogram(bins_local, hist_local_tangential_avg, "Local force tangential contact point", symmetry=True)
        # # plotter.plot_polar_histogram(bins_local, hist_local_tangential_avg/hist_local_normal_avg, "Ratio tangential to normal force", symmetry=True)
        # plotter.plot_histogram_ellipsoid(pdfs['contacts_hist_cont_point_local']/2, bins_local, "Local contact point density",'$pdf$')
        # plotter.plot_histogram_ellipsoid(hist_local_normal_avg, bins_local, "Local force normal contact point",'$F_n [N] $')
        # plotter.plot_histogram_ellipsoid(hist_local_tangential_avg, bins_local, "Local force tangential contact point",'$F_t [N] $')

        # plotter.plot_histogram_ellipsoid(hist_local_normal_avg/(avgcsv['p_yy']*total_area), bins_local,
        #                                   "Local force normal contact point normalized pressure",'$F_n/p A_p $')
        # plotter.plot_histogram_ellipsoid(hist_local_tangential_avg/(avgcsv['p_yy']*total_area), bins_local, 
        #                                  "Local force tangential contact point normalized pressure",'$F_t/p A_p$')

        # plotter.plot_histogram_ellipsoid(pdfs['contacts_hist_cont_point_local']*hist_local_normal_avg/(avgcsv['p_yy']*total_area*area_adjustments_ellipsoid), bins_local,
        #                                   "Average Local force normal contact point normalized pressure",'$|F_n|/p A_p $')
        # plotter.plot_histogram_ellipsoid(pdfs['contacts_hist_cont_point_local']*hist_local_tangential_avg/(avgcsv['p_yy']*total_area*area_adjustments_ellipsoid), bins_local, 
        #                                  "Average Local force tangential contact point normalized pressure",'$|F_t|/p A_p$')

        # plotter.plot_histogram_ellipsoid(hist_local_tangential_avg/hist_local_normal_avg, bins_local, "Ratio tangential to normal force",'$F_t/F_N$')
        # # plotter.plot_histogram_ellipsoid_flat(hist_weigh_avg['normal_force_hist_mixed'] , 10,10, "Normal force mixed", '$F_n [N]$')
        # # plotter.plot_histogram_ellipsoid_flat(hist_weigh_avg['tangential_force_hist_mixed'] , 10,10, "Tangential force mixed", '$F_t [N]$')
        # plotter.plot_histogram_ellipsoid(pdfs['bin_counts_power'], bins_local, "Power dissipation density", '$p(contacts)$')
        # plotter.plot_histogram_ellipsoid(hist_weigh_avg['power_dissipation_normal'], bins_local, "Power dissipation normal",'$P_n [W]$')
        # plotter.plot_histogram_ellipsoid(hist_weigh_avg['power_dissipation_tangential'], bins_local, "Power dissipation tangential",'$P_t [W]$')
        # plotter.plot_histogram_ellipsoid(hist_weigh_avg['power_dissipation_tangential']/hist_weigh_avg['power_dissipation_normal'], bins_local, "Power dissipation tangential to normal ratio", '$P_t/P_n$')
        # total_dissipation_histogram = hist_weigh_avg['power_dissipation_normal']+hist_weigh_avg['power_dissipation_tangential']
        # plotter.plot_histogram_ellipsoid(total_dissipation_histogram/total_average_dissipation_local, bins_local, "Total power dissipation",'$P/P_{tot}$')
        # plotter.plot_polar_histogram(bins_local, hist_weigh_avg['power_dissipation_normal'], "Power diss normal", symmetry=True)
        # power_normalized_area = total_dissipation_histogram*total_area/(2*area_adjustments_ellipsoid*total_average_dissipation_local)
        # plotter.plot_histogram_ellipsoid(power_normalized_area, bins_local, "Power dissipation total (area)",'$P_j A_p /P_{tot} A_j $')

        # ellipsoid_force_pdf = pdfs['contacts_hist_cont_point_local']*np.sqrt(hist_weigh_avg['local_normal_force_hist_cp']**2 + hist_weigh_avg['local_tangential_force_hist_cp']**2)
        # plotter.plot_histogram_ellipsoid(ellipsoid_force_pdf/(avgcsv['p_yy']*total_area), bins_local, "Force density", '$\\langle F_j \\rangle / \\sigma_{yy} A_j$')

        plt.ion()

    else:
        #import the data with pickle
        importer = DataExporter(ap, cof, simulation_type ,param)
        averages, averages_dump = importer.import_with_pickle()

        #plotter = DataPlotter(ap, cof, simulation_type ,param)
        #plotter.plot_data(averages, averages_dump)
        #plotter.plot_eulerian_velocities(averages)

        # plot ellipsois in 3d
        plotter.plot_ellipsoids(500, averages, averages_dump)