import multiprocessing
import numpy as np

import config
from analysis_utils import *
from ReaderVtk import ReaderVtk
from ReaderDump import ReaderDump
from ProcessorVtk import ProcessorVtk
from ProcessorDump import ProcessorDump
from CombinedProcessor import CombinedProcessor
from ProcessorCsv import ProcessorCsv
from ProcessorDat import ProcessorDat
from DataExporter import DataExporter
from histogram_utils import area_adjustment_ellipsoid, compute_circular_mean, compute_pdf_orientation

class SimulationAnalyzer:
    """Orchestrates the entire simulation post-processing workflow."""
    def __init__(self, cof, ap, param, pressure, num_processes):
        self.cof = cof
        self.ap = ap
        self.param = param
        self.pressure = pressure
        self.num_processes = num_processes
        
        # This will hold the final processed data
        self.results_data = {}

    def run_analysis(self):
        """Execute the full analysis pipeline."""
        print("1. Loading and initializing data readers and processors...")
        self._initialize_processors()

        print("2. Running parallel processing of simulation steps...")
        parallel_results = self._run_parallel_processing()

        print("3. Aggregating and post-processing results...")
        self._post_process(parallel_results)

        print("4. Exporting final data...")
        self._export_data()
        
        print("\nAnalysis complete.")

    def _initialize_processors(self):
        """Initializes all data readers and processors."""
        # VTK Reader
        self.vtk_reader = ReaderVtk(self.cof, self.ap, I=self.param, pressure=self.pressure)
        self.vtk_reader.read_data((config.GLOBAL_PATH + "/"), 'simple_shear_', box=True)
        self.vtk_reader.no_wall_atoms()
        self.vtk_reader.get_initial_velocities_and_orientations()
        
        # Other Readers and Processors
        self.df_csv, self.shear_rate = self.vtk_reader.read_csv_file()
        self.df_dat = self.vtk_reader.read_dat_file()
        self.particles_volume = self.vtk_reader.get_particles_volume()
        
        dt_hertz = time_step_used(self.shear_rate, self.ap, 1.5 * 0.08)
        
        dump_reader = ReaderDump(self.cof, self.ap, I=self.param, pressure=self.pressure)
        dump_reader.read_data((config.GLOBAL_PATH + "/"), 'simple_shear_contact_data_')
        
        vtk_processor = ProcessorVtk(self.vtk_reader)
        dump_processor = ProcessorDump(dump_reader, self.vtk_reader.n_wall_atoms, self.vtk_reader.n_central_atoms)  
        
        self.combined_processor = CombinedProcessor(vtk_processor, dump_processor, self.vtk_reader.file_list_box, self.shear_rate, dt_hertz)

    def _run_parallel_processing(self):
        """Manages the multiprocessing pool to process steps."""
        if self.param >= config.INERTIAL_NUMBER_THRESHOLD:
            self.shear_one_index = config.SHEAR_ONE_INDEX_HIGH_I
        else:
            self.shear_one_index = config.SHEAR_ONE_INDEX_LOW_I
        
        print(f"Starting analysis from shear step index: {self.shear_one_index}")
        
        # Define the range of steps to process
        steps_to_process = range(self.shear_one_index, self.combined_processor.n_sim)
        self.n_sim_processed = len(steps_to_process)

        with multiprocessing.Pool(self.num_processes) as pool:
            results = pool.map(self.combined_processor.process_single_step, steps_to_process)
        return results

    def _post_process(self, parallel_results):
        """Aggregates parallel results and computes final metrics."""
        area_adjust, total_area = area_adjustment_ellipsoid(config.NUM_BINS_LOCAL, self.ap)
        
        # Process primary results from simulation steps
        averages, hist_avg, pdfs, distributions = process_results(
            parallel_results, config.BINS_CONFIG, config.CORRELATION_KEYS, area_adjust, self.n_sim_processed
        )

        # Compute autocorrelations
        auto_corr_vel = compute_autocorrelation_function(distributions['vy_velocity'])
        auto_corr_omega = compute_autocorrelation_function(distributions['omegaz_velocity'])
        
        final_strain = self.n_sim_processed / 100
        strain = np.linspace(0, final_strain, auto_corr_vel.size)
        
        averages['auto_corr_vel'] = auto_corr_vel[:config.AUTOCORR_STRAIN_POINTS]
        averages['auto_corr_omega'] = auto_corr_omega[:config.AUTOCORR_STRAIN_POINTS]
        averages['strain'] = strain[:config.AUTOCORR_STRAIN_POINTS]
        
        # Process supplementary CSV and DAT files
        csv_processor = ProcessorCsv(self.df_csv)
        csv_processor.exclude_initial_strain_cycle(self.param)
        avg_csv = csv_processor.get_averages(self.shear_rate)
        
        dat_processor = ProcessorDat(self.df_dat)
        avg_dat = dat_processor.compute_averages(self.shear_one_index / self.combined_processor.n_sim)
        avg_dat = dat_processor.compute_max_vx_diff(avg_dat)
        
        # Combine all results into a single dictionary
        self.results_data = {**averages, **hist_avg, **pdfs, **avg_csv, **avg_dat}
        
        # Add final computed values
        self.results_data['shear_rate'] = self.shear_rate
        self.results_data['thetax_mean'] = compute_circular_mean(distributions['thetax'], config.NUM_BINS_ORIENTATION)
        self.results_data['thetaz_mean'] = compute_circular_mean(distributions['thetaz'], config.NUM_BINS_ORIENTATION)
        self.results_data['pdf_thetax'] = compute_pdf_orientation(distributions['thetax'], config.NUM_BINS_ORIENTATION)
        self.results_data['pdf_thetaz'] = compute_pdf_orientation(distributions['thetaz'], config.NUM_BINS_ORIENTATION)

    def _export_data(self):
        """Exports the final processed data using DataExporter."""
        exporter = DataExporter(self.ap, self.cof, I=self.param)
        exporter.export_with_pickle(self.results_data)