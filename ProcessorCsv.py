import pandas as pd
import numpy as np
from DataProcessor import DataProcessor

class ProcessorCsv(DataProcessor):
    def __init__(self, data):
        super().__init__(data)


    def exclude_initial_strain_cycle(self, Inertial_number):
        
        if Inertial_number >= 0.01:
            strain = 6
            total_strain = 16
        else:
            strain = 4
            total_strain = 14

        starting_index = int(self.data_reader.shape[0] / total_strain*strain)  # to start from steady state

        # Slice the DataFrame from the starting index
        df_sliced = self.data_reader.iloc[starting_index:]

        self.df = df_sliced

    def get_averages(self):
        
        df_filtered = self.df.drop(columns=['time', 'shear_strain', 'msdY'], errors='ignore')
        avg_dict = df_filtered.mean().to_dict()     
        avg_dict['Dyy'] = self.compute_diffusion_coefficient("Y")
        avg_dict['Dzz'] = self.compute_diffusion_coefficient("Z")
        return avg_dict
    
    def compute_dissipation_mu_I_average(self, shear_rate, volume_particles):
        """
        Compute the dissipation rate mu_I
        """
        # Compute the dissipation rate
        mu_I_diss = shear_rate * self.df['p_xy'] *volume_particles/self.df['phi']

        return mu_I_diss.mean()


    def compute_diffusion_coefficient(self, direction="Y"):
        """
        Compute the diffusion coefficient in the y direction
        """
        label = 'msd' + direction
        # Compute the mean square displacement in the y direction
        msd_scaled = self.df[label]-self.df[label].iloc[0] #subtract the initial value
        time_scaled = self.df['time']-self.df['time'].iloc[0] #subtract the initial value
        
        # Compute the diffusion coefficient
        diffusion_coefficient = np.polyfit(time_scaled, msd_scaled, 1)[0]/2

        return diffusion_coefficient