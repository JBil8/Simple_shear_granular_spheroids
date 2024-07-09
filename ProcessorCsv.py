import pandas as pd
import numpy as np
from DataProcessor import DataProcessor

class ProcessorCsv(DataProcessor):
    def __init__(self, data):
        super().__init__(data)


    def exclude_initial_strain_cycle(self, strain=1, total_strain=11):
        

        starting_index = int(self.data_reader.shape[0] / total_strain*strain)  # to start from steady state

        # Slice the DataFrame from the starting index
        df_sliced = self.data_reader.iloc[starting_index:]

        self.df = df_sliced

    def get_averages(self):
        
        df_filtered = self.df.drop(columns=['time', 'shear_strain', 'msd'], errors='ignore')
        avg_dict = df_filtered.mean().to_dict()     
        avg_dict['Dyy'] = self.compute_y_diffusion_coefficient()
        return avg_dict
    
    def compute_y_diffusion_coefficient(self):
        """
        Compute the diffusion coefficient in the y direction
        """
        # Compute the mean square displacement in the y direction
        msd_scaled = self.df['msd']-self.df['msd'].iloc[0] #subtract the initial value
        time_scaled = self.df['time']-self.df['time'].iloc[0] #subtract the initial value
        
        # Compute the diffusion coefficient
        diffusion_coefficient = np.polyfit(time_scaled, msd_scaled, 1)[0]/2

        return diffusion_coefficient
