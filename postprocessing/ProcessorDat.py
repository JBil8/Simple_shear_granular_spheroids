import pandas as pd
import numpy as np
from DataProcessor import DataProcessor


class ProcessorDat(DataProcessor):
    def __init__(self, data):
        super().__init__(data)

    def compute_averages(self, ratio_to_skip):

        bins = self.data_reader['bin_index'].unique()
        num_bins_to_skip = int(np.ceil(len(bins)*ratio_to_skip))

        bins_array = np.array(bins[num_bins_to_skip:])

        results_dict = {'bins': bins_array}

        numeric_columns = self.data_reader.select_dtypes(
            include=[np.number]).columns

        for variable in numeric_columns:
            if variable == 'bin_index':
                continue
            if variable in ['vx', 'vy', 'vz']:
                # Compute and subtract the average across bins for each timestep
                self.data_reader[variable] = self.data_reader.groupby('timestep').apply(
                    lambda g: g[variable] - np.average(g[variable], weights=g['Ncount'])).reset_index(level=0, drop=True)

            averages = []
            std_devs = []

            for b in bins[num_bins_to_skip:]:
                bin_data = self.data_reader[self.data_reader['bin_index']
                                            == b][variable]
                averages.append(bin_data.mean())
                std_devs.append(bin_data.std())

            averages_array = np.array(averages)
            std_devs_array = np.array(std_devs)

            results_dict[f'{variable}_avg'] = averages_array
            results_dict[f'{variable}_std_dev'] = std_devs_array

        return results_dict

    def compute_max_vx_diff(self, dict):

        max_vx = np.max(dict['vx_avg'])
        min_vx = np.min(dict['vx_avg'])
        dict['max_vx_diff'] = max_vx - min_vx
        return dict
