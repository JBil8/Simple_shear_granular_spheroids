import os
import re
import numpy as np
import vtk
from abc import ABC, abstractmethod
import pandas as pd


class DataReader:
    def __init__(self, cof, ap, parameter=None, value=None, pressure=None, muw=None, vwall=None, fraction=None, phi=None, I=None):
        """
        Specify the parameters that vary in the parametric study
        cof: coefficient of friction
        ap: aspect ratio
        I: inertial number
        phi: volume fraction
        """
        self.cof = cof
        self.ap = ap
        self.I = I
        self.pressure = pressure
        self.muw = muw
        self.vwall = vwall
        self.fraction = fraction
        self.phi = phi
        self.n_sim = None
        self.step = None
        self.directory = None
        self.file_list = None

    def get_number_of_time_steps(self):
        """
        Deduce number of time steps from the number of files and the step between them
        """
        digits = [int(''.join(re.findall(r'\d+', filename)))
                  for filename in self.file_list]
        self.n_sim = len(self.file_list)
        self.step = int((max(digits) - min(digits)) / (self.n_sim - 1))

    def prepare_data(self, global_path):

        self.directory = global_path + \
            f'alpha_{self.ap}_cof_{self.cof}_pressure_{self.pressure}_I_{self.I}/'
        self.file_list = os.listdir(self.directory)

    @abstractmethod
    def filter_relevant_files(self, prefix='shear_ellipsoids_'):
        pass

    def sort_files_by_time(self, box=False):
        """
        Sort the files by time
        """
        digits = [int(''.join(re.findall(r'\d+', filename)))
                  for filename in self.file_list]
        time_idxs = np.argsort(digits)
        self.file_list = list(np.asarray(self.file_list)[time_idxs])
        if box:
            digits = [int(''.join(re.findall(r'\d+', filename)))
                      for filename in self.file_list_box]
            time_idxs = np.argsort(digits)
            self.file_list_box = list(
                np.asarray(self.file_list_box)[time_idxs])

    def get_data(self):
        """
        Read the data from the csv files
        """
        data = np.genfromtxt(
            self.directory + "time_series/data.csv", delimiter=',', skip_header=1)
        self.time = data[:, 0]
        Fx = data[:, 1]
        Fy = data[:, 2]
        Fz = data[:, 3]
        tke = data[:, 4]  # translational kinetic energy
        rke = data[:, 5]  # rotational kinetic energy

        return self.time, Fx, Fy, Fz, tke, rke

    def read_csv_file(self):
        """
        Read data from the only csv file in the directory
        """
        # Find the csv file in the directory
        file_list = os.listdir(self.directory)
        csv_file = [
            filename for filename in file_list if filename.endswith('.csv')][0]

        shear_rate = float(
            re.search(r"shear_([0-9.eE+-]+)\.csv$", csv_file).group(1))

        # Read the csv file into a DataFrame
        df = pd.read_csv(os.path.join(self.directory, csv_file))

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        return df, shear_rate

    def read_dat_file(self):
        """
        Read data from the only dat file in the directory
        """
        # find the dat file in the directory
        file_list = os.listdir(self.directory)
        dat_file = [
            filename for filename in file_list if filename.endswith('.dat')][0]
        df = self.load_data(dat_file)
        return df

    def load_data(self, file_path):
        with open(self.directory+file_path, 'r') as file:
            lines = file.readlines()

        data = []
        current_step = None

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue

            parts = line.split()

            if len(parts) == 2:
                # New timestep
                current_step = int(parts[0])
            elif len(parts) > 2:
                # Data line
                bin_data = list(map(float, parts))
                data.append([current_step] + bin_data)

        columns = ['timestep', 'bin_index', 'coord', 'Ncount', 'vx', 'vy', 'vz', 'c_omegaz', 'density_mass',
                   'v_pxx_loc', 'v_pyy_loc', 'v_pzz_loc', 'v_pxy_loc']
        df = pd.DataFrame(data, columns=columns)

        # Remove rows where all data values are zero for the last bin
        df = df[~((df['bin_index'] == df['bin_index'].max()) & (
            df.drop(['timestep', 'bin_index', 'coord'], axis=1) == 0).all(axis=1))]

        # Multiply 'v_pxx_loc', 'v_pyy_loc', 'v_pzz_loc', 'v_pxy_loc' by 'Ncount'
        df['v_pxx_loc'] = df['v_pxx_loc'] * df['Ncount']
        df['v_pyy_loc'] = df['v_pyy_loc'] * df['Ncount']
        df['v_pzz_loc'] = df['v_pzz_loc'] * df['Ncount']
        df['v_pxy_loc'] = df['v_pxy_loc'] * df['Ncount']

        # Divide 'density_mass' by 1000
        df['density_mass'] = df['density_mass'] / 1000.0

        return df
