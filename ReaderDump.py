import os 
import re
import numpy as np
from DataReader import DataReader


class ReaderDump(DataReader):
    def __init__(self,  cof, ap, parameter=None, pressure = None, value=None, muw=None, vwall=None, fraction=None, phi = None, I = None, o=False):
        super().__init__(cof, ap, parameter, value, pressure, muw, vwall, fraction, phi, I, o)

    def read_data(self, global_path, prefix):
        self.prepare_data(global_path)
        self.filter_relevant_files(prefix)
        self.sort_files_by_time()
        self.get_number_of_time_steps()

    def filter_relevant_files(self, prefix='simple_shear_contact_data'):
        self.file_list = [filename for filename in self.file_list if filename.startswith(prefix) and filename[len(prefix):len(prefix)+1].isdigit() and filename.endswith('.dump')]
