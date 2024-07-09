import numpy as np
import pickle as pkl

class DataExporter:
    def __init__(self, ap , cof, parameter=None, value=None, muw=None, vwall=None, fraction=None, phi = None, I = None):
        self.ap = str(ap) 
        self.cof = str(cof)
        self.parameter = str(parameter)
        self.value = str(value)
        self.muw = str(muw)
        self.vwall = str(vwall)
        self.fraction = str(fraction)
        self.phi = str(phi)
        self.I = str(I)

    def export_with_pickle(self, data):
        # export the data with pickle for further analysis with appropriate name
        with open('output_data/simple_shear_ap' + self.ap + '_cof_' + self.cof + '_I_' + self.I + '.pkl', 'wb') as f:
            pkl.dump(data, f)

    def import_with_pickle(self):
        # import the data with pickle for further analysis with appropriate name
        with open('output_data/simple_shear_ap' + self.ap + '_cof_' + self.cof + '_I_' + self.I + '.pkl', 'wb') as f:
            data = pkl.load(f)
        return data

    def export_with_pickle_obstructed(self, averages, msd):
        # export the data with pickle for further analysis
        with open('output_data/obstruction_shear_ap' + self.ap + '_cof_' + self.cof + 
                  '_vw_' + self.vwall + '_obfr_' +self.fraction + '_phi_' + self.phi +'.pkl', 'wb') as f:
            pkl.dump(averages, f)
        with open('output_data/obstruction_shear_ap' + self.ap + '_cof_' + self.cof +
                    '_vw_' + self.vwall + '_obfr_' +self.fraction + '_phi_' + self.phi +'_msd.pkl', 'wb') as f:
                pkl.dump(msd, f)

    def import_with_pickle_obstructed(self):
        # import the data with pickle for further analysis
        with open('output_data/obstruction_shear_ap' + self.ap + '_cof_' + self.cof + 
                  '_vw_' + self.vwall + '_obfr_' +self.fraction + '_phi_' + self.phi +'.pkl', 'rb') as f:
            averages = pkl.load(f)
        return averages

    def export_data(self, data_vtk, data_dump):
        # export data to csv files for further analysis
        # export the data from data_vtk and data_dump
        strain = np.arange(1, data_vtk['theta_x'].shape[0]+1)*20/data_vtk['theta_x'].shape[0]
        
        #combine the two dictionaries into one with all the keys
        data_vtk.update(data_dump)
        all_data = [strain]
        key_list = []
        #export the data to csv files for all the keys except the last one in the dictionary
        for key in data_vtk.keys() - ['trackedGrainsContactData']:
            if data_vtk[key].ndim == 1:
                key_list.append(key)
                # for 1D arrays, add the data to the list
                all_data.append(data_vtk[key])

        np.savetxt(
            'output_data/simple_shear_ap' + self.ap + '_cof_' + self.cof + '_' + self.parameter + '_' + self.value + '.csv',
            np.transpose(all_data),
            delimiter=',',
            header=','.join(['strain'] + key_list)
        )

        # export eulerian velocities as csv with rows for each time step
        np.savetxt(
            'output_data/simple_shear_ap' + self.ap + '_cof_' + self.cof + '_' + self.parameter + '_' + self.value + '_eulerian_velocities.csv',
            data_vtk['eulerian_vx'],
            delimiter=','
        )
        
    def export_force_distribution(self, force_normal_distribution, force_tangential_distribution):
        # export the force distribution numpy histograms to csv files for further analysis
        with open('output_data/simple_shear_ap' + self.ap + '_cof_' + self.cof + '_' + self.parameter + '_' + self.value + '_force_normal.pkl', 'wb') as f:
            pkl.dump(force_normal_distribution, f)

        with open('output_data/simple_shear_ap' + self.ap + '_cof_' + self.cof + '_' + self.parameter + '_' + self.value + '_force_tangential.pkl', 'wb') as f:
            pkl.dump(force_tangential_distribution, f)

       

