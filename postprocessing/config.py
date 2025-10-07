import os
import yaml

# Load YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(os.path.expandvars(f.read()))

# Environment / cluster configs
GLOBAL_PATH = cfg["sim_data_dir"]
DEFAULT_NUM_PROCESSES = cfg["cpus_per_task"]

# --- Simulation Parameters ---
INERTIAL_NUMBER_THRESHOLD = 0.01
SHEAR_ONE_INDEX_HIGH_I = 1200  # 600
SHEAR_ONE_INDEX_LOW_I = 1200  # 400

# --- Analysis Parameters ---
NUM_BINS_GLOBAL = 144
NUM_BINS_LOCAL = 10
NUM_BINS_ORIENTATION = 180
AUTOCORR_STRAIN_POINTS = 600

BINS_CONFIG = {
    'global_normal_force_hist': NUM_BINS_GLOBAL,
    'global_tangential_force_hist': NUM_BINS_GLOBAL,
    'local_normal_force_hist_cp': NUM_BINS_LOCAL,
    'local_tangential_force_hist_cp': NUM_BINS_LOCAL,
    'global_normal_force_hist_cp': NUM_BINS_GLOBAL,
    'global_tangential_force_hist_cp': NUM_BINS_GLOBAL,
    'contacts_hist_cont_point_global': NUM_BINS_GLOBAL,
    'contacts_hist_cont_point_local': NUM_BINS_LOCAL,
    'contacts_hist_global_normal': NUM_BINS_GLOBAL,
    'contacts_hist_global_tangential': NUM_BINS_GLOBAL,
    'power_dissipation_normal': NUM_BINS_LOCAL,
    'power_dissipation_tangential': NUM_BINS_LOCAL,
    'bin_counts_power': NUM_BINS_LOCAL,
    'normal_force_hist_mixed': NUM_BINS_LOCAL**2,
    'tangential_force_hist_mixed': NUM_BINS_LOCAL**2,
    'counts_mixed': NUM_BINS_LOCAL**2,
}

CORRELATION_KEYS = {
    'c_y_values': 'c_density_y',
    'c_r_values': ['c_delta_vy', 'c_delta_omega_z']
}
