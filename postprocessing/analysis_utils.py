import numpy as np
import argparse
from histogram_utils import *


def parse_argument(value):
    try:
        int_value = int(value)
        if float(value) == int_value:
            return int_value
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid argument value: {value}")


def time_step_used(shear_rate, ap, factor, Young=5e6, rho=1000, nu=0.3, small_axis=1.0, shear_mod=1e6):

    long_axis = ap * small_axis
    if ap > 1:
        avg_diameter = (2 * small_axis + long_axis) / 3 * 2
    else:
        avg_diameter = (small_axis + 2 * long_axis) / 3 * 2
    v_collision = 2 * shear_rate * avg_diameter
    hertz_factor = 2.943 * (5 * np.sqrt(2) * np.pi *
                            rho * (1 - nu**2) / (4 * Young))**(2 / 5)
    min_dimen = min(small_axis, long_axis)
    dt_hertz = 2 * factor * hertz_factor * min_dimen / (v_collision)**(1 / 5)
    dt_rayleigh = 2 * factor * np.pi * min_dimen * \
        np.sqrt(rho / shear_mod) * (1 / (0.8766 + 0.1631 * nu))
    return min(dt_hertz, dt_rayleigh)


def average_correlation_data(results, y_value_key, density_key,
                             box_length_key='box_y_length', num_points=100, grid_type='uniform'):
    """
    Correctly averages time-series correlation data from simulations with variable box sizes.

    This function can create either a uniform grid (for data like C(y)) or a
    non-uniform grid that respects an initial data gap (for radial correlations).

    Parameters:
        results (list[dict]):
            A list of dictionary objects, where each dict represents a single time step's output.
        y_value_key (str):
            The dictionary key for the correlation distance values (the x-axis).
        density_key (str):
            The dictionary key for the correlation function values (the y-axis).
        box_length_key (str, optional):
            The dictionary key for the box lengths. Used for the 'uniform' grid type.
            Assumes the y-dimension is the second element. Defaults to 'box_lengths'.
        num_points (int, optional):
            The number of points for the 'uniform' grid. Defaults to 100.
        grid_type (str, optional):
            The type of common grid to generate for interpolation.
            - 'uniform': Creates a uniformly spaced grid from 0 to half the average box length.
                        Ideal for density correlation C(y).
            - 'non_uniform_from_data': Creates a grid that mimics the data's structure,
                        including a potential gap after r=0. Ideal for radial correlations
                        where short distances are skipped.
            Defaults to 'uniform'.

    Returns:
        tuple[np.ndarray, np.ndarray] or tuple[None, None]:
            A tuple containing:
            - common_grid (np.ndarray): The common, standardized grid for the x-axis.
            - averaged_correlation (np.ndarray): The correctly averaged correlation data.
            Returns (None, None) if the specified keys are not found in the results.
    """
    # --- 1. Check if data exists ---
    if not results or y_value_key not in results[0] or density_key not in results[0]:
        print(
            f"Warning: Keys '{y_value_key}' or '{density_key}' not found in results. Skipping averaging.")
        return None, None

    # --- 2. Define a common grid for interpolation based on grid_type ---
    if grid_type == 'uniform':
        # This method is ideal for the C(y) density correlation
        avg_length = np.mean([res[box_length_key][1]
                             for res in results if box_length_key in res])
        common_grid = np.linspace(0, avg_length / 2, num_points)

    elif grid_type == 'non_uniform_from_data':
        # This method is ideal for your spatial velocity autocorrelation
        min_dists, spacings, max_dists = [], [], []

        for res in results:
            r_values = res[y_value_key]
            if len(r_values) > 1:
                # Find the first non-zero distance to determine the gap size
                first_positive_idx = np.searchsorted(r_values, 0, side='right')
                if first_positive_idx < len(r_values):
                    min_dists.append(r_values[first_positive_idx])

                # Calculate the average spacing of the rest of the points
                if len(r_values) > first_positive_idx + 1:
                    spacings.append(
                        np.mean(np.diff(r_values[first_positive_idx:])))

            if len(r_values) > 0:
                max_dists.append(r_values[-1])

        # Calculate the average properties of the grid
        avg_min_dist = np.mean(min_dists) if min_dists else 0.1
        avg_spacing = np.mean(spacings) if spacings else 0.1
        avg_max_dist = np.mean(max_dists) if max_dists else 1.0

        # Construct the non-uniform grid
        tail_grid = np.arange(avg_min_dist, avg_max_dist, avg_spacing)
        # Assume the data always includes a point for r=0
        common_grid = np.concatenate(([0], tail_grid))
    else:
        raise ValueError(
            "grid_type must be 'uniform' or 'non_uniform_from_data'")

    # --- 3. Interpolate each result onto the common grid ---
    interpolated_correlation_list = []
    for res in results:
        original_x_values = res[y_value_key]
        original_y_values = res[density_key]

        # np.interp handles both uniform and non-uniform grids perfectly
        interpolated_correlation = np.interp(
            common_grid, original_x_values, original_y_values)
        interpolated_correlation_list.append(interpolated_correlation)

    # --- 4. Average the interpolated data ---
    averaged_correlation = np.mean(interpolated_correlation_list, axis=0)

    return common_grid, averaged_correlation


def process_results(results, bins_config, correlation_keys, area_adjustments_ellipsoid, n_sim):
    """
    Processes raw parallel results to compute averaged values, distributions, and PDFs.
    """
    # --- 1. Initialization ---
    histogram_sums = {key: np.zeros(bins) for key, bins in bins_config.items()}
    averages = {}
    distributions = {}

    # --- 2. Process Correlation Data ---
    for x_key, y_keys in correlation_keys.items():
        y_keys_list = y_keys if isinstance(y_keys, list) else [y_keys]
        for y_key in y_keys_list:
            avg_x, avg_y = average_correlation_data(
                results, x_key, y_key, grid_type='non_uniform_from_data')
            if avg_x is not None:
                averages[x_key] = avg_x
                averages[y_key] = avg_y

    # --- 3. Aggregate Histograms, Distributions, and Scalar Averages ---
    processed_keys = set(correlation_keys.keys())
    for y_list in correlation_keys.values():
        processed_keys.update(y_list if isinstance(y_list, list) else [y_list])

    if results:
        for key in results[0].keys():
            if key in processed_keys:
                continue
            if key in histogram_sums:
                histogram_sums[key] = np.sum(
                    [res[key] for res in results], axis=0)
            elif key in ['trackedGrainsOrientation', 'trackedGrainsPosition', 'thetax', 'thetaz']:
                distributions[key] = np.concatenate(
                    [res[key] for res in results])
            elif key in ['vy_velocity', 'omegaz_velocity']:
                distributions[key] = np.stack(
                    [res[key] for res in results], axis=1)
            else:
                averages[key] = np.mean([res[key] for res in results], axis=0)

    distributions['thetax_particles'] = np.stack(
        [result['thetax'] for result in results], axis=1)

    # --- 4. Compute Weighted Average Histograms ---
    histograms_weighted_avg = {}

    weight_map = {
        'global_normal_force_hist': 'contacts_hist_global_normal',
        'global_tangential_force_hist': 'contacts_hist_global_tangential',
        'local_normal_force_hist_cp': 'contacts_hist_cont_point_local',
        'local_tangential_force_hist_cp': 'contacts_hist_cont_point_local',
        'global_normal_force_hist_cp': 'contacts_hist_cont_point_global',
        'global_tangential_force_hist_cp': 'contacts_hist_cont_point_global',
        'normal_force_hist_mixed': 'counts_mixed',
        'tangential_force_hist_mixed': 'counts_mixed',
    }

    # Process all standard weighted averages in a loop
    for key, weight_key in weight_map.items():
        histograms_weighted_avg[key] = compute_weighted_average_hist(
            histogram_sums[key], histogram_sums[weight_key]
        )

    # Handle the special cases that use n_sim as the weight
    for key in ['power_dissipation_normal', 'power_dissipation_tangential']:
        histograms_weighted_avg[key] = compute_weighted_average_hist(
            histogram_sums[key], n_sim
        )

    # --- 5. Compute Probability Density Functions (PDFs) ---
    pdfs = {}
    bins_global = bins_config['contacts_hist_global_normal']
    pdfs['contacts_hist_global_normal'] = compute_pdf(
        histogram_sums['contacts_hist_global_normal'], 360 / bins_global)
    pdfs['contacts_hist_global_tangential'] = compute_pdf(
        histogram_sums['contacts_hist_global_tangential'], 360 / bins_global)
    pdfs['contacts_hist_cont_point_global'] = compute_pdf(
        histogram_sums['contacts_hist_cont_point_global'], 360 / bins_global)
    pdfs['contacts_hist_cont_point_local'] = compute_pdf_on_ellipsoid(
        histogram_sums['contacts_hist_cont_point_local'], area_adjustments_ellipsoid)
    pdfs['bin_counts_power'] = compute_pdf_on_ellipsoid(
        histogram_sums['bin_counts_power'], area_adjustments_ellipsoid)

    return averages, histograms_weighted_avg, pdfs, distributions


def compute_autocorrelation_function(property_data):  # Renamed here
    """
    Compute the autocorrelation function as a function of strain for the y-velocity component.
    """
    n_particles, n_strains = property_data.shape  # And here

    fft_property = np.fft.fft(property_data, n=2*n_strains, axis=1)  # And here
    power_spectrum = fft_property * np.conjugate(fft_property)

    autocorr = np.fft.ifft(power_spectrum, axis=1).real[:, :n_strains]

    autocorr_per_particle = autocorr / n_strains

    avg_autocorr = np.mean(autocorr_per_particle, axis=0)

    mean_squared_fluctuations = np.mean(property_data**2)  # And here

    avg_autocorr /= mean_squared_fluctuations

    return avg_autocorr
