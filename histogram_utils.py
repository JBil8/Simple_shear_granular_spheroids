import numpy as np
from scipy.integrate import quad


def compute_histogram_median(hist):
    """Compute the median value of a histogram."""
    # Calculate the cumulative sum of the histogram
    cumulative_hist = np.cumsum(hist)

    # Find the index where the cumulative sum crosses half of the total count
    total_count = cumulative_hist[-1]
    median_index = np.searchsorted(cumulative_hist, total_count / 2)

    # Get the value corresponding to the median index
    return hist[median_index]

def compute_histogram_sum(results, key):
    """Compute the sum of histograms for a given key in a vectorized manner."""
    # Stack all histograms into a 2D array
    histograms = np.array([result[key] for result in results])
    
    # Sum the histograms along the first axis
    hist_sum = np.sum(histograms, axis=0)
    
    return hist_sum

def compute_histogram_avg(hist_sum, num_results):
    """Compute the average histogram."""
    return np.divide(hist_sum, num_results, where=num_results != 0)

def compute_weighted_average_hist(hist_values, hist_counts):
    return np.divide(hist_values, hist_counts , where=hist_counts != 0)

def compute_pdf(hist_sum):
    """Compute the PDF for a histogram."""
    total_count = np.sum(hist_sum)
    return hist_sum / total_count if total_count != 0 else hist_sum

def compute_pdf_on_ellipsoid(hist_sum, max_angle = 90):
    """Compute the PDF for a histogram, adjusted for ellipsoid surface area."""
    bin_edges = np.linspace(0, max_angle, len(hist_sum)+1)
    
    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate the surface area adjustment (proportional to sin(theta))
    surface_area_adjustment = np.sin(np.radians(bin_centers))  # Assuming bin_centers are in degrees
    
    # Adjust the histogram by the surface area
    adjusted_hist = hist_sum / surface_area_adjustment
    
    # Normalize the adjusted histogram to get the PDF
    total_count = np.sum(adjusted_hist)
    pdf = adjusted_hist / total_count if total_count != 0 else adjusted_hist
    
    return pdf

def area_adjustment_ellipsoid(n_bins, ap):
    """
    assuming symmetry along the positive and negative axis of revolution
    Gives the surface area adjustment for each bin of the histogram
    and the total surface area of the ellipsoid
    """

    bin_edges = np.linspace(0, np.pi/2, n_bins+1)
    
    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate total surface area considering small dimension is equal to one
    if ap>1: #prolate
        ecc = np.sqrt(1-1/ap**2)
        total_area = 2*np.pi*(1+ap/ecc*np.arcsin(ecc))
    elif ap==1: #sphere
        total_area = 4*np.pi
    else: #oblate
        ecc = np.sqrt(1-ap**2)
        total_area = 2*np.pi/ap**2*(1+(1-ecc**2)/ecc*np.arctanh(ecc))

    # Calculate the surface area adjustment for each polar section
    if ap>=1:
        # surface_area_function = lambda theta: 2*np.pi*np.sin(theta)*np.sqrt(np.cos(theta)**2 + ap**2*np.sin(theta)**2)
        surface_area_function = lambda theta: 2*np.pi*np.sin(theta)*np.sqrt((1+ap**2+(1-ap**2)*np.cos(2*theta))/2)
    else:
        surface_area_function = lambda theta: 2*np.pi/ap*np.sin(theta)*np.sqrt((1+1/ap**2+(1/ap**2-1)*np.cos(2*theta))/2)
        
    surface_area_adjustment = np.diff(bin_edges)*surface_area_function(bin_centers) #trapezoidal rule

    return surface_area_adjustment, total_area

def normalize_histogram_forces(force_hist_avg, pdf, pressure, box_length_x, box_length_z):
    """
    Same normalization as in Rheophysics (Da Cruz et al. 2005)
    """
    force_hist_avg_normalized = pdf*force_hist_avg/(pressure*box_length_x*box_length_z)
    return force_hist_avg_normalized
