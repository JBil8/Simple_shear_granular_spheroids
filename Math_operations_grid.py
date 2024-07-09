import numpy as np


class MathOperationsGrid:
# This class can be used to compute the divergence and curl of a 2D vector field
# Example usage:
# vx and vy are arrays representing the x and y components of the velocity field
# dx and dy are the grid spacing in the x and y directions

# Compute divergence
# divergence = compute_divergence(vx, vy, dx, dy)

# Compute curl
# curl = compute_curl(vx, vy, dx, dy)
    def __init__(self, nx, ny, dx, dy):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy   

    def compute_divergence(self,vx, vy):
        div_x = np.gradient(vx, axis=0) / self.dx
        div_y = np.gradient(vy, axis=1) / self.dy
        return div_x + div_y
        
    def compute_curl(self, vx, vy):
        curl_x = np.gradient(vy, axis=0) / self.dx - np.gradient(vx, axis=1) / self.dy
        return curl_x

    def compute_eulerian_rmsd(self,results_vtk, averages, n_sim):
        # Initialize dictionary to store cumulative square deviations
        square_deviations = {key: np.zeros_like(results_vtk[0][key]) for key in averages.keys()}

        # Compute cumulative square deviations
        for step_data in results_vtk:
            for key in averages.keys():
                deviation = step_data[key] - averages[key]
                squared_deviation = deviation ** 2
                square_deviations[key] += squared_deviation

        # Divide cumulative square deviations by n_sim to get mean square deviation
        mean_square_deviations = {key: value / n_sim for key, value in square_deviations.items()}
        rmsd = {key: np.sqrt(value) for key, value in mean_square_deviations.items()}
        return rmsd
    
    def compute_spatial_gradient(self, field):
        grad_x = np.gradient(field, axis=0) / self.dx
        grad_y = np.gradient(field, axis=1) / self.dy
        return grad_x, grad_y
