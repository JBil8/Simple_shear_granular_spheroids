import argparse
import matplotlib.pyplot as plt

# Import the main analysis class and helper functions
from analyzer import SimulationAnalyzer
from analysis_utils import parse_argument
import config


def main():
    """Parses command-line arguments and runs the simulation analysis."""
    parser = argparse.ArgumentParser(
        description='Process granular simulation data.')
    parser.add_argument('-c', '--cof', required=True, type=float,
                        help='Coefficient of friction (particle-particle)')
    parser.add_argument('-a', '--ap', required=True,
                        type=float, help='Aspect ratio')
    parser.add_argument('-v', '--value', required=True,
                        type=float, help='Packing fraction or Inertial number')
    parser.add_argument('-s', '--pressure', required=True,
                        type=parse_argument, help='Confining pressure')
    parser.add_argument('-np', '--num_processes', type=int, default=config.DEFAULT_NUM_PROCESSES,
                        help='Number of processes for parallel execution')

    args = parser.parse_args()

    # Disable interactive plotting for server-side execution
    plt.ioff()

    # Create an analyzer instance and run the workflow
    analyzer = SimulationAnalyzer(
        cof=args.cof,
        ap=args.ap,
        param=args.value,
        pressure=args.pressure,
        num_processes=args.num_processes
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
