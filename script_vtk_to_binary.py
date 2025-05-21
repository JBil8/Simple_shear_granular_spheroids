import os
import vtk

def is_ascii_vtk(file_path):
    """Check if a VTK file is in ASCII format by reading the first few bytes safely."""
    with open(file_path, "rb") as file:  # Open in binary mode
        first_bytes = file.read(100)  # Read first 100 bytes

    return b'ASCII' in first_bytes.upper()  # Check if 'ASCII' appears

def get_vtk_dataset_type(file_path):
    """Determine the dataset type of a VTK file."""
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip().lower()
            if line.startswith("dataset"):
                return line.split()[1]  # Extract dataset type (e.g., "polydata" or "unstructured_grid")
    return None  # Unknown type

def convert_vtk_ascii_to_binary(input_vtk_file, output_vtk_file):
    """Convert an ASCII VTK file to binary format while handling different dataset types."""
    dataset_type = get_vtk_dataset_type(input_vtk_file)

    if dataset_type == "polydata":
        reader = vtk.vtkPolyDataReader()
        writer = vtk.vtkPolyDataWriter()
    elif dataset_type == "unstructured_grid":
        reader = vtk.vtkUnstructuredGridReader()
        writer = vtk.vtkUnstructuredGridWriter()
    else:
        print(f"Skipping (Unsupported Dataset Type: {dataset_type}): {input_vtk_file}")
        return

    reader.SetFileName(input_vtk_file)
    reader.Update()
    
    data = reader.GetOutput()
    
    writer.SetFileName(output_vtk_file)
    writer.SetInputData(data)
    writer.SetFileTypeToBinary()
    writer.Write()

def find_and_convert_vtk_files(root_dir, overwrite=False):
    """Recursively find ASCII VTK files and convert them to binary."""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".vtk"):
                # print(f"Processing: {os.path.join(dirpath, filename)}")
                ascii_vtk_file = os.path.join(dirpath, filename)

                if is_ascii_vtk(ascii_vtk_file):
                    binary_vtk_file = ascii_vtk_file if overwrite else os.path.join(dirpath, f"binary_{filename}")
                    convert_vtk_ascii_to_binary(ascii_vtk_file, binary_vtk_file)
                else:
                    print(f"Skipping (Already Binary): {ascii_vtk_file}")

root_directory = "/home/jacopo/Documents/phd_research/Liggghts_simulations/cluster_simulations/"
find_and_convert_vtk_files(root_directory, overwrite=True)  # Set overwrite=False to keep both versions
