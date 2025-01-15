import os
from PIL import Image

folder_path = "/home/jacopo/Documents/PhD_research/Data_processing_simple_shear/output_plots_stress_hertz"  
output_folder = "/home/jacopo/Documents/PhD_research/Data_processing_simple_shear/gifs"  # Folder to save GIFs
os.makedirs(output_folder, exist_ok=True)

# Parameters
# prefix = "ellipsoid_Local force normal contact point normalized pressure"
prefix = "ellipsoid_Ratio tangential to normal force"
cofs = [0.4, 1.0]
Is = [0.1, 0.046, 0.022, 0.01, 0.0046, 0.0022, 0.001]
alphas = [0.33, 0.40, 0.50, 0.56, 0.67, 0.83, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

# Function to create GIFs
def create_gif(file_list, output_path, frame_duration=2):
    images = []
    for file_name in file_list:
        # Open image using Pillow to ensure consistency
        img = Image.open(os.path.join(folder_path, file_name)).convert("RGB")
        images.append(img)

    # Save GIF with the specified duration
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration * 500,  # Frame duration in milliseconds
        loop=0,
    )

# Loop over configurations and create GIFs
# 1. Loop over alphas for fixed I and cof
for cof in cofs:
    for I in Is:
        matching_files = []
        for alpha in alphas:
            # Construct the exact filename
            file_name = f"{prefix}_alpha_{alpha}_cof_{cof}_I_{I}.png"
            if file_name in os.listdir(folder_path):  # Check if file exists
                matching_files.append(file_name)
        if matching_files:
            output_file = os.path.join(output_folder, f"{prefix}_cof_{cof}_I_{I}.gif")
            create_gif(matching_files, output_file)

# # 2. Loop over cofs for fixed I and alpha
# for I in Is:
#     for alpha in alphas:
#         matching_files = []
#         for cof in cofs:
#             # Construct the exact filename
#             file_name = f"{prefix}_alpha_{alpha}_cof_{cof}_I_{I}.png"
#             if file_name in os.listdir(folder_path):  # Check if file exists
#                 matching_files.append(file_name)
#         if matching_files:
#             output_file = os.path.join(output_folder, f"{prefix}_alpha_{alpha}_I_{I}.gif")
#             create_gif(matching_files, output_file)

# # 3. Loop over Is for fixed alpha and cof
# for alpha in alphas:
#     for cof in cofs:
#         matching_files = []
#         for I in Is:
#             # Construct the exact filename
#             file_name = f"{prefix}_alpha_{alpha}_cof_{cof}_I_{I}.png"
#             if file_name in os.listdir(folder_path):  # Check if file exists
#                 matching_files.append(file_name)
#         if matching_files:
#             output_file = os.path.join(output_folder, f"{prefix}_alpha_{alpha}_cof_{cof}.gif")
#             create_gif(matching_files, output_file)



print(f"GIFs have been created and saved in {output_folder}.")
