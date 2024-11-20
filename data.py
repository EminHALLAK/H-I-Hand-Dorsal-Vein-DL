import os
import shutil

# Path to the main directory that contains the 60 folders
main_directory = "Veins_Dataset"

# Source and new file names
source_name = "4"
new_name = "10"

# Iterate through each folder in the main directory
for folder in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder)

    # Ensure it's a directory
    if os.path.isdir(folder_path):
        # Find the source image
        for file in os.listdir(folder_path):
            if file.startswith(source_name):
                # Construct full file paths for the source and new files
                source_file_path = os.path.join(folder_path, file)
                new_file_path = os.path.join(folder_path, new_name + os.path.splitext(file)[1])

                # Copy and rename the image
                shutil.copy2(source_file_path, new_file_path)
                print(f"Copied and renamed '{source_file_path}' to '{new_file_path}'")
