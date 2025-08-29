import h5py
import numpy as np
import os

def merge_h5(data_folders, file_name, output_folder, chunk_size=1000):
    """Function that merges h5 files efficiently by processing in chunks.
    INPUT:
        data_folders: list of paths to folders containing data
        file_name: 'train_data.h5' or 'val_data.h5' name of the h5 file to merge
        output_folder: path to the final dataset folder
        chunk_size: number of entries to process at once, adjust for memory efficiency
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, file_name)

    # Determine dataset shapes from the first file to initialize datasets in output file
    first_file = os.path.join(data_folders[0], file_name)
    with h5py.File(first_file, 'r') as file:
        coeff_shape = file['coefficients'].shape[1:]  # excluding first dim (num samples)
        control_shape = file['control'].shape[1:]
        data_shape = file['data'].shape[1:]

    # Open the output file for writing
    with h5py.File(output_file, 'w') as outfile:
        # Create datasets with unknown initial size, allowing resizing along the first dimension
        coeff_ds = outfile.create_dataset('coefficients', shape=(0,) + coeff_shape, maxshape=(None,) + coeff_shape)
        control_ds = outfile.create_dataset('control', shape=(0,) + control_shape, maxshape=(None,) + control_shape)
        data_ds = outfile.create_dataset('data', shape=(0,) + data_shape, maxshape=(None,) + data_shape)

        # Append data in chunks
        for folder in data_folders:
            file_path = os.path.join(folder, file_name)
            with h5py.File(file_path, 'r') as file:
                num_samples = file['coefficients'].shape[0]
                
                # Process in chunks
                for i in range(0, num_samples, chunk_size):
                    end_i = min(i + chunk_size, num_samples)
                    
                    # Load chunk from current file
                    coeff_chunk = file['coefficients'][i:end_i]
                    control_chunk = file['control'][i:end_i]
                    data_chunk = file['data'][i:end_i]
                    
                    # Resize datasets in output file
                    coeff_ds.resize(coeff_ds.shape[0] + coeff_chunk.shape[0], axis=0)
                    control_ds.resize(control_ds.shape[0] + control_chunk.shape[0], axis=0)
                    data_ds.resize(data_ds.shape[0] + data_chunk.shape[0], axis=0)

                    # Write chunk to output file
                    coeff_ds[-coeff_chunk.shape[0]:] = coeff_chunk
                    control_ds[-control_chunk.shape[0]:] = control_chunk
                    data_ds[-data_chunk.shape[0]:] = data_chunk

    print(f"Data merged successfully into {output_file}")

def merge_prefix(data_folders, file_name, output_folder):
    """Function that merges prefix text files by appending each file’s contents.
    INPUT:
        data_folders: list of paths to folders containing data
        file_name: 'train_text.prefix' or 'val_text.prefix' name of the file to merge
        output_folder: path to final dataset folder
    """
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, file_name)

    # Open the output file in write mode
    with open(output_file, 'w') as outfile:
        # Loop through each folder and concatenate text files
        for folder in data_folders:
            file_path = os.path.join(folder, file_name)
            
            # Check if file exists before processing
            if os.path.exists(file_path):
                with open(file_path, 'r') as infile:
                    # Read file content and strip trailing newlines
                    content = infile.read().rstrip()
                    # Write content to output file with a newline
                    outfile.write(content + '\n')
            else:
                print(f"File {file_path} does not exist, skipping.")

    print(f"Text files merged successfully into {output_file}")

# Paths for datasets
folders = ['/home/elisa/code/icon-gen/Paper_datasets/1D_ODE/', 
           '/home/elisa/code/icon-gen/Paper_datasets/2D_ODE/',
           '/home/elisa/code/icon-gen/Paper_datasets/3D_ODE/',
           "/home/elisa/code/icon-gen/Paper_datasets/PDE",
           "/home/elisa/code/icon-gen/Paper_datasets/Cons_laws",
           "/home/elisa/code/icon-gen/Paper_datasets/Cons_laws_shocks"]

output_folder = '/home/elisa/code/icon-gen/Paper_datasets/All_ODEsPDEs'

# Run merge functions
merge_h5(folders, 'train_data.h5', output_folder)
merge_h5(folders, 'val_data.h5', output_folder)
merge_prefix(folders, "train_text.prefix", output_folder)
merge_prefix(folders, "val_text.prefix", output_folder)




# import h5py
# import numpy as np
# import os
# from glob import glob


# def merge_h5(data_folders, file_name, output_folder):
#     """functioin that merges h5 files
#     INPUT: data_folders: list containing the path to the folders to the data
#     file_name: 'train_data.h5' or 'val_data.h5' this is also the name of the merged data 
#     output_folder: path to final dataset folder"""
#     # Initialize lists to collect data from all files
#     all_coefficients = []
#     all_control = []
#     all_data = []

#     # Loop through each folder and load the val_data.h5 file
#     for folder in data_folders:
#         file_path = os.path.join(folder, file_name)
        
#         with h5py.File(file_path, 'r') as file:
#             # Load the datasets from each val_data.h5 file
#             coefficients = file['coefficients'][:]
#             control = file['control'][:]
#             data = file['data'][:]
            
#             # Append the data to the corresponding lists
#             all_coefficients.append(coefficients)
#             all_control.append(control)
#             all_data.append(data)

#     # Concatenate the datasets from all files
#     merged_coefficients = np.concatenate(all_coefficients, axis=0)
#     merged_control = np.concatenate(all_control, axis=0)
#     merged_data = np.concatenate(all_data, axis=0)

#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Write the concatenated data to a new val_data.h5 file in the fullDataset folder
#     output_file = os.path.join(output_folder, file_name)

#     with h5py.File(output_file, 'w') as outfile:
#         outfile.create_dataset('coefficients', data=merged_coefficients)
#         outfile.create_dataset('control', data=merged_control)
#         outfile.create_dataset('data', data=merged_data)

#     print(f"Data merged successfully into {output_file}")
#     return

# def merge_prefix(data_folders, file_name, output_folder):
#     """functioin that merges prefix files
#     INPUT: data_folders: list containing the path to the folders to the data
#     file_name: 'train_text.prefix' or 'val_text.prefix' this is also the name of the merged data 
#     output_folder: path to final dataset folder"""
    
#     os.makedirs(output_folder, exist_ok=True)
#     output_file = os.path.join(output_folder, file_name)

#     # Open the output file in write mode
#     with open(output_file, 'w') as outfile:
#         # Loop through each folder and concatenate the train_text.prefix files
#         for folder in data_folders:
#             file_path = os.path.join(folder, file_name)
            
#             # Check if the file exists before processing
#             if os.path.exists(file_path):
#                 with open(file_path, 'r') as infile:
#                     # Read the file content, strip any trailing newlines
#                     content = infile.read().rstrip()
#                     # Write the stripped content to the output file
#                     outfile.write(content + '\n')  # Add a single newline to separate each file's content
#             else:
#                 print(f"File {file_path} does not exist, skipping.")
        

#     print(f"Text files merged successfully into {output_file}")

# # List of directories containing val_data.h5 files (adjust the pattern to match your folder structure)
# folders = ['/home/elisa/code/icon-gen/Paper_datasets/1D_ODE/', '/home/elisa/code/icon-gen/Paper_datasets/2D_ODE/',"/home/elisa/code/icon-gen/Paper_datasets/PDE",
#            "/home/elisa/code/icon-gen/Paper_datasets/Cons_laws","/home/elisa/code/icon-gen/Paper_datasets/Cons_laws_shocks"] #sorted(glob('/home/elisa/code/icon-gen/try_merge/*/'))  # Adjust the path to your folder structure
# print(folders)
# output_folder = '/home/elisa/code/icon-gen/ODEs_dataset/ODEPDE'
# merge_h5(folders, 'train_data.h5', output_folder)
# merge_h5(folders, 'val_data.h5', output_folder)
# merge_prefix(folders, "train_text.prefix", output_folder)
# merge_prefix(folders, "val_text.prefix", output_folder)