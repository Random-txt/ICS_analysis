import pandas as pd
import os

def excel_save(file_path, global_stats, local_stats, output_file='output_data.xlsx'):
    # Create a dictionary for the data to be saved
    data = {
        'Filepath': [file_path],
        'NbreImages': [len(global_stats.get('raw_tif', []))],  # Number of frames
        'F': [local_stats.get('F')],
        'S': [local_stats.get('S')],
        'Conv': [local_stats.get('Conv')],
        'waist': [global_stats.get('Waist_glob')],
        'BG': [None],  # Set NaN as per the request
        'CRmoy': [local_stats.get('CR_moy')],
        'CRstd': [local_stats.get('CR_std')],
        'CRsem': [local_stats.get('CR_sem')],
        'Nglob': [global_stats.get('N_glob')],
        'Nmoy': [local_stats.get('N_moy')],
        'Nstd': [local_stats.get('N_std')],
        'Nsem': [local_stats.get('N_sem')],
        'CRMglob': [global_stats.get('CRM_glob')],
        'CRMmoy': [local_stats.get('CRM_moy')],
        'CRMstd': [local_stats.get('CRM_std')],
        'CRMsem': [local_stats.get('CRM_sem')],
        'BlanchRelat': [None],  # Add NaN or any value as per the requirement
        'CR_glob': [global_stats.get('CR_glob')],
        'CR_glob_std': [global_stats.get('CR_glob_std')],
        'CR_moy_glob': [global_stats.get('CR_moy_glob')],
        'CR_std_glob': [global_stats.get('CR_std_glob')],
        'N_glob': [global_stats.get('N_glob')],
        'N_std_glob': [global_stats.get('N_std_glob')],
        'Waist_moy': [global_stats.get('Waist_moy')],
        'Waist_std': [global_stats.get('Waist_std')],
        'CRM_std_glob': [global_stats.get('CRM_std_glob')]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    
    # If the Excel file exists, append to it; otherwise, create a new file
    if os.path.exists(output_file):
        # Read the existing Excel file
        existing_df = pd.read_excel(output_file)
        # Concatenate the new data with the existing data
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Save the DataFrame to an Excel file, overwriting it with the updated data
    df.to_excel(output_file, index=False)

    print(f"Data successfully saved to {output_file}")
