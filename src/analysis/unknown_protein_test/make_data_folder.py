# holoのprotein: 
#  ligand, 
# water in ligand
# 予測水:pos, neg

# data_folder/apo_name/holo_name/に保存

import os
import shutil

def copy_holo_files(protein_name, apo_name, holo_name, classifying_rule):
    """
    Copy holo files (protein, ligand, water in ligand, pos, neg) to a new folder.
    
    Parameters:
    - data_folder (str): Path to the data folder
    - apo_name (str): Name of the apo folder
    - holo_name (str): Name of the holo folder
    """
    # Define source and destination paths
    holo_source_folder = f"/mnt/ito/test/{protein_name}/holo/{holo_name}"
    prediction_folder = f"/mnt/ito/test/{protein_name}/predicted_labeled_water/{holo_name}/{classifying_rule}"
    dest_folder = f"/mnt/ito/test/{protein_name}/data_folder/{apo_name}/{holo_name}"
    
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # List of files to be copied
    holo_files_to_copy = [f'{holo_name}.pdb', f'{holo_name}_ligand.pdb', f'{holo_name}_{apo_name}_HOH_in_ligand.pdb']
    
    # Copy each file to the destination folder
    for file_name in holo_files_to_copy:
        source_file = os.path.join(holo_source_folder, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, dest_file)
        else:
            print(f"Warning: {source_file} does not exist and will not be copied.")
    
    holo_files_to_copy = [f'neg_pred_O_placed_{apo_name}_3.0.pdb', f'pos_pred_O_placed_{apo_name}_3.0.pdb']
    for file_name in holo_files_to_copy:
        source_file = os.path.join(prediction_folder, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, dest_file)
        else:
            print(f"Warning: {source_file} does not exist and will not be copied.")





