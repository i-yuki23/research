def get_ligand_path(pdb_name):
    return f"/mnt/dandan/3drism/dir_3DRISM_20181213_155211/{pdb_name}/{pdb_name}_ligand.pdb"

def get_water_path(pdb_name):
    return f"/mnt/ito/3DRISM220308/{pdb_name}/pred_O_placed_{pdb_name}_3.0.pdb"

def get_original_ligand_pocket_path(pdb_name, ligand_voxel_num):
    return f"/home/ito/research/data/ligand_pocket/{pdb_name}/VOXEL_NUM_{ligand_voxel_num}.npy"

def get_ghecom_ligand_pocket_path(pdb_name):
    return f"/home/ito/research/data/ghecom/ligand_pocket/{pdb_name}/ligand_pocket.npy"

def get_protein_path(pdb_name):
    return f"/mnt/dandan/3drism/dir_3DRISM_20181213_155211/{pdb_name}/{pdb_name}_min.pdb"