import sys
sys.path.append('..')

from preprocess.Class.WaterClassifier.LigandPocketDefinerOriginal import LigandPocketDefinerOriginal
from lib.voxel import get_voxel_info
from lib.pdb import get_all_pdb_names
import numpy as np

ligand_pocket_size = []
for pdb_name in get_all_pdb_names():
    grid_dims, grid_origin = get_voxel_info(pdb_name)
    ligand_pocket_definer = LigandPocketDefinerOriginal(pdb_name, grid_dims, grid_origin, 8)
    ligand_pocket = ligand_pocket_definer.define_ligand_pocket()
    ligand_pocket_size.append(ligand_pocket.sum())

np.save('ligand_pocket_size.npy', np.array(ligand_pocket_size))

