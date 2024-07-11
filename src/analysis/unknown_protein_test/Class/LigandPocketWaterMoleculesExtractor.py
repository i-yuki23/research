import sys
sys.path.append('../..')

from lib.pdb import get_coordinates_from_pdb, get_atom_id_from_pdb
from lib.path import get_test_gr_path, get_test_water_path
from lib.voxel import get_voxel_info
from lib.voxel import coordinate_to_voxel_index, get_voxel_info
from lib.pdb import get_coordinates_from_pdb, filter_atoms_and_create_new_pdb
import numpy as np
from typing import Tuple

class LigandPocketWaterMoleculesExtractor:

    def __init__(self, protein_name, apo_name):
        self.protein_name = protein_name
        self.apo_name = apo_name
        self.water_coordinates = get_coordinates_from_pdb(get_test_water_path(protein_name, apo_name), type="ATOM")
        self.water_ids = get_atom_id_from_pdb(get_test_water_path(protein_name, apo_name), type="ATOM")
        self.grid_dims, self.grid_origin = get_voxel_info(dx_path=get_test_gr_path(protein_name, apo_name))
        self.__set_convert_dict(self.water_coordinates, self.water_ids, self.grid_origin)

    def __set_convert_dict(self, water_coordinates: np.ndarray, water_ids, grid_origin: np.ndarray) -> None:
        water_voxel_index = coordinate_to_voxel_index(water_coordinates, grid_origin)
        self.water_index_to_coordinate = {tuple(water_voxel_index[i]): water_coordinates[i] for i in range(water_voxel_index.shape[0])}
        self.water_coordinate_to_id = {tuple(water_coordinates[i]): water_id for i, water_id in enumerate(water_ids)}

    def __get_voxelized_water_center(self, water_coordinates: np.ndarray, grid_dims: np.ndarray, grid_origin: np.ndarray) -> np.ndarray:
        water_voxel_index = coordinate_to_voxel_index(water_coordinates, grid_origin)

        voxelized_water_center = np.zeros(grid_dims)
        for index in water_voxel_index:
            voxelized_water_center[index[0], index[1], index[2]] = 1
        return voxelized_water_center
        
    def __convert_voxel_to_water_coordinates(self, water_voxelized: np.ndarray) -> np.ndarray:
        water_index_tmp = np.where(water_voxelized == 1)
        water_index = []
        for i in zip(water_index_tmp[0], water_index_tmp[1], water_index_tmp[2]):
            water_index.append([i[0], i[1], i[2]])

        water_coordinates = []
        for index in water_index:
            water_coordinates.append(self.water_index_to_coordinate[tuple(index)])
        return np.vstack(water_coordinates)
    
    def __convert_coordinates_to_water_ids(self, water_coordinates: np.ndarray) -> list:
        target_water_ids = []
        for coordinate in water_coordinates:
            target_water_ids.append(self.water_coordinate_to_id[tuple(coordinate)])
        return target_water_ids
    

    def __get_water_coordinates_inside_ligand_pocket(self, ligand_pocket) -> np.ndarray:
        voxelized_water_center = self.__get_voxelized_water_center(self.water_coordinates, self.grid_dims, self.grid_origin)
        voxelized_water_center_inside_ligand_pocket = np.where((voxelized_water_center == 1) & (ligand_pocket == 1), 1, 0)
        water_coordinates_inside_ligand_pocket = self.__convert_voxel_to_water_coordinates(voxelized_water_center_inside_ligand_pocket)

        return water_coordinates_inside_ligand_pocket


    def get_water_coords_with_ids_inside_ligand_pocket(self, ligand_pocket) -> Tuple[list, list]:
        water_coordinates_inside_ligand_pocket = self.__get_water_coordinates_inside_ligand_pocket(ligand_pocket)
        water_ids_inside_ligand_pocket = self.__convert_coordinates_to_water_ids(water_coordinates_inside_ligand_pocket)
        return water_coordinates_inside_ligand_pocket, water_ids_inside_ligand_pocket


    def save_water_coordinates_inside_ligand_pocket(self, output_pdb_path: str, water_ids_inside_ligand_pocket: np.ndarray, type: str):
        filter_atoms_and_create_new_pdb(input_pdb_path=get_test_water_path(self.protein_name, self.apo_name), output_pdb_path=output_pdb_path, target_atom_ids=water_ids_inside_ligand_pocket, type=type)

