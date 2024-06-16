import sys 
sys.path.append('..')

import numpy as np
from lib.dx import read_dx
from lib.voxel import get_voxel_info
from lib.pdb import get_atom_id_from_pdb, get_coordinates_from_pdb
from modules.fetch_neighboring_voxel import fetch_neighboring_voxel
from modules.get_water_coords_inside_ligand_pocket import get_water_coordinates_inside_ligand_pocket

def _get_base_voxel_data(gr_path):
    gr_voxel, _, _ = read_dx(gr_path)
    return gr_voxel

def __extract_training_voxel_data(water_coordinates, base_voxel, grid_origin, grid_dims, data_voxel_num):
    training_voxel_data_list = []
    for water_coordinate in water_coordinates:
        training_voxel_data = fetch_neighboring_voxel(water_coordinate, base_voxel, grid_origin, grid_dims, data_voxel_num)
        training_voxel_data_list.append(training_voxel_data[:, :, :, np.newaxis])
    training_voxel_data_array = np.array(training_voxel_data_list)
    return training_voxel_data_array

def _get_training_data(water_path, gr_path, grid_origin, grid_dims, data_voxel_num):

    water_coordinates = get_coordinates_from_pdb(water_path)
    gr_voxel = _get_base_voxel_data(gr_path)
    training_data_displaceable = __extract_training_voxel_data(water_coordinates, gr_voxel, grid_origin, grid_dims, data_voxel_num)
    return training_data_displaceable

def save_training_data(water_path, holo_name, gr_path, grid_origin, grid_dims, data_voxel_num=10) -> None:
    
    training_data_displaceable = _get_training_data(water_path, gr_path, grid_origin, grid_dims, data_voxel_num)
    displaceable_water_ids = get_atom_id_from_pdb(water_path)

    for index, one_training_data_displaceable in enumerate(training_data_displaceable):
        save_path_dis = "/mnt/ito/test/WD5/test_data/" + holo_name + f"/water_id_{displaceable_water_ids[index]}.npy"
        np.save(save_path_dis, one_training_data_displaceable[:, :, :, :])

def main():

    test_dir = "/mnt/ito/test/WD5/"
    pdb_name = "2h14"
    holo_name = "3smrA"

    dx_path = test_dir + pdb_name + "_min.dx"
    ligand_pocket_path = test_dir + "ligand_pocket/" + holo_name + ".npy"
    water_inside_ligand_pocket_path = test_dir + f"water_inside_ligand_pocket/{holo_name}/" + f"pred_O_placed_{pdb_name}_3.0.pdb"


    # water_coords_inside_ligand_pocket = get_coordinates_from_pdb(water_inside_ligand_pocket_path)
    # ligand_pocket = np.load(ligand_pocket_path)
    grid_dims, grid_origin = get_voxel_info(dx_path=dx_path)


    save_training_data(water_inside_ligand_pocket_path, holo_name, dx_path, grid_origin, grid_dims, data_voxel_num=10)
    # print(water_coords_inside_ligand_pocket.shape)

if __name__ == '__main__':
    main()