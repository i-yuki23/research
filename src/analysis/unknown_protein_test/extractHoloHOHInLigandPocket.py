from Class.LigandPocketDefiner import LigandPocketDefiner
from modules.get_water_coords_inside_ligand_pocket import save_water_coordinates_inside_ligand_pocket
from lib.pdb import get_coordinates_from_pdb, get_atom_id_from_pdb
from lib.path import get_test_gr_path
from lib.voxel import get_voxel_info

def main():
    protein_name = 'CDK2'
    apo_name = "1pw2"
    holo_name = '1e1v'
    ligand_voxel_num = 10

    ligand_pocket_definer = LigandPocketDefiner(protein_name=protein_name, apo_name=apo_name, holo_name=holo_name, voxel_num=ligand_voxel_num)
    ligand_pocket = ligand_pocket_definer.define_ligand_pocket()

    input_pdb_path = f"/mnt/ito/test/{protein_name}/holo/{holo_name}/{holo_name}_HOH.pdb"
    output_pdb_path = f"/mnt/ito/test/{protein_name}/holo/{holo_name}/{holo_name}_{apo_name}_HOH_in_ligand.pdb"
    water_coordinates = get_coordinates_from_pdb(input_pdb_path, type="HETATM")
    water_ids = get_atom_id_from_pdb(input_pdb_path, type="HETATM")
    grid_dims, grid_origin = get_voxel_info(dx_path=get_test_gr_path(protein_name, apo_name))


    save_water_coordinates_inside_ligand_pocket(input_pdb_path, output_pdb_path, water_coordinates, water_ids, ligand_pocket, grid_dims, grid_origin)    


if __name__ == '__main__':
    main()