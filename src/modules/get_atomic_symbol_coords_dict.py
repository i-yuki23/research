from lib.pdb import get_coordinates_from_pdb, get_atomic_symbols_from_pdb

def get_atomic_symbol_coords_dict_from_pdb(pdb_path):
    coords = get_coordinates_from_pdb(pdb_path)
    atomic_symbols = get_atomic_symbols_from_pdb(pdb_path)

    atomic_symbol_coords_dict = {}
    for atomic_symbol, coord in zip(atomic_symbols, coords):
        if atomic_symbol in atomic_symbol_coords_dict:
            atomic_symbol_coords_dict[atomic_symbol].append(coord)
        else:
            atomic_symbol_coords_dict[atomic_symbol] = [coord]
    return atomic_symbol_coords_dict