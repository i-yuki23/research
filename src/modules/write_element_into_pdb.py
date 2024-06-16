import sys
sys.path.append('..')
from lib.pdb_Bio import get_atom_names

def write_element_into_pdb(pdb_file):
    atom_names = get_atom_names(pdb_file)
    protein_atomic_symbols = [atom_name[0] for atom_name in atom_names]
    index_to_insert = 77  # 挿入したいインデックス

    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    i = 0
    with open(pdb_file, 'w') as file:
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                line = line.rstrip('\n')  # 末尾の改行を削除
                if len(line) < index_to_insert + 1:
                    line = line.ljust(index_to_insert + 1)
                
                new_line = line[:index_to_insert] + protein_atomic_symbols[i] + line[index_to_insert + 1:] + "\n"
                print(new_line[77])
                file.write(new_line)
                i += 1
            else:
                file.write(line)

write_element_into_pdb('/mnt/ito/test/WD5/2h14.pdb')