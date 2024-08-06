import sys
sys.path.append('../..')
from lib.helper import make_dir
from lib.pdb import get_all_pdb_names
from lib.path import get_protein_path, get_ligand_path

def combine_pdb_files(pdb_file1, pdb_file2, output_file):
    def read_pdb(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return [line for line in lines if line.startswith(('ATOM', 'HETATM'))]

    def write_pdb(lines, output_path):
        with open(output_path, 'w') as file:
            for line in lines:
                file.write(line)
            file.write('END\n')

    # Read the PDB files
    pdb1_lines = read_pdb(pdb_file1)
    pdb1_lines.append('TER\n')
    pdb2_lines = read_pdb(pdb_file2)

    # Get max atom serial and residue sequence number from pdb1
    max_atom_serial = max(int(line[6:11]) for line in pdb1_lines if line.startswith(('ATOM', 'HETATM')))
    max_residue_seq = max(int(line[22:26].strip()) for line in pdb1_lines if line.startswith(('ATOM', 'HETATM')))

    # Adjust atom serial numbers and residue sequence numbers for pdb2
    updated_pdb2_lines = []
    for line in pdb2_lines:
        if line.startswith(('ATOM', 'HETATM')):
            atom_serial = int(line[6:11].strip()) + max_atom_serial
            residue_seq_str = line[22:26].strip()
            if residue_seq_str:
                residue_seq = int(residue_seq_str) + max_residue_seq
            else:
                residue_seq = max_residue_seq + 1
                max_residue_seq += 1

            updated_line = f"{line[:6]}{atom_serial:5}{line[11:17]}{'LIG':>3}{line[20:22]}{residue_seq:4}{line[26:]}"
            symbol = line[76:78].strip()
            if symbol == 'A':
                symbol = ' C'
                updated_line = updated_line[:76] + symbol + updated_line[78:]
            updated_pdb2_lines.append(updated_line)
        elif line.startswith('TER'):
            # updated_line = f"{line[:6]}{int(line[6:11]):5} {line[11:]}".rstrip() + '\n'
            # updated_pdb2_lines.append(updated_line)
            updated_pdb2_lines.append('TER\n')

    # Combine the lines
    combined_lines = pdb1_lines + updated_pdb2_lines

    # Write the combined PDB file
    write_pdb(combined_lines, output_file)

# Usage example
protein_names = get_all_pdb_names()

for pdb_name in protein_names:
    output_path = f'/home/ito/research/data/protein_ligand_complex/{pdb_name}/{pdb_name}_complex.pdb'
    # make_dir(output_path)
    combine_pdb_files(get_protein_path(pdb_name), get_ligand_path(pdb_name), output_path)

