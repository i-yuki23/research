import sys
sys.path.append('..')
from lib.helper import make_dir
from lib.pdb import get_all_pdb_names
from lib.path import get_protein_path, get_ligand_path

import os

def combine_pdb_files_by_chain(pdb_file1, pdb_file2, output_dir):
    def read_pdb(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines

    def write_pdb(lines, output_path):
        with open(output_path, 'w') as file:
            for line in lines:
                file.write(line)
            file.write('END\n')

    def split_chains(lines):
        chains = {}
        current_chain = []
        current_chain_id = None
        for line in lines:
            if line.startswith('ATOM'):
                chain_id = line[21]
                if current_chain_id is None:
                    current_chain_id = chain_id
                if chain_id != current_chain_id:
                    chains[current_chain_id] = current_chain
                    current_chain = []
                    current_chain_id = chain_id
                current_chain.append(line)
        if current_chain:
            chains[current_chain_id] = current_chain
        return chains

    # Read the PDB files
    pdb1_lines = read_pdb(pdb_file1)
    pdb2_lines = read_pdb(pdb_file2)

    # Split chains for pdb1
    chains_pdb1 = split_chains(pdb1_lines)

    for chain_id, chain_lines in chains_pdb1.items():
        chain_lines.append('TER\n')

        # Get max atom serial and residue sequence number from the chain
        max_atom_serial = max(int(line[6:11]) for line in chain_lines if line.startswith('ATOM'))
        max_residue_seq = max(int(line[22:26].strip()) for line in chain_lines if line.startswith('ATOM'))

        # Adjust atom serial numbers and residue sequence numbers for pdb2
        updated_pdb2_lines = []
        for line in pdb2_lines:
            if line.startswith('ATOM'):
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
                updated_pdb2_lines.append('TER\n')

        # Combine the lines
        combined_lines = chain_lines + updated_pdb2_lines

        # Write the combined PDB file for each chain
        output_file = os.path.join(output_dir, f'chain_{chain_id}.pdb')
        write_pdb(combined_lines, output_file)

# Usage example
protein_names = get_all_pdb_names()

for pdb_name in protein_names:
    # pdb_name = '4lkk'
    output_dir = f'/home/ito/research/data/protein_ligand_complex/{pdb_name}/'
    # os.makedirs(output_dir, exist_ok=True)
    combine_pdb_files_by_chain(f'/mnt/ito/pdbbind_raw/refined_set/{pdb_name}/{pdb_name}_protein.pdb', get_ligand_path(pdb_name), output_dir)
    # break