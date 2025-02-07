#!/bin/bash
while read PDB
do
    echo "${PDB}"
    cp /mnt/ito/pdbbind_raw/general_set/${PDB}/${PDB}_min.dx /mnt/ito/data/pdb_bind/${PDB}/${PDB}_min.dx
done</home/ito/research/data/general_valid_proteins.txt
