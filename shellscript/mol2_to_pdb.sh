#!/bin/bash
while read PDB
do 
	  obabel /mnt/ito/pdbbind_raw/general_set/${PDB}/${PDB}_ligand.mol2 -O /mnt/ito/pdbbind_raw/general_set/${PDB}/${PDB}_ligand.pdb
done</mnt/ito/pdbbind_raw/general_set/index/general_high_resolution_pdb_ids_without_ion.txt
