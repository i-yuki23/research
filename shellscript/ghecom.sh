#!/bin/bash
while read PDB
do
    ghecom -M P -ipdb "/mnt/dandan/3drism/dir_3DRISM_20181213_155211/${PDB}/${PDB}_min.pdb" -opocpdb "/home/ito/research/data/ghecom/clusters/${PDB}/${PDB}_ghecom.pdb"
done<list
