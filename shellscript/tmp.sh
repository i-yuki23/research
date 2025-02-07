#!/bin/bash
while read PDB
do
    echo "${PDB}"
    mkdir -p ./displaceable/${PDB}
    mkdir -p ./non_displaceable/${PDB}
    cp ./${PDB}/displaceable/* ./displaceable/${PDB}/
    cp ./${PDB}/non_displaceable/* ./non_displaceable/${PDB}/
done</home/ito/research/data/all_valid_proteins
