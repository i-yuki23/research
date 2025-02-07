#!/bin/bash
while read PDB
do
    cd displaceable
	cd $PDB
    rm *.npy
    cd ../..
    cd non_displaceable
    cd $PDB
    rm *.npy
    cd ../..
done</home/ito/research/shellscript/list
