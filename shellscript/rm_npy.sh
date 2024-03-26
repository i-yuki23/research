#!/bin/bash
while read PDB
do
	cd $PDB
    rm *.npy
    cd .. 
done</home/ito/research/shellscript/list
