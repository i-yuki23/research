#!/bin/bash
CUTOFF=5.0
while read PDB
do

    cd $PDB/
    echo $PDB
    if [[ $PDB == *gz* ]]; then
	PDB_NO_GZ=${PDB//gz/}  # Remove 'gz' from the PDB name
	mv ${PDB}_min.dx ${PDB_NO_GZ}_min.dx  # Rename the original file
	python2 /home/ito/placevent-main/placevent.py ${PDB_NO_GZ}_min.dx 55.5 $CUTOFF > pred_O_placed_${PDB}_${CUTOFF}.pdb
	mv ${PDB_NO_GZ}_min.dx ${PDB}_min.dx  # Rename the file back to the original name
    else
	python2 /home/ito/placevent-main/placevent.py ${PDB}_min.dx 55.5 $CUTOFF > pred_O_placed_${PDB}_${CUTOFF}.pdb
    fi
    cd ../
done<shellscript/list_gz
