# %%
import sys
sys.path.append('../..')
from modules.get_voxelized_ligand import get_voxelized_ligand
from lib.voxel import get_voxel_info
from lib.pdb import get_all_pdb_names
import matplotlib.pyplot as plt
import numpy as np

# %%
pdb_names = get_all_pdb_names()

# %%
ligand_voxelized_list = []
for pdb_name in pdb_names:
    grid_dims, grid_origin = get_voxel_info(pdb_name)
    ligand_vozelized = get_voxelized_ligand(pdb_name, grid_dims, grid_origin)
    ligand_voxelized_list.append(ligand_vozelized.sum())

# %%
np.save('ligand_size.npy', np.array(ligand_voxelized_list))

# %%
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(ligand_voxelized_list, bins=50, color='skyblue', edgecolor='black')
peak_index = np.argmax(counts)
peak_x = (bins[peak_index] + bins[peak_index + 1]) / 2
plt.title('Distribution of Average pocket SASA')
plt.xlabel('Average pocket SASA')
plt.ylabel('Frequency')
plt.show()
print(peak_x)

# %%



