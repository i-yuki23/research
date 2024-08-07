import sys
sys.path.append('..')
from lib.pdb import get_atom_id_from_pdb, get_all_pdb_names
import pandas as pd
import numpy as np
import os

LIGAND_POCKET_VOXEL_NUM = 8
included_protein_list = 

def get_water0_pdb(pdb_names):

    def count_water_number(pdb_file):
        ids = get_atom_id_from_pdb(pdb_file)
        return len(ids)
    
    water_0_pdb = []
    for pdb_name in pdb_names:
        base_path = f'../../data/LIGAND_POCKET_VOXEL_NUM_{LIGAND_POCKET_VOXEL_NUM}/labeled_water/'
        file_path = f'{pdb_name}/pred_O_placed_{pdb_name}_3.0.pdb'

        displaceable_water_num = count_water_number(os.path.join(base_path, 'displaceable/', file_path))
        non_displaceable_water_num = count_water_number(os.path.join(base_path, 'non_displaceable/', file_path))
        if displaceable_water_num == 0 and non_displaceable_water_num == 0:
            water_0_pdb.append(pdb_name)

    return water_0_pdb

exsheet = pd.ExcelFile('clusterNum.xlsx')
exsheet_df = exsheet.parse(exsheet.sheet_names[0])
# filtered_exsheet_df = exsheet_df[~exsheet_df['PDB'].isin(water_0_pdb)]
filtered_exsheet_df = exsheet_df[exsheet_df['PDB'].isin(included_protein_list)]
exsheet_array = filtered_exsheet_df.values

cluster_num = filtered_exsheet_df["Cluster Num"][:]
cluster_protein_counts = np.bincount(cluster_num) # それぞれのクラスターに何個のたんぱく質があるか

# クラスターのインデックスを作成
cluster_indices = np.arange(len(cluster_protein_counts))
# クラスターをタンパク質数でソート
sorted_clusters = cluster_indices[np.argsort(-cluster_protein_counts)]

# グループAとBのクラスター数とタンパク質数を初期化
clusters_a = []
clusters_b = []
proteins_a = []
proteins_b = []

# クラスターをグループAとBに分配
for cluster_index in sorted_clusters:
    # 現在のグループAとBのクラスター数とタンパク質数を取得
    count_a = len(clusters_a)
    count_b = len(clusters_b)
    proteins_count_a = sum(proteins_a)
    proteins_count_b = sum(proteins_b)

    # グループAとBの条件を満たす場合にクラスターを追加
    if (count_a < count_b or 3.0 * proteins_count_a < 7.0 * proteins_count_b) and count_a * 3 < int(len(cluster_protein_counts)*7):
        clusters_a.append(cluster_index)
        proteins_a.append(cluster_protein_counts[cluster_index])
    else:
        clusters_b.append(cluster_index)
        proteins_b.append(cluster_protein_counts[cluster_index])

# 結果を出力
print("Group A clusters:", clusters_a)
print("Group A cluster kinds:", len(clusters_a))
print("Group A protein count:", sum(proteins_a))

print("Group B clusters:", clusters_b)
print("Group B cluster kinds:", len(clusters_b))
print("Group B protein count:", sum(proteins_b))

train = []
test = []

for data in exsheet_array:
    for a_index in clusters_a:
        if data[3] == a_index:
            train.append(data[1])
            # continue
            
for data in exsheet_array:
    for b_index in clusters_b:
        if data[3] == b_index:
            test.append(data[1])

# %%
with open("train.txt", mode = "w") as f:
    for pdb in train:
        f.write(pdb+"\n")

# ### 上と同様にtestをtestとvalに分ける

# %%
## trainデータの数を数える
counts_train = np.zeros(len(clusters_b))
for index, num in enumerate(clusters_b):
    for i in exsheet_array:
        if i[3] == num:
            counts_train[index] += 1

# クラスターの要素数を表す配列

# クラスターをタンパク質数でソート
sorted_clusters = clusters_b

# グループAとBのクラスター数とタンパク質数を初期化
clusters_train = []
clusters_val = []
proteins_train = []
proteins_val = []

# クラスターをグループAとBに分配
for index, cluster_index in enumerate(sorted_clusters):
    # 現在のグループAとBのクラスター数とタンパク質数を取得
    count_train = len(clusters_train)
    count_val = len(clusters_val)
    proteins_count_train = sum(proteins_train)
    proteins_count_val = sum(proteins_val)

    # グループAとBの条件を満たす場合にクラスターを追加
    if (count_train < count_val or 5.0 * proteins_count_train < 5.0 * proteins_count_val) and count_train * 5 < int(len(counts_train)*5):
        clusters_train.append(cluster_index)
        proteins_train.append(counts_train[index])
    else:
        clusters_val.append(cluster_index)
        proteins_val.append(counts_train[index])

# 結果を出力
print("Group train clusters:", clusters_train)
print("Group train cluster kinds:", len(clusters_train))
print("Group train protein count:", sum(proteins_train))

print("Group val clusters:", clusters_val)
print("Group val cluster kinds:", len(clusters_val))
print("Group val protein count:", sum(proteins_val))

# %%
train = []
val = []

for data in exsheet_array:
    for train_index in clusters_train:
        if data[3] == train_index:
            train.append(data[1])
            
for data in exsheet_array:
    for val_index in clusters_val:
        if data[3] == val_index:
            val.append(data[1])

with open("val.txt", mode = "w") as f:
    for PDB in train:
        f.write(PDB+"\n")
        
with open("test.txt", mode = "w") as f:
    for PDB in val:
        f.write(PDB+"\n")


