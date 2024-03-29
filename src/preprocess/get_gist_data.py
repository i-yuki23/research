# gistのデータをゲットしたい
# 水分子を中心として南北せるとかじゃなくて、水が存在しているところはpredでそれ以外は０にしたい
# ボクセル数がどんくらいになるかを確認→ボクセル数を合わせる

import os
import pandas as pd
import numpy as np
import sys

sys.path.append("/home/ito/GIST/Deep-GIST/lib/")
sys.path

        
 #replaceを計算する範囲となる、リガンドボクセルの閾値の設定
threshold = 10**(-6)
threshold1 = 1-np.exp(-1)


dir_testdata = "/mnt/yusaku/gist/testdata/"
dxname="gist-dA"

dict_sum = {}
for pdb_name in pdb_names[10:20]:
    print(pdb_name)

    # GIST計算の結果をよりこみボクセルを取得
    # displaceable と non-displaceable の水分子の座標を取得
    # 水分子１っこをそれぞれボクセル化
    # 閾値10^-6でカットオフをかけ、ΔG_replaceをとってくる

    #GIST計算結果のDXファイルのパスを指定し、読み込む
    gist_path = f"/mnt/ito/3DRISM220308/{pdb_name}/{pdb_name}_gist-pred.dx"
    gist_map = ReadDX(gist_path)
    len_file, grid, init = dxinfo(path_dx_pred)

    #リガンド込みのPDBファイルを読み込む
    path_pro = "/mnt/dandan/3drism/dir_3DRISM_20181213_155211/{}/{}_min.pdb".format(protein, protein)
    path_lig = "/mnt/dandan/3drism/dir_3DRISM_20181213_155211/{}/{}_ligand.pdb".format(protein, protein)
    path_O = "/mnt/ito/3DRISM220308/{}/pred_O_placed_{}.pdb".format(protein, protein)
    path_lig_pocket = "/mnt/ito/3DRISM220308/{}/interface_5A.npy".format(protein)

    
    pro_xyz, pro_type, pro_amino_type = ReadPDB2(path_pro)
    lig_xyz, lig_type, lig_amino_type = ReadPDB2(path_lig)
    O_xyz, O_type, O_amino_type = ReadPDB2(path_O)
    
    # print(O_xyz)
    
    channel = 5
    voxel = np.array([0.0]*grid[0]*grid[1]*grid[2]*channel).reshape(channel,grid[0], grid[1], grid[2])
    voxel_ligand = np.array([0.0]*grid[0]*grid[1]*grid[2]*1).reshape(1, grid[0], grid[1], grid[2])
    vdw = {1:1.20,6:1.69984,7:1.62500,8:1.51369,9:1.656,16:1.78180,15:1.871,35:1.978,53:2.094}
    cutoff = 10
    
    # atoms = voxelizer_atom_np(atoms_atomic_num=pro_type, 
    #                 atoms_xyz=pro_xyz,
    #                 origin=init,
    #                 lengths_index_voxelize=grid,
    #                 half_length_index_cutoff=cutoff,
    #                 length_voxel=0.5,
    #                 factor=12,   
    #                 )
    
    ligand = voxelizer_ligand_np(atoms_atomic_num=lig_type, 
                    atoms_xyz=lig_xyz,
                    origin=init,
                    lengths_index_voxelize=grid,
                    half_length_index_cutoff=10,
                    length_voxel=0.5,
                    factor=12,   
                    radiuses=vdw,
                    dtype=np.float64)
    
    lig_pocket = np.load(path_lig_pocket)

    
    # リガンドポケットのボクセル ＆ Oの中心のボクセル & リガンドのボクセル
    # の条件でOの中心ボクセルを取得して、それを座標に戻す
    # →その座標についてボクセル化→ 10^-6でカットオフをかけ、こいつらに対してΔG_replaceを計算
    
    voxelized_O_xyz = ((O_xyz - init) // 0.5).astype(int)
    
    
    # ボクセル番号から座標を取り出す辞書を作成（※keyにリストは使えないのでタプルに変換）
    dict_O = {}
    for i in range(voxelized_O_xyz.shape[0]):
        dict_O[tuple(voxelized_O_xyz[i])] = O_xyz[i]
        
    # ボクセル形式に変換
    O_atoms_temp = np.zeros((grid[0],grid[1],grid[2]))
    for i in voxelized_O_xyz:
        # if i[0] > grid[0] or i[1] > grid[1] or i[2] > grid[2]:
        #     continue
        O_atoms_temp[i[0], i[1], i[2]] = 1
    
    
    replaced_voxelized_O_xyz = []
    replaced_voxelized_O_xyz_temp = np.where(((ligand[0]>threshold1) & (O_atoms_temp==1) & (lig_pocket==1)), 1, 0)
    replaced_voxelized_O_xyz_index = np.where(replaced_voxelized_O_xyz_temp == 1)
    for i in zip(replaced_voxelized_O_xyz_index[0], replaced_voxelized_O_xyz_index[1], replaced_voxelized_O_xyz_index[2]):
        replaced_voxelized_O_xyz.append([i[0], i[1], i[2]])
    
    replaced_O_xyz = []
    for i in replaced_voxelized_O_xyz:
        replaced_O_xyz.append(dict_O[tuple(i)])
    # print(replaced_O_xyz)
     
    replaced_O_atoms = voxelizer_atom_np(atoms_atomic_num=(np.ones(len(replaced_O_xyz))*8),
                    atoms_xyz=replaced_O_xyz,
                    origin=init,
                    lengths_index_voxelize=grid,
                    half_length_index_cutoff=cutoff,
                    length_voxel=0.5,
                    factor=12,   
                    )
    
    remain_voxelized_O_xyz = []
    remain_voxelized_O_xyz_temp = np.where(((ligand[0]<=threshold1) & (O_atoms_temp==1) & (lig_pocket==1)), 1, 0)
    remain_voxelized_O_xyz_index = np.where(remain_voxelized_O_xyz_temp == 1)
    for i in zip(remain_voxelized_O_xyz_index[0], remain_voxelized_O_xyz_index[1], remain_voxelized_O_xyz_index[2]):
        remain_voxelized_O_xyz.append([i[0], i[1], i[2]])
        
    remain_O_xyz = []
    for i in remain_voxelized_O_xyz:
        remain_O_xyz.append(dict_O[tuple(i)])
    
    # print(remain_O_xyz)
    
    dG_replaced_O_list = []
    for O in replaced_O_xyz: 
        # print(O)
        A_replaced_O_atom = voxelizer_atom_np(atoms_atomic_num=[8],
                atoms_xyz=[O],
                origin=init,
                lengths_index_voxelize=grid,
                half_length_index_cutoff=cutoff,
                length_voxel=0.5,
                factor=12,   
                )
        gist_replaced_O = np.where(A_replaced_O_atom[2]>threshold, pred, 0)
        # num_replaced_O = np.where(A_replaced_O_atom[2]>threshold1, 1, 0).sum()
        dG_replaced = gist_replaced_O.sum()*(0.5**3)
        dG_replaced_O_list.append(dG_replaced)


    dG_remain_O_list = []
    for O_ in remain_O_xyz: 
        A_remain_O_atom = voxelizer_atom_np(atoms_atomic_num=[8],
                atoms_xyz=[O_],
                origin=init,
                lengths_index_voxelize=grid,
                half_length_index_cutoff=cutoff,
                length_voxel=0.5,
                factor=12,   
                )
        gist_remain_O = np.where(A_remain_O_atom[2]>threshold, pred, 0)
        # num_remain_O = np.where(A_remain_O_atom[2]>threshold1, 1, 0).sum()
        dG_remain_O = gist_remain_O.sum()*(0.5**3)
        dG_remain_O_list.append(dG_remain_O)

    df1 = pd.DataFrame({"dG_replaced_O": dG_replaced_O_list})
    df2 = pd.DataFrame({"dG_remain_O": dG_remain_O_list})

    df1.to_csv("/mnt/ito/3DRISM220308/{}/dG_replaced_O_eLig.csv".format(protein))
    df2.to_csv("/mnt/ito/3DRISM220308/{}/dG_remain_O_eLig.csv".format(protein))
    print(protein)