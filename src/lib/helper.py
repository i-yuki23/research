from scipy.spatial import cKDTree
import numpy as np
import os
import glob

def extract_points_within_threshold(coords1, coords2, threshold):
    """閾値以内にあるポケットの点を抽出する
    Parameters
    ----------
    coords1 : np.ndarray
        基準となる方の点のセット（例 リガンドの座標）
    coords2 : np.ndarray
        抽出したい方の点のセット (例 ポケットの座標)
    threshold : float
        閾値
    """
    
    # ポケット領域を定義する点のセットに対してk-d treeを構築
    pocket_tree = cKDTree(coords2)
    
    # リガンドの原子に対して、閾値以内にあるポケットの点を検索し、そのインデックスを抽出
    indices = pocket_tree.query_ball_point(coords1, threshold)
    
    # インデックスから重複を除き、ユニークなものだけを抽出
    unique_indices = set(sum(indices, []))  # インデックスリストのフラット化とユニーク化
    
    # ユニークなインデックスを使用して、条件を満たすポケットの点の座標を抽出
    extracted_points = [coords2[i] for i in unique_indices]
    if extracted_points:
        extracted_points = np.vstack(extracted_points)
    return np.array(extracted_points)

# MODIFY: time complexity is O(M * N), optimize this algorithm
def calculate_closest_distance(coords1, coords2):
    """
    Calculate the shortest distance between any two points in two sets of 3D coordinates.
    
    Parameters:
    coords1 (np.ndarray): A 2D array of shape (N, 3) representing N points in 3D space.
    coords2 (np.ndarray): A 2D array of shape (M, 3) representing M points in 3D space.
    
    Returns:
    float: The shortest distance between any two points in the two sets.
    """
    # Ensure the inputs are numpy arrays
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    # Calculate the pairwise distances between all points in coords1 and coords2
    distances = np.linalg.norm(coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :], axis=2)
    
    # Find the minimum distance
    min_distance = np.min(distances)
    
    return min_distance

import numpy as np

def calculate_average_of_top_n_closest_distances(coords1, coord2, n=10):
    """
    coords2 が1原子(1つの3次元座標)の場合の、
    coords1 の各点との距離を求め、その中から最小距離上位 n 個を平均する。

    Parameters
    ----------
    coords1 : np.ndarray
        形状 (N, 3) の 3次元座標の配列
    coord2 : np.ndarray
        形状 (3,)
    n : int
        平均をとる対象とする最小距離の個数

    Returns
    -------
    float
        最も小さい距離上位 n 個の平均値
    """
    # coords1 各点と、coord2 との距離を一次元配列で得る → shape (N,)
    distances = np.linalg.norm(coords1 - coord2, axis=1)

    # 全距離をソートして、最小値から n 個を取り出して平均
    # ただし N < n の場合は全距離の平均
    if len(distances) >= n:
        # np.partition を使った高速な方法
        top_n_partition = np.partition(distances, n - 1)[:n]
        return float(np.mean(top_n_partition))
    else:
        # 全ての距離数が n に満たない場合は、そのまま平均
        return float(np.mean(distances))
    
def calculate_top_n_closest_coords(coords1, coord2, n=10):
    """
    coords2 が1原子(1つの3次元座標)の場合の、
    coords1 の各点との距離を求め、その中から最小距離上位 n 個の座標と距離を返す。

    Parameters
    ----------
    coords1 : np.ndarray
        形状 (N, 3) の 3次元座標の配列
    coord2 : np.ndarray
        形状 (3,)
    n : int
        最近接の上位 N 個を取得する

    Returns
    -------
    np.ndarray
        最も小さい距離上位 n 個
    """
    # coords1 各点と coord2 との距離を計算 → shape (N,)
    distances = np.linalg.norm(coords1 - coord2, axis=1)

    # 距離をソートして上位 N 個のインデックスを取得
    if len(distances) >= n:
        top_n_indices = np.argpartition(distances, n - 1)[:n]  # 上位 N 個のインデックスを取得
        top_n_coords = coords1[top_n_indices]  # 上位 N 個の対応する座標
    else:
        # N 個未満の場合、全てを対象とする
        top_n_indices = np.arange(len(distances))
        top_n_coords = coords1

    return top_n_coords



def make_dir(save_file_path):
    """ファイルを保存するためのディレクトリを作成し、ファイルを保存する
    Parameters
    ----------
    save_file_path : str
        保存するファイルのパス
    """
    directory = os.path.dirname(save_file_path)
            
    # ディレクトリが存在するか確認
    if not os.path.exists(directory):
        # ディレクトリが存在しない場合、ディレクトリを作成（親ディレクトリも含めて）
        os.makedirs(directory, exist_ok=True)

def get_latest_checkpoint(checkpoint_dir):
    # List all files in the directory
    checkpoint_files = os.listdir(checkpoint_dir)

    # Filter and sort the checkpoint files based on the numbering
    checkpoint_files = [f for f in checkpoint_files if f.startswith('cp-') and f.endswith('.weights.h5')]
    checkpoint_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))

    # Get the latest checkpoint file
    latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None

    return os.path.join(checkpoint_dir, latest_checkpoint)

def remove_all_checkpoints(checkpoint_dir):
    # List all files in the directory
    files = glob.glob(os.path.join(checkpoint_dir, '*'))

    for file in files:
        os.remove(file)

def get_list_from_pdb_text(pdb_test_path):
    with open(pdb_test_path, mode='r') as f:
        pdb_list = f.read().splitlines()
    return pdb_list
