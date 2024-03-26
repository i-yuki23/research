from scipy.spatial import cKDTree
import numpy as np

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