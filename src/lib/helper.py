from scipy.spatial import cKDTree
import numpy as np
import os

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