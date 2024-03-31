import sys
sys.path.append('/home/ito/research/src')
from lib.pdb import get_clusters_from_pdb
from lib.path import get_ghecom_path

def get_clusters_from_ghecom(pdb_name):
    ghecom_path = get_ghecom_path(pdb_name)
    clusters = get_clusters_from_pdb(ghecom_path, type="HETATM")
    return clusters


