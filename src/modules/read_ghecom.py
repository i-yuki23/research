import sys
sys.path.append('/home/ito/research/src')
from lib.pdb import get_clusters_from_pdb

def get_clusters_from_ghecom(pdb_name):
    ghecom_path = f'/home/ito/research/data/ghecom/clusters/{pdb_name}/{pdb_name}_ghecom.pdb'
    clusters = get_clusters_from_pdb(ghecom_path, type="HETATM")
    return clusters


