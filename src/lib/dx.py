import numpy as np
import pandas as pd

#len(file)とグリッド数、原点の座標を返す
def dxinfo(path_dx):
    with open(path_dx, 'r') as f:
        file = f.readlines()
    grid_dims = file[0].strip().split()[5:8]
    grid_origin = file[1].strip().split()[1:4]
    return np.array([int(grid_dims[i])for i in range(3)]), np.array([float(grid_origin[i])for i in range(3)])

def dx_to_ndarray(path_dx, grid_dims):
    with open(path_dx, 'r') as f:
        file = f.readlines()
        
    #.dxファイルは7行目から値が書かれている。
    file = file[7:]
    
    #最後の一行にコメントがある場合除外する
    try:
        i = float(file[-1].replace("\n", "").split()[0]) # -1は最後の要素を表す
    
    # ValueErrorが起きた時は、以下の処理を行う
    except ValueError as e:
        file = file[:-1]                                 # :-1にしているので最後の行が除かれる
        print(file[-1])
        i = float(file[-1].replace("\n", "").split()[0])
    
    table = [line.replace("\n", "").split() for line in file]
    table = [[float(s)for s in line] for line in table]
    df = pd.DataFrame(table)
    
    gist = df.values.ravel()
    gist = gist[~np.isnan(gist)]
    gist_vox= gist.reshape([int(grid_dims[i])for i in range(3)], order='C')

    return gist_vox

def read_dx(path_dx):
    grid_dims, grid_origin = dxinfo(path_dx)
    voxel = dx_to_ndarray(path_dx, grid_dims)
    return voxel, grid_dims, grid_origin

def np2dx(data, grid_origin, save_path):

    import re
    nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]

    with open(save_path, "w") as f:
        f.write("object 1 class gridpositions counts{:8d}{:8d}{:8d}\n".format(nx, ny, nz))
        f.write("origin {:15.8f}{:15.8f}{:15.8f}\n".format(grid_origin[0], grid_origin[1], grid_origin[2]))
        f.write("delta       0.50000000 0 0\ndelta  0      0.50000000 0\ndelta  0 0      0.50000000\n")
        f.write("object 2 class gridconnections counts{:9d}{:9d}{:9d}\n".format(nx, ny, nz))
        f.write("object 3 class array type double rank 0 items {:27d} data follows\n".format(nx * ny * nz))
        idx=0
        amari = (nx * ny * nz) % 3
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    idx+=1
                    moji = "    0." + "{:.4E}".format(data[x, y, z]*10.).replace(".","").replace("E+","E+0").replace("E-","E-0")
                    try : 
                        if moji.find("0.-") != -1 : moji = re.sub("0.-(....).E", "-0.\\1E", moji)
                        if moji[moji.rfind("E")+5] != None : moji = moji.replace("+0","+").replace("-0","-")
                    except : pass
                    if idx % 3 == 0 :
                        f.write(moji+"\n")
                    else : 
                        f.write(moji)
                    if nx * ny * nz == idx and amari!=0:
                        f.write("\n")
        f.write("object \"Untitled\" class field\n")
