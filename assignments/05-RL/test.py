import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

N = 100
M = 40
input = np.zeros((N, M, M, M))
test_split_num = 5 * M * M * M // 100
# total points = 40*40*40: 5% = 3200
# need to generate 3200 random points between 0 to 40, exclusive

for i in range(N):
    np.random.seed(0)
    occupied_cells = np.random.randint(0, 40, size=(3200, 3))
    input[i, occupied_cells] = 1
    occ_list = list(map(tuple, occupied_cells))

    oset = set(occ_list)
    while len(oset) < test_split_num:
        diff = test_split_num - len(oset)
        add_occ_cells = np.random.randint(0, 40, size=(diff, 3))
        add_occ_list = list(map(tuple, add_occ_cells))
        oset.update(add_occ_list)
        input[i, add_occ_cells] = 1
    occ_cells_array = np.array(list(map(list, oset)))
    rows = list(occ_cells_array[:, 0])
    cols = list(occ_cells_array[:, 1])
    height = list(occ_cells_array[:, 2])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(rows, cols, height)
    major_ticks = np.arange(0, 40, 10)
    minor_ticks = np.arange(0, 40, 5)

    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_zticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_zticks(major_ticks)

    # And a corresponding grid
    ax.grid(which="both")
    # Or if you want different settings for the grids:
    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", alpha=0.001)
    ax.grid(True)

    # plt.show()
    plt.savefig("point_cloud_" + str(i) + ".png")
