# 程序用于check原始轨迹计算是否正确
# 选择一条原始轨迹 可视化

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():

    rfile = "/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/r_values.csv"
    ffile = "/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/f_values.csv"

    r_values = pd.read_csv(rfile, skiprows=1, header=None)
    f_values = pd.read_csv(ffile, skiprows=1, header=None)

    r = r_values[20].tolist()
    f = f_values[20].tolist()

    plt.figure()
    plt.plot(r[0: 1800], f[0: 1800], 'b-', alpha = 1.0)
    plt.savefig("/home/tyt/project/Single-chain/opt+R/Rand_length/column_format/Single_traj.png", dpi = 300)


if __name__ == "__main__":
    main()
