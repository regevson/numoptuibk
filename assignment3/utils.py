import tqdm
import numpy as np
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt


class GridEval:
    def __init__(self, grid_evaluate, x1_min, x1_max, x2_min, x2_max, x1steps=10, x2steps=10):
        k1 = np.linspace(x1_min, x1_max, x1steps)
        k2 = np.linspace(x2_min, x2_max, x2steps)
        grid_points = cartesian([k1, k2])
        result = np.vstack([grid_evaluate(gp) for gp in tqdm.tqdm(grid_points)])
        grid_result = result.reshape((k1.shape[0], k2.shape[0])).T
        self.k1 = k1
        self.k2 = k2
        self.grid_result = grid_result

    def plot(self, close=True):
        k1 = self.k1
        k2 = self.k2
        if close:
            plt.close()
        plt.imshow(self.grid_result, extent=[k1[0], k1[-1], k2[0], k2[-1]], aspect="auto", origin="lower")
        plt.xlabel("k1")
        plt.ylabel("k2")
        plt.colorbar()
        plt.show()

