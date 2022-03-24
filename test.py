from biopar.de import DE
from biopar.benchmark import schaffer
import numpy as np
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    limits = np.array([
        [-10, 10],
        [-10, 10]
    ])
    abc = DE(schaffer, limits, 100, "cpu")
    abc.iterate(500, 0.1, 0.5)
    print(abc.getResult())