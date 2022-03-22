import numpy as np
from typing import *

#import de.pso_cpu as pso_cpu
import biopar.abc.abc_gpu as abc_gpu

class ABC:
    def __init__(self, target: Callable[[np.ndarray], float], limits : np.ndarray, n_bees : int, mode : str = "cpu", lazy_min : bool = True):
        self.__target = target
        self.__limits = limits
        self.__n_bees = n_bees
        self.__dim, _ = limits.shape
        self.__mode = "gpu" if mode == "gpu" else "cpu"
        self.__best, self.__best_value = None, None

        if self.__mode == "gpu":
            self.__bees, self.__fitness = abc_gpu.generate_bees_gpu(n_bees, target, limits)
        else:
            #self.__agents, self.__best_global = pso_cpu.generate_particles_cpu(n_particles, limits, target)
            x = 1

    def iterate(self, n_iterations : int, max_trials : int = 5):
        if self.__mode == "gpu":
            self.__bees, self.__fitness, self.__best, self.__best_value = abc_gpu.abc_run_iterations_gpu(self.__bees, self.__limits, self.__target, self.__fitness, max_trials, n_iterations, best=self.__best, best_value=self.__best_value)
        else:
            pass #self.__particles, self.__best_global  = pso_cpu.run_cpu_pso_iterations(self.__particles, self.__target, n_iterations, self.__n_particles, self.__dim, w, phi_p, phi_g)

    def getResult(self) -> Tuple[np.ndarray, float]:
        v = self.__best
        return v.copy_to_host() if self.__mode == "gpu" else v, self.__best_value[0]