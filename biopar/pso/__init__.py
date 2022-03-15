import numpy as np
from typing import *

import pso.pso_cpu as pso_cpu
import pso.pso_gpu as pso_gpu

class PSO:
    def __init__(self, target: Callable[[np.ndarray], float], limits : np.ndarray, n_particles : int, mode : str = "cpu", lazy_min : bool = True):
        self.__target = target
        self.__limits = limits
        self.__n_particles = n_particles
        self.__dim, _ = limits.shape
        self.__mode = "gpu" if mode == "gpu" else "cpu"
        self.__best_global : np.ndarray = None
        self.__lazy_min : bool = lazy_min

        if self.__mode == "gpu":
            self.__particles, self.__best_global = pso_gpu.generate_particles_gpu(n_particles, limits, target, self.__lazy_min)
        else:
            self.__particles, self.__best_global = pso_cpu.generate_particles_cpu(n_particles, limits, target)
            x = 1

    def iterate(self, n_iterations : int, w : float, phi_g : float, phi_p : float, tqdm = True):
        if self.__mode == "gpu":
            self.__particles, self.__best_global = pso_gpu.run_gpu_pso_iterations(self.__particles, self.__target, n_iterations, self.__n_particles, self.__dim, w, phi_p, phi_g, self.__lazy_min, tqdm)
        else:
            self.__particles, self.__best_global  = pso_cpu.run_cpu_pso_iterations(self.__particles, self.__target, n_iterations, self.__n_particles, self.__dim, w, phi_p, phi_g, enable_tqdm = tqdm)

    def getResult(self) -> Tuple[np.ndarray, float]:
        v = self.__particles[int(self.__best_global[0]), 2*self.__dim:3*self.__dim]
        return v.copy_to_host() if self.__mode == "gpu" else v, self.__best_global[1]