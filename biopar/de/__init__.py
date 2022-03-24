import numpy as np
from typing import *

import biopar.de.de_cpu as de_cpu
import biopar.de.de_gpu as de_gpu

class DE:
    def __init__(self, target: Callable[[np.ndarray], float], limits : np.ndarray, n_agents : int, mode : str = "cpu", lazy_min : bool = True):
        self.__target = target
        self.__limits = limits
        self.__n_agents = n_agents
        self.__dim, _ = limits.shape
        self.__mode = "gpu" if mode == "gpu" else "cpu"
        self.__best_global : np.ndarray = None
        self.__lazy_min : bool = lazy_min

        if self.__mode == "gpu":
            self.__agents = de_gpu.generate_agents_de_gpu(n_agents, limits, target)
        else:
            self.__agents, self.__best_global = de_cpu.generate_agents_cpu(n_agents, limits, target)

    def iterate(self, n_iterations : int, CR : float, F : float, tqdm = False):
        if self.__mode == "gpu":
            self.__agents, self.__best_global = de_gpu.run_gpu_de_iterations(self.__agents, n_iterations, self.__limits, self.__target, CR, F, enable_tqdm = tqdm)
        else:
            self.__agents, self.__best_global  = de_cpu.run_cpu_de_iterations(self.__agents, self.__target, n_iterations, CR, F, self.__best_global, enable_tqdm = tqdm)

    def getResult(self) -> Tuple[np.ndarray, float]:
        v = self.__agents[int(self.__best_global[0]), :self.__dim]
        return v.copy_to_host() if self.__mode == "gpu" else v, self.__best_global[1]