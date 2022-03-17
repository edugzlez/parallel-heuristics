import numpy as np
from typing import *

#import de.pso_cpu as pso_cpu
import de.de_gpu as de_gpu

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
            #self.__agents, self.__best_global = pso_cpu.generate_particles_cpu(n_particles, limits, target)
            x = 1

    def iterate(self, n_iterations : int, CR : float, F : float):
        if self.__mode == "gpu":
            self.__agents, self.__best_global = de_gpu.run_gpu_de_iterations(self.__agents, n_iterations, self.__limits, self.__target, CR, F, self.__lazy_min)
        else:
            pass #self.__particles, self.__best_global  = pso_cpu.run_cpu_pso_iterations(self.__particles, self.__target, n_iterations, self.__n_particles, self.__dim, w, phi_p, phi_g)

    def getResult(self) -> Tuple[np.ndarray, float]:
        v = self.__agents[int(self.__best_global[0]), :self.__dim]
        return v.copy_to_host() if self.__mode == "gpu" else v, self.__best_global[1]