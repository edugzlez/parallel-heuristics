import numpy as np
import numba
from typing import *
from numpy import random
import tqdm

def generate_agents_cpu(num_agents : int, limits : np.ndarray, fun : Callable[[np.ndarray], float]) -> np.ndarray:
    ldim, _ = limits.shape
    agents = np.random.uniform(size=(num_agents, ldim), low = limits[:, 0], high = limits[:, 1])
    
    mfun = numba.njit(fun)
    
    agents = np.hstack([
        agents, 
        agents.copy(),
        np.apply_along_axis(mfun, 1, agents).reshape((-1, 1))
    ])

    x = np.argmin(agents[:, -1])
    best_global = np.array([x, agents[x, -1]])
    return agents, best_global


def run_cpu_de_iterations(agents : np.array, fun : Callable[[np.ndarray], float], iterations : int, CR : float, F : float, best_global : np.ndarray = None, enable_tqdm = True):
    
    @numba.njit(parallel=True)
    def agent_iteration_cpu(agents : np.array, CR : float, F : float) -> None:
        m, n = agents.shape
        ldim = (n-1)//2
        for x in numba.prange(m):
            
            d = np.random.randint(0, ldim)
            a = np.random.randint(0, m)
            b = np.random.randint(0, m)
            c = np.random.randint(0, m)
            agents[x, ldim + d] = agents[a, d] + F * (agents[b, d] -  agents[c, d])

            for j in range(ldim):
                if j != x:
                    rj = np.random.rand()
                    if rj < CR:
                        agents[x, ldim + j] = agents[a, j] + F * (agents[b, j] -  agents[c, j])

            v = mfun(agents[x, ldim:2*ldim])
            if v < agents[x, -1]:
                agents[x, -1] = v
                for j in range(ldim):
                    agents[x, j] = agents[x, ldim + j]

    mfun = numba.njit(fun)

    if best_global is None:
        best_global = np.array([0, agents[0, -1]])

    if enable_tqdm:
        rg = tqdm.tqdm(range(iterations))
    else:
        rg = range(iterations)

    for _ in rg:
        agent_iteration_cpu(agents, CR, F)
    

    # TODO: parallelize
    best_global[0] = np.argmin(agents[:, -1])
    best_global[1] = agents[int(best_global[0]), -1]

    return agents, best_global
