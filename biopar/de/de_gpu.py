from re import X
from numba import cuda
import numpy as np
from typing import *
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_next, xoroshiro128p_uniform_float32
from time import time
import warnings
from tqdm import tqdm
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import math
warnings.filterwarnings('ignore')

gpu = cuda.get_current_device()
MAGIC_NUM = 64

def generate_agents_de_gpu(num_agents : int, limits : DeviceNDArray, fun : Callable[[np.ndarray], float]) -> DeviceNDArray:

    @cuda.jit
    def generate_agent(agents : DeviceNDArray, limits : DeviceNDArray, rng_states : DeviceNDArray):
        x, y = cuda.grid(2)
        num_agents, _ = agents.shape
        ldim, _ = limits.shape

        if x < num_agents and y < ldim:
            tid = x * ldim + y
            agents[x, y] = xoroshiro128p_uniform_float32(rng_states, tid)*(limits[y, 1]-limits[y, 0]) + limits[y, 0]

    @cuda.jit
    def eval_agent(agents : DeviceNDArray):
        x = cuda.grid(1)
        num_agents, n = agents.shape
        ldim = (n-1)//2

        if x < num_agents:
            agents[x, -1] = fun_gpu(agents[x, :ldim])

    limits = cuda.to_device(limits)
    ldim, _ = limits.shape

    fun_gpu = cuda.jit(fun, device=True)
    
    threadsPerBlock = (MAGIC_NUM, ldim)
    blocks = (math.ceil(num_agents/threadsPerBlock[0]), 1)

    seed = int(time())
    rng_states = create_xoroshiro128p_states(num_agents * ldim, seed = seed)

    agents = cuda.device_array(shape = (num_agents, 2 * ldim + 1))
    generate_agent[blocks, threadsPerBlock](agents, limits, rng_states)
    eval_agent[math.ceil(num_agents/32), 32](agents)

    return agents

def run_gpu_de_iterations(agents : DeviceNDArray, num_iterations : int, limits : DeviceNDArray, fun : Callable[[np.ndarray], float], CR : float, F : float, lazy_min = True) -> DeviceNDArray:
    
    @cuda.jit
    def de_agent_iteration(agents : DeviceNDArray, rng_states : DeviceNDArray, CR : float, F : float):
        x = cuda.grid(1)
        num_agents, _ = agents.shape
        ldim, _ = limits.shape

        if x < num_agents:
            # TODO: Seleccionar a,b,c diferentes
            
            # RANDOM DIMENSION
            tid = x
            d = xoroshiro128p_next(rng_states, tid) % ldim

            # RANDOM AGENT A
            a = xoroshiro128p_next(rng_states, tid) % num_agents

            # RANDOM AGENT B
            b = xoroshiro128p_next(rng_states, tid) % num_agents
            
            # RANDOM AGENT C
            c = xoroshiro128p_next(rng_states, tid) % num_agents


            agents[x, ldim + d] = agents[a, d] + F * (agents[b, d] -  agents[c, d])

            for j in range(ldim):
                if j != x:
                    rj = xoroshiro128p_uniform_float32(rng_states, tid)
                    if rj < CR:
                        agents[x, ldim + j] = agents[a, j] + F * (agents[b, j] -  agents[c, j])
            
            v = fun_gpu(agents[x, ldim:2*ldim])
            if v < agents[x, -1]:
                agents[x, -1] = v
                for j in range(ldim):
                    agents[x, j] = agents[x, ldim + j]

    @cuda.jit
    def compute_min(agents : DeviceNDArray, best_global : DeviceNDArray, mutex : DeviceNDArray, lazy_min : bool):
        x = cuda.grid(1)
        num_agents, _ = agents.shape

        if x < num_agents:
            v = agents[x, -1]
            if v < best_global[1]:
                
                if lazy_min:
                    while cuda.atomic.compare_and_swap(mutex, 0, 1) == 1: # mutex
                        continue

                ## START CRITICAL SECTION
                if v < best_global[1]:
                    best_global[0] = x
                    best_global[1] = v
                ## END CRITICAL SECTION

                if lazy_min:
                    mutex[0] = 0       

    fun_gpu = cuda.jit(fun, device=True)

    num_agents, _ = agents.shape
    ldim, _ = limits.shape

    seed = int(time())
    rng_states = create_xoroshiro128p_states(num_agents, seed = seed)

    for _ in tqdm(range(num_iterations)):
        de_agent_iteration[math.ceil(num_agents/MAGIC_NUM), MAGIC_NUM](agents, rng_states, CR, F)
    
    best_global = cuda.to_device([0, agents[0, -1]])
    
    mutex = cuda.to_device(np.array([0]))

    compute_min[math.ceil(num_agents/MAGIC_NUM), MAGIC_NUM](agents, best_global, mutex, True)

    return agents, best_global


