from numba import cuda
import math
import numpy as np
from typing import *
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from time import time
import warnings
import tqdm
from numba.cuda.cudadrv.devicearray import DeviceNDArray
warnings.filterwarnings('ignore')
MAGIC_NUM = 64

def generate_particles_gpu(n_particles : int, limits : np.ndarray, fun : Callable[[np.ndarray], float], lazy_min : bool = True) -> DeviceNDArray:
    limits = cuda.to_device(limits)
    ldim, _ = limits.shape

    @cuda.jit
    def partial_generate_particles(particles : DeviceNDArray, limits : DeviceNDArray, rng_states : DeviceNDArray):
        #x, y = cuda.grid(2)
        x = cuda.grid(1)
        num_particles, _ = particles.shape
        ldim, _ = limits.shape

        if x < num_particles: #and y < ldim:
            tid = x
            for y in range(ldim):
                particles[x, y] = xoroshiro128p_uniform_float32(rng_states, tid)*(limits[y, 1]-limits[y, 0]) + limits[y, 0]
                particles[x, 2*ldim + y] = particles[x, y]
        
    @cuda.jit      
    def evaluate_pso_particles_gpu(particles : DeviceNDArray, mutex : DeviceNDArray, best_global : DeviceNDArray, lazy_min : bool):
        x = cuda.grid(1)
        m, n = particles.shape
        n = (n-1)//3

        if x < m:
            v = fun_gpu(particles[x, :n])
        
            for k in range(n):
                particles[x, 2*n+k] = particles[x, k]
            
            particles[x, -1] = v
                
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
    threadsPerBlock = (MAGIC_NUM, 2)
    blocks = (math.ceil(n_particles/threadsPerBlock[0]), ldim)

    seed = int(time())
    rng_states = create_xoroshiro128p_states(n_particles, seed = seed)

    particles = cuda.device_array(shape=(n_particles, 3*ldim + 1))
    best_global = np.array([0, particles[0, -1]])
    best_global = cuda.to_device(best_global)
    mutex = cuda.to_device(np.array([0]))

    partial_generate_particles[math.ceil(n_particles/MAGIC_NUM) , MAGIC_NUM](particles, limits, rng_states)
    evaluate_pso_particles_gpu[math.ceil(n_particles/MAGIC_NUM) , MAGIC_NUM](particles, mutex, best_global, lazy_min)

    return particles, best_global
@cuda.jit
def addOne(mat):
    x, y = cuda.grid(2)
    if x < mat.shape[0] and y < mat.shape[1]:
        mat[x, y] = mat[x, y] + 1


def run_gpu_pso_iterations(particles : DeviceNDArray, fun : Callable[[np.ndarray], float], iterations : int, n_particles : int, l_dim : int, w : float, phi_p : float, phi_g : float, lazy_min : bool = True, enable_tqdm : bool = True) -> DeviceNDArray:
    @cuda.jit
    def pso_particle_iteration_gpu(particles : DeviceNDArray, best_global, w : float, phi_p : float, phi_g : float, rng_states : DeviceNDArray) -> None:
        x, y = cuda.grid(2)
        stridex, stridey = cuda.gridsize(2)
        m, n = particles.shape
        n = (n-1)//3
        if x < m and y < n:
            tid_p =  (y * m) + x
            r_p = xoroshiro128p_uniform_float32(rng_states, tid_p)
            r_g = xoroshiro128p_uniform_float32(rng_states, tid_p)
            vy = n + y
            by = 2 * n + y
            particles[x, vy] = w * particles[x, vy] + phi_p * r_p * (particles[x, by] - particles[x, y]) + phi_g * r_g * (particles[int(best_global[0]), y] - particles[x, y])
            particles[x, y] = particles[x, y] + particles[x, n+y]

    @cuda.jit      
    def evaluate_pso_particles_gpu(particles : DeviceNDArray, mutex : DeviceNDArray, best_global : DeviceNDArray, lazy_min : bool):
        x = cuda.grid(1)
        m, n = particles.shape
        n = (n-1)//3

        if x < m:
            v = fun_gpu(particles[x, :n])
            if v < particles[x, -1]:
                
                for k in range(n): # improve
                    particles[x, 2*n+k] = particles[x, k]
                
                particles[x, -1] = v
                
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

    threadsPerBlock = (MAGIC_NUM, 2)
    blocks = (math.ceil(n_particles/threadsPerBlock[0]), math.ceil(l_dim/threadsPerBlock[1]))
    
    best_global = np.array([0, particles[0, -1]])
    best_global = cuda.to_device(best_global)
    mutex = cuda.to_device(np.array([0]))

    seed = int(time())
    rng_states = create_xoroshiro128p_states(n_particles * l_dim, seed = seed)

    if enable_tqdm:
        rg = tqdm.tqdm(range(iterations))
    else:
        rg = range(iterations)
        
    for _ in rg:
        pso_particle_iteration_gpu[blocks, threadsPerBlock](particles, best_global, w, phi_p, phi_g, rng_states)
        evaluate_pso_particles_gpu[math.ceil(n_particles/MAGIC_NUM), MAGIC_NUM](particles, mutex, best_global, lazy_min)
    
    return particles, best_global