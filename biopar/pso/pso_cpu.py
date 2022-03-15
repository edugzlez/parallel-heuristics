from typing import Callable
import numba
import numpy as np
import tqdm
import random

def generate_particles_cpu(n_particles : int, limits : np.ndarray, fun : Callable[[np.ndarray], float]) -> np.ndarray:
    ldim, _ = limits.shape
    particles = np.random.uniform(size=(n_particles, ldim), low = limits[:, 0], high = limits[:, 1])
    
    mfun = numba.njit(fun)
    
    particles = np.hstack([
        particles, 
        np.zeros((n_particles, ldim)),
        particles.copy(),
        np.apply_along_axis(mfun, 1, particles).reshape((-1, 1))
    ])

    x = np.argmin(particles[:, -1])
    best_global = np.array([x, particles[x, -1]])
    return particles, best_global


def run_cpu_pso_iterations(particles : np.array, fun : Callable[[np.ndarray], float], iterations : int, n_particles : int, l_dim : int, w : float, phi_p : float, phi_g : float, best_global : np.ndarray = None, enable_tqdm = True):
    @numba.njit(parallel=True)
    def particle_iteration_cpu(particles : np.ndarray, best_global : np.ndarray, w : float, phi_p : float, phi_g : float) -> None:
        m, n = particles.shape
        n = (n-1)//3
        for x in numba.prange(m):
            for y in numba.prange(n):
                r_p = random.uniform(0, 1)
                r_g = random.uniform(0, 1)
                vy = n + y
                by = 2 * n + y
                particles[x, vy] = w * particles[x, vy] + phi_p * r_p * (particles[x, by] - particles[x, y]) + phi_g * r_g * (particles[int(best_global[0]), y] - particles[x, y])
                particles[x, y] = particles[x, y] + particles[x, vy]

    @numba.njit(parallel=True)
    def evaluate_funs_cpu(particles, best_global):
        m, n = particles.shape
        n = (n-1)//3
        for x in numba.prange(m):
            v = mfun(particles[x, :n])
            if v < particles[x, -1]:
                particles[x, 2*n:3*n] = particles[x, :n]
                particles[x, -1] = v
                
        x = np.argmin(particles[:, -1])
        best_global[0] = x
        best_global[1] = particles[x, -1]

    mfun = numba.njit(fun)

    if best_global is None:
        best_global = np.array([0, particles[0, -1]])

    if enable_tqdm:
        rg = tqdm.tqdm(range(iterations))
    else:
        rg = range(iterations)

    for _ in rg:
        particle_iteration_cpu(particles, best_global, w, phi_p, phi_g)
        evaluate_funs_cpu(particles, best_global)

    return particles, best_global
