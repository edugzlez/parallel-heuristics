from ast import Call
from os import device_encoding
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np
from typing import *
import math
from time import time
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_next, xoroshiro128p_uniform_float32
from tqdm import tqdm 

MAGIC_NUM = 64

def generate_bees_gpu(num_bees : int, fun : Callable[[np.ndarray], float], limits : np.ndarray) -> Tuple[DeviceNDArray, DeviceNDArray]: 

    @cuda.jit
    def generate_bee(bees : DeviceNDArray, limits : DeviceNDArray, num_bees : int, ldim : int, rng_states : DeviceNDArray):
        x, y = cuda.grid(2)
        
        if x < num_bees and y < ldim:
            tid = x * ldim + y
            bees[x, y] = xoroshiro128p_uniform_float32(rng_states, tid)*(limits[y, 1]-limits[y, 0]) + limits[y, 0]
    
    @cuda.jit
    def evaluate_bee(bees : DeviceNDArray, num_bees : int, ldim : int, fitness : DeviceNDArray):
        x = cuda.grid(1)
        
        if x < num_bees:
            fitness[x] = fun_gpu(bees[x, :ldim])

    ldim, _ = limits.shape

    fun_gpu = cuda.jit(device=True)(fun)

    bees = cuda.device_array(shape=(num_bees, ldim))
    fitness = cuda.device_array(shape=num_bees)
    limits = cuda.to_device(limits)

    threadsPerBlock = (MAGIC_NUM, 2)

    seed = int(time())
    rng_states = create_xoroshiro128p_states(num_bees * ldim, seed = seed)

    blocks = (math.ceil(num_bees/threadsPerBlock[0]), math.ceil(ldim/threadsPerBlock[1]))

    generate_bee[blocks, threadsPerBlock](bees, limits, num_bees, ldim, rng_states)
    evaluate_bee[math.ceil(num_bees/MAGIC_NUM), MAGIC_NUM](bees, num_bees, ldim, fitness)
   
    return bees, fitness


def abc_run_iterations_gpu(bees : DeviceNDArray, limits : np.ndarray, fun : Callable[[np.ndarray], float], fitness : DeviceNDArray, max_trials : float, num_iterations : int, best : DeviceNDArray = None, best_value : DeviceNDArray = None, enable_tqdm : bool = True):

    @cuda.jit
    def bee_employer_iteration(
        bees : DeviceNDArray,
        new_bees : DeviceNDArray,
        fitness : DeviceNDArray,
        trials : DeviceNDArray,
        max_trials : int,
        rng_states : DeviceNDArray,
        num_bees : int,
        ldim : int,
        minmax_values : DeviceNDArray,
        sum_value : DeviceNDArray,
        mutex : DeviceNDArray,
        best : DeviceNDArray,
        best_value : DeviceNDArray
    ):
        x = cuda.grid(1)

        if x < math.ceil(num_bees/2):

            # Selecting random dimension
            tid = x
            k = xoroshiro128p_next(rng_states, tid) % ldim

            # Selecting random distinct bee b'

            b = xoroshiro128p_next(rng_states, tid) % (num_bees - 1)
            if b == x: 
                b = b + 1

            r = xoroshiro128p_uniform_float32(rng_states, tid) * 2 - 1 # r € [-1, 1]

            new_bees[x, k] = bees[x, k] + r * (bees[x, k] - bees[b, k])
            for j in range(ldim):
                if j != k:
                    new_bees[x, j] = bees[x, j]
    
            v = fun_gpu(new_bees[x, :ldim])

            if v < fitness[x]:
                for j in range(ldim):
                    bees[x, j] = new_bees[x, j]
                fitness[x] = v
                cuda.atomic.add(sum_value, 0, -v)
                cuda.atomic.min(minmax_values, 0, -v)
                trials[x] = max_trials
            else:
                trials[x] = trials[x] - 1

    @cuda.jit
    def compute_probabilities(
        bees : DeviceNDArray,
        fitness : DeviceNDArray,
        probs : DeviceNDArray,
        minmax_values : DeviceNDArray,
        sum_value : DeviceNDArray
    ):
        x = cuda.grid(1)
        employer_bees = math.ceil(bees.shape[0]/2)
        if x < employer_bees:
            probs[x] = ((-fitness[x]) - minmax_values[0])/(sum_value[0]-employer_bees*minmax_values[0])

    @cuda.jit
    def bee_onlooker_iteration(
        bees : DeviceNDArray,
        fitness : DeviceNDArray,
        rng_states : DeviceNDArray,
        ldim : int,
        num_bees : int,
        probs : DeviceNDArray,
        mutex : DeviceNDArray,
        best : DeviceNDArray,
        best_value : DeviceNDArray
    ):
        x = cuda.grid(1)

        if x < math.floor(num_bees/2):
            _x = x
            tid = _x
            x = int(math.ceil(num_bees/2) + _x)

            r = xoroshiro128p_uniform_float32(rng_states, x)
            s = probs[0]
            i = 0

            while s > r and i < num_bees + 1:
                i = i + 1
                s += probs[i]

            for j in range(ldim):
                bees[x, j] = bees[i, j]

            # Selecting random dimension
            
            k = xoroshiro128p_next(rng_states, tid) % ldim

            # Selecting random distinct bee b'
            b = xoroshiro128p_next(rng_states, tid) % (num_bees - 1)
            if b == x: 
                b = b + 1
            
            r = xoroshiro128p_uniform_float32(rng_states, tid) * 2 - 1 # r € [-1, 1]

            bees[x, k] = bees[x, k] + r * (bees[x, k] - bees[b, k])
            v = fun_gpu(bees[x, :ldim])
            fitness[x] = v
            
            if v < best_value[0]:
                while cuda.atomic.compare_and_swap(mutex, 0, 1) == 1: # mutex
                    continue
                
                ## START CRITICAL SECTION
                if v < best_value[0]:
                    for i in range(ldim):
                        best[i] = bees[x, i]
                best_value[0] = v
                ## END CRITICAL SECTION
                
                mutex[0] = 0

    @cuda.jit
    def bee_abandoned_iteration(
        bees : DeviceNDArray,
        limits : DeviceNDArray,
        ldim : int,
        trials : DeviceNDArray,
        num_bees : int,
        fitness : DeviceNDArray,
        rng_states : DeviceNDArray,
        max_trials : int
    ):
        x = cuda.grid(1)

        if x < math.ceil(num_bees/2) and trials[x] <= 0:
            tid = x
            for y in range(ldim):
                bees[x, y] = xoroshiro128p_uniform_float32(rng_states, tid)*(limits[y, 1]-limits[y, 0]) + limits[y, 0]
            fitness[x] = fun_gpu(bees[x, :])
            trials[x] = max_trials

    fun_gpu = cuda.jit(device=True)(fun)

    num_bees, ldim = bees.shape

    sum_value = cuda.device_array(shape=1)
    if best is None:
        best = cuda.device_array(shape=ldim)
        bees[0].copy_to_device(best)
    if best_value is None:
        best_value = cuda.device_array(shape=1)
        best_value[0] = fitness[0]
    mutex = cuda.to_device(np.array([0]))
    minmax_values = cuda.device_array(shape=2)

    employer_bees = math.ceil(num_bees/2)
    onlooker_bees = num_bees - employer_bees

    new_bees = cuda.device_array((employer_bees, ldim))
    trials = cuda.device_array(shape=employer_bees, dtype=np.int8)
    probs = cuda.device_array(shape=employer_bees)

    threadsPerBlock = MAGIC_NUM
    blocks_employer = math.ceil(employer_bees/threadsPerBlock)
    blocks_onlooker = math.ceil(onlooker_bees/threadsPerBlock)

    seed = int(time())
    rng_states = create_xoroshiro128p_states(employer_bees, seed = seed + 1)

    if enable_tqdm:
        rg = tqdm(range(num_iterations))
    else:
        rg = range(num_iterations)


    for _ in rg:
        bee_employer_iteration[blocks_employer, threadsPerBlock](bees, new_bees, fitness, trials, max_trials, rng_states, num_bees, ldim, minmax_values, sum_value, mutex, best, best_value)
        compute_probabilities[blocks_employer, threadsPerBlock](bees, fitness, probs, minmax_values, sum_value)
        bee_onlooker_iteration[blocks_onlooker, threadsPerBlock](bees, fitness, rng_states, ldim, num_bees, probs, mutex, best, best_value)
        bee_abandoned_iteration[blocks_employer, threadsPerBlock](bees, limits, ldim, trials, num_bees, fitness, rng_states, max_trials)
    

    return bees, fitness, best, best_value

