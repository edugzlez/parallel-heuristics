from biopar.benchmark import schaffer
import numpy as np
import warnings
from biopar.benchmark import functions
from tqdm import tqdm
import pandas as pd
from biopar.abc import ABC 
from biopar.de import DE 
from biopar.pso import PSO
from time import time
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    ns = [100, 1000, 10000]
    results = pd.DataFrame()
    iterations_limit = 2000


    CRs = [np.random.uniform(0, 1)]
    Fs = [np.random.uniform(0, 2)]
    ws = [np.random.uniform(0, 1)]
    phi_gs = [np.random.uniform(0, 1)]
    phi_ps = [np.random.uniform(0, 1)]
    max_trials_s = [5]

    h = len(CRs)*len(Fs)+len(ws)*len(phi_ps)*len(phi_ps)+len(max_trials_s)
    s = [n*h for n in ns]
    
    sits = 1000

    def mygen():
        pbar = tqdm(functions.items(), desc="functions")
        mbar = tqdm(total = sum(s), desc = "ns")

        for name, args in pbar:
            target = args['fun']
            limits = args['limits']
            min_value = args['min_value']
            ext = args['ext']
            if ext:
                n = np.random.randint(2, 10)
                limits = np.vstack([limits]*n)
            
            mbar.reset()

            for n in ns:
                ## DE
                for CR in CRs:
                    for F in Fs:
                        exp_name = f"{name}|{n}|DE|{CR}|{F}"
                        de = DE(target, limits, n, "gpu")
                        better_value = float('inf')
                        num_iterations = 0

                        i = time()
                        while num_iterations < iterations_limit and abs(better_value - min_value) > 0.01:
                            de.iterate(sits, CR, F)
                            _, better_value = de.getResult()
                            num_iterations += sits
                        j = time()

                        yield {
                            "exp_name" : exp_name,
                            "num_iterations" : num_iterations,
                            "better_value" : better_value,
                            "abs" : abs(better_value - min_value),
                            "elapsed" : j-i
                        }
                        mbar.update(n)
                ## PSO
                for w in ws:
                    for phi_p in phi_ps:
                        for phi_g in phi_gs:
                            exp_name = f"{name}|{n}|PSO|{w}|{phi_p}|{phi_g}"
                            pso = PSO(target, limits, n, "gpu")
                            better_value = float('inf')
                            num_iterations = 0

                            i = time()
                            while num_iterations < iterations_limit and abs(better_value - min_value) > 0.01:
                                pso.iterate(sits, w, phi_g, phi_p)
                                _, better_value = de.getResult()
                                num_iterations += sits
                            j = time()

                            yield {
                                "exp_name" : exp_name,
                                "num_iterations" : num_iterations,
                                "better_value" : better_value,
                                "abs" : abs(better_value - min_value),
                                "elapsed" : j-i
                            }
                            mbar.update(n)

                ## ABC
                for max_trials in max_trials_s:
                    exp_name = f"{name}|{n}|ABC|{max_trials}"
                    abc = ABC(target, limits, n, "gpu")
                    better_value = float('inf')
                    num_iterations = 0

                    i = time()
                    while num_iterations < iterations_limit and abs(better_value - min_value) > 0.01:
                        abc.iterate(sits, max_trials)
                        _, better_value = de.getResult()
                        num_iterations += sits
                    j = time()

                    yield {
                        "exp_name" : exp_name,
                        "num_iterations" : num_iterations,
                        "better_value" : better_value,
                        "abs" : abs(better_value - min_value),
                        "elapsed" : j-i
                    }
                    mbar.update(n)
    
    # results = pd.DataFrame(mygen())
    
    args = functions['styblinkski']
    target = args['fun']
    limits = args['limits']
    min_value = args['min_value']
    ext = args['ext']

    agents = [100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
    y_cpu = []
    y_gpu = []
    for n_agents in 
        CR = np.random.random()
        F = np.random.random()

        i = time()
        de = DE(target, limits, n_agents, "cpu")
        de.iterate(10000, CR, F, True)
        _, better_value = de.getResult()
        j = time()
        y_cpu.append(j-i)
        print(f"CPU: {j-i}s")

        i = time()
        de = DE(target, limits, n_agents, "gpu")
        de.iterate(10000, CR, F, True)
        _, better_value = de.getResult()
        j = time()
        y_gpu.append(j-i)
        print(f"GPU: {j-i}s")
    