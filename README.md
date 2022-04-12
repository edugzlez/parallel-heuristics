# Paralelización de metaheurísticas

Este repositorio es resultado de un periodo de prácticas como becario en el Departamento de Sistemas Informáticos y Computación de la Universidad Complutense de Madrid.

## Cómo instalar
Debes instalar las librerías necesarias
```
pip install -r requirements.txt
```

Además debes disponer de una GPU NVIDIA y del CUDA Toolkit instalado: https://developer.nvidia.com/cuda-downloads

Se hace notar que la versión de `numpy` que se utiliza es a lo sumo la 1.21, pues la 1.22 aún no es compatible con `numba`.

```
pip install git+https://github.com/edugzlez/parallel-heuristics
```

## Cómo usar
### Differential Evolution
```python
from biopar.de import DE
from biopar.benchmark import functions


fun_data = functions['styblinsky']
num_agents = 1000
CR = 0.2
F = 0.8

de = DE(fun_data['target'], fun_data['limits'], num_agents, mode="gpu")
de.iterate(100, CR, F, enable_tqdm=True)
best_vector, best_value = de.getResult()
```

### Particle Swarm Optimization
```python
from biopar.pso import PSO
from biopar.benchmark import functions


fun_data = functions['styblinsky']
num_particles = 1000
w = 0.7
phi_p = 0.4
phi_g = 0.6

pso = PSO(fun_data['target'], fun_data['limits'], num_particles, mode="gpu")
pso.iterate(100, w, phi_p, phi_g, enable_tqdm=True)
best_vector, best_value = pso.getResult()
```

### Artificial Bee Colony
```python
from biopar.abc import ABC
from biopar.benchmark import functions


fun_data = functions['styblinkski']
num_bees = 1000
max_trials = 5

abc_ = ABC(fun_data['fun'], fun_data['limits'], num_bees, mode="gpu")
abc_.iterate(100, max_trials, enable_tqdm=True)
best_vector, best_value = abc_.getResult()
```


### Utilizar funciones propias

Las funciones que se utilicen deben ser muy limitadas y utilizar operaciones matemáticas básicas o que estén presentes en la librería `math`. Puedes ver todas las operaciones permitidas en: https://numba.pydata.org/numba-doc/latest/cuda/cudapysupported.html.

```python
def my_function(input):
    x = input[0]
    y = input[1]

    return (x-5)**2 + (y-3.5)**2

limits = np.array([
    [-10, 0],   # coordenada x
    [ -7, 0]     # coordenada y
])
```
