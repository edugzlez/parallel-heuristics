from types import NoneType
import numpy as np
from typing import *

class GeneticAlgorithm:

    population : np.ndarray # population[i, :] = value of person number i
    evaluation_function : Callable[[np.ndarray], float] # evaluation_function(person)

    cross_probability : float
    cross : Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], NoneType] # cross

    mutate_probability : float
    mutate : Callable[[np.ndarray], NoneType]

    def __init__(self, evaluation_function):
        pass