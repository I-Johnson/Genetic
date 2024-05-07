from __future__ import annotations
from typing import Type
from abc import abstractmethod, ABC
import random

class Individual:
    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def fitness(self) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def __lt__(self, other: Individual) -> bool:
        '''Implements < operator. Returns True if calling object is more fit than other.'''
        raise NotImplementedError
    
    @abstractmethod
    def crossover_with(self, other: Individual) -> list[Individual]:
        raise NotImplementedError
    
    @abstractmethod
    def mutate_with_probability(self, p: float) -> None:
        '''Mutates the calling object with probability p.'''
        raise NotImplementedError


class GeneticAlgorithm(ABC):

    def __init__(self, population_size: int, mutation_prob: float):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.population = [ 
            self.create_new_individual()
            for _ in range(self.population_size)
        ]
        self.population.sort() # uses Individual.__lt__()

    @staticmethod
    @abstractmethod
    def create_new_individual() -> Individual:
        # Create a new Individual object, typically involving randomness.
        raise NotImplementedError

    @abstractmethod
    def reproducing_subset(self) -> list[Individual]:
        raise NotImplementedError

    def run_genetic_algorithm(self, max_generations: int):
        # NOTE: self.population initalized in constructor
        for _ in range(max_generations):
            reproducers = self.reproducing_subset()
            
            while len(reproducers) >= 2:
                parent1 = reproducers.pop(random.randint(0, len(reproducers)-1))
                parent2 = reproducers.pop(random.randint(0, len(reproducers)-1))
                children = parent1.crossover_with(parent2)
                for child in children:
                    child.mutate_with_probability(self.mutation_prob)
            
                self.population.extend(children)
                self.population.sort()
                self.population = self.population[:self.population_size]

        return self.population[0]
    
if __name__ == "__main__":
    print("Are you lost? You look lost.")
