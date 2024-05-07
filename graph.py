'''
graph.py -- William Bailey -- April 2024
A basic class implementation of a weighted, directed graph.
'''
import numpy as np

class Graph:
    '''
    Represents a weighted, directed graph.
    '''
    def __init__(self, node_names: list, weights: np.ndarray, city_locations=None):
        '''
        Construct a Graph object from a list of node names, 
        and the weights on the edges between them.
        '''
        assert len(weights) == len(weights[0]) and len(weights) == len(node_names), 'Size mismatch!'
        self.node_names = node_names
        self.nodes = list(range(len(self.node_names)))
        self.weights = weights
        self.city_locations = city_locations


    def weight_from_i_to_j(self, i: int, j: int) -> int:
        '''
        Returns the weight on the edge from vertex i to vertex j.
        '''
        return self.weights[i][j]


    def get_tour_weight(self, tour):
        '''
        Returns the total weight of traveling the entire circuit from 
        tour[0] tour[1], ..., tour[N-1], tour[0]
        '''
        return sum(self.weights[tour[i-1]][tour[i]] for i in range(0, len(tour)))


    def get_node_locations(self) -> np.ndarray:
        return self.city_locations


    def get_city_name_for_index(self, i):
        return self.node_names[i]


    def generate_nna_tour(self) -> list:
        N = len(self.nodes)
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                # if self.weight_from_i_to_j(i, j): # ignore zero edges (self loops not permitted)
                edges.append([i, j, self.weight_from_i_to_j(i, j)]) # Assumes undirected graph.
        
        edges.sort(key=lambda tpl: tpl[2]) # Sort by weight

        tour = []
        for edge in edges:
            if len(tour) == N:
                break

            v1, v2, weight = edge
            if degree_in_tour(v1, tour) < 2 and degree_in_tour(v2, tour) < 2:
                tour.append(edge)
                print(tour)

        return tour

def degree_in_tour(i: int, tour: list[tuple[int,int,float]]) -> int:
    if len(tour) == 0:
        return 0
    
    return sum(i in edge[:2] for edge in tour)