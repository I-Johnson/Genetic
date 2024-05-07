import numpy as np
import random
import matplotlib.pyplot as plt

# generate random gaussian graph with num_cities cities
def make_random_gaussian_graph(num_cities):
    rng = np.random.default_rng(seed=42)  # Seed for reproducibility
    cities = rng.normal(0, 1, (num_cities, 2))
    dists = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dists[i, j] = dists[j, i] = np.linalg.norm(cities[i] - cities[j])
    return cities, dists

# generate a child from two parents using crossover 
def crossover(parent1, parent2):
    split = random.randint(1, len(parent1) - 2)
    child = parent1[:split] + [city for city in parent2 if city not in parent1[:split]]
    return child
# mutate a route with a mutation rate
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# run the genetic algorithm with the given parameters and return the best route, best distance, distance history and plot progress
def genetic_algorithm(num_cities, dists, population_size, num_generations, mutation_rate):
    population = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]
    distance_history = []
    # used to calculate the distance of a tour 
    def calculate_distance(tour):
        return sum(dists[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
 # used to plot the progress of the genetic algorithm
    def plot_progress():
        plt.figure(figsize=(10, 5))
        plt.plot(distance_history, 'b-o', label='Distance (Fitness)')
        plt.xlabel('Generation')
        plt.ylabel('Total Distance')
        plt.title('Improvement over Generations')
        plt.legend()
        plt.grid(True)
        plt.show()
## Run the genetic algorithm over the specified number of generations
    for generation in range(num_generations):
        distances = [calculate_distance(route) for route in population]
        best_index = np.argmin(distances)
        distance_history.append(distances[best_index])
        new_population = [population[best_index]]  # Elitism
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    best_route = population[np.argmin([calculate_distance(route) for route in population])]
    return best_route, calculate_distance(best_route), distance_history, plot_progress

# Parameters and setup
num_cities = 85
cities, dists = make_random_gaussian_graph(num_cities)
population_size = 50
num_generations = 200
mutation_rate = 0.1

# Running the genetic algorithm
best_route, best_distance, distance_history, plot_progress = genetic_algorithm(num_cities, dists, population_size, num_generations, mutation_rate)
plot_progress()

# Plotting the best route
plt.figure(figsize=(10, 6))
plt.scatter(cities[:, 0], cities[:, 1], c='red', label='Cities')
for i in range(num_cities):
    start_pos = cities[best_route[i]]
    end_pos = cities[best_route[(i + 1) % num_cities]]
    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k--')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Best Route Found')
plt.legend()
plt.show()

print("Best route distance:", best_distance)
