import random
import math

# define the function to be optimized
def function_to_optimize(x):
    return ((2*x)-(x**2))/16

# define the search interval
search_interval = (0, 31)

# define the genetic algorithm parameters
population_size = int(input("Enter the population size: "))
mutation_rate = float(input("Enter the mutation rate: "))
generations = int(input("Enter the number of generations: "))
#population_size = 10
#mutation_rate = 0.1
#generations = 5

# define the individual representation and initialization
def create_individual():
    return random.uniform(search_interval[0], search_interval[1])

# define the fitness function
def fitness(individual):
    return function_to_optimize(individual)

# define the selection operator
def selection(population):
    return random.sample(population, 2)

# define the crossover operator
def crossover(parents):
    return (parents[0] + parents[1]) / 2

# define the mutation operator
def mutation(individual):
    return individual + random.uniform(-1, 1) * mutation_rate

# initialize the population
population = [create_individual() for i in range(population_size)]

# iterate over the generations
for i in range(generations):
    # evaluate the fitness of the population
    fitness_values = [fitness(individual) for individual in population]
    # select the parents for reproduction
    parents = [selection(population) for j in range(population_size // 2)]
    # perform crossover to generate the offspring
    offspring = [crossover(parents[j]) for j in range(population_size // 2)]
    # perform mutation on the offspring
    offspring = [mutation(individual) for individual in offspring]
    # replace the least fit individuals with the offspring
    population = sorted(list(zip(population, fitness_values)), key=lambda x: -x[1])[:population_size // 2]
    population = [individual for individual, fitness in population] + offspring

# print the best solution found
best_individual = max(population, key=fitness)
print("The maximum value of the function is", fitness(best_individual), "at x =", best_individual)
