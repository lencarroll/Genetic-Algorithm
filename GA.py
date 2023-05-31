import random
from random import choice
import numpy as np
import multiprocessing
import math

class geneticAlgorithm:
    def __init__(self, PARAMETERS, QUIT_VAL, FITNESS_LOC, DATA_VALS, POPULATION_SIZE = None, MUTATION_PERCENTAGE = None, CROSSOVER_RATE = None, ELITISM_RATE = None, NUMBER_OF_GENERATIONS = None, GA_TYPE = None, VERBOSE = None):

        #Here we assign the default arguments:
        if POPULATION_SIZE==None:
            POPULATION_SIZE = 100
        if MUTATION_PERCENTAGE == None:
            MUTATION_PERCENTAGE = 0.01
        if CROSSOVER_RATE == None:
            CROSSOVER_RATE = 0.8
        if ELITISM_RATE == None:
            ELITISM_RATE = 0.1
        if NUMBER_OF_GENERATIONS == None:
            NUMBER_OF_GENERATIONS = -1
        if GA_TYPE == None:
            GA_TYPE = "Random"
        if VERBOSE == None:
            VERBOSE = 1

        POPULATION_SIZE = int(POPULATION_SIZE)
        NUMBER_OF_GENERATIONS = int(NUMBER_OF_GENERATIONS)

        if MUTATION_PERCENTAGE < 0 or CROSSOVER_RATE < 0 or ELITISM_RATE < 0 or (VERBOSE !=0 or VERBOSE != 1) or POPULATION_SIZE == 0 or NUMBER_OF_GENERATIONS == 0:
            print("Your arguments are incorrect! Please fix it")
            exit()

        #Assign the user's fitness function to self.fitness
        self.fitness = FITNESS_LOC

        #Generate an initial population from the parameters defined by the user.
        population = []
        for i in range(POPULATION_SIZE):
            pop_val = []
            for j in range(len(PARAMETERS)):
                pop_val.append(random.choice(PARAMETERS[j]))
            population.append(pop_val)


        # The Crossover function is defined herer
        def crossover(parent1, parent2):
            # Crossover is done in such a way that if the parent has an even number of parameters, the child will have equal traits from both parents. But if the parent has odd parameters
            # the child will have more parameters from one parent than the other

            child1 = []
            child2 = []

            for i in range(len(parent1)):
                if i%2==0:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
                elif i%2==1:
                    child1.append(parent2[i])
                    child2.append(parent1[i])

            return child1, child2


        # The mutation function is defined here
        def mutate(individual, mutation_rate):
            # Here we take the value of the individual's parameters and then multiply that parameter with the mutation rate. We then construct a negative to positive range from this value and
            # add it to the individual. For example
            # Let's say the parameter is 5 and the mutation rate is 1%, then the range of values will be [-0.05,0.05)
            # A random number is then selected between [-0.05, 0.05), let's say -0.002, and is added to the individual
            # 5 - 0.002 = -4.998
            for i in range(len(individual)):
                individual[i] += random.uniform(-np.abs(individual[i]*mutation_rate),np.abs(individual[i]*mutation_rate))
                return individual

        # Here we define the elitism function, where we decide how many individuals from the population go through to the next generation unchanged
        def elitism(population, fitness_scores,elitism_rate, population_size):
            new_population_index = []
            new_population = []
            # The user defines a value elitism_rate, which is the percentage of the fittest individuals that will move straight to the next generation.
            # Let's say the population has 50 individuals and the rate is 10%, that means the fittest 50*0.1 = 5 indivuals from this generation will go
            # to the next generation wouldn't crossing or being mutated
            for i in range(int(population_size*elitism_rate)):
                temp_max = max(fitness_scores)
                if temp_max not in new_population:
                    new_population_index.append(fitness_scores.index(temp_max))
                    new_population.append(temp_max)
                else:
                    temp_max = 0
                    for j in range(len(fitness_scores)):
                        if fitness_scores[j]>temp_max and fitness_scores[j] not in new_population:
                            temp_max = fitness_scores[j]

                    new_population_index.append(fitness_scores.index(temp_max))
                    new_population.append(temp_max)

            return new_population_index


        # This function deals with the creation of the offspring and the mutation of them
        def createchildren(population, fitness_scores, crossover_rate, elitism_rate, mutation_rate, population_size, ga_type):
            new_population = []
            # The user defines values for mutation rate, population size, elitism rate and crossover rate.
            # Above is already explained what all these mean except crossover rate. Here it means the total percentage of individuals who will move to the next generation
            # either unaletered or via their offspring
            # So if elitism rate is 0.1 and crossover rate is 0.8, then 0.8 individuals move on to the next generation
            # That being said, we already dealt with the elite individuals, so we need to deal with the 0.8-0.1 = 0.7 or 70% of other individuals

            # If selector random is chosen, then N number of offspring will be generated for the next genration from N random individuals of the current generation.
            # N has to be an even number
            if ga_type == "Random" or ga_type == "random":
                range_val = math.floor(population_size*(crossover_rate-elitism_rate)/2)
                if range_val > math.floor(len(population)/2):
                    range_val = math.floor(len(population)/2)
                for i in range(range_val):
                    parent1 = random.choices(population)[0]
                    j = 0
                    while j==0:
                        parent2 = random.choices(population)[0]
                        if parent2==parent1:
                            j+=0
                        else:
                            j+=1

                    child1, child2 = crossover(parent1, parent2)
                    child1 = mutate(child1, mutation_rate)
                    child2 = mutate(child2, mutation_rate)
                    new_population.append(child1)
                    new_population.append(child2)
                    parent1_index = population.index(parent1)

                    population.remove(parent1)
                    fitness_scores.remove(fitness_scores[parent1_index])
                    
                    parent2_index = population.index(parent2)
                    population.remove(parent2)
                    fitness_scores.remove(fitness_scores[parent2_index])
                  
                return (population,new_population,fitness_scores)

            # If selector ranked is chosen, then N number of offspring will be generated for the next genration from the N fittest individuals of the current remaining generation.
            # N has to be an even number
            elif ga_type == "Ranked" or ga_type == "ranked":
                range_val = math.floor(population_size*(crossover_rate-elitism_rate)/2)
                if range_val > math.floor(len(population)/2):
                    range_val = math.floor(len(population)/2)
                for i in range(range_val):
                    max_parent1 = fitness_scores.index(max(fitness_scores))
                    parent1 = population[max_parent1]
                    population.remove(parent1)
                    fitness_scores.remove(max(fitness_scores))
                    max_parent2 = fitness_scores.index(max(fitness_scores))
                    parent2 = population[max_parent2]
                    population.remove(parent2)
                    fitness_scores.remove(max(fitness_scores))

                    child1, child2 = crossover(parent1, parent2)
                    child1 = mutate(child1, mutation_rate)
                    child2 = mutate(child2, mutation_rate)
                    new_population.append(child1)
                    new_population.append(child2)

                return (population, new_population, fitness_scores)

            # If selector weighted is chosen, then N number of offspring will be generated for the next genration from N random individuals of the current generation, but.
            # weights are applied, giving preference to the fittest individuals, however weaker individuals could still be selected
            # N has to be an even number
            elif ga_type == "Weighted" or ga_type == "weighted":
                range_val = math.floor(population_size*(crossover_rate-elitism_rate)/2)
                if range_val > math.floor(len(population)/2):
                    range_val = math.floor(len(population)/2)
                for i in range(range_val):
                    parent1 = random.choices(population, weights=fitness_scores)[0]
                    j = 0
                    while j==0:
                        parent2 = random.choices(population, weights=fitness_scores)[0]
                        if parent2==parent1:
                            j+=0
                        else:
                            j+=1

                    child1, child2 = crossover(parent1, parent2)
                    child1 = mutate(child1, mutation_rate)
                    child2 = mutate(child2, mutation_rate)
                    new_population.append(child1)
                    new_population.append(child2)
                    parent1_index = population.index(parent1)

                    population.remove(parent1)
                    fitness_scores.remove(fitness_scores[parent1_index])
                    
                    parent2_index = population.index(parent2)

                    population.remove(parent2)
                    fitness_scores.remove(fitness_scores[parent2_index])

                return (population, new_population, fitness_scores)

            # If the crossover rate plus the elitism rate exceeds the value of 1, the crossover rate will be adjusted to be equal to 1 - elitism rate.

        
        k = 1
        while k!=0:
            # Evaluate the fitness of each individual in the population
            # The generation number is outputted here. Unlike some other (probably more efficient) GA codes out there, you can let the generations
            # continue until you get your desired result
            if VERBOSE == 1:
                print("Generation Step:",k)

            # Here the fitness score for each individual is calculated
            fitness_scores = [self.fitness(individual, DATA_VALS) for individual in population]

            #The max fitness score is determined
            max_fitness = max(fitness_scores)
            max_fittest = population[fitness_scores.index(max(fitness_scores))]

            # Here we check that if the max fitness value is above a certain criterion, stop the GA process.
            if max(fitness_scores) >= QUIT_VAL:
                fittest_individual = population[fitness_scores.index(max(fitness_scores))]
                print("Best Score:",max(fitness_scores))
                print("Fittest Individual:",fittest_individual)
                self.fittest = fittest_individual
                k = 0
            # If for some reason the algorithm stop, this is just to be safe
            if k==0:
                break

            # Assuming the max fitness value is not good enough, we generate a new generation

            # The most elite individuals are determined and sent through to the next generation
            new_population_index = []
            if ELITISM_RATE != 0:
                new_population_index = elitism(population, fitness_scores, ELITISM_RATE, POPULATION_SIZE)

            # A new population is created
            new_population = []
            for i in new_population_index:
                new_population.append(population[i])

            # Individuals from the new population is removed from the old population
            for i in new_population:
                pop_index = population.index(i)
                population.remove(i)
                fitness_scores.remove(fitness_scores[pop_index])

            # We create new individuals through a crossover and mutation. This function removes old population members.
            crossover_population = createchildren(population, fitness_scores, CROSSOVER_RATE, ELITISM_RATE, MUTATION_PERCENTAGE, POPULATION_SIZE, GA_TYPE)
            population = crossover_population[0]
            fitness_scores = crossover_population[2]

            # Add the new individuals to the new population
            for i in range(len(crossover_population[1])):
                new_population.append(crossover_population[1][i])

            # If the crossover and elitism didn't create enough individuals, we make up the rest here.
            if (POPULATION_SIZE-len(new_population))>=1:
                for i in range(int(POPULATION_SIZE-len(new_population))):
                    pop_val = []
                    for j in range(len(PARAMETERS)):
                        pop_val.append(random.choice(PARAMETERS[j]))
                    new_population.append(pop_val)

            # We reset population to be the new population
            population = new_population

            # The current best score and current fittest individual is outputted if wanted.
            if VERBOSE == 1:
                print("Current Best Score:",max_fitness)
                print("Current Fittest Individual:",max_fittest)

            # If the user sets a finite number of generations, if that number is reached we quit
            if NUMBER_OF_GENERATIONS != -1:
                if k==NUMBER_OF_GENERATIONS:
                    fittest_individual = population[fitness_scores.index(max(fitness_scores))]
                    self.fittest = fittest_individual
                    print("Max Generation Reached!")
                    print("Best Score:",max_fitness)
                    print("Fittest Individual:",max_fittest)
                    k = 0

            if k == 0:
                break
            
            k+=1
