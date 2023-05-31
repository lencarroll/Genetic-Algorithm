# Genetic-Algorithm

This is a basic genetic algorithm script. Elitism, crossover and mutation is included. Crossover is done randomized, based on weights or based on rank. Number of generations do not have to be set. 

All you need to do is set up a main python file which includes a fitness function, the parameters and the particular data you want to learn from. An example and explanation follows.

```
import random
from random import choice
import numpy as np
import math

# This is my test X and Y data points
X = np.arange(0.1, 10, 0.1)
Y = np.array(2*(np.array(X))**2 - 3*(np.array(X)) + 5)

# Create a list which contains all your data. 
DATA_VALS = [X,Y]

# These are my GA parameters, you can have less or more.
p0 = range(-10,10)
p1 = range(-10,10)
p2 = range(-10,10)

#Create a list of the range of GA parameters
PARAMETERS = [p0,p1,p2]

#Write your fitness function. It must have the following:
#Only two arguments, one being individual and the other one your list of all your data.
#The fitness score must give preference to larger is better. In the example below we use the RMSD function to calculate the fitness score.
#The lower the RMSD value, the better, but since this GA code gives preference to a larger fitness value, the RMSD value has to be inverted.
#Since the RMSD value can be 0, I used 1/(1+RMSD), meaning the maximum value is 1, while the minimum value is 0.
#In this fitness function, I split the data from DATA_VALS into two separate fragments, X and Y. 
#The individual parameter is a particular list of parameters that the GA script will produce at that moment. If 3 parameters are required, then
#individual will be a list of three values. Separate those parameters and use them in your fitness function.
#In my example, I am trying to figure out what the parameters should be for the equation:
#aX^2 + bX + c
#I set them above as 2, -3, 5, so the hope is that the code will get it right.
#I use my X values I loaded into the function and the parameters generated, and calculated new Y values.
#The new Y values are compared to the correct Y values with a RMSD function. If the values are spot on, the RMSD value will be 0.
def fitness(individual, DATA_VALS):
    X = np.array(DATA_VALS[0])
    Y = np.array(DATA_VALS[1])
    
    P1 = individual[0]
    P2 = individual[1]
    P3 = individual[2]

    Y_new = P1*(X)**2 + P2*(X) + P3

    RMSD = []
    for i in range(len(Y)):
        RMSD_value = np.sqrt((Y_new[i]-Y[i])**2)
        if math.isnan(RMSD_value):
            return 0
        else:
            RMSD.append(RMSD_value)
    RMSD = np.mean(np.array(RMSD))
    return 1/(1+RMSD)

#What is the minimum fitness score you are looking for, I set mine as 0.95
FIT_PARAM = 0.95
#What is your population size per generation, I set mine as 1000
POPULATION_SIZE = 1000
#I chose a mutation rate of 1% or 0.01
MUTATION_RATE = 0.01
#I chose a crossover rate of 80% or 0.8
CROSSOVER_RATE = 0.8
#Here an elitism rate of 10% or 0.1 is chosen
ELITISM_RATE = 0.1
#Insert the name of your fitness function, mine was just fitness
FITNESS_FUNC_NAME = fitness
#Insert the name of your list of data
DATA_VALUES = DATA_VALS
#Insert the number of generations you want to use. -1 means don't stop until a particular result is achieved, otherwise choose a value bigger than 1
NUMBER_OF_GENERATIONS = -1
#What type of crossover method do you want to use, Random, Weighted or Ranked?
CROSSOVER_METHOD = "Weighted"
#If verbose is 1, the result of each generation is outputted. If the value is 0, only the final result is outputted
VERBOSE = 1

from GA import geneticAlgorithm
GA = geneticAlgorithm(PARAMETERS, FIT_PARAM, FITNESS_FUNC_NAME, DATA_VALUES, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, ELITISM_RATE, NUMBER_OF_GENERATIONS, CROSSOVER_METHOD, VERBOSE)
```
