import random, time, matplotlib.pyplot as plt, numpy as np, copy
""" 
set up variables
"""
noFriends, noIndividuals, generations, mutationRate, probabilityGeneSwap, tournamentSize = 100, 100, 100, 0.05, 0.6, 3
fitness_over_generations = np.zeros(generations)  # for use in graph
start_time = time.time()  # start timer

# set genome for individuals (row = individual genome, columns = individual genes)
population = np.random.choice([-1, 1], size=(noIndividuals, noFriends))

# relationship matrix
relationship = np.random.choice([-1, 1], size=(noFriends, noFriends))
for i in range(noFriends):
    relationship[i, i] = 0

# initialize fitness array
fitness = np.zeros(noIndividuals)

""" 
Iterate through generations 
"""
for i in range(generations):
    """ initialize generation """
    counter = 0  # newPop counter
    newPop = np.zeros((noIndividuals, noFriends))

    """ Calculate fitness """
    for ff in range(noIndividuals):
        fitness[ff] = np.matmul(np.matmul(population[ff, :].transpose(), relationship), population[ff, :])
    fitness_over_generations[i] = max(fitness)

    while counter < noIndividuals - 1:
        """ Tournament """
        # select competitors and find fitness
        competitorsIndex = np.array(random.sample(range(len(population)), tournamentSize))  # choose competitor index
        competitorsFitness = [fitness[index] for index in competitorsIndex]  # select competitors

        # index of fittest and lowest value
        highest, lowest = np.where(fitness == max(competitorsFitness)), np.where(fitness == min(competitorsFitness))
        winner, loser = highest[0][0], lowest[0][0]

        # assign winner to newPop for elitism
        newPop[counter, :] = population[winner]

        """ mate """
        for g in range(noFriends):
            # assign genes fromm winner to loser based on a probability
            if random.random() < probabilityGeneSwap:
                newPop[counter + 1, g] = population[winner, g]
            else:
                newPop[counter + 1, g] = population[loser, g]

            # mutation
            if random.random() < mutationRate:
                if newPop[counter + 1, g] == 1:
                    newPop[counter + 1, g] = -1
                else:
                    newPop[counter + 1, g] = 1
        # update counter
        counter += 2

    """ assign newPop to population"""
    population = copy.deepcopy(newPop)

"""
Display results
"""
# time taken and top fitness
time_taken = time.time() - start_time
print('Time taken was:', time_taken)
print('Top fitness is', max(fitness_over_generations))

# graph of fittest values over time
plt.plot(range(generations), fitness_over_generations)
plt.xlabel('Generations'), plt.ylabel('Fitness'), plt.title('Max fitness over generations'), plt.show()
