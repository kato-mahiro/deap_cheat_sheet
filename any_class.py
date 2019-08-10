import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

class myclass:
    def __init__(self):
        self.gene1 = numpy.random.rand(2,2)
        self.gene2 = "gene"
        self.gene3 = random.choice([0,1])

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("create_ind", myclass, )
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.create_ind,n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    #return sum(individual),
    print(float(numpy.sum(individual[0].gene1)))
    return float(numpy.sum(individual[0].gene1)),

def cxTwoPointCopy(ind1, ind2):
    #size = len(ind1)
    #cxpoint1 = random.randint(1, size)
    #cxpoint2 = random.randint(1, size - 1)
    #if cxpoint2 >= cxpoint1:
        #cxpoint2 += 1
    #else: # Swap the two cx points
        #cxpoint1, cxpoint2 = cxpoint2, cxpoint1
#
    ##ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
    
def myMutation(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])
            individual[i][0].gene1 = numpy.random.rand(2,2)
    return individual,
    
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", myMutation, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    
    pop = toolbox.population(n=300)
    import pdb; pdb.set_trace()
    
    hof = tools.HallOfFame(1, similar=numpy.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats,halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    main()
