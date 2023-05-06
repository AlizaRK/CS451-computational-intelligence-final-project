import random
import math
import matplotlib.pyplot as plt
import numpy as np

class Rastrigin:
    '''
        Finding the gloal minimum of Rastrigin function using Genetic Algorithm
    '''
    def __init__(self,population_size, num_gen,num_parent, dim):
        self.pop_size = population_size
        self.pop=np.random.uniform(low=-10.0, high=10.0, size=(population_size,dim))
        self.num_gen=num_gen
        self.num_parents=num_parent
        self.dim=dim
        # stores best fitness of each generation
        self.BFs = []

    
    def rastrigin_func(self,x):
        '''
        Rastrigin func - Returns fitness of the chromosome
        '''
        sum = 0
        for a in x:
            sum += ((a*a)-(10*np.cos(2*np.pi*a)))
        sum += 10*len(x)
        return sum

    def calculate_fitness(self,population):
        '''
        Returns fitness scores of all chromosomes in current population
        '''
        fitness=[]
        for chromosome in population:
            fitness.append(self.rastrigin_func(chromosome))
        return fitness

    def truncation(self, fitness_scores, population, selection_size):
        """
        Returns fittest chromosomes, of size = selection_size
        """
        parents = np.empty((selection_size, population.shape[1]))
        for parent_idx in range(selection_size):
            fitness_idx = np.argmin(fitness_scores)
            parents[parent_idx] = population[fitness_idx]
            fitness_scores[fitness_idx] = np.inf
        return parents


    def crossover(self, parents, num_children):
        '''
        Crossover at middle and return relevant child chromosome
        '''
        children = np.empty((num_children, self.dim))
        cp = self.dim // 2
        for i in range(0,num_children, 2):
            parent1 = parents[i]
            parent2 = parents[ i + 1]
            children[i, :cp] = parent1[:cp] 
            children[i, cp:] = parent2[cp:]
            children[i +1, cp:] = parent1[cp:]
            children[i +1, :cp] = parent2[:cp]
        return children

    def mutation(self,children):
        '''
        Mutates a random numerical value of each child chromosome 
        '''
        for i in range(len(children)):
            chromosome_idx = np.random.randint(0, self.dim)
            random_value = np.random.uniform(-1.0, 1.0)
            children[i, chromosome_idx] -= random_value
        return children



    def many_gens(self):
        '''
        Run a single generation of the problem  - [parent selection, crossover, mutation, survivor selection]
        Update the chromose population as the evolutionary process takes place
        '''

        for generation in range(self.num_gen):
            fitness=self.calculate_fitness(self.pop)

            # parent selection
            parents=self.truncation(fitness,self.pop,self.num_parents)

            # create offspring from these parents
            children = self.crossover(parents,num_children=self.num_parents)
            children_mutated =self.mutation(children)

            # survivor selection
            self.pop = np.append(self.pop, children_mutated,axis=0 )
            fitness=self.calculate_fitness(self.pop)
            self.pop=self.truncation(fitness,self.pop,self.pop_size)

            # The best fitness in the current generation
            bf=np.min(self.calculate_fitness(self.pop))
            self.BFs.append(bf)
            if(generation%100==0):
                print("Best Fitness in generation {0}:{1}".format(generation,bf ))
        # Getting the best solution in last generation
        final_fitness_scores = self.calculate_fitness(self.pop)

        # Return the the best fitness.
        bf_index = np.argmin(final_fitness_scores)
        print("Best Solution : ", self.pop[bf_index])
        print("Best Solution Fitness :   ", np.min(final_fitness_scores))
        return self.BFs

    def plot_graph(self):
        '''
        Plots graphs for best fitness
        '''

        # BFs plotting
        plt.semilogy(self.BFs)
        # x = np.array([x+1 for x in range(self.num_gen)])
        # y = np.array(self.BFs)
        # plt.plot(x, y)
        plt.title("Rastrigin Function")
        plt.xlabel("Number Of Generations")
        plt.ylabel("Best Fitness")
        plt.show()

       


dimension = 2
population_size= 100
num_parents = 10
num_generations = 3000

print("Running:")
obj=Rastrigin(population_size,num_generations,num_parents, dimension)
obj.many_gens()
obj.plot_graph()


# dim = 2
# Best Solution :  [3.64594717e-04 1.98397440e-05]
# Best Solution Fitness :    2.6450201566774467e-05

#  dim = 4
# Best Solution :  [ 0.00014637 -0.00011371  0.0007459  -0.00023474]
# Best Solution Fitness :    0.00012812679662488335
