import random
import numpy as np
import math
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, velocity, position, n, w, c1, c2) -> None:
        self.velocity = velocity
        self.position = position
        self.lbest = None
        self.fitness = None
        self.maxfit = None
        self.n = n
        self.w = w
        self.c1 = c1
        self.c2 = c2
    
    def update(self, gbest):
        # updating velocity
        newval = []
        for i in range(self.n):
            newval.append((self.w * self.velocity[i]) + (self.c1 * random.random() * (self.lbest[i] - self.position[i])) + (self.c2 * random.random() * (gbest[i] - self.position[i])))
        self.velocity = (newval[0], newval[1])
        
        # updating position
        newval = []
        for i in range(self.n):
            newval.append(self.position[i] + self.velocity[i])
        self.position = (newval[0], newval[1])

class PSO:
    def __init__(self, pop_size, iterations, n, w, c1, c2) -> None:
        self.pop_size = pop_size
        self.iterations = iterations
        self.n = n
        self.particles = []
        self.gbest = None
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bests = []
    
    def population_initialization(self):
        for i in range(self.pop_size):
            randomx = np.random.uniform(-5.12, 5.12)
            randomy = np.random.uniform(-5.12, 5.12)
            randomvx = np.random.uniform(-1, 1)
            randomvy = np.random.uniform(-1, 1)
            p = Particle((randomx, randomy), (randomvx, randomvy), self.n, self.w, self.c1, self.c2)
            self.particles.append(p)
    
    def rastrigin_function(self, x):
        summ = 0
        for i in range(len(x)):
            summ += (x[i]**2) - (10 * math.cos(2 * math.pi * x[i])) + 10
        return summ

    def evaluate_fitness(self):
        '''
        Evaluating fitness based on the function.
        Updating gbest and lbest
        
        '''
        allfitness = []
        for i in self.particles:
            x = i.position
            i.fitness = self.rastrigin_function(x)
            if i.maxfit != None:
                if i.fitness < i.maxfit:
                    i.maxfit = i.fitness
                    i.lbest = i.position
            else:
                i.maxfit = i.fitness
                i.lbest = i.position
            allfitness.append(i.maxfit)
        idx = allfitness.index(min(allfitness))
        self.gbest = self.particles[idx].lbest
        self.bests.append(min(allfitness))
        
    def updating_vel_pos(self):
        for i in self.particles:
            i.update(self.gbest)
    
    def run_algorithm(self):
        self.population_initialization()
        for i in range(self.iterations):
            self.evaluate_fitness()
            self.updating_vel_pos()
        return self.bests, self.gbest

def main():
    ps = PSO(pop_size=50, iterations=100, n=2, w=np.random.uniform(0, 1), c1=np.random.uniform(1.5, 2.5), c2=np.random.uniform(1.5, 2.5))
    bests, best = ps.run_algorithm()
    iters = np.arange(100)

        # plot x and y
    print(best, bests[-1])
    plt.plot(iters, bests)

    # set labels for x and y axes and title
    plt.xlabel('Iterations')
    plt.ylabel('gbest')
    plt.title('PSO on the function')

    # display the plot
    plt.show()



if __name__ == "__main__":
    main()     

