import random
import numpy as np
import math

class Particle:
    def ___init___(self, velocity, position):
        self.velocity = velocity
        self.position = position
        self.lbest = None
        self.fitness = None
    
    def update_velocity(self):
        pass

class PSO:
    def ___init___(self, pop_size, iterations, n):
        self.pop_size = pop_size
        self.iterations = iterations
        self.n = n
        self.particles = []
    
    def population_initialization(self):
        for i in range(self.pop_size):
            randomx = np.random.uniform(-5.12, 5.12)
            randomy = np.random.uniform(-5.12, 5.12)
            randomvx = np.random.uniform(-1, 1)
            randomvy = np.random.uniform(-1, 1)
            p = Particle((randomx, randomy), (randomvx, randomvy))
            self.particles.append(p)
    
    def rastrigin_function(self, x):
        summ = 0
        for i in range(len(x)):
            summ += (x[i]**2) - (10 * math.cos(2 * math.pi * x[i])) + 10
        return summ

    def evaluate_fitness(self):
        for i in self.particles:
            x = i.position
            i.fitness = rastrigin_function(x)
            print(i.fitness)
        

ps = PSO(5, 2, 2)
ps.population_initialization()
ps.evaluate_fitness()