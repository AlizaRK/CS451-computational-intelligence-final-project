import random
import numpy as np
import math
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, velocity, position, dim, w, c1, c2) -> None:
        self.velocity = velocity
        self.position = position
        self.lbest = None
        self.fitness = None
        self.maxfit = None
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2
    
    def update(self, gbest):
        """
        Updates the velocity and position of the particle.

        Parameters:
        ----------
        gbest : list of coordinates
            A list of coordinates in multiple planes. 

        Returns:
        -------
        None
        """
        # updating velocity
        newval = []
        for i in range(self.dim):
            newval.append((self.w * self.velocity[i]) + (self.c1 * random.random() * 
                                                         (self.lbest[i] - self.position[i])) + 
                                                         (self.c2 * random.random() * (gbest[i] - 
                                                                                       self.position[i])))
        self.velocity = newval
        
        # updating position
        newval = []
        for i in range(self.dim):
            newval.append(self.position[i] + self.velocity[i])
        self.position = newval

class PSO:
    def __init__(self, pop_size, iterations, dim, w, c1, c2) -> None:
        self.pop_size = pop_size
        self.iterations = iterations
        self.dim = dim
        self.particles = []
        self.gbest = None
        self.gbestfit = None
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bests = []
    
    def population_initialization(self):
        """
        Initializes particles with random positions and velocities in the search space of the function.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """
        for i in range(self.pop_size):
            pos = []
            vel = []
            for j in range(self.dim):
                randomx = np.random.uniform(-5.12, 5.12)
                randomvx = np.random.uniform(-1, 1)
                pos.append(randomx)
                vel.append(randomvx)
            p = Particle(pos, vel, self.dim, self.w, self.c1, self.c2)
            self.particles.append(p)
    
    def rastrigin_function(self, x):
        """
        Calculates the value by using rastrigin's function.

        Parameters:
        ----------
        x : list of coordinates
            A list of position coordinates of a single particle.

        Returns:
        -------
        summ : float value
                Value returned by the rastrigin's function.
        """
        summ = 0
        for i in range(len(x)):
            summ += (x[i]**2) - (10 * math.cos(2 * math.pi * x[i])) + 10
        return summ

    def evaluate_fitness(self):
        """
        Evaluates the fitness and updates local best and global best. 

        Parameters:
        ----------
        None.

        Returns:
        -------
        None
        """
        allfitness = []
        # Calculating particle fitness and updating local best.
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

        # Updating global best
        idx = allfitness.index(min(allfitness))
        fit = min(allfitness)
        newbest = self.particles[idx].lbest
        if self.gbestfit != None:
            if fit < self.gbestfit:
                self.gbest = self.particles[idx].lbest
                self.gbestfit = fit
        else:
            self.gbest = self.particles[idx].lbest
            self.gbestfit = fit
        self.bests.append(self.gbestfit)
        
    def updating_vel_pos(self):
        """
        Calls individual particle instances to update the velocity and position of the particle.

        Parameters:
        ----------
        None.

        Returns:
        -------
        None
        """
        for i in self.particles:
            i.update(self.gbest)
    
    def run_algorithm(self):
        """
        Carries out the necessary steps for PSO.

        Parameters:
        ----------
        None

        Returns:
        -------
        self.bests : A list of floats - size of iterations
                        The list contains all the fitness values of gbests 
        gbest : A list of coordinates
                    The coordinates of best minimized position in the solution space.
        """
        self.population_initialization()
        for i in range(self.iterations):
            self.evaluate_fitness()
            self.updating_vel_pos()
        print("Best solution: ", self.gbest)
        print("Best fitness: ", self.gbestfit)
        return self.bests, self.gbest

def main(pop_size, iterations, dim, w, c1, c2):
    pass
    """
    Starts the algorithm and visualizes the results.

    Parameters:
    ----------
    pop_size : Number of particles - integer
    iterations : Number of epochs - integer
    dim : Dimensions of the function - integer
    w : intertia factor - float
    c1 : Acceleration constant - float
    c2 : Acceleration constant - float

    Returns:
    -------
    None
        """
    