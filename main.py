from rastrigin_ga import Rastrigin
from rastrigin_pso import PSO
import numpy as np
import matplotlib.pyplot as plt
'''Alter the following parameters'''
iterations = 2000
pop_size = 100
dim = 12

# PSO Hyperparameters
w = 0.5
c1 = 2
c2 = 2

# EA Hyperparameters
num_parents = 10

# --------------------------------------------------------------
pso_2 = PSO(pop_size, iterations, dim, w, c1, c2)
pso_2_output, pso_2_best = pso_2.run_algorithm()


ea_2 = Rastrigin(pop_size, iterations, num_parents, dim)
ea_2_output = ea_2.many_gens()

iters = np.arange(iterations)
plt.plot(iters, pso_2_output, label=f"PSO - dim {dim}")
plt.plot(iters, ea_2_output, label=f"Genetic Algo - dim {dim}")
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Comparison of PSO and GA on the function based on dimensions')
plt.show()