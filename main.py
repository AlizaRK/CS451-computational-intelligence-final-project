from rastrigin_ga import Rastrigin
from rastrigin_pso import PSO
import numpy as np
import matplotlib.pyplot as plt

iterations = 2000
pop_size = 100
dim = 5

# PSO Hyperparameters
w = 0.5
c1 = 2
c2 = 2
pso_2 = PSO(pop_size, iterations, 5, w, c1, c2)
pso_2_output, pso_2_best = pso_2.run_algorithm()
# pso_5 = PSO(pop_size, iterations, 5, w, c1, c2)
# pso_5_output, pso_5_best = pso_5.run_algorithm()

# EA Hyperparameters
num_parents = 10
ea_2 = Rastrigin(pop_size, iterations, num_parents, dim)
ea_2_output = ea_2.many_gens()
# ea_5 = Rastrigin(pop_size, iterations, num_parents, 2)
# ea_5.many_gens()

iters = np.arange(2000)
plt.plot(iters, pso_2_output, label="PSO - dim 2")
plt.plot(iters, ea_2_output, label="Genetic Algo - dim 2")
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Comparison of PSO and GA on the function')
plt.show()