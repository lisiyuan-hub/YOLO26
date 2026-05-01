import numpy as np

n_dots = 100000

x = np.random.random(n_dots)

y = np.random.random(n_dots)

distance = np.sqrt(x**2 + y**2)

in_circle = distance[distance < 1]

pi = 4 * float(len(in_circle)) / n_dots

print(pi)
