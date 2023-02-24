import numpy as np
from numpy.random import default_rng

def dist(p1, p2):
    """
    Calculate Euclidean distance between p1 and p2 on a 2D plane.
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def initial_tour(p):
    """
    Generates an initial tour that visits all cities in each country exactly once.
    p: a list of integers representing the country for each city
    """
    tour = {}
    for i,pi in enumerate(p):
        if pi in tour:
            tour[pi].append((i,pi))
        else:
            tour[pi] = [(i,pi)]
    x = []
    for t in np.unique(p):#tour:
        x += tour[t]
    x.append((0,p[0]))
    return x

def neighbor_tour(x):
    """
    Generates a neighbor tour that visits all cities in each country exactly once.
    x: current tour
    """

    rng = default_rng()
    i,j = rng.choice(np.arange(1,len(x)-1), size=2, replace=False)
    m,M = min(i,j),max(i,j)
    while x[i][1]!=x[j][1] and (x[m-1][1]==x[m][1] or x[M+1][1]==x[M][1]):
        i,j = rng.choice(np.arange(1,len(x)-1), size=2, replace=False)
        m,M = min(i,j),max(i,j)
    return x[:m]+x[M:m-1:-1]+x[M+1:]

def tour_cost(x, D):
    """
    Calculates the total distance of a given tour.
    tour: a list of (city, country) tuples representing the tour
    distances: a 2D list of distances between cities
    """
    f = 0
    for i in range(len(x)-1):
        j = (i + 1) % (len(x)-1)
        Vi, _ = x[i]
        Vj, _ = x[j]
        f += D[Vi][Vj]
    return f

def simulated_annealing(D, p, T0, alpha, stopping_temp, max_iter, x0=None):
    """
    Finds the best tour using the simulated annealing algorithm.
    D: a 2D list of distances between cities
    p: a list of integers representing the country for each city
    T0: the starting temperature
    alpha: the temperature scale
    stopping_temp: the temperature at which to stop the algorithm
    max_iter: the number of iterations at each temperature
    """
    if x0==None:
        x = initial_tour(p)
    else:
        x = x0[:]
    step_cost = [[0,tour_cost(x, D)]]
    best_cost = [[0,tour_cost(x, D)]]
    best_x = x[:]
    T = T0
    i,accept = 1,0
    while T > stopping_temp and i < max_iter:
   
        xp = neighbor_tour(x)
        delta = tour_cost(xp, D) - tour_cost(x, D)
        if delta < 0 or np.exp(-delta / T) > np.random.random():
            x = xp[:]
            step_cost.append([i,delta+step_cost[-1][-1]])
            accept += 1
            if tour_cost(x, D) < tour_cost(best_x, D):
                best_x = x[:]
                best_cost.append([i,step_cost[-1][-1]])
        #T *= alpha
        #T = 4 * ((max_iter - i)/max_iter)
        # Decrease temperature based on logarithmic cooling schedule
        T = T0 / (alpha*np.log(i+1))
        i += 1.0
    best_cost.append([i,best_cost[-1][-1]])
    return best_x, np.array(best_cost), np.array(step_cost), accept/i