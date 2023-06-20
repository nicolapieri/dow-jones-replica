import staging as stage
import pandas as pd
import numpy as np
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.single.global_best import GlobalBestPSO


def find_particle_loss(coeffs):
    benchmark_tracking_error = np.std(stage.trainX.dot(coeffs) - stage.trainY['^DJI'])
    return benchmark_tracking_error


def swarm(x):
    n_particles = x.shape[0]
    particle_loss = [find_particle_loss(x[i]) for i in range(n_particles)]
    return particle_loss


# Given a matrix of position options search for best position
g = GridSearch(GlobalBestPSO,
               objective_func=swarm,
               n_particles=100,
               dimensions=len(stage.trainX.columns),
               options={'c1': [1.5, 2.5], 'c2': [1, 2], 'w': [0.4, 0.5]},
               bounds=(len(stage.trainX.columns) * [0], len(stage.trainX.columns) * [1]),
               iters=100)
best_cost, best_pos = g.search()

# Given the best position option optimize the cost
optimizer = GlobalBestPSO(n_particles=1000,
                          dimensions=len(stage.trainX.columns),
                          options=best_pos,
                          bounds=(len(stage.trainX.columns) * [0], len(stage.trainX.columns) * [1]))
cost, pos = optimizer.optimize(swarm, iters=100)

# portfolio allocation with Particle Swarm (PSO)
leverage = sum(pos)
weights = dict(zip(list(stage.trainX.columns), list(pos / leverage)))
allocation_PSO = pd.DataFrame({'Component': stage.trainX.columns,
                               'PSOweight(%)': np.multiply(list(weights.values()), 100)}).sort_values('PSOweight(%)', ascending=False)
allocation_PSO.set_index('Component', inplace=True)
allocation_PSO.reset_index(inplace=True)

# PSO optimization evaluation
stage.testY['PSO'] = leverage * stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'PSO')
print("\nLeverage Factor:", leverage, "\n")
