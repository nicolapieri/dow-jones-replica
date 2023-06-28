import staging as stage
import pandas as pd
import numpy as np
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.single.global_best import GlobalBestPSO


# portfolio allocation with Particle Swarm (PSO)
def train_particle_loss(coeffs):
    benchmark_tracking_error = np.std(stage.X.dot(coeffs) - stage.Y['^DJI'])
    return benchmark_tracking_error


def train_swarm(x):
    n_particles = x.shape[0]
    particle_loss = [train_particle_loss(x[i]) for i in range(n_particles)]
    return particle_loss


g = GridSearch(GlobalBestPSO,
               objective_func=train_swarm,
               n_particles=100,
               dimensions=len(stage.X.columns),
               options={'c1': [1.5, 2.5], 'c2': [1, 2], 'w': [0.4, 0.5]},
               bounds=(len(stage.X.columns) * [0],
                       len(stage.X.columns) * [1]),
               iters=100)

best_cost, best_pos = g.search()  # Given a matrix of position options search for best position

optimizer = GlobalBestPSO(n_particles=1000,
                          dimensions=len(stage.X.columns),
                          options=best_pos,
                          bounds=(len(stage.X.columns) * [0],
                                  len(stage.X.columns) * [1]))

cost, pos = optimizer.optimize(train_swarm, iters=100)  # Given the best position option optimize the cost

PSO_leverage = sum(pos)
PSO_weights = dict(zip(list(stage.X.columns), list(pos / PSO_leverage)))
PSO_allocation = pd.DataFrame({'Component': stage.X.columns,
                               'PSO-30wg(%)': np.multiply(list(PSO_weights.values()), 100)}).sort_values('PSO-30wg(%)',
                                                                                                         ascending=False)
PSO_allocation.set_index('Component', inplace=True)
PSO_allocation.reset_index(inplace=True)

# PSW portfolio optimization evaluation
stage.Opt['PSO'] = PSO_leverage * stage.X.dot(list(PSO_weights.values()))
stage.evaluate(stage.Opt, 'PSO')
