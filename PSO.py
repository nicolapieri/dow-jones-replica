import staging as stage
import pandas as pd
import numpy as np
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.single.global_best import GlobalBestPSO


# portfolio allocation with Particle Swarm (PSO)
def train_particle_loss(coeffs):
    benchmark_tracking_error = np.std(stage.trainX.dot(coeffs) - stage.trainY['^DJI'])
    return benchmark_tracking_error


def train_swarm(x):
    n_particles = x.shape[0]
    particle_loss = [train_particle_loss(x[i]) for i in range(n_particles)]
    return particle_loss


train_g = GridSearch(GlobalBestPSO,
                     objective_func=train_swarm,
                     n_particles=100,
                     dimensions=len(stage.trainX.columns),
                     options={'c1': [1.5, 2.5], 'c2': [1, 2], 'w': [0.4, 0.5]},
                     bounds=(len(stage.trainX.columns) * [0],
                             len(stage.trainX.columns) * [1]),
                     iters=100)

train_best_cost, train_best_pos = train_g.search()  # Given a matrix of position options search for best position

train_optimizer = GlobalBestPSO(n_particles=1000,
                                dimensions=len(stage.trainX.columns),
                                options=train_best_pos,
                                bounds=(len(stage.trainX.columns) * [0],
                                        len(stage.trainX.columns) * [1]))

train_cost, train_pos = train_optimizer.optimize(train_swarm,
                                                 iters=100)  # Given the best position option optimize the cost

PSOtrain_leverage = sum(train_pos)
PSOtrain_weights = dict(zip(list(stage.trainX.columns), list(train_pos / PSOtrain_leverage)))
PSOtrain_allocation = pd.DataFrame({'Component': stage.trainX.columns,
                                    'PSO-30wg(%)': np.multiply(list(PSOtrain_weights.values()), 100)}).sort_values(
    'PSO-30wg(%)', ascending=False)
PSOtrain_allocation.set_index('Component', inplace=True)
PSOtrain_allocation.reset_index(inplace=True)


# validation with top 10 weighted stocks
def val_particle_loss(coeffs):
    benchmark_tracking_error = np.std(stage.valX[PSOtrain_allocation['Component'][0:10]].dot(coeffs) - stage.valY['^DJI'])
    return benchmark_tracking_error


def val_swarm(x):
    n_particles = x.shape[0]
    particle_loss = [val_particle_loss(x[i]) for i in range(n_particles)]
    return particle_loss


val_g = GridSearch(GlobalBestPSO,
                   objective_func=val_swarm,
                   n_particles=100,
                   dimensions=len(stage.valX[PSOtrain_allocation['Component'][0:10]].columns),
                   options={'c1': [1.5, 2.5], 'c2': [1, 2], 'w': [0.4, 0.5]},
                   bounds=(len(stage.valX[PSOtrain_allocation['Component'][0:10]].columns) * [0],
                           len(stage.valX[PSOtrain_allocation['Component'][0:10]].columns) * [1]),
                   iters=100)

val_best_cost, val_best_pos = val_g.search()  # Given a matrix of position options search for best position

val_optimizer = GlobalBestPSO(n_particles=1000,
                              dimensions=len(stage.valX[PSOtrain_allocation['Component'][0:10]].columns),
                              options=val_best_pos,
                              bounds=(len(stage.valX[PSOtrain_allocation['Component'][0:10]].columns) * [0],
                                      len(stage.valX[PSOtrain_allocation['Component'][0:10]].columns) * [1]))

val_cost, val_pos = val_optimizer.optimize(val_swarm, iters=100)  # Given the best position option optimize the cost

PSOval_leverage = sum(val_pos)
PSOval_weights = dict(zip(list(stage.valX[PSOtrain_allocation['Component'][0:10]].columns), list(val_pos / PSOval_leverage)))
PSOval_allocation = pd.DataFrame({'Component': stage.valX[PSOtrain_allocation['Component'][0:10]].columns,
                                  'PSO-10wg(%)': np.multiply(list(PSOval_weights.values()), 100)}).sort_values(
    'PSO-10wg(%)', ascending=False)
PSOval_allocation.set_index('Component', inplace=True)
PSOval_allocation.reset_index(inplace=True)

# PSW portfolio optimization evaluation
stage.testY['PSO'] = PSOval_leverage * stage.testX[PSOtrain_allocation['Component'][0:10]].dot(list(PSOval_weights.values()))
stage.evaluate(stage.testY, 'PSO')

print(PSOval_allocation)
print("-" * 50)
print(f"Leverage Factor: {PSOval_leverage}")
