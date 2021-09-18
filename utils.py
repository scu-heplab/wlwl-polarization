import numpy as np


def particle_swarm_optimization(fitness_func, particle_num, dim, weight=0.6, c1=2, c2=2, iter_num=1000, eq_func=None, eq_weight=None, neq_func=None, neq_weight=None):
    particle = np.random.uniform(0, 1, (particle_num, dim))
    velocity = np.random.uniform(0, 1, (particle_num, dim))

    unit_fitness = np.squeeze(np.square(fitness_func(particle)))
    if eq_func is not None:
        if type(eq_func) is list:
            for func in eq_func:
                unit_fitness += np.squeeze(np.square(func(particle))) * eq_weight
        else:
            unit_fitness += np.squeeze(np.square(eq_func(particle))) * eq_weight
    if neq_func is not None:
        if type(neq_func) is list:
            for func in neq_func:
                unit_fitness += np.squeeze(np.abs(np.minimum(func(particle), 0))) * neq_weight
        else:
            unit_fitness += np.squeeze(np.abs(np.minimum(neq_func(particle), 0))) * neq_weight
    unit_optimal = particle
    global_fitness = np.min(unit_fitness)
    arg_index = np.argmin(unit_fitness)
    global_optimal = unit_optimal[arg_index]
    for _ in range(iter_num - 1):
        fitness = np.squeeze(np.square(fitness_func(particle)))
        if eq_func is not None:
            if type(eq_func) is list:
                for func in eq_func:
                    fitness += np.squeeze(np.square(func(particle))) * eq_weight
            else:
                fitness += np.squeeze(np.square(eq_func(particle))) * eq_weight
        if neq_func is not None:
            if type(neq_func) is list:
                for func in neq_func:
                    fitness += np.squeeze(np.abs(np.minimum(func(particle), 0))) * neq_weight
            else:
                fitness += np.squeeze(np.abs(np.minimum(neq_func(particle), 0))) * neq_weight
        velocity = weight * velocity
        velocity += c1 * np.random.rand(particle_num, dim) * (unit_optimal - particle)
        velocity += c2 * np.random.rand(particle_num, dim) * (global_optimal - particle)

        logical = unit_fitness < fitness
        unit_fitness = np.where(logical, unit_fitness, fitness)
        unit_optimal = np.where(logical[:, np.newaxis], unit_optimal, particle)
        min_unit_fitness = np.min(unit_fitness)
        if min_unit_fitness < global_fitness:
            global_fitness = min_unit_fitness
            arg_index = np.argmin(unit_fitness)
            global_optimal = unit_optimal[arg_index]

        particle += velocity

    return global_optimal


def chi_square(observe, template):
    def calc_loss(factor):
        theory = np.sum(factor[:, np.newaxis, np.newaxis] * template[np.newaxis], -1)
        loss = np.sum(np.square(observe[np.newaxis] - theory) / theory, (1, 2)) / (10 ** 2 - 4)
        return loss
    return calc_loss


def fit_fraction(theta, model_name, energy_level):
    eq_func = [lambda args: np.sum(args, -1) - 1]
    neq_func = [lambda args: args[:, 0], lambda args: args[:, 1], lambda args: args[:, 2], lambda args: args[:, 3],
                lambda args: 1 - args[:, 0], lambda args: 1 - args[:, 1], lambda args: 1 - args[:, 2], lambda args: 1 - args[:, 3]]
    temp = np.load('./temp/' + model_name + '/' + 'temp_' + energy_level + '.npy')
    param = particle_swarm_optimization(chi_square(theta, temp), 1000, 4, iter_num=1000,
                                        eq_func=eq_func, eq_weight=100, neq_func=neq_func, neq_weight=100)
    print('pred:', np.round(param, 5), 'loss:', chi_square(theta, temp)(param[np.newaxis]))
