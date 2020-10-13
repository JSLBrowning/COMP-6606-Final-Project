# import from project functions
import initial as init
import evaluation as eval

# import from outside functions
import math
import random
import statistics
from copy import deepcopy
from operator import itemgetter
from random import randint

import numpy


# create an initial population pool by randomly put bulbs into white cells
def create_population_pool(net_white_cell, size):
    r_puzzle_data = []
    total_number_bulb = len(net_white_cell)

    loop = 0
    loop_warning = 0

    while loop < size:
        number_bulb = randint(1, total_number_bulb)
        new_bulb_set = random.sample(net_white_cell, k=number_bulb)
        new_bulb_set.sort()
        loop_warning += 1

        if new_bulb_set not in r_puzzle_data:
            r_puzzle_data.append(new_bulb_set)
            loop += 1

        if loop_warning > 100000:
            print("Creating population pool failure...")
            print("reduce the number of population size: mu")
            loop_warning = 0

    return r_puzzle_data


# recombine both parents to generate an offspring
# single crossover
def recombination(gene_space, parent_one, parent_two, cross_point_percent):
    cross_point = int(len(gene_space) * cross_point_percent)
    # print(f"cross point: {cross_point}")
    offspring = []
    for i in range(0, len(parent_one)):
        if parent_one[i] in gene_space[0:cross_point]:
            offspring.append(parent_one[i])
    for i in range(0, len(parent_two)):
        if parent_two[i] in gene_space[cross_point:]:
            offspring.append(parent_two[i])

    return offspring


# mutate an offspring
def mutation(gene_space, offspring, number_flip):
    mutation_offspring = deepcopy(offspring)

    mutation_gene = random.sample(gene_space, k=number_flip)

    for i in range(0, len(mutation_gene)):
        if mutation_gene[i] in offspring:
            mutation_offspring.remove(mutation_gene[i])
        else:
            mutation_offspring.append(mutation_gene[i])

    mutation_offspring.sort()

    # print(f"mutation list: {mutation_gene}")
    # print(f"before:{offspring}  after: {mutation_offspring}")

    return mutation_offspring


# select parents by fitness proportional
def roulette_wheel_selection(population_pool, number_pair_parents):
    pool = deepcopy(population_pool)
    parents_pool = []
    sum_fitness_mu = 0

    for i in range(0, len(pool)):
        sum_fitness_mu += pool[i].get("evaluation_fitness")

    for x in range(0, number_pair_parents):
        parents = []
        pool_run = deepcopy(pool)
        sum_fitness = sum_fitness_mu
        for i in range(0, 2):
            sum_temporary = 0
            parent_selection_value = random.uniform(0, sum_fitness)
            for j in range(0, len(pool_run)):
                sum_temporary += pool_run[j].get("evaluation_fitness")
                if sum_temporary >= parent_selection_value:
                    sum_fitness = sum_fitness - pool_run[j].get("evaluation_fitness")
                    parents.append(pool_run[j])
                    del pool_run[j]
                    break
        parents_pool.append(parents)

    return parents_pool


# select parents by fitness proportional
def stochastic_universal_sampling(population_pool, number_pair_parents):
    pool = deepcopy(population_pool)
    parents_pool = []
    sum_fitness_mu = 0

    for i in range(0, len(pool)):
        sum_fitness_mu += pool[i].get("evaluation_fitness")
    sampling_r = random.uniform(0, float(sum_fitness_mu / number_pair_parents))

    sum_temporary = pool[0].get("evaluation_fitness")
    i = 0
    for _ in range(0, number_pair_parents):
        parents = []
        for _ in range(0, 2):
            sum_fitness_one_round = 0
            while sum_temporary < sampling_r:
                i += 1
                if i >= len(population_pool):
                    i = 0
                sum_temporary += pool[i].get("evaluation_fitness")
                sum_fitness_one_round += pool[i].get("evaluation_fitness")
            parents.append(pool[i])
            sampling_r += sum_fitness_one_round
            # print(sum_fitness_one_round)
            # print(sampling_r)

            if sampling_r >= sum_fitness_mu:
                i = 0
                sampling_r = random.uniform(0, float(sum_fitness_mu / number_pair_parents))
                sum_temporary = pool[0].get("evaluation_fitness")

        parents_pool.append(parents)

    return parents_pool


# select parents by fitness proportional
def roulette_wheel_selection_survival(population_pool, number_survival):
    pool = deepcopy(population_pool)
    survival_pool = []
    sum_fitness_mu = 0

    for i in range(0, len(pool)):
        sum_fitness_mu += pool[i].get("evaluation_fitness")

    for x in range(0, number_survival):
        sum_temporary = 0
        parent_selection_value = random.uniform(0, sum_fitness_mu)
        for j in range(0, len(pool)):
            sum_temporary += pool[j].get("evaluation_fitness")
            if sum_temporary >= parent_selection_value:
                survival_pool.append(pool[j])
                sum_fitness_mu -= pool[j].get("evaluation_fitness")
                pool.remove(pool[j])
                break

    # for i in range(0, len(survival_pool)):
    #     print(survival_pool[i])

    return survival_pool


# uniform random
def uniform_random_parents_selection(population_pool, number_pair_selection, replacement):
    number_selection = number_pair_selection * 2
    if not replacement:
        if number_selection <= len(population_pool):
            pool = random.sample(population_pool, k=number_selection)
        else:
            pool = random.choices(population_pool, k=number_selection)
    else:
        pool = random.choices(population_pool, k=number_selection)

    parent_pool = []
    for i in range(0, number_pair_selection):
        parents = [pool[2 * i], pool[2 * i + 1]]
        parent_pool.append(parents)

    return parent_pool


# uniform random survival selection
def uniform_random_survival_selection(population_pool, number_selection, replacement):
    if not replacement:
        if number_selection <= len(population_pool):
            pool = random.sample(population_pool, k=number_selection)
        else:
            pool = random.choices(population_pool, k=number_selection)
    else:
        pool = random.choices(population_pool, k=number_selection)

    return pool


# k-Tournament Selection with or without replacement
def k_tournament_selection(population_pool, k, number_selection, replacement):
    pool = deepcopy(population_pool)
    survival = []
    for i in range(0, number_selection):
        survival_selection_pool = random.sample(pool, k=k)
        survival_selection_pool = sorted(survival_selection_pool, key=itemgetter('evaluation_fitness'), reverse=True)
        survival.append(survival_selection_pool[0])
        if not replacement:
            pool.remove(survival_selection_pool[0])
    return survival


# parent selections
def parent_selection(popular_pool, number_pair_parents, config):

    k_tournament_parents = config["k_tournament_parents"]
    parent_selection_mode = config["parent_selection"]
    if parent_selection_mode == "roulette_wheel":
        parents_pool = roulette_wheel_selection(popular_pool,
                                                number_pair_parents)

    elif parent_selection_mode == "stochastic_universal_sampling":
        parents_pool = stochastic_universal_sampling(popular_pool,
                                                     number_pair_parents)

    elif parent_selection_mode == "uniform_random":
        parents_pool = uniform_random_parents_selection(popular_pool,
                                                        number_pair_parents,
                                                        replacement=True)

    else:  # elif parent_selection == "k_tournament":
        parents_pool = []
        for i in range(0, number_pair_parents):
            parents_pool.append(k_tournament_selection(popular_pool,
                                                       k=k_tournament_parents,
                                                       number_selection=2,
                                                       replacement=False))

    return parents_pool


def mutation_self_adaptive(config, mutation_rate, lambda_size, net_white_cells):
    t = 1 / math.sqrt(len(net_white_cells))

    if len(mutation_rate) < lambda_size:
        mean = float(config["initial_mutation_rate"])
        standard_deviation = 0.1

    else:
        standard_deviation = statistics.stdev(mutation_rate)
        mean = statistics.mean(mutation_rate)

    random_factor = float(numpy.random.random(1)[0])
    standard_deviation_new = standard_deviation * math.exp(random_factor * t)

    new_mutation_rate = abs(float(numpy.random.normal(mean, standard_deviation_new, 1)))

    return new_mutation_rate


# generate offspring, under recombination and mutation
def generate_offspring(net_white_cells, lambda_size, parents_pool, puzzle_map, puzzle_initial, config, mutation_rate):
    population_offspring = []
    for i in range(0, lambda_size):
        offspring = recombination(net_white_cells,
                                  parents_pool[i][0].get("bulbs"),
                                  parents_pool[i][1].get("bulbs"),
                                  cross_point_percent=0.5)

        if config["self_adaptive"]:
            evaluation_1 = eval.evaluate_puzzle_map(puzzle_map,
                                                    puzzle_initial,
                                                    offspring, config=config)

            current_mutation_rate = mutation_self_adaptive(config, mutation_rate, lambda_size, net_white_cells)

            # print(f"mutation rate is {current_mutation_rate}")
            offspring = mutation(net_white_cells, offspring, int(current_mutation_rate * len(net_white_cells)))
            # evaluate offsprings
            evaluation_2 = eval.evaluate_puzzle_map(puzzle_map,
                                                    puzzle_initial,
                                                    offspring, config=config)

            if evaluation_1.get("evaluation_fitness") < evaluation_2.get("evaluation_fitness"):
                mutation_rate.append(current_mutation_rate)

        else:
            offspring = mutation(net_white_cells, offspring,
                                 int(config["initial_mutation_rate"] * len(net_white_cells)))
            evaluation_2 = eval.evaluate_puzzle_map(puzzle_map,
                                                    puzzle_initial,
                                                    offspring, config=config)
        if config["repair_function"]:
            if evaluation_2.get("total_conflict"):
                puzzle = eval.insert_bulbs(puzzle_map, offspring)

                # print(f' before repair')
                # print(offspring)
                # for x in range(0, len(puzzle)):
                #     print(puzzle[len(puzzle) - x - 1])
                invalid_bulbs = repair_bulb_shining(puzzle, puzzle_initial)
                for x in range(0, len(invalid_bulbs)):
                    if invalid_bulbs[x] in offspring:
                        offspring.remove(invalid_bulbs[x])
                # print(offspring)
                puzzle = eval.insert_bulbs(puzzle_map, offspring)
                # print(f' repair shining')
                # for x in range(0, len(puzzle)):
                #     print(puzzle[len(puzzle) - x - 1])
                bulbs = repair_black_bulb(puzzle, puzzle_initial)
                # puzzle = insert_bulbs(puzzle_map, bulbs)
                # print(f' repair black')
                # for x in range(0, len(puzzle)):
                #     print(puzzle[len(puzzle) - x - 1])

                evaluation_2 = eval.evaluate_puzzle_map(puzzle_map,
                                                        puzzle_initial,
                                                        bulbs, config=config)

        population_offspring.append(evaluation_2)

    return population_offspring


# survival selections
def survival_selection(config, population_mu, population_lambda):
    mu_size = len(population_mu)

    if config["survival_strategy"] == "plus":
        survival_pool = population_mu + population_lambda
    else:  # comma
        survival_pool = population_lambda

    if config["survival_selection"] == "truncation":
        population_mu = truncation(survival_pool, number_survival=mu_size)

    elif config["survival_selection"] == "uniform_random":
        population_mu = uniform_random_survival_selection(survival_pool,
                                                          mu_size,
                                                          replacement=False)
    elif config["survival_selection"] == "roulette_wheel":
        population_mu = roulette_wheel_selection_survival(survival_pool, mu_size)

    else:  # puzzle_config["survival_selection"] == "truncation":
        population_mu = k_tournament_selection(survival_pool,
                                               k=config["k_tournament_survivals"],
                                               number_selection=mu_size,
                                               replacement=False)

    return population_mu


# survival selections for MOEA
def survival_selection_MOEA(config, survival_pool):

    mu_size = config["mu_size"]

    if config["survival_selection"] == "truncation":
        population_mu = truncation(survival_pool, number_survival=mu_size)

    elif config["survival_selection"] == "uniform_random":
        population_mu = uniform_random_survival_selection(survival_pool,
                                                          mu_size,
                                                          replacement=False)
    elif config["survival_selection"] == "roulette_wheel":
        population_mu = roulette_wheel_selection_survival(survival_pool, mu_size)

    else:  # puzzle_config["survival_selection"] == "truncation":
        population_mu = k_tournament_selection(survival_pool,
                                               k=config["k_tournament_survivals"],
                                               number_selection=mu_size,
                                               replacement=False)

    return population_mu


# fix bulbs shining each other
# return - the valid bulbs
def repair_bulb_shining(puzzle_map, initial_puzzle):
    # validation = True
    # check the bulbs by every row
    puzzle = deepcopy(puzzle_map)
    row = initial_puzzle[0][0]
    col = initial_puzzle[0][1]
    invalid_bulbs = []
    for i in range(0, row):
        bulb = 0
        bulb_in_row = []
        for j in range(0, col):
            cell = puzzle[i][j]
            if cell == init.CELL_BULB:
                bulb += 1
                bulb_in_row.append([i, j])
            if (cell < init.CELL_EMPTY) or (j == col - 1):
                if bulb > 1:
                    delete_bulbs = random.sample(bulb_in_row, k=bulb - 1)
                    for x in range(0, len(delete_bulbs)):
                        invalid_bulbs.append(delete_bulbs[x])
                bulb = 0
                bulb_in_row = []

    # check the bulbs by every column
    for i in range(0, col):
        bulb = 0
        bulb_in_column = []
        for j in range(0, row):
            cell = puzzle[j][i]
            if cell == init.CELL_BULB:
                bulb += 1
                bulb_in_column.append([j, i])
            if (cell < init.CELL_EMPTY) or (j == row - 1):
                if bulb > 1:
                    delete_bulbs = random.sample(bulb_in_column, k=bulb - 1)
                    for x in range(0, len(delete_bulbs)):
                        if delete_bulbs[x] not in invalid_bulbs:
                            invalid_bulbs.append(delete_bulbs[x])
                bulb_in_column = []
                bulb = 0

    return invalid_bulbs


# check black cell value by the number of surrounded bulbs
def repair_black_bulb(puzzle_map, initial_puzzle):
    bulbs_surrounding = []
    number_rows = initial_puzzle[0][0]
    number_cols = initial_puzzle[0][1]
    black_cell_number = len(initial_puzzle)
    bulbs = []
    for i in range(1, black_cell_number):
        col = initial_puzzle[i][0] - 1
        row = initial_puzzle[i][1] - 1
        value = initial_puzzle[i][2]
        bulbs_surround = 0
        empty_cells = []
        bulb_cells = []
        if col < number_cols - 1:
            if puzzle_map[row][col + 1] == init.CELL_BULB:
                bulbs_surround += 1
                bulb_cells.append([row, col + 1])
            elif puzzle_map[row][col + 1] == init.CELL_EMPTY:
                empty_cells.append([row, col + 1])
        if col > 0:
            if puzzle_map[row][col - 1] == init.CELL_BULB:
                bulbs_surround += 1
                bulb_cells.append([row, col - 1])
            elif puzzle_map[row][col - 1] == init.CELL_EMPTY:
                empty_cells.append([row, col - 1])
        if row < number_rows - 1:
            if puzzle_map[row + 1][col] == init.CELL_BULB:
                bulbs_surround += 1
                bulb_cells.append([row + 1, col])
            elif puzzle_map[row + 1][col] == init.CELL_EMPTY:
                empty_cells.append([row + 1, col])
        if row > 0:
            if puzzle_map[row - 1][col] == init.CELL_BULB:
                bulbs_surround += 1
                bulb_cells.append([row - 1, col])
            elif puzzle_map[row - 1][col] == init.CELL_EMPTY:
                empty_cells.append([row - 1, col])

        if value != init.CELL_BLACK_FIVE:

            if bulbs_surround > value:
                bulbs_surrounding = random.sample(bulb_cells, k=value)
            elif bulbs_surround < value:
                bulbs_surrounding = random.sample(empty_cells, k=value - bulbs_surround) + bulb_cells
            else:
                bulbs_surrounding = bulb_cells
        else:
            bulbs_surrounding = bulb_cells

        for x in range(0, len(bulbs_surrounding)):
            bulbs.append(bulbs_surrounding[x])

    return bulbs


# truncation for survival
def truncation(population_pool, number_survival):
    pool = sorted(population_pool, key=itemgetter('evaluation_fitness'), reverse=True)[:number_survival]
    random.shuffle(pool)
    return pool
