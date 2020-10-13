# import from project functions
from copy import deepcopy

import initial as init
import evaluation as evaluation
import evolutionary as ea
import logs

# --------imports from system------
import sys
from operator import itemgetter
import numpy as np
import json


# domination, minor is better
def dominates(x1, x2):
    if np.any(x1 < x2) and np.all(x1 <= x2):
        return True
    else:
        return False


# pareto front
# non-dominated sorting
def non_dominated_sorting(puzzles_input, config):
    puzzles = deepcopy(puzzles_input)

    number_objectives = config["number_objectives"]
    source = np.array([])

    # print("Before dominated...........")
    # for i in range(0, len(puzzles)):
    #     print(f'No. {i}  {puzzles[i]}')

    # create arrays for domination
    white_cells = puzzles[0].get("white_cells")
    if number_objectives == 3:
        for i in range(0, len(puzzles)):
            initial_dominates = {'dominates': [],
                                 'front_line': 0,
                                 'domination_count': 0}

            puzzles[i].update(initial_dominates)

            x = np.array([
                puzzles[i].get("empty_cells"),
                puzzles[i].get("shining_conflict"),
                puzzles[i].get("black_conflict")
            ])
            source = np.append(source, x, axis=0)
            objectives = {'objectives': [white_cells - x[0], x[1], x[2]]}
            puzzles[i].update(objectives)
            # print(f'{x[0]}   {x[1]}   {x[2]}')
    elif number_objectives == 4:
        for i in range(0, len(puzzles)):
            initial_dominates = {'dominates': [],
                                 "front_line": 0,
                                 'domination_count': 0}
            puzzles[i].update(initial_dominates)

            x = np.array([
                puzzles[i].get("empty_cells"),
                puzzles[i].get("shining_conflict"),
                puzzles[i].get("black_conflict"),
                puzzles[i].get("bulb_cells_total")
            ])
            source = np.append(source, x, axis=0)
            objectives = {'objectives': [white_cells - x[0], x[1], x[2], x[3]]}
            puzzles[i].update(objectives)
    else:
        print("config: number_objectives must be 3 or 4!")
        exit(0)

    source = np.reshape(source, (-1, number_objectives))

    # print("Before count domination...........")
    # for i in range(0, len(puzzles)):
    #     print(f'No. {i}  {puzzles[i]}')
    # print(source)

    # domination
    for i in range(0, len(puzzles)):
        for j in range(0, len(puzzles)):
            if dominates(source[i], source[j]):
                new_domination_count = puzzles[j].get("domination_count")
                current_domination = {"domination_count": (new_domination_count + 1)}
                puzzles[j].update(current_domination)
                dominates_list = puzzles[i].get('dominates')
                if j in dominates_list:
                    print(f'error: duplicated domination count:{i}   {j}')
                    # print(puzzles[i])
                    # print(puzzles[j])
                    # print(source[i])
                    # print(source[j])
                    # print(len(puzzles))
                    # print(dominates_list)
                    exit(0)
                dominates_list.append(j)
                new_dominates = {'dominates': dominates_list}
                puzzles[i].update(new_dominates)

    # sorting
    front_line = 1
    front_line_data = []
    unsorted_count = len(puzzles)

    # print("\n\n Before sorting...........\n\n")
    # for i in range(0, len(puzzles)):
    #     print(f'No. {i}  {puzzles[i]}')

    while unsorted_count:
        if front_line > 50:
            unsorted_count = 0
        for i in range(0, len(puzzles)):
            if puzzles[i].get('domination_count') == 0:
                if puzzles[i].get('front_line') == 0:
                    current_front = {"front_line": front_line}
                    puzzles[i].update(current_front)
                    unsorted_count -= 1
                    if front_line == 1:
                        front_line_data.append(source[i])

        for i in range(0, len(puzzles)):
            if puzzles[i].get('front_line') == front_line:
                sub_sets = puzzles[i].get("dominates")
                if sub_sets:
                    for j in range(0, len(sub_sets)):
                        new_domination_count = puzzles[sub_sets[j]].get('domination_count') - 1
                        if new_domination_count < 0:
                            print(f'No.{i} Error: domination_count < 0 ')
                            new_domination_count = 0
                            print(sub_sets)
                            exit(0)
                        current_domination_count = {"domination_count": new_domination_count}
                        puzzles[sub_sets[j]].update(current_domination_count)

        front_line += 1

    # update fitness by front line
    step = int(100 / front_line)
    for i in range(0, front_line):
        for j in range(0, len(puzzles)):
            if puzzles[j].get('front_line') == i + 1:
                new_fitness = {"evaluation_fitness": 100 - step * i}
                puzzles[j].update(new_fitness)

    # adjust the data format for return
    mean_source = np.round(source.mean(axis=0), 2)
    best_source = source.min(axis=0)

    mean_source[0] = np.round(puzzles[0].get("white_cells") - mean_source[0], 2)
    best_source[0] = puzzles[0].get("white_cells") - best_source[0]

    log = np.array([])
    for i in range(0, number_objectives):
        log = np.append(log, mean_source[i])
        log = np.append(log, best_source[i])

    front_line_population = []
    for i in range(0, len(puzzles)):
        if puzzles[i].get('front_line') == 1:
            front_line_population.append(puzzles[i])

    result_log = {
        'log': log,
        'front_line_data': np.sort(front_line_data, axis=0),
        'population': puzzles,
        'front_population': front_line_population
    }

    # for i in range(0, number_objectives):
    #     result_log.append([mean_source[i], min_source[i]])
    #
    # for i in range(0, len(front_line_data)):
    #     print(front_line_data[i])

    # print(f"\n\n after dominated...........\n \n")
    # for i in range(0, len(puzzles)):
    #     print(f'No. {i}  {puzzles[i]}')

    return result_log


# comparison the solution
def solution_comparison(solution_previous, solution_new, config):
    s1 = deepcopy(solution_previous)
    s2 = deepcopy(solution_new)
    number_objectives = config["number_objectives"]

    # create arrays for domination
    source_1 = np.array([])
    source_2 = np.array([])
    if number_objectives == 3:
        for i in range(0, len(s1)):
            x = np.array([s1[i].get("empty_cells"), s1[i].get("shining_conflict"), s1[i].get("black_conflict")])
            source_1 = np.append(source_1, x, axis=0)

        for i in range(0, len(s2)):
            x = np.array([s2[i].get("empty_cells"), s2[i].get("shining_conflict"), s2[i].get("black_conflict")])
            source_2 = np.append(source_2, x, axis=0)
            # print(f'{x[0]}   {x[1]}   {x[2]}')
    elif number_objectives == 4:
        for i in range(0, len(s1)):
            x = np.array([s1[i].get("empty_cells"), s1[i].get("shining_conflict"),
                          s1[i].get("black_conflict"), s1[i].get("bulb_cells_total")
                          ])
            source_1 = np.append(source_1, x, axis=0)

        for i in range(0, len(s2)):
            x = np.array([s2[i].get("empty_cells"), s2[i].get("shining_conflict"),
                          s2[i].get("black_conflict"), s2[i].get("bulb_cells_total")])
            source_2 = np.append(source_2, x, axis=0)
    else:
        print("config: number_objectives must be 3 or 4!")
        exit(0)

    source_1 = np.reshape(source_1, (-1, number_objectives))
    source_2 = np.reshape(source_2, (-1, number_objectives))

    p1 = 0
    p2 = 0
    for i in range(0, len(s1)):
        for j in range(0, len(s2)):
            if dominates(source_1[i], source_2[j]):
                p1 += 1
            elif dominates(source_2[j], source_1[i]):
                p2 += 1

    # print("Previous....")
    # print(source_1)
    # print("New......")
    # print(source_2)
    # print("p1 and p2")
    # print(p1)
    # print(p2)
    if p1 >= p2:
        return s1
    else:
        return s2


# read arguments
def config_argv():
    if len(sys.argv) != 3:
        print("Error")
        sys.exit(0)
    # reading arguments from arg vector
    problem_filepath = sys.argv[1]
    config_filepath = sys.argv[2]
    print(problem_filepath, config_filepath)


# ======== ========= =========
# ======== Main body =========
# ======== ========= =========
def main():
    # problem_filepath = "../problems/d1.lup"
    # config_filepath = "config_green_d1.json"

    if len(sys.argv) != 3:
        print("Error")
        sys.exit(0)
    # reading arguments from arg vector
    problem_filepath = sys.argv[1]
    config_filepath = sys.argv[2]

    print(problem_filepath, config_filepath)

    # loading config
    with open(config_filepath) as config_file:
        puzzle_config = json.load(config_file)

    log_file_path = puzzle_config["log_file"]
    logs.log_text_add(puzzle_config, log_file_path)

    # set the random seed
    init.set_random_seed(puzzle_config.get("random_seed"))

    # reading problem info
    print("Reading Map....")

    print(problem_filepath)
    puzzle_initial = init.read_map(problem_filepath)
    print(puzzle_initial)

    number_cols = puzzle_initial[0][0]
    number_rows = puzzle_initial[0][1]

    # create an initial map by col/row
    puzzle_map = init.initialize_map(number_cols, number_rows)

    # update myMap by loading the config
    puzzle_map = init.load_map_data(puzzle_map, puzzle_initial)

    for i in range(0, len(puzzle_map)):
        print(puzzle_map[len(puzzle_map) - i - 1])

    # $$$$$$$  config begins here  $$$$$$$$$$$$$$$$$
    mu_size = puzzle_config["mu_size"]
    lambda_size = puzzle_config["lambda_size"]

    termination_constant = {"number_evaluation": puzzle_config["termination_evaluation"],
                            "no_change": puzzle_config["termination_no_change"]
                            }
    # $$$$$$$  config ends here  $$$$$$$$$$$$$$$$$

    # implement constraints or not
    if puzzle_config["black_cell_constraints"]:
        puzzle_constraints = init.initialize_validation_map(puzzle_map, number_rows, number_cols)
        print("Printing after initializing by unique bulbs...")
        for i in range(0, len(puzzle_constraints)):
            print(puzzle_constraints[len(puzzle_constraints) - i - 1])
        net_white_cells = evaluation.check_net_cells(puzzle_constraints, number_rows, number_cols)
        puzzle_map = puzzle_constraints
    else:
        net_white_cells = evaluation.check_net_cells(puzzle_map, number_rows, number_cols)

    solution = []

    for runs in range(0, puzzle_config["number_runs"]):

        print(f'Run   {runs + 1}')

        run_log = []
        mutation_rate = []

        # initialize population pool of mu - parents
        initial_mu = ea.create_population_pool(net_white_cells, mu_size)

        population_mu = []
        for i in range(len(initial_mu)):
            evaluations = evaluation.evaluate_puzzle_map(puzzle_map,
                                                         black_cells=puzzle_initial,
                                                         bulb_cells=initial_mu[i],
                                                         config=puzzle_config)
            population_mu.append(evaluations)

        log_data = non_dominated_sorting(population_mu, puzzle_config)
        population_mu = log_data.get("population")

        # for i in range(0, len(population_mu)):
        #     print(population_mu[i])
        # exit(0)

        number_evaluation = mu_size

        run_log.append([mu_size, log_data.get("log")])

        # prepare variables for EA loop
        termination = False

        front_line_data = log_data.get("front_line_data")
        no_change_front = 0
        # start to EA loop
        while not termination:

            parents_pool = ea.parent_selection(population_mu, lambda_size, puzzle_config)

            population_lambda = ea.generate_offspring(net_white_cells, lambda_size,
                                                      parents_pool, puzzle_map, puzzle_initial,
                                                      puzzle_config, mutation_rate)

            # update population lambda by NSGA-II
            if puzzle_config["survival_strategy"] == "plus":
                survival_pool = population_lambda + population_mu
            else:
                survival_pool = population_lambda

            survival_pool = non_dominated_sorting(survival_pool, puzzle_config)
            survival_pool = survival_pool.get("population")

            population_mu = ea.survival_selection_MOEA(puzzle_config, survival_pool)

            # update population survival by NSGA-II
            log_data = non_dominated_sorting(population_mu, puzzle_config)
            population_mu = log_data.get("population")

            number_evaluation += lambda_size

            run_log.append([number_evaluation, log_data.get("log")])
            print(f'{number_evaluation}  {log_data.get("log")}')

            front_line_data_new = log_data.get("front_line_data")

            if np.array_equiv(front_line_data, front_line_data_new):
                no_change_front += lambda_size
                if no_change_front >= puzzle_config["termination_no_change"]:
                    termination = True
            else:
                front_line_data = front_line_data_new
                no_change_front = 0

            if number_evaluation >= termination_constant["number_evaluation"]:
                termination = True
        # print(mutation_rate)

        # write running log into file
        # for i in range(0, len(run_log)):
        #     print(run_log[i])
        # for i in range(0, len(population_mu)):
        #     print(population_mu[i])

        front_p = log_data.get("front_population")
        # print("\n front line population \n")
        # for i in range(0, len(front_p)):
        #     print(front_p[i])

        logs.logs_write(log_file_path, runs, run_log)

        if runs == 0:
            solution = front_p
        else:
            solution = solution_comparison(solution, front_p, puzzle_config)
        for i in range(0, len(solution)):
            print(solution[i])
    logs.write_solution(puzzle_initial, puzzle_map, solution, puzzle_config["solution_file"])
    return


if __name__ == "__main__":
    main()
