# import from project functions
from copy import deepcopy

import initial as init
import hill
import logs

# --------imports from system------
import sys
from operator import itemgetter
import numpy as np


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

    if len(sys.argv) != 3:
        print("Error")
        sys.exit(0)
    # reading arguments from arg vector
    map_file = sys.argv[1]
    config_file = sys.argv[2]
    # config_file = "default2.cfg"
    # map_file = "./maps/map2.txt"
    configurations = init.Config(config_file)
    configurations.set_filename(map_file)

    print(configurations.annealing)
    print(configurations.black_constraints)
    print(configurations.log_file)
    print(configurations.solution_file)

    game_map = hill.Map(map_file, configurations)

    print("initial board.....")

    for i in range(0, game_map.column):
        print(game_map.board[game_map.column - i - 1])

    print("optimized board.....")

    for i in range(0, game_map.column):
        print(game_map.optimized_board[game_map.column - i - 1])

    exit(0)


if __name__ == "__main__":
    main()
