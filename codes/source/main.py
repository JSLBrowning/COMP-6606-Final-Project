# import from project functions
from copy import deepcopy

import initial as init
import hill
import logs
from board import GameBoard
from utils import Generate_Board

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

    #Define Images Directory to locate board Pieces
    pieces_dir = '/home/rob/Desktop/Laptop_rob/Work/Auburn/Artificial_Intelligence/'\
                'Final_Project/GitHub/COMP-6606-Artificial-Intelligence-Final-Project/codes/images/'
    board_i = np.zeros([game_map.row, game_map.column])
    board_opt = np.zeros([game_map.row, game_map.column])

    print("initial board.....")

    for i in range(0, game_map.column):
        board_i[i, :] = game_map.board[game_map.column - i - 1] 
        print(game_map.board[game_map.column - i - 1])

    #Generate Initialized Board
    Generate_Board(pieces_dir, board_i)

    print("optimized board.....")

    for i in range(0, game_map.column):
        board_opt[i, :] = game_map.optimized_board[game_map.column - i - 1] 
        print(game_map.optimized_board[game_map.column - i - 1])

    #Generate Interactive Optimized Board
    Generate_Board(pieces_dir, board_opt)    

    exit(0)


if __name__ == "__main__":
    main()
