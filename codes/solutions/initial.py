# import from outside functions
import random
from copy import deepcopy
from datetime import datetime

# -------- constants  --------

# import from project functions
from evaluation import check_bulb_shining

CELL_BLACK_ZERO = 0
CELL_BLACK_ONE = 1
CELL_BLACK_TOW = 2
CELL_BLACK_THREE = 3
CELL_BLACK_FOUR = 4
CELL_BLACK_FIVE = 5

# the value indicates that a cell can't put a bulb due to a zero black cell adjacent.
CELL_BULB_ZERO = 9

# the value indicates that a white cell
CELL_EMPTY = 6

# the value indicates that a bulb in a cell
CELL_BULB = 7

# the value indicates that a cell is light up by a bulb
CELL_LIGHT = 8


# set random seed
def set_random_seed(puzzle_config):
    # set the random seed
    if puzzle_config == "time":
        random.seed(datetime.now())
    else:
        seed_int = int(puzzle_config)
        random.seed(seed_int)


# initialize the map with white cell value
# return - array: initialize all cells as empty cell
def initialize_map(number_of_rows, number_of_cols):
    r_array = []  # array will be returned
    for x in range(0, number_of_rows):
        r_array.append([])
        for y in range(0, number_of_cols):
            r_array[x].append(CELL_EMPTY)
    return r_array


# loading the map data of the problem
# return - array:
#   0:number of rows and columns
#   >0:black cell info of row/col/value
def read_map(filepath):
    read_file = []  # original file data read
    initial_puzzle = []  # initial puzzle will be returned

    with open(filepath) as fp:
        line = fp.readline()
        read_file.append(line)
        while line:
            line = fp.readline()
            read_file.append(line)

    #  store first/second lines of data
    temp_cols = int(read_file[0])
    temp_rows = int(read_file[1])

    initial_puzzle.append([])
    initial_puzzle[0].append(temp_cols)
    initial_puzzle[0].append(temp_rows)

    length = len(read_file)

    # store data from the 3rd lines
    for i in range(2, length - 1):
        initial_puzzle.append([])
        message = read_file[i].split(" ")
        for j in range(0, len(message)):
            initial_puzzle[i - 1].append(int(message[j]))

    return initial_puzzle


# load the black cells into the map
# return - array: map_data (every rows * cols)with black cells value in it
def load_map_data(updating_map, black_cells):
    r_map = deepcopy(updating_map)  # data will be returned
    numbers = len(black_cells)
    for i in range(1, numbers):
        col = black_cells[i][0] - 1
        row = black_cells[i][1] - 1
        value = black_cells[i][2]
        r_map[row][col] = value
    return r_map


# initialize the map under validation
# all cells will fill up by bulbs if only unique way to do it
def initialize_validation_map(puzzle_map, rows, cols):
    # set Zero adjacent constraints
    for i in range(0, rows):
        for j in range(0, cols):
            if puzzle_map[i][j] == CELL_BLACK_ZERO:
                if i < rows - 1:
                    if puzzle_map[i + 1][j] == CELL_EMPTY:
                        puzzle_map[i + 1][j] = CELL_BULB_ZERO
                if j < cols - 1:
                    if puzzle_map[i][j + 1] == CELL_EMPTY:
                        puzzle_map[i][j + 1] = CELL_BULB_ZERO
                if i > 0:
                    if puzzle_map[i - 1][j] == CELL_EMPTY:
                        puzzle_map[i - 1][j] = CELL_BULB_ZERO
                if j > 0:
                    if puzzle_map[i][j - 1] == CELL_EMPTY:
                        puzzle_map[i][j - 1] = CELL_BULB_ZERO

    # validating all unique bulbs
    new_bulb = True
    while new_bulb:
        new_bulb = False
        for i in range(0, rows):
            for j in range(0, cols):
                if 0 < puzzle_map[i][j] < CELL_BLACK_FIVE:
                    grids = []
                    cell_empty = False
                    if i < rows - 1:
                        if puzzle_map[i + 1][j] == CELL_EMPTY:
                            cell_empty = True
                            grids.append([i + 1, j])
                        elif puzzle_map[i + 1][j] == CELL_BULB:
                            grids.append([i + 1, j])
                    if j < cols - 1:
                        if puzzle_map[i][j + 1] == CELL_EMPTY:
                            cell_empty = True
                            grids.append([i, j + 1])
                        elif puzzle_map[i][j + 1] == CELL_BULB:
                            grids.append([i, j + 1])
                    if i > 0:
                        if puzzle_map[i - 1][j] == CELL_EMPTY:
                            cell_empty = True
                            grids.append([i - 1, j])
                        elif puzzle_map[i - 1][j] == CELL_BULB:
                            grids.append([i - 1, j])
                    if j > 0:
                        if puzzle_map[i][j - 1] == CELL_EMPTY:
                            cell_empty = True
                            grids.append([i, j - 1])
                        elif puzzle_map[i][j - 1] == CELL_BULB:
                            grids.append([i, j - 1])
                    if puzzle_map[i][j] == len(grids):
                        for x in range(0, len(grids)):
                            puzzle_map[grids[x][0]][grids[x][1]] = CELL_BULB

                        # set cells lighted
                        set_lighted = check_bulb_shining(puzzle_map, rows, cols)

                        if set_lighted:
                            print("Error in lightening white cells")
                            # breakpoint()

                        # ask for more loop in while since a new bulb has been set
                        if cell_empty:
                            new_bulb = True

                    # check the empty cell which will be set as can't put a bulb in
                    # because the number of black cell also reaches the requirement
                    # only active when a new bulb has been set
                    if new_bulb:
                        if 0 < puzzle_map[i][j] < CELL_BLACK_FOUR:
                            cell_empty_adjacent = []
                            cell_bulb_adjacent = 0
                            if i < rows - 1:
                                if puzzle_map[i + 1][j] == CELL_EMPTY:
                                    cell_empty_adjacent.append([i + 1, j])
                                elif puzzle_map[i + 1][j] == CELL_BULB:
                                    cell_bulb_adjacent += 1
                            if j < cols - 1:
                                if puzzle_map[i][j + 1] == CELL_EMPTY:
                                    cell_empty_adjacent.append([i, j + 1])
                                elif puzzle_map[i][j + 1] == CELL_BULB:
                                    cell_bulb_adjacent += 1
                            if i > 0:
                                if puzzle_map[i - 1][j] == CELL_EMPTY:
                                    cell_empty_adjacent.append([i - 1, j])
                                elif puzzle_map[i - 1][j] == CELL_BULB:
                                    cell_bulb_adjacent += 1
                            if j > 0:
                                if puzzle_map[i][j - 1] == CELL_EMPTY:
                                    cell_empty_adjacent.append([i, j - 1])
                                elif puzzle_map[i][j - 1] == CELL_BULB:
                                    cell_bulb_adjacent += 1

                            if cell_bulb_adjacent and len(cell_empty_adjacent):
                                if puzzle_map[i][j] == cell_bulb_adjacent:
                                    for x in range(0, len(cell_empty_adjacent)):
                                        row_update = cell_empty_adjacent[x][0]
                                        col_update = cell_empty_adjacent[x][1]
                                        puzzle_map[row_update][col_update] = CELL_BULB_ZERO

    return puzzle_map
