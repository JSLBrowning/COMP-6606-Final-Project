
# import from project functions
import initial as init

# import from outside functions
import random
from copy import deepcopy


# set bulbs into white cells
# return - array: map_data (every rows * cols)with bulbs in it
def set_random_bulbs(puzzle_map, white_cells, bulb_number):
    r_bulb_map = deepcopy(puzzle_map)

    # random select bulbs placement
    r_choice = random.sample(white_cells, k=bulb_number)

    # update map data with bulbs
    for i in range(0, bulb_number):
        r_row = r_choice[i][0]
        r_col = r_choice[i][1]
        r_bulb_map[r_row][r_col] = init.CELL_BULB

    return r_bulb_map


# count all numbers of :
# 1. black cells
# 2. shining cells
# 3. bulb cells
def evaluate_puzzle_map(puzzle_map, black_cells, bulb_cells, config):
    number_cells_black = len(black_cells) - 1
    number_cells_empty = 0
    number_cells_bulb = 0

    cols = black_cells[0][0]
    rows = black_cells[0][1]

    available_placement = 0
    for i in range(0, rows):
        for j in range(0, cols):
            if puzzle_map[i][j] == init.CELL_EMPTY:
                available_placement += 1

    puzzle = insert_bulbs(puzzle_map, bulb_cells)

    fitness_shining_conflict = check_bulb_shining(puzzle, rows, cols)
    for i in range(0, rows):
        for j in range(0, cols):
            if puzzle[i][j] == init.CELL_BULB:
                number_cells_bulb += 1
            if puzzle[i][j] == init.CELL_EMPTY:
                number_cells_empty += 1
            elif puzzle[i][j] == init.CELL_BULB_ZERO:
                number_cells_empty += 1

    number_cells_shining = rows * cols - number_cells_empty - number_cells_black
    fitness_black_conflict = check_black_bulb(puzzle, rows, cols, black_cells)

    # penalty function
    total_conflict = fitness_shining_conflict + fitness_black_conflict
    total_available_cell = rows * cols - number_cells_black
    shrink = config["penalty_shrink_factor"]
    minus = config["penalty_minus_factor"]
    if total_conflict > available_placement:
        total_conflict = available_placement - 1
    minor_factor = (available_placement - total_conflict * minus) / available_placement
    if minor_factor < 0:
        minor_factor = 0
    evaluation_fitness = int(100 * number_cells_shining * minor_factor / total_available_cell)
    original_fitness = int(100 * number_cells_shining / total_available_cell)
    if total_conflict:
        if config["fitness_function"] == "original":
            original_fitness = evaluation_fitness
            evaluation_fitness = 0
        else:
            evaluation_fitness = int(evaluation_fitness * shrink)

    puzzle_eval_data = {
        "black_cells": number_cells_black,
        "white_cells": rows * cols - number_cells_black,
        "empty_cells": number_cells_empty,
        "total_conflict": total_conflict,
        "shining_conflict": fitness_shining_conflict,
        "black_conflict": fitness_black_conflict,
        "domination_count": 0,
        "front_line": 0,
        "dominates": [],
        "original_fitness": original_fitness,
        "evaluation_fitness": evaluation_fitness,
        "number_cells_shining": number_cells_shining,
        "bulb_cells": len(bulb_cells),
        "bulb_cells_total": number_cells_bulb,
        "bulbs": bulb_cells
    }
    #    print(puzzle_eval_data)
    #    breakpoint()

    return puzzle_eval_data


# put the bulb array into puzzle map
def insert_bulbs(puzzle_map, bulbs):
    puzzle_copied = deepcopy(puzzle_map)
    for i in range(0, len(bulbs)):
        row = bulbs[i][0]
        col = bulbs[i][1]
        puzzle_copied[row][col] = init.CELL_BULB
    return puzzle_copied


# check black cell value by the number of surrounded bulbs
# return - True: bulbs adjacent fit
# return - False: bulbs adjacent don't fit
def check_black_bulb(map_data, number_rows, number_cols, black_data):
    conflict = 0
    black_cell_number = len(black_data)
    for i in range(1, black_cell_number):
        col = black_data[i][0] - 1
        row = black_data[i][1] - 1
        value = black_data[i][2]
        bulbs_surround = 0
        if col < number_cols - 1:
            if map_data[row][col + 1] == init.CELL_BULB:
                bulbs_surround += 1
        if col > 0:
            if map_data[row][col - 1] == init.CELL_BULB:
                bulbs_surround += 1
        if row < number_rows - 1:
            if map_data[row + 1][col] == init.CELL_BULB:
                bulbs_surround += 1
        if row > 0:
            if map_data[row - 1][col] == init.CELL_BULB:
                bulbs_surround += 1

        if bulbs_surround != value and value != init.CELL_BLACK_FIVE:
            conflict += abs(bulbs_surround - value)

    return conflict


# check the status of white cells whether light up or not
# return - True: all white cells are light up
# return - False: not all white cells are light up
def check_light_up(map_data, number_rows, number_cols):
    # count_white_cells = 0
    net_white_cells = []
    for i in range(0, number_rows):
        for j in range(0, number_cols):
            if map_data[i][j] == init.CELL_EMPTY:
                net_white_cells.append([i, j])

    return net_white_cells


# check white cells which can put a bulb in
def check_net_cells(map_data, number_rows, number_cols):
    # count_white_cells = 0
    net_white_cells = []
    for i in range(0, number_rows):
        for j in range(0, number_cols):
            if map_data[i][j] == init.CELL_EMPTY:
                net_white_cells.append([i, j])

    return net_white_cells


# read the map if bulbs are placed as valid
# return array: bulbs data with row/col info
#   0: number of bulbs
#   >0: col/row indicates bulb cell
def read_valid_map(map_data, number_rows, number_cols):
    puzzle_data = []
    count_bulb = 0  # count of bulbs
    count_lighted = 0  # count of lighted cell
    puzzle_data.append([0, 0])
    for i in range(0, number_rows):
        for j in range(0, number_cols):
            if map_data[i][j] == init.CELL_BULB:
                puzzle_data.append([])
                puzzle_data[count_bulb + 1].append(j + 1)
                puzzle_data[count_bulb + 1].append(i + 1)
                count_bulb += 1
            elif map_data[i][j] == init.CELL_LIGHT:
                count_lighted += 1
    count_lighted += count_bulb
    puzzle_data[0][0] = count_bulb
    puzzle_data[0][1] = count_lighted
    return puzzle_data


# check all white cells which has a bulb in it
# return - True: no two bulbs shine on each other
# return - False: at least two bulbs shine on each other
# @@ note @@ : puzzle_map will be updated by lighted info
def check_bulb_shining(puzzle_map, row, col):
    # validation = True
    # check the bulbs by every row
    conflict = 0
    for i in range(0, row):
        start = 0
        bulb = 0
        for j in range(0, col):
            cell = puzzle_map[i][j]
            if cell == init.CELL_BULB:
                bulb += 1
            if (cell < init.CELL_EMPTY) or (j == col - 1):
                if bulb > 1:
                    conflict += bulb - 1
                if bulb:
                    for x in range(start, j + 1):
                        if puzzle_map[i][x] == init.CELL_EMPTY or puzzle_map[i][x] == init.CELL_BULB_ZERO:
                            puzzle_map[i][x] = init.CELL_LIGHT
                    bulb = 0
                start = j

    # check the bulbs by every column
    for i in range(0, col):
        start = 0
        bulb = 0
        for j in range(0, row):
            cell = puzzle_map[j][i]
            if cell == init.CELL_BULB:
                bulb += 1
            if (cell < init.CELL_EMPTY) or (j == row - 1):
                if bulb > 1:
                    conflict += bulb - 1
                if bulb:
                    for x in range(start, j + 1):
                        if puzzle_map[x][i] == init.CELL_EMPTY or puzzle_map[x][i] == init.CELL_BULB_ZERO:
                            puzzle_map[x][i] = init.CELL_LIGHT
                    bulb = 0
                start = j

    return conflict
