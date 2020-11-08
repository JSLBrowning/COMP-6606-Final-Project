# import from outside functions
import configparser
import random
# from copy import deepcopy
from datetime import datetime

# -------- constants  --------

# import from project functions
from hill import check_bulb_shining

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


# class for config parser
class Config:
    black_constraints = False
    annealing = False
    runs = 0
    constance = 0
    evaluation_number = 0
    random_seed = []
    log_path = []
    solution_path = []
    log_file = []
    solution_file = []

    def __init__(self, config_file):
        self.default = configparser.ConfigParser()
        self.default.sections()
        self.default.read(config_file)
        self.black_constraints = self.default["default"]["black_constraints"]
        self.annealing = self.default["default"]["annealing"]
        self.evaluation_number = int(self.default["default"]["termination_evaluation"])
        self.runs = int(self.default["default"]["number_runs"])
        self.random_seed = self.default["default"]["random_seed"]
        self.log_path = self.default["default"]["log_path"]
        self.solution_path = self.default["default"]["solution_path"]

        self.log_file = []
        self.solution_file = []
        self.set_random_seed()

    def set_random_seed(self):
        # set the random seed
        if self.random_seed == "time":
            random.seed(datetime.now())
        else:
            try:
                seed_int = int(self.random_seed)
                random.seed(seed_int)
            except ValueError:
                pass

    def get_random_seed(self):
        return self.random_seed

    def get_number_runs(self):
        return self.runs

    def get_termination_evaluation(self):
        return self.evaluation_number

    def get_log_file(self):
        return self.default["default"]["log_path"]

    def get_solution_file(self):
        return self.default["default"]["solution_path"]

    def get_constant(self):
        return self.constance

    def set_filename(self, map_name):
        text_word = map_name.split('/')
        self.log_file = self.log_path + text_word[len(text_word) - 1]
        self.solution_file = self.solution_path + text_word[len(text_word) - 1]
        # print(self.log_file)
        # print(self.solution_file)



