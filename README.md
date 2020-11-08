# COMP-6606-Artificial-Intelligence-Final-Project
The purpose of this project was to apply the techniques we learned in Auburn University's COMP 6600/6606 Artificial Intelligence course to a new problem that we were interested in.

We chose to focus on the [Light Up puzzle](https://en.wikipedia.org/wiki/Light_Up_(puzzle)), and solved it using hill climbing, simulated annealing, and a deep neural network.

# Running Codes #
cd codes

python3 source/main.py problems/a2.lup source/default.cfg

# Board Legend #

CELL_BLACK_ZERO = 0

CELL_BLACK_ONE = 1

CELL_BLACK_TWO = 2

CELL_BLACK_THREE = 3

CELL_BLACK_FOUR = 4

CELL_BLACK_FIVE = 5

CELL_BULB_ZERO = 9 (This value indicates that a bulb cannot be placed adjacent to this cell.)

CELL_EMPTY = 6 (This value indicates an empty white cell.)

CELL_BULB = 7 (This value indicates a white cell with a bulb inside it.)

CELL_LIGHT = 8 (This value indicates that a cell is lit up by a bulb.)