# COMP-6606-Artificial-Intelligence-Final-Project
The purpose of this project was to apply the techniques we learned in Auburn University's COMP 6600/6606 Artificial Intelligence course to a new setting that we were interested in.


############################
# running codes ###
############################

cd codes

python3 source/main.py problems/a2.lup source/default.cfg

############################
# Board number meanings  ###
############################

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