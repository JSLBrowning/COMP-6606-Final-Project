import tkinter as tk
from tkinter.ttk import *
from PIL import ImageTk, Image
from board import GameBoard
import os
from numpy import loadtxt, savetxt
import csv
import numpy as np

def Generate_Board(img_dir, board):

    gameboard = board

    #Define image directory for pieces
    home_dir = img_dir

    black_0 = Image.open(home_dir + "Black_0.png")
    b0 = black_0.resize((31, 31), Image.ANTIALIAS) 
    black_1 = Image.open(home_dir + "Black_1.png")
    b1 = black_1.resize((31, 31), Image.ANTIALIAS) 
    black_2 = Image.open(home_dir + "Black_2.png")
    b2 = black_2.resize((31, 31), Image.ANTIALIAS) 
    black_3 = Image.open(home_dir + "Black_3.png")
    b3 = black_3.resize((31, 31), Image.ANTIALIAS) 
    black_4 = Image.open(home_dir + "Black_4.png")
    b4 = black_4.resize((31, 31), Image.ANTIALIAS) 
    black_none = Image.open(home_dir + "Black_none.png")
    bn = black_none.resize((31, 31), Image.ANTIALIAS) 
    bulb = Image.open(home_dir + "bulb1.png")
    bb = bulb.resize((31, 31), Image.ANTIALIAS)
    redbulb = Image.open(home_dir + "redbulb1.png")
    rb = redbulb.resize((31, 31), Image.ANTIALIAS) 
    yellow = Image.open(home_dir + "yellow.png")
    y = yellow.resize((31, 31), Image.ANTIALIAS) 
    yellow_cross = Image.open(home_dir + "yellow_cross.png")
    yc = yellow_cross.resize((31, 31), Image.ANTIALIAS) 
    white = Image.open(home_dir + "white.png")
    w = white.resize((31, 31), Image.ANTIALIAS) 
    white_cross = Image.open(home_dir + "white_cross.png")
    wc = white_cross.resize((31, 31), Image.ANTIALIAS) 

    #Initialize Board
    root = tk.Tk()
    board = GameBoard(root)
    row , column = gameboard.shape
    board.pack(side="top", fill="both", expand="true", padx=row, pady=column)

    #Add the respective Pieces
    board_idx = {}
    _names = {}
    check = 0
    for i in range(row):
        
        for j in range(column):
            check+=1    
            _names["name{0}".format(check)] = "spot" + str(check)

            if gameboard[i,j] == 0:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(b0)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)
            elif gameboard[i,j] == 1:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(b1)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)    
            elif gameboard[i,j] == 2:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(b2)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)
            elif gameboard[i,j] == 3:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(b3)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)    
            elif gameboard[i,j] == 4:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(b4)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)
            elif gameboard[i,j] == 5:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(bn)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)
            elif gameboard[i,j] == 6:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(w)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)
            elif gameboard[i,j] == 7:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(bb)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)
            elif gameboard[i,j] == 8:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(y)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)
            elif gameboard[i,j] == 9:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(wc)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)
            elif gameboard[i,j] == 10:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(yc)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)    
            elif gameboard[i,j] == 11:
                board_idx["spot{0}".format(check)] = ImageTk.PhotoImage(rb)
                board.addpiece(_names["name{0}".format(check)], board_idx["spot{0}".format(check)], i,j)
      
    root.mainloop()
            
'''def make_dataset(cases):

    game_map = hill.Map(map_file, configurations)

    #Define Images Directory to locate board Pieces  
    pieces_dir = get_img_path()

    board_i = np.zeros([game_map.row, game_map.column])
    board_opt = np.zeros([game_map.row, game_map.column])

    print("initial board.....")

    for i in range(0, game_map.column):
        board_i[i, :] = game_map.board[game_map.column - i - 1] 
        print(game_map.board[game_map.column - i - 1])


    print("optimized board.....")

    for i in range(0, game_map.column):
        board_opt[i, :] = game_map.optimized_board[game_map.column - i - 1] 
        print(game_map.optimized_board[game_map.column - i - 1])
'''

def get_img_path():
    path = os.getcwd() + '/images/'
    return path

def get_play_path():
    path = os.getcwd() + '/playlist/hakuna.mp3'
    return path     

	
def create_csv(data, name_str):
    # save to csv file
    save_str = os.getcwd() + '/logs/' + name_str + '.csv'
    savetxt(save_str, data.astype(np.uint8), delimiter=',')

def get_csv(name_str):
    # save to csv file
    load_str = os.getcwd() + '/logs/' + name_str + '.csv'
    data = loadtxt(load_str, delimiter=',')
    return data

