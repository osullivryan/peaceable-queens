from mlxtend.plotting import checkerboard_plot
from peaceablequeens.peacable_queens_types import BLACK_PIECES, WHITE_PIECES
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse


def plot_board(board, file):
    board = np.char.mod('%d', board)
    board = np.where(board==str(BLACK_PIECES), 'B', board) 
    board = np.where(board==str(WHITE_PIECES), 'W', board) 
    board = np.where(board==str(0), '', board) 

    brd = checkerboard_plot(board)
    plt.savefig(f"{file}.svg")
    plt.savefig(f"{file}.png")


def plot_board_from_file(board_file, output_file):
    with open(board_file, 'rb') as f:
        files = pickle.load(f)
    board = files['board']
    plot_board(board, output_file)



if __name__ == "__main__":
    
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Plot a file')

    # Add the arguments
    my_parser.add_argument('-file',
                        '-f',
                        metavar='file',
                        type=str,
                        help='The board file')

    my_parser.add_argument('-output',
                        '-o',
                        metavar='output',
                        type=str,
                        help='The output file')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    plot_board_from_file(args.file, args.output)
