from numba.core.errors import NumbaPendingDeprecationWarning
from peaceablequeens.optimize import main 
from peaceablequeens.board import create_board
from peaceablequeens.plotting import plot_board
import random
import numpy as np
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning) 
import datetime


class Individual(object):
    def __init__(self, name):
        self.name = name


if __name__ == "__main__":

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Run the Peacable Queens')

    # Add the arguments
    my_parser.add_argument('-board',
                        '-b',
                        metavar='board',
                        type=int,
                        help='The board size')

    my_parser.add_argument('-pieces',
                        '-p',
                        metavar='pieces',
                        type=int,
                        help='The number of pieces')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    board_size = args.board
    n_pieces = args.pieces

    last_pop, hof, logbook = main(board_size, n_pieces)
    if hof[0].fitness.values[0] == 0:
        print('++++++++++++++')
        print('SOLUTION FOUND')
        print('++++++++++++++')
        time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        black_pieces = hof[0][:n_pieces]
        white_pieces = hof[0][n_pieces:]
        board = create_board(black_pieces, white_pieces, board_size)
        np.set_printoptions(threshold=np.inf)

        cp = dict(
            rndstate=random.getstate(),
            black_pieces=black_pieces,
            white_pieces=white_pieces,
            board=board,
        )
        with open(f"results/SOLUTION_{board_size}_{n_pieces}_{time_stamp}.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)

        plot_board(board, f"results/SOLUTION_{board_size}_{n_pieces}_{time_stamp}")
