from peaceable_queens.peacable_queens_types import BLACK_PIECES, WHITE_PIECES
from typing import List
from peaceable_queens.board import create_board
import numpy as np
from numba import jit, njit


@njit(fastmath=True)
def count_vertical_and_horizontal(board: np.ndarray) -> int:
    count = 0
    for i in range(board.shape[0]):
        column = board[:, i]
        row = board[i, :]
        # count the rows
        n_white = np.count_nonzero(row == WHITE_PIECES)
        n_black = np.count_nonzero(row == BLACK_PIECES)
        if n_white > 0 and n_black > 0:
            count += 1
        # count the columns
        n_white = np.count_nonzero(column == WHITE_PIECES)
        n_black = np.count_nonzero(column == BLACK_PIECES)
        if n_white > 0 and n_black > 0:
            count += 1
    return count

@njit(fastmath=True)
def count_diagonals(board: np.ndarray) -> int:
    count = 0
    diags_f = [
        np.diag(board,i) for i in range(-board.shape[0], board.shape[1] - 1)
    ]
    diags_b = [
        np.diag(np.flipud(board),i) for i in range(-board.shape[0], board.shape[1] - 1)
    ]
    for diag in diags_f:
        # count the pieces
        n_white = np.count_nonzero(diag == WHITE_PIECES)
        n_black = np.count_nonzero(diag == BLACK_PIECES)
        if n_white > 0 and n_black > 0:
            count += 1
    for diag in diags_b:
        # count the pieces
        n_white = np.count_nonzero(diag == WHITE_PIECES)
        n_black = np.count_nonzero(diag == BLACK_PIECES)
        if n_white > 0 and n_black > 0:
            count += 1

    return count

@njit(fastmath=True)
def count_same_space(board: np.ndarray, n_pieces: int) -> int:
    n_spaces_filled = np.sum(board > 0)
    return abs(n_pieces - n_spaces_filled)

@njit(fastmath=True)
def cost(iteration, n_pieces_each: int, board_size: int) -> int:
    black_pieces = iteration[:n_pieces_each]
    white_pieces = iteration[n_pieces_each:]
    board = create_board(black_pieces, white_pieces, board_size)
    return (
        count_vertical_and_horizontal(board)
        + count_diagonals(board)
        + count_same_space(board, len(black_pieces)*2)
    ),
