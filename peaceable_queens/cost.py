from typing import List
from peaceable_queens.board import create_board
import numpy as np


def count_vertical_and_horizontal(board: np.ndarray) -> int:
    count = 0
    for i in range(board.shape[0]):
        column = board[:, i]
        row = board[i, :]
        # count the rows
        n_white = np.count_nonzero(row == "w")
        n_black = np.count_nonzero(row == "b")
        if n_white > 0 and n_black > 0:
            count += 1
        # count the columns
        n_white = np.count_nonzero(column == "w")
        n_black = np.count_nonzero(column == "b")
        if n_white > 0 and n_black > 0:
            count += 1
    return count


def count_diagonals(board: np.ndarray) -> int:
    count = 0
    diags_f = [
        board[::-1, :].diagonal(i) for i in range(-board.shape[0], board.shape[1] - 1)
    ]
    diags_b = [
        np.flipud(board)[::-1, :].diagonal(i)
        for i in range(-board.shape[0], board.shape[1] - 1)
    ]
    for diag in diags_f + diags_b:
        # count the piences
        n_white = np.count_nonzero(diag == "w")
        n_black = np.count_nonzero(diag == "b")
        if n_white > 0 and n_black > 0:
            count += 1

    return count


def count_same_space(black_pieces: List[int], white_pieces: List[int]) -> int:
    pieces = np.array([*black_pieces, *white_pieces])
    _, counts = np.unique(pieces, return_counts=True)
    return np.sum(counts - np.ones_like(counts))


def cost(iteration, n_pieces_each: int, board_size: int) -> int:
    black_pieces = iteration[:n_pieces_each]
    white_pieces = iteration[n_pieces_each:]
    board = create_board(black_pieces, white_pieces, board_size)
    return (
        count_vertical_and_horizontal(board)
        + count_diagonals(board)
        + count_same_space(black_pieces, white_pieces)
    ),
