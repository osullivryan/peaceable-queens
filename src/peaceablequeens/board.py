from peaceablequeens.peacable_queens_types import BLACK_PIECES, WHITE_PIECES
import numpy as np
from typing import List
from numba import njit, int64

@njit(fastmath=True)
def create_board(
    black_pieces: List[int], white_pieces: List[int], board_size: int
) -> np.ndarray:
    board = np.zeros((board_size**2, 1), dtype=int64)
    for i in range(len(black_pieces)):
        board[black_pieces[i]] = BLACK_PIECES
    for i in range(len(white_pieces)):
        board[white_pieces[i]] = WHITE_PIECES
    board = board.reshape((board_size, board_size))
    return board
