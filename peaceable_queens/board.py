import numpy as np
from typing import List


def create_board(
    black_pieces: List[int], white_pieces: List[int], board_size: int
) -> np.ndarray:
    board = np.empty((board_size, board_size), dtype=str)
    for piece in black_pieces:
        board.itemset(piece, "black")
    for piece in white_pieces:
        board.itemset(piece, "white")
    return board
