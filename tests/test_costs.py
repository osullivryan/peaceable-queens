from peaceablequeens.types import BLACK_PIECES, WHITE_PIECES
from peaceablequeens.cost import count_same_space, count_diagonals, count_vertical_and_horizontal
import numpy as np

def test_count_same_space():
    board = np.array([
        [WHITE_PIECES, WHITE_PIECES, 0],
        [0, 0, 0],
        [0, BLACK_PIECES, BLACK_PIECES],
    ])
    assert count_same_space(board, 4) == 0


def test_count_diagonals():
    board = np.array([
        [WHITE_PIECES, WHITE_PIECES, 0],
        [0, 0, 0],
        [0, BLACK_PIECES, BLACK_PIECES],
    ])
    cost = count_diagonals(board)
    assert cost == 1


def test_count_vertical_and_horizontal():
    board = np.array([
        [WHITE_PIECES, WHITE_PIECES, 0],
        [0, 0, 0],
        [0, BLACK_PIECES, BLACK_PIECES],
    ])
    cost = count_vertical_and_horizontal(board)
    assert cost == 1
