from peaceable_queens.board import create_board

def test_create_board():
    black_pieces = [i for i in range(3)]
    white_pieces = [i + 5 for i in range(3)]
    board_size = 5

    board = create_board(black_pieces, white_pieces, board_size)
    assert board
