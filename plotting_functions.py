import jax
import jax.numpy as jnp
import chess
import chess.svg
from IPython.display import display, HTML

def bitboard_to_fen(bitboard):
    piece_order = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']


    board = chess.Board(None)

    for i, piece_symbol in enumerate(piece_order):
        piece_bitboard = bitboard[i]

        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                if piece_bitboard[row, col]:
                    piece = chess.Piece.from_symbol(piece_symbol)
                    board.set_piece_at(square, piece)

    return board.fen()

def print_chessboard_from_bitboard(batch, grid_width=4):
    html = "<table><tr>"
    for i, bitboard in enumerate(batch):
        fen = bitboard_to_fen(bitboard)
        board = chess.Board(fen)
        board_svg = chess.svg.board(board, size=100)
        html += f"<td>{board_svg}</td>"
        if (i + 1) % grid_width == 0 and (i + 1) != len(batch):
            html += "</tr><tr>"
    html += "</tr></table>"
    display(HTML(html))
