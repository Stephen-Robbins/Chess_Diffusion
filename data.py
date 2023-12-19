import pandas as pd
import jax
import jax.numpy as jnp
import chess
import chess.svg
from IPython.display import SVG
jax.config.update("jax_enable_x64", True)


def bitboards_to_array(bb: jnp.array) -> jnp.array:
    bb = jnp.asarray(bb, dtype=jnp.uint64)[:, jnp.newaxis]
    s = 8 * jnp.arange(7, -1, -1, dtype=jnp.uint64)
    b = (bb >> s).astype(jnp.uint8)
    b = jnp.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)

def fen_to_bitboard(fen):
    board = chess.Board(fen)
    black, white = board.occupied_co

    bitboards = jnp.array([
        black & board.pawns,
        black & board.knights,
        black & board.bishops,
        black & board.rooks,
        black & board.queens,
        black & board.kings,
        white & board.pawns,
        white & board.knights,
        white & board.bishops,
        white & board.rooks,
        white & board.queens,
        white & board.kings,
       
    ], dtype=jnp.uint64)

    board_array = bitboards_to_array(bitboards)
    # Create the 13th layer for empty spaces
    empty_spaces = jnp.ones((8, 8), dtype=jnp.uint8)
    for bb in board_array:
        empty_spaces = empty_spaces & ~bb

    # Add the 13th layer to the board array
    board_array = jnp.concatenate((board_array, empty_spaces[jnp.newaxis, :, :]), axis=0)

    return jax.device_put(board_array)
def sample_bitboards(n):
    df = pd.read_csv('chessData.csv')
    # Sample n FEN strings randomly from the DataFrame
    sampled_fens = df['FEN'].sample(n, replace=True).tolist()

    # Convert each FEN string to its bitboard representation
    bitboards = [fen_to_bitboard(fen) for fen in sampled_fens]

    # Stack all the bitboards into a single JAX array
    return jnp.stack(bitboards)