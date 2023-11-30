#Chess Diffusion

## Overview
This repository contains an implementation of the paper 'Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning' specifically applied to chess data. Our approach involves casting discrete chess data into bits, performing diffusion, and then reverting the data back to its original discrete categorical form. This work builds upon and modifies the code from [lucidrains' bit-diffusion implementation](https://github.com/lucidrains/bit-diffusion), originally designed for image data.

## Chess Data Representations
We experiment with two different representations of a chessboard:

1. **8x8 Tensor Representation**: Each square is assigned an integer (0-12) to represent the chess piece or an empty space. This method has shown promise in generating realistic chess positions, capturing some pawn structures and correct king placements. However, it sometimes suffers from mode collapse and a tendency to converge to an empty board.

2. **Bitboard Representation (12x8x8 Tensor)**: Each square is represented by a 12-element vector, with each element corresponding to a chess piece. The value is 1 if the piece occupies the square; otherwise, 0. This representation tends to converge quickly to an empty board and currently does not capture meaningful structure as effectively as the first method.

## Challenges and Potential Improvements
- The model occasionally generates very similar samples, suggesting mode collapse.
- A tendency to converge to an empty board reflects the statistical average of an actual chess game.
- The complexity of the UNet structure, borrowed from image diffusion models, might be excessive for the 8x8 chessboard.
- Refinement of the bit diffusion algorithm to be more specifically tailored for chess data.
- Adjustments in data representation and model architecture are under consideration for better performance.


## Contributing
We welcome contributions and suggestions for improvements. Please feel free to fork the repository and submit pull requests.

## Acknowledgments
- Original implementation of discrete diffusion for images by [lucidrains](https://github.com/lucidrains/bit-diffusion).
- Inspiration from the paper 'Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning'.

