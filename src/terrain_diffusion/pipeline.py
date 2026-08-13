"""Model Pipeline

Overview

Turns an uncleaned patch into a finished full resolution elevation patch, and hides how that is
done from the rest of the system.
It exposes one operation: given an uncleaned patch and any conditioning, return a cleaned patch
of the same size. Behind that operation it runs whatever models are needed and applies the
elevation transforms, so callers never see latent maps or model internals.
For the first working version it can be a stand-in that just smooths the patch. For real terrain
it runs the core model to get a latent map and a low resolution summary, then the decoder to
produce the detail layer, then reassembles the base and detail into real elevations with
Elevation Encoding.

Neighbours and communication

- The Windowed Blending Sampler sends an uncleaned patch and any conditioning and receives a
  cleaned patch of the same size.
- It asks Model Inference to run a named model on a patch.
- It asks Elevation Encoding to turn the models' base and detail outputs into real elevations.
"""
import numpy as np
import itertools

from terrain_diffusion.inference import load_model
from terrain_diffusion import encoding

class ModelPipeline:

  def generate(self, patch: np.ndarray) -> np.ndarray:
    return self.blur(patch)

  def blur(self, grid: np.ndarry) -> np.ndarray:
    ret_grid = np.zeros(grid.shape)
    for (i,j) in np.ndindex(grid.shape):

      top = max(0, i-1)
      bottom = min(grid.shape[0] - 1, i+1)
      left = max(0, j-1)
      right = min(grid.shape[1] - 1, j+1)
      all_indices = set(itertools.product(range(top, bottom + 1), range(left, right+1)))
      valid_indices = [(x,y) for (x,y) in all_indices if (x, y) != (i, j)]

      ret_grid[i][j] = sum(grid[x][y] for (x,y) in valid_indices) / len(valid_indices)
    return ret_grid


  def clean_patch(patch: np.ndarray) -> np.ndarray:
    core, decoder = load_model("core"), load_model("decoder")

    core_output = core.predict(patch)
    decoder_output = decoder.predict(core_output.latent_map)

    #TODO: update functions and output once elevation encoding is complete
    some_output = encoding.some_function(core_output.low_res_grid, decoder_output.full_res_grid)
    return some_output