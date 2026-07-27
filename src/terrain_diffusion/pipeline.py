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
