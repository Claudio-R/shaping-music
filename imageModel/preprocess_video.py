from __future__ import division

import numpy as np
import vggish_params
import vgg_params
import resampy

def preprocess_video(data, sample_rate):
    """Converts video into an array of examples for VGG.

    Args:
        data: np.array of images of dimension one (B/N) or three (RGB).
        Each pixel in the image is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
        sample_rate: Sample rate of data.

    Returns:
        3D np.array of shape [num_examples, num_pixels, num_pixels] which represents
        a sequence of examples, each of which contains a patch of images
        where num_examples is len(data) / vgg_params.EXAMPLE_WINDOW_SECONDS * vgg_params.SAMPLE_RATE
        and num_pixels is equal to vgg_params.EXAMPLE_WINDOW_SECONDS * vgg_params.SAMPLE_RATE
    """

      # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vgg_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    step = vgg_params.EXAMPLE_WINDOW_SECONDS * vgg_params.SAMPLE_RATE
    frame_examples = data[:, :, 0::step]
    return frame_examples