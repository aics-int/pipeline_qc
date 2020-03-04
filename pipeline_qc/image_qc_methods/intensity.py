from lkaccess import LabKey, contexts
import math
from aicsimageio import AICSImage
import numpy as np
import pandas as pd
from scipy import signal, stats
from skimage import exposure, filters


def intensity_stats_single_channel(single_channel_im):
    # Intensity stat function
    # Calculates mean, min, max, stdev, 0.5% percentile, and 99.5% percentile intensity
    # of a single channel 3D image and outputs as a dict
    # Input-single_channel_im: 3D numpy array of a single color channel (ZYX)

    result = dict()
    result.update({'mean': single_channel_im.mean()})
    result.update({'max': single_channel_im.max()})
    result.update({'min': single_channel_im.min()})
    result.update({'std': single_channel_im.std()})
    result.update({'99.5%': np.percentile(single_channel_im, 99.5)})
    result.update({'0.5%': np.percentile(single_channel_im, 0.5)})

    return result
