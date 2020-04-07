import numpy as np


def intensity_stats_single_channel(single_channel_im):
    # Intensity stat function
    # Input: Numpy array of a single color channel (2D array)
    # Calculates mean, median, min, max, and std of an image and outputs as a dict

    result = dict()
    result.update({'mean': single_channel_im.mean()})
    result.update({'median': single_channel_im.median()})
    result.update({'max': single_channel_im.max()})
    result.update({'min': single_channel_im.min()})
    result.update({'std': single_channel_im.std()})
    result.update({'99.5%': np.percentile(single_channel_im, 99.5)})
    result.update({'0.5%': np.percentile(single_channel_im, 0.5)})

    return result
