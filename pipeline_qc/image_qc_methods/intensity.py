import numpy as np

def intensity_stats_single_channel(single_channel_im, cell_mask):
    # Intensity stat function
    # Input(single_channel_im) : Numpy array of a single color channel (2D array)
    # Input(cell_mask): Numpy array of 1s and 0s showing where cells exist in FOV (2D array)
    # Calculates mean, median, min, max, and std of an image and outputs as a dict

    single_channel_im_seg = single_channel_im * cell_mask
    result = dict()
    result.update({'mean': single_channel_im_seg[np.nonzero(single_channel_im_seg)].mean()})
    result.update({'median': np.median(single_channel_im_seg[np.nonzero(single_channel_im_seg)])})
    result.update({'max': single_channel_im_seg[np.nonzero(single_channel_im_seg)].max()})
    result.update({'min': single_channel_im_seg[np.nonzero(single_channel_im_seg)].min()})
    result.update({'std': single_channel_im_seg[np.nonzero(single_channel_im_seg)].std()})
    result.update({'99.5%': np.percentile(single_channel_im_seg[np.nonzero(single_channel_im_seg)], 99.5)})
    result.update({'0.5%': np.percentile(single_channel_im_seg[np.nonzero(single_channel_im_seg)], 0.5)})

    return result
