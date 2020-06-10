import math
import matplotlib.pyplot as plt
import os
from aicsimageio import AICSImage
import numpy as np
import pandas as pd
from scipy import signal, stats
from skimage import exposure, filters


def plot_profile_clip (detect_bottom, detect_top, profiles, output_path, true_bottom=None, true_top=None,):
    """
    Plots intesity/contrast profile along with z-stack-cut-off labels
    :param detect_bottom: an integer indicating bottom that fov_clip method detects
    :param detect_top: an integer indicating top that fov_clip method detects
    :param profiles: a list of profiles to plot (e.g. intensity, contrast)
    :param output_path: output path to save the figure
    :param true_bottom: an integer manual annotation of the true bottom of z-stack
    :param true_top: an integer manual annotation of the true top of z-stack
    :return: a plot
    """
    plt.figure()
    for profile in profiles:
        plt.scatter(y=profile, x=np.linspace(0, len(profile) - 1, len(profile)))
    if true_bottom is not None:
        plt.axvline(x=true_bottom, color='k', linestyle='-')
    if true_top is not None:
        plt.axvline(x=true_top, color='k', linestyle='-')
    if detect_bottom is not None:
        plt.axvline(x=detect_bottom, color='r', linestyle='--')
    if detect_top is not None:
        plt.axvline(x=detect_top, color='r', linestyle='--')
    plt.ylim((0, 1.1))
    plt.savefig(output_path)


def detect_false_clip_bf(bf_z, threshold=(0.01, 0.073)):
    """
    NOT READ TO USE -- This method FAILS when a small portion of FOV contains floating/popping/mitotic cell on top
    Run a detect_floating/popping/mitotic cell filter before use.

    Detect top and bottom of z-stack with bright field using laplace transforms and filters
    :param bf_z: a z-stack image in bright field
    :param threshold: a tuple to set threshold for peak prominence and laplace range cutoff
    :return:
        detect_bottom: an integer of index of bottom-z-stack or None
        detect_top: an integer of index of top-z-stack or None
        crop_top: a boolean if crop top is True or False
        crop_bottom: a boolean if crop bottom is True or False
        flag_top: a boolean if the top should be flagged
        flag_bottom: a boolean if the bottom should be flagged
        laplace_range: a list of range of laplace transform intensity (between 99.5th percentile and 0.5th percentile)
    """
    # Initialize values:
    crop_top = True
    crop_bottom = True
    flag_bottom = False
    flag_top = False
    detect_top = None
    detect_bottom = None

    laplace_range = []
    for z in range(0, bf_z.shape[0]):
        bf = bf_z[z, :, :]
        laplace = filters.laplace(bf, ksize=3)
        # diff = np.max(laplace) - np.min(laplace)
        diff = (np.percentile(laplace, 99.5) - np.percentile(laplace, 0.5))
        laplace_range.append(diff)

    if np.max(laplace_range) < 0.08:
        # peak = np.where (laplace_range == np.max(laplace_range))[0][0]
        all_peaks = signal.argrelmax(np.asarray(laplace_range))
        # Check if it is a 'good peak'
        peak_prom = signal.peak_prominences(laplace_range, all_peaks[0])[0]
        if peak_prom[np.where(peak_prom == np.max(peak_prom))][0] > threshold[0]:
            peak = all_peaks[0][np.where(peak_prom == np.max(peak_prom))][0]
        else:
            peak = np.where(laplace_range == np.max(laplace_range))[0][0]
    else:
        peak = np.where(laplace_range == np.max(laplace_range))[0][0]

    if peak is not None:
        for z in range(peak, len(laplace_range)):
            if laplace_range[z] <= (np.max(laplace_range) - 2.5 * np.std(laplace_range)):
                detect_top = z
                crop_top = flag_top = False
                break
        for z in reversed(range(0, peak)):
            if laplace_range[z] <= (np.max(laplace_range) - 2.5 * np.std(laplace_range)):
                detect_bottom = z
                crop_bottom = flag_bottom = False
                break

        if detect_top is None:
            for z in range(peak + 1, len(laplace_range)):
                if laplace_range[z] < threshold[1]:
                    detect_top = z
                    crop_top = flag_top = False
                    break
        if detect_bottom is None:
            for z in reversed(range(0, peak)):
                if laplace_range[z] < threshold[1]:
                    detect_bottom = z
                    crop_bottom = flag_bottom = False
                    break

    stat_dict = dict()
    stat_dict.update({'detect_bottom': detect_bottom})
    stat_dict.update({'detect_top': detect_top})
    stat_dict.update({'crop_top': crop_top})
    stat_dict.update({'crop_bottom': crop_bottom})
    stat_dict.update({'flag_top': flag_top})
    stat_dict.update({'flag_bottom': flag_bottom})
    stat_dict.update({'laplace_range': laplace_range})
    return stat_dict


def initialize_peak(z_aggregate):
    """
    Initializes intensity peaks from median intensity profile
    Parameters
    ----------
    z_aggregate
        contrast profile in z

    Returns
    -------
    top_peak
        identified index of top z-slice
    bottom_peak
        identified index of bottom z-slice
    """
    try:
        all_peaks = signal.argrelmax(np.array(z_aggregate))[0]
    except:
        all_peaks = []

    # Initialize top and bottom peaks from all_peaks
    if len(all_peaks) == 2:
        bottom_peak = all_peaks[0]
        top_peak = all_peaks[1]
    elif len(all_peaks) > 2:
        # Get the peak with highest intensity and initialize top/bottom with the same peak
        indexed = stats.rankdata(all_peaks, method='ordinal')
        refined_z = []
        for index in all_peaks:
            refined_z.append(z_aggregate[index])
        top_peak = bottom_peak = all_peaks[np.where(refined_z == np.max(refined_z))][0]
        peak_info = 'more than 2 peaks'
        print('more than 2 peaks')
    elif len(all_peaks) == 1:
        # Set bottom peak and top peak as the same peak
        bottom_peak = all_peaks[0]
        top_peak = all_peaks[0]
    else:
        # Report cannot find peak
        bottom_peak = 0
        top_peak = 0
        print('cannot find peak')
        peak_info = 'cannot find peak'

    return top_peak, bottom_peak


def refine_bottom(real_bottom, contrast_99_percentile, ref_slope=-0.005, ref_r=0.8):
    """
    Refines the index of detected bottom z-slice
    Parameters
    ----------
    real_bottom
        Initial detected index of bottom z-slice
    contrast_99_percentile
        a list of contrast profile in z
    ref_slope
        a float of reference slope to set crop equals true/false
    ref_r
        a float of reference r2 value threshold to set crop equals true/false
    Returns
    -------
    real_bottom
        an integer of index of bottom z-stack or None
    crop_bottom
        a boolean if crop bottom is True or False
    flag_bottom
        a boolean if the bottom should be flagged

    """
    bottom_range = np.linspace(0, real_bottom - 1, real_bottom)
    real_bottom = real_bottom
    crop_bottom = False
    flag_bottom = False
    if len(bottom_range) >= 5:
        # Get linear regression
        slope, y_int, r, p, err = stats.linregress(x=list(range(0, 5)), y=contrast_99_percentile[0:5])
        # Set criteria with slope and r-value to determine if the bottom is cropped
        if slope <= ref_slope:
            real_bottom = real_bottom
            crop_bottom = False
        elif (slope <= 0) & (math.fabs(r) > ref_r):
            real_bottom = real_bottom
            crop_bottom = False
    else:
        # The z-stack might not be cropped, but should require validation
        print('flag bottom, too short')
        flag_bottom = True
        crop_bottom = False
    return real_bottom, crop_bottom, flag_bottom


def refine_top(real_top, contrast_99_percentile, ref_slope=-0.015, ref_r=0.8):
    """
    Refines the index of detected top z-slice
    Parameters
    ----------
    real_top
        Initial detected index of top z-slice
    contrast_99_percentile
        a list of contrast profile in z
    ref_slope
        a float of reference slope to set crop equals true/false
    ref_r
        a float of reference r2 value threshold to set crop equals true/false
    Returns
    -------
    real_top
        an integer of index of top-z-stack or None
    crop_top
        a boolean if crop top is True or False
    flag_top
        a boolean if the top should be flagged
    """
    top_range = np.linspace(real_top, len(contrast_99_percentile) - 1, len(contrast_99_percentile) - real_top)
    real_top = real_top
    crop_top = False
    flag_top = False
    if len(top_range) >= 5:
        # get linear regression
        slope, y_int, r, p, err = stats.linregress(x=list(range(real_top, real_top + 5)),
                                                   y=contrast_99_percentile[real_top:real_top + 5])
        # Set criteria with slope and r-value to determine if the top is cropped
        if slope <= ref_slope:
            real_top = real_top
            crop_top = False
        elif (slope <= 0) & (math.fabs(r) > ref_r):
            real_top = real_top
            crop_top = False
    else:
        # The z-stack might not be cropped, but should require validation
        print('flag top, too short')
        flag_top = True
    return real_top, crop_top, flag_top


def check_top(contrast_99_percentile, ref_slope_range=(-0.01, 0), ref_r=0.8):
    """
    Checks the top 5 z-slices of a z-stack to finally determine if it is cropped or not
    Parameters
    ----------
    contrast_99_percentile
        a list of contrast profile in z
    ref_slope_range
        a tuple of acceptable slope range to set crop equals true/false
    ref_r
        a float of reference r2 value threshold to set crop equals true/false
    Returns
    -------
    crop_top
        a boolean if crop top is True or False
    """
    crop_top = True
    top_range = np.linspace(len(contrast_99_percentile)-5, len(contrast_99_percentile)-1, 5)
    # for index in top_range:
    #     #Calysta, I added this in because there were erors popping up when there were negative top_range values
    #     if index < 0:
    #         real_top = []
    #         print('here')
    #         crop_top = True
    #         break
    #     else:

    slope, y_int, r, p, err = stats.linregress(x=top_range,
                                               y=np.take(contrast_99_percentile,
                                                         indices=top_range.astype(int)))
    if (ref_slope_range[0] < slope <= ref_slope_range[1]) & (math.fabs(r) > ref_r):
        crop_top = False
    return crop_top


def detect_false_clip_cmdr(cmdr, contrast_threshold=(0.2, 0.19), ref_r=0.8):
    """
    Detects top/bottom clipping in a z-stack. (The method will fail if you have small debris/floating cells on top. )
    Parameters
    ----------
    cmdr
        a (z, y, x) cmdr image
    contrast_threshold
        a tuple of contrast threshold (threshold for finding bottom, threshold for finding top)
    ref_r
        a float of reference r2 value threshold to set crop equals true/false
    Returns
    -------
    real_bottom
        an integer of index of bottom-z-stack or None
    real_top
        an integer of index of top-z-stack or None
    crop_top
        a boolean if crop top is True or False
    crop_bottom
        a boolean if crop bottom is True or False
    flag_top
        a boolean if the top should be flagged
    flag_bottom
        a boolean if the bottom should be flagged
    contrast_99_percentile
        contrast profile in z
    z_aggregate
        median intensity profile in z
    """

    # Initialize values
    crop_top = True
    crop_bottom = True
    real_top = None
    real_bottom = None
    flag_bottom = False
    flag_top = False
    peak_info = ''

    # Rescale image
    cmdr = exposure.rescale_intensity(cmdr, in_range='image')

    # Generate contrast and median intensity curve along z
    z_aggregate = []
    contrast_99_percentile = []
    for z in range(0, cmdr.shape[0]):
        z_aggregate.append(np.median(cmdr[z, :, :]) / np.max(cmdr[:, :, :]))
        contrast_99_percentile.append(
            (np.percentile(cmdr[z, :, :], 99.9) - np.min(cmdr[z, :, :])) / np.percentile(cmdr[:, :, :], 99.9))

    # Find intensity peaks in bottom and top of z-stack. A perfect z-stack should return 2 peaks,
    # the peak at lower index is the bottom of z-stack in focus, and the peak at higher index is the top of z-stack
    # in focus

    top_peak, bottom_peak = initialize_peak(z_aggregate)

    # From bottom and top peak, find the z plane at contrast threshold to the bottom and top of z-stack
    bottom_range = contrast_99_percentile[0: bottom_peak]

    # Iterate from bottom peak to the bottom of z-stack, find the closest z-stack that reaches lower contrast threshold
    for z in reversed(range(0, bottom_peak)):
        contrast_value = contrast_99_percentile[z]
        if contrast_value <= contrast_threshold[0]:
            real_bottom = z
            crop_bottom = False
            break
        else:
            real_bottom = np.where(bottom_range == np.min(bottom_range))[-1][-1]

    # Iterate from top peak to the top of z-stack, find the closest z-stack that reaches upper contrast threshold
    top_range = contrast_99_percentile[top_peak:len(z_aggregate)]
    for z in range(top_peak, len(z_aggregate)):
        contrast_value = contrast_99_percentile[z]
        if contrast_value <= contrast_threshold[1]:
            real_top = z
            crop_top = False
            break

    # # For logging purposes only
    # if real_top is None:
    #     print('crop top')
    # if real_bottom is None:
    #     print('crop bottom')

    # Refine crop bottom identification with the slope and fit of the contrast curve
    # Linear fit for first five z-stacks from bottom
    if real_bottom is not None:
        real_bottom, crop_bottom, flag_bottom = refine_bottom(real_bottom, contrast_99_percentile,
                                                              ref_slope=-0.005, ref_r=ref_r)

    # Refine crop top identification with the slope and fit of the contrast curve
    # Linear fit for first five z-stacks from top
    if real_top is not None:
        real_top, crop_top, flag_top = refine_top(real_top, contrast_99_percentile,
                                                  ref_slope=-0.015, ref_r=ref_r)

    # last check for top. Could be more tolerant with crop top cut-off, but cannot predict the top slice
    if real_top is None:
        crop_top = check_top(contrast_99_percentile, ref_slope_range=(-0.01, 0), ref_r=ref_r)

    stat_dict = dict()
    stat_dict.update({'real_bottom': real_bottom})
    stat_dict.update({'real_top': real_top})
    stat_dict.update({'crop_top': crop_top})
    stat_dict.update({'crop_bottom': crop_bottom})
    stat_dict.update({'flag_top': flag_top})
    stat_dict.update({'flag_bottom': flag_bottom})
    stat_dict.update({'contrast_99_percentile': contrast_99_percentile})
    stat_dict.update({'z_aggregate': z_aggregate})
    stat_dict.update({'peak_info': peak_info})
    return stat_dict

#=======================================================================================================================
# Validating false clip methods

# csv = r'C:\Users\calystay\Downloads\aics94_fovqc.csv'
# df = pd.read_csv(csv)
# df = df[['fovid', 'sourceimagefileid', 'fovimagedate', 'localfilepath', 'barcode', '638nm crop_top-false clip',
#         '638nm crop_bottom-false clip']]
# df_check = df[df['638nm crop_top-false clip'] == True]

# df_out = pd.DataFrame()
#
# df_val = pd.read_csv(r'\\allen\aics\microscopy\Calysta\test\crop_top_bottom\false_clip.csv', header=None)
# df_val = df_val.rename(columns={0: 'file_name'})
#
# for index, row in df_val.iterrows():
#
#     plate = row['file_name'].split('_')[0]
#     plate_path = os.path.join(r'\\allen\aics\microscopy\PRODUCTION\PIPELINE_4_4', plate)
#
#     for root, directories, files in os.walk(plate_path):
#         for file in files:
#             if file == (row['file_name'].split('.ome')[0] + '.czi'):
#                 path = root
#                 img = file
#
#     image_data = AICSImage(os.path.join(path, img))
#     new_row = {'localfilepath': os.path.join(path, img)}
#     # new_row = {'file_name': row['localfilepath']}
#     # image_data = AICSImage('\\' + row['localfilepath'].replace('/', '\\'))
#     image = image_data.data
#
#     channel_names = np.asarray(image_data.get_channel_names())
#     cmdr_index = np.where(channel_names == 'CMDRP')[0][0]
#
#     cmdr = image[0, 0, cmdr_index, :, :, :]
#     for upper_contrast in np.linspace(0.18, 0.25, 30):
#         # detect_bottom, detect_top, crop_top, crop_bottom, flag_top, flag_bottom, contrast, intensity =
#
#         stat_dict = detect_false_clip_cmdr(cmdr, contrast_threshold=(0.2, upper_contrast))
#
#         # Plot to compare
#         # plt_save_path = os.path.join(r'\\allen\aics\microscopy\Calysta\test\crop_top_bottom\test_0608_1',
#         #                              str(row['fovid']) + '_plot.png')
#         # plot_profile_clip(detect_bottom=None, detect_top=None,
#         #                   profiles=[stat_dict['contrast_99_percentile'], stat_dict['z_aggregate']],
#         #                   output_path=plt_save_path)
#
#         new_row['crop_bottom'] = stat_dict['crop_bottom']
#         new_row['crop_top' + '_' + str(upper_contrast)] = stat_dict['crop_top']
#
#         # row = {  # 'full_path': os.path.join(path, img),
#         #     #'detect_bottom': stat_dict['detect_bottom]'],
#         #     #'detect_top': stat_dict['detect_top'],
#         #     'crop_bottom' + '_' + str(upper_contrast): stat_dict['crop_bottom'],
#         #     'crop_top' + '_' + str(upper_contrast): stat_dict['crop_top'],
#         #     'flag_bottom' + '_' + str(upper_contrast): stat_dict['flag_bottom'],
#         #     'flag_top' + '_' + str(upper_contrast): stat_dict['flag_top'],
#         # }
#
#     df_out = df_out.append(new_row, ignore_index=True)
#     print(new_row)
# df_out.to_csv(r'\\allen\aics\microscopy\Calysta\test\crop_top_bottom\false_clip_out_0609_1.csv')