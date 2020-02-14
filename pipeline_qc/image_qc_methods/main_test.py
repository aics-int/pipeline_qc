from lkaccess import LabKey, contexts
import math
from aicsimageio import AICSImage
import numpy as np
import pandas as pd
from scipy import signal, stats
from skimage import exposure, filters


def intensity_stats_single_channel(single_channel_im):
    # Intensity stat function
    # Calculates mean, min, max, and std of an image and outputs as a dict

    result = dict()
    result.update({'mean': single_channel_im.mean()})
    result.update({'max': single_channel_im.max()})
    result.update({'min': single_channel_im.min()})
    result.update({'std': single_channel_im.std()})

    return result

def query_from_fms(cellline):
    # Queries FMS (only using cellline right now) for image files that we would QC

    server_context = LabKey(contexts.PROD)

    sql = f'''
     SELECT fov.fovid, fov.sourceimagefileid, well.wellname.name as wellname, plate.barcode,
        instrument.name as instrument, fcl.celllineid.name as cellline, fov.created, file.localfilepath
        FROM microscopy.fov as fov
        INNER JOIN microscopy.well as well on fov.wellid = well.wellid
        INNER JOIN microscopy.plate as plate on well.plateid = plate.plateid
        INNER JOIN microscopy.instrument as instrument on fov.instrumentid = instrument.instrumentid
        INNER JOIN celllines.filecellline as fcl on fov.sourceimagefileid = fcl.fileid
        INNER JOIN fms.file as file on fov.sourceimagefileid = file.fileid
        WHERE fov.objective = 100
        AND fov.qcstatusid.name = 'Passed'
        AND fcl.celllineid.name = '{cellline}'
    '''

    result = server_context.execute_sql('microscopy', sql)
    df = pd.DataFrame(result['rows'])
    return df[['sourceimagefileid', 'created', 'fovid', 'instrument', 'localfilepath', 'wellname', 'barcode',
                    'cellline']]


def split_image_into_channels(im_path, sourceimagefileid):
    # Splits image data into all requisite channels, as 3D images (405, 488, 561, 638, bf are normal channels)
    # Uses the context table in labkey to find the channel number and then splits the aicsimage loaded file accordingly
    # This allows us to oly load an image once and then use it for as many qc steps as we want to later

    server_context = LabKey(contexts.PROD)
    im = AICSImage(im_path)
    np_im = im.data[0, 0, :, :, :, :]
    sql = f'''
      SELECT content.contenttypeid.name, content.channelnumber
        FROM processing.content as content
        WHERE content.fileid = '{sourceimagefileid}'
    '''

    result = server_context.execute_sql('microscopy', sql)
    df = pd.DataFrame(result['rows'])

    split_channels = dict()

    for index, row in df.iterrows():
        for channel in ['405nm', '488nm', '561nm', '638nm', 'brightfield']:
            if row['name'] == 'Raw ' + channel:
                channel_number = row['channelnumber']
                exec('ch' + channel + "= np_im[channel_number, :, :, :]")
                exec("split_channels.update({channel: ch" + channel + "})")
            else:
                pass

    return split_channels


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


def detect_false_clip_cmdr(cmdr, contrast_threshold=(0.2, 0.19)):
    """
    Detects top/bottom clipping in a z-stack. (The method will fail if you have small debris/floating cells on top. )
    :param cmdr: a (z, y, x) cmdr image
    :param contrast_threshold: a tuple of contrast threshold (threshold for finding bottom, threshold for finding top)
    :return:
        real_bottom: an integer of index of bottom-z-stack or None
        real_top: an integer of index of top-z-stack or None
        crop_top: a boolean if crop top is True or False
        crop_bottom: a boolean if crop bottom is True or False
        flag_top: a boolean if the top should be flagged
        flag_bottom: a boolean if the bottom should be flagged
        contrast_99_percentile: contrast profile in z
        z_aggregate: median intensity profile in z
    """

    # Initialize values
    crop_top = True
    crop_bottom = True
    real_top = None
    real_bottom = None
    flag_bottom = False
    flag_top = False

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
        top_peak = bottom_peak = all_peaks[np.where(refined_z==np.max(refined_z))][0]
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

    # For logging purposes only
    # if real_top is None:
    #     print('crop top')
    # if real_bottom is None:
    #     print('crop bottom')

    # Refine crop bottom identification with the slope and fit of the contrast curve
    # Linear fit for first five z-stacks from bottom
    if real_bottom is not None:
        bottom_range = np.linspace(0, real_bottom - 1, real_bottom)
        if len(bottom_range) >= 5:
            # Get linear regression
            slope, y_int, r, p, err = stats.linregress(x=list(range(0, 5)), y=contrast_99_percentile[0:5])
            # Set criteria with slope and r-value to determine if the bottom is cropped
            if slope <= -0.005:
                real_bottom = real_bottom
                crop_bottom = False
            elif (slope <= 0) & (math.fabs(r) > 0.8):
                real_bottom = real_bottom
                crop_bottom = False
        else:
            # The z-stack might not be cropped, but should require validation
            print('flag bottom, too short')
            flag_bottom = True
            crop_bottom = False

    # Refine crop top identification with the slope and fit of the contrast curve
    # Linear fit for first five z-stacks from top
    if real_top is not None:
        top_range = np.linspace(real_top, len(z_aggregate) - 1, len(z_aggregate) - real_top)
        if len(top_range) >= 5:
            # get linear regression
            slope, y_int, r, p, err = stats.linregress(x=list(range(real_top, real_top + 5)),
                                                       y=contrast_99_percentile[real_top:real_top + 5])
            # Set criteria with slope and r-value to determine if the top is cropped
            if slope <= -0.005:
                real_top = real_top
                crop_top = False
            elif (slope <= 0) & (math.fabs(r) > 0.8):
                real_top = real_top
                crop_top = False
        else:
            # The z-stack might not be cropped, but should require validation
            print('flag top, too short')
            flag_top = True

    stat_dict = dict()
    stat_dict.update({'real_bottom': real_bottom})
    stat_dict.update({'real_top': real_top})
    stat_dict.update({'crop_top': crop_top})
    stat_dict.update({'crop_bottom': crop_bottom})
    stat_dict.update({'flag_top': flag_top})
    stat_dict.update({'flag_bottom': flag_bottom})
    stat_dict.update({'contrast_99_percentile': contrast_99_percentile})
    stat_dict.update({'z_aggregate': z_aggregate})
    return stat_dict


def batch_qc(query_df):
    # Runs qc steps and collates all data into a single dataframe for easy sorting and plotting
    # Runs on multiple files, to be used with the query_fms function

    pd.options.mode.chained_assignment = None
    stat_list = list()
    # Step to split 6D image into separate z-stack images
    for index, row in query_df.iterrows():
        channel_dict = split_image_into_channels(row['localfilepath'], row['sourceimagefileid'])
        stat_dict = dict()
        # Iterates over each z-stack image and runs qc_steps, and then adds each stat generated to the stat_dict
        for key, value in channel_dict.items():
            # Runs the intensity metrics on all z-stack images
            intensity_dict = intensity_stats_single_channel(value)
            for intensity_key, intensity_value, in intensity_dict.items():
                stat_dict.update({key + ' ' + intensity_key + '-intensity': intensity_value})
        # Runs the false_clip scripts on bf and 638(cmdr)
        if key == 'brightfield':
            bf_false_clip_dict = detect_false_clip_bf(value)
            for false_clip_key, false_clip_value in bf_false_clip_dict.items():
                stat_dict.update({key + ' ' + false_clip_key: false_clip_value})
        elif key == '638nm':
            bf_false_clip_dict = detect_false_clip_cmdr(value)
            for false_clip_key, false_clip_value in bf_false_clip_dict.items():
                stat_dict.update({key + ' ' + false_clip_key: false_clip_value})
        stat_list.append(stat_dict)

    return pd.concat([query_df, pd.DataFrame(stat_list)], axis=1)


# Testing the code on a small subset of data
cellline = 'AICS-7'
query_df = query_from_fms(cellline)
query_df_small = query_df[0:10]
stat_df = batch_qc(query_df_small)

