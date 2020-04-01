from lkaccess import LabKey, contexts
from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
import numpy as np
import pandas as pd
from scipy import signal, stats
import argparse
import os
from scipy import ndimage
from skimage import exposure, filters, morphology, measure
import math

def intensity_stats_single_channel(single_channel_im):
    # Intensity stat function
    # Calculates mean, min, max, and std of an image and outputs as a dict

    result = dict()
    result.update({'mean': single_channel_im.mean()})
    result.update({'mean': single_channel_im.median()})
    result.update({'max': single_channel_im.max()})
    result.update({'min': single_channel_im.min()})
    result.update({'std': single_channel_im.std()})
    result.update({'99.5%': np.percentile(single_channel_im, 99.5)})
    result.update({'0.5%': np.percentile(single_channel_im, 0.5)})

    return result


def query_fovs_from_fms(workflows = [], cell_lines = [], plates = [], fovids = []):
    # Queries FMS (only using cell line right now) for image files that we would QC
    # Inputs all need to be lists of strings
    if not workflows:
        workflow_query = ''
    else:
        workflow_query = f"AND fov.wellid.plateid.workflow.name IN {str(workflows).replace('[','(').replace(']',')')}"
    if not cell_lines:
        cell_line_query = ""
    else:
        cell_line_query = f"AND fcl.celllineid.name IN {str(cell_lines).replace('[','(').replace(']',')')}"
    if not plates:
        plate_query = ""
    else:
        plate_query = f"AND plate.barcode IN {str(plates).replace('[','(').replace(']',')')}"
    if not fovids:
        fovid_query = ""
    else:
        fovid_query = f"AND fov.fovid IN {str(fovids).replace('[','(').replace(']',')')}"
    server_context = LabKey(contexts.PROD)

    sql = f'''
     SELECT fov.fovid, fov.sourceimagefileid, well.wellname.name as wellname, plate.barcode,
        instrument.name as instrument, fcl.celllineid.name as cellline, fov.fovimagedate, file.localfilepath,
        fov.wellid.plateid.workflow.name as workflow
        FROM microscopy.fov as fov
        INNER JOIN microscopy.well as well on fov.wellid = well.wellid
        INNER JOIN microscopy.plate as plate on well.plateid = plate.plateid
        INNER JOIN microscopy.instrument as instrument on fov.instrumentid = instrument.instrumentid
        INNER JOIN celllines.filecellline as fcl on fov.sourceimagefileid = fcl.fileid
        INNER JOIN fms.file as file on fov.sourceimagefileid = file.fileid
        WHERE fov.objective = 100
        AND fov.qcstatusid.name = 'Passed'
        {workflow_query}
        {cell_line_query}
        {plate_query} 
        {fovid_query}
    '''

    result = server_context.execute_sql('microscopy', sql)
    df = pd.DataFrame(result['rows'])
    if df.empty:
        print("Query from FMS returned no fovids")
        return pd.DataFrame()
    else:
        return df[['sourceimagefileid', 'fovimagedate', 'fovid', 'instrument', 'localfilepath', 'wellname', 'barcode',
                    'cellline', 'workflow']]


def query_fovs_from_filesystem(plates, workflows = ['PIPELINE_4_4', 'PIPELINE_4_5', 'PIPELINE_4_6', 'PIPELINE_4_7', 'PIPELINE_5.2', 'PIPELINE_6', 'PIPELINE_7', 'RnD_Sandbox']):
    # Querying the filesystem for plates, and creating a list of all filepaths needing to be processed
    #  plate is a string, workflows is a list of strings
    #

    prod_dir = '/allen/aics/microscopy/'
    pipeline_dirs = ['PIPELINE_4_4', 'PIPELINE_4_5', 'PIPELINE_4_6', 'PIPELINE_4_7', 'PIPELINE_5.2', 'PIPELINE_6', 'PIPELINE_7']
    data_dirs = ['RnD_Sandbox']
    paths = list()
    for dir in pipeline_dirs:
        if dir not in workflows:
            pass
        else:
            for subdir in os.listdir(prod_dir + 'PRODUCTION/' + dir):
                if subdir in plates:
                    paths.append({dir:prod_dir + 'PRODUCTION/' + dir + '/' + subdir})
                else:
                    pass

    for rnd_dir in data_dirs:
        if dir not in workflows:
            pass
        else:
            for rnd_subdir in os.listdir(prod_dir + 'Data/' + rnd_dir):
                if rnd_subdir in plates:
                    paths.append({'RnD':prod_dir + 'Data/' + rnd_dir + '/' + rnd_subdir})

    supported_folders = ['100X_zstack', '100XB_zstack']
    image_metadata_list = list()
    for row in paths:
        for workflow, path in row.items():
            for instrument in os.listdir(path):
                for folder in os.listdir(path + '/' + instrument):
                    if folder in supported_folders:
                        image_dir = path + '/' + instrument + '/' + folder
                        for image in os.listdir(image_dir):
                            if image.endswith('czi'):
                                image_metadata_dict = dict()
                                image_metadata_dict.update({'workflow': workflow})
                                image_metadata_dict.update({'barcode': path[-10:]})
                                image_metadata_dict.update({'instrument': instrument})
                                image_metadata_dict.update({'localfilepath': image_dir + '/' + image})
                                image_metadata_list.append(image_metadata_dict)
                            else:
                                pass

    return pd.DataFrame(image_metadata_list)


def query_fovs(workflows=[], cell_lines=[], plates=[], fovids=[], only_from_fms=True):
    # Script that can query multiple parameters and join those tables into one query dataframe
    # workflows, cell_lines, plates, and fovs are all lists of strings
    # options: only_from_fms means you can only query fms. If false, will call the filesystem query as well
    df = query_fovs_from_fms(workflows, cell_lines, plates, fovids)
    if only_from_fms == False:
        df_2 = query_fovs_from_filesystem(plates)
        df = pd.concat([df, df_2], axis=0, ignore_index=True)

    return df


def split_image_into_channels(im_path, source_image_file_id):
    # Splits image data into all requisite channels, as 3D images (405, 488, 561, 638, bf are normal channels)
    # Uses the context table in labkey to find the channel number and then splits the aicsimage loaded file accordingly
    # This allows us to oly load an image once and then use it for as many qc steps as we want to later

    server_context = LabKey(contexts.PROD)
    im = AICSImage(im_path)
    np_im = im.data[0, 0, :, :, :, :]
    if source_image_file_id == 'nan':
        df = pd.DataFrame()
    else:
        sql = f'''
          SELECT content.contenttypeid.name, content.channelnumber
            FROM processing.content as content
            WHERE content.fileid = '{source_image_file_id}'
        '''
        result = server_context.execute_sql('microscopy', sql)
        df = pd.DataFrame(result['rows'])

    if df.empty:
        channels = im.get_channel_names()
        channel_info_list = list()
        for channel in channels:
            channel_info_dict = dict()
            if channel in ['Bright_2']:
                channel_info_dict.update({'name': 'Raw brightfield'})
                channel_info_dict.update({'channelnumber':channels.index(channel)})
            elif channel in ['EGFP']:
                channel_info_dict.update({'name': 'Raw 488nm'})
                channel_info_dict.update({'channelnumber':channels.index(channel)})
            elif channel in ['CMDRP']:
                channel_info_dict.update({'name': 'Raw 638nm'})
                channel_info_dict.update({'channelnumber': channels.index(channel)})
            elif channel in ['H3342']:
                channel_info_dict.update({'name': 'Raw 405nm'})
                channel_info_dict.update({'channelnumber':channels.index(channel)})
            elif channel in ['TaRFP']:
                channel_info_dict.update({'name': 'Raw 561nm'})
                channel_info_dict.update({'channelnumber':channels.index(channel)})
            channel_info_list.append(channel_info_dict)
        df = pd.DataFrame(channel_info_list)

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


def segment_colony_area(bf, gaussian_thresh):
    """
    From a 2D bright field image (preferably the center slice), determine foreground vs background to segment
    colony area coverage
    :param bf: a 2D bright field image
    :param gaussian_thresh: a threshold cutoff for gaussian to separate foreground from background
    :return: a 2D segmented image
    """
    p2, p98 = np.percentile(bf, (2, 98))
    rescale = exposure.rescale_intensity(bf, in_range=(p2, p98))
    dist_trans = filters.sobel(rescale)
    gaussian_2 = filters.gaussian(dist_trans, sigma=15)

    mask = np.zeros(bf.shape, dtype=bool)
    mask[gaussian_2 <= gaussian_thresh] = True

    mask_erode = morphology.erosion(mask, selem=morphology.disk(3))
    remove_small = filter_small_objects(mask_erode, 750)
    dilate = morphology.dilation(remove_small, selem=morphology.disk(10))

    new_mask = np.ones(bf.shape, dtype=bool)
    new_mask[dilate == 1] = False
    return new_mask


def filter_small_objects(bw_img, area):
    """
    From a segmented image, filter out segmented objects smaller than a certain area threshold
    :param bw_img: a 2D segmented image
    :param area: an integer of object area threshold (objects with size smaller than that will be dropped)
    :return: a 2D segmented, binary image with small objects dropped
    """
    label_objects, nb_labels = ndimage.label(bw_img)
    sizes = np.bincount(label_objects.ravel())
    max_area = max(sizes)
    # Selecting objects above a certain size threshold
    # size_mask = (sizes > area) & (sizes < max_area)
    size_mask = (sizes > area)
    size_mask[0] = 0
    filtered = label_objects.copy()
    filtered_image = size_mask[filtered]

    int_img = np.zeros(filtered_image.shape)
    int_img[filtered_image == True] = 1
    int_img = int_img.astype(int)
    return int_img


def detect_edge_position(bf_z, segment_gauss_thresh=0.045, area_cover_thresh=0.9):
    """
    From a 3D bright field image, determine if the z-stack is in an edge position
    :param bf_z: a 3D bright field image
    :param segment_gauss_thresh: a float to set gaussian threshold for 2D segmentation of colony area
    :param area_cover_thresh: a float to represent the area cutoff of colony coverage to be considered as an edge
    :return: a boolean indicating the image is an edge position (True) or not (False)
    """
    edge = True
    z_center = find_center_z_plane(bf_z)
    bf = bf_z[z_center, :, :]
    segment_bf = segment_colony_area(bf, segment_gauss_thresh)

    if (np.sum(segment_bf)) / (bf.shape[0] * bf.shape[1]) > area_cover_thresh:
        edge = False

    return {'edge fov?': edge}


def find_center_z_plane(image):
    mip_yz = np.amax(image, axis=2)
    mip_gau = filters.gaussian(mip_yz, sigma=2)
    edge_slice = filters.sobel(mip_gau)
    contours = measure.find_contours(edge_slice, 0.005)
    new_edge = np.zeros(edge_slice.shape)
    for n, contour in enumerate(contours):
        new_edge[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1

    # Fill empty spaces of contour to identify as 1 object
    new_edge_filled = ndimage.morphology.binary_fill_holes(new_edge)

    # Identify center of z stack by finding the center of mass of 'x' pattern
    z = []
    for i in range(100, mip_yz.shape[1] + 1, 100):
        edge_slab = new_edge_filled[:, i - 100:i]
        # print (i-100, i)
        z_center, x_center = ndimage.measurements.center_of_mass(edge_slab)
        z.append(z_center)

    z = [z_center for z_center in z if ~np.isnan(z_center)]
    z_center = int(round(np.median(z)))
    return z_center


def generate_images(image):
    center_plane = find_center_z_plane(image)
    # panels: top, bottom, center
    top = image[-1, :, :]
    bottom = image[0, :, :]
    center = image[center_plane, :, :]
    # panels: mip_xy, mip_xz, mip_yz
    mip_xy = np.amax(image, axis=0)
    mip_xz = np.amax(image, axis=1)
    mip_yz = np.amax(image, axis=2)

    return top, bottom, center, mip_xy, mip_xz, mip_yz


def generate_qc_images(single_channel_im, output_path, fov_id, channel_name):

    # Generate diretories that are needed for saving files
    directories = ['fuse', 'mip_xy', 'mip_xz', 'mip_yz', 'top', 'bottom', 'center']
    for directory in directories:
        try:
            os.mkdir(os.path.join(output_path, directory))
        except:
            pass

    # Generates 6 images
    top, bottom, center, mip_xy, mip_xz, mip_yz = generate_images(single_channel_im)
    img_height = top.shape[0]
    img_width = top.shape[1]
    z_height = mip_xz.shape[0]

    settings = (np.min(mip_xy), np.max(mip_xy))

    # set display for mip_images
    rescaled_xy = exposure.rescale_intensity(mip_xy, in_range=settings)
    rescaled_xz = exposure.rescale_intensity(mip_xz, in_range=settings)
    rescaled_yz = exposure.rescale_intensity(mip_yz, in_range=settings)
    # Create fuse image combining 3 mips
    fuse = np.zeros(shape = ((img_height + z_height), (img_width + z_height)))
    fuse[0:z_height, 0:img_width] = rescaled_xz
    fuse[z_height:z_height+img_height, 0:img_width] = rescaled_xy
    fuse[z_height:z_height+img_height, img_width:img_width+z_height] = np.rot90(rescaled_yz)

    # Create qc image combining fuse and center_TL
    qc = np.zeros(((img_height + z_height), (2*img_width + z_height)))
    qc[:, 0:img_width+z_height] = fuse
    qc[z_height:img_height+z_height, img_width+z_height:2*img_width+z_height] = center

    # Save and reformat images in a dictionary
    new_images_dict = {'top': np.reshape(top, (1, img_height, img_width)),
                       'bottom': np.reshape(bottom, (1, img_height, img_width)),
                       'center': np.reshape(center, (1, img_height, img_width)),
                       'mip_xy': np.reshape(rescaled_xy, (1, img_height, img_width)),
                       'mip_xz': np.reshape(rescaled_xz, (1, z_height, img_width)),
                       'mip_yz': np.reshape(rescaled_yz, (1, z_height, img_height)),
                       'fuse': np.reshape(fuse, (1, fuse.shape[0], fuse.shape[1]))}

    # Save images in output directory, with specific structure
    file_name = str(fov_id) + '_' + channel_name

    for key, image in new_images_dict.items():
        writer = ome_tiff_writer.OmeTiffWriter(os.path.join(output_path, key,
                                                                   file_name + '-' + key + '.tif'),
                                                      overwrite_file=True)
        writer.save(image.astype(np.uint16))


def batch_qc(output_dir, workflows=[], cell_lines=[], plates=[], fovids=[], only_from_fms=True, image_gen=False):
    # Runs qc steps and collates all data into a single dataframe for easy sorting and plotting
    # Runs on multiple files, to be used with the query_fms function
    pd.options.mode.chained_assignment = None

    # Run the query fn on specified cell line
    query_df = query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids, only_from_fms=only_from_fms)

    stat_list = list()

    # Iterates through all fovs identifies by query dataframe
    for index, row in query_df.iterrows():

        # Splits 6D image into single channel images for qc algorithm processing
        channel_dict = split_image_into_channels(row['localfilepath'], str(row['sourceimagefileid']))

        # Initializes a dictionary where all stats for an fov are saved
        stat_dict = dict()

        # Iterates over each z-stack image and runs qc_algorithms, and then adds each stat generated to the stat_dict
        for channel_name, channel_array in channel_dict.items():

            # Runs the intensity metrics on all z-stack images. Put here since run on all channels
            intensity_dict = intensity_stats_single_channel(channel_array)
            for intensity_key, intensity_value, in intensity_dict.items():
                stat_dict.update({channel_name + ' ' + intensity_key + '-intensity': intensity_value})

            # Runs all metrics to be run on brightfield (edge detection, false clip bf) and makes bf qc_images
            if channel_name == 'brightfield':
                bf_edge_detect = detect_edge_position(channel_array)
                for edge_key, edge_value in bf_edge_detect.items():
                    stat_dict.update({channel_name + ' ' + edge_key: edge_value})
                bf_false_clip_dict = detect_false_clip_bf(channel_array)
                for false_clip_key, false_clip_value in bf_false_clip_dict.items():
                    stat_dict.update({channel_name + ' ' + false_clip_key + '-false clip': false_clip_value})

                # PUT QC_IMAGES FOR BF HERE
                if image_gen:
                    generate_qc_images(channel_array, output_dir, row['fovid'], channel_name)

            # Runs all metrics to be run on 638 (false clip 638) and makes 638 qc_images
            elif channel_name == '638nm':
                bf_false_clip_dict = detect_false_clip_cmdr(channel_array)
                for false_clip_key, false_clip_value in bf_false_clip_dict.items():
                    stat_dict.update({channel_name + ' ' + false_clip_key + '-false clip': false_clip_value})

                # PUT QC_IMAGES FOR 638 HERE
                if image_gen:
                    generate_qc_images(channel_array, output_dir, row['fovid'], channel_name)

        # Adds stat_dict to a list of dictionaries, which corresponds to the query_df.
        stat_list.append(stat_dict)
        print(f"Added {str(row['fovid'])} to stat dictionary")

    # Joins query_df to stat_list, and then writes out a csv of all the data to an output folder
    result = pd.concat([query_df, pd.DataFrame(stat_list)], axis=1)
    result.to_csv(output_dir + '/fov_qc_metrics.csv')

    return result


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='directory which all files should be saved', required=True)
parser.add_argument('--workflows', type=str, help="Array of workflows to run qc on. E.g. ['PIPELINE4', 'PIPELINE4.4'] ",default = '[]', required=False)
parser.add_argument('--cell_lines', type=str, help="Array of Cell-lines to run qc on. E.g. ['AICS-11', 'AICS-7'] ", required=False)
parser.add_argument('--plates', type=str, help="Array of plates to run qc on. E.g. ['3500003813', '3500003642'] ", required=False)
parser.add_argument('--fovids', type=str, help="Array of fovids to run qc on. E.g. ['123', '6'] ", required=False)
parser.add_argument('--only_fms', type=str, help="Boolean to say whether to only run query on data in fms (default is True)", default=True, required=False)

args = parser.parse_args()

batch_qc(args.output_dir, args.workflows, args.cell_lines)

