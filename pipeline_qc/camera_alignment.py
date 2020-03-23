from collections import OrderedDict
import os
import math
import matplotlib.pyplot as plt
import numpy as np

from aicsimageio import AICSImage
from skimage import transform as tf, exposure as exp, filters, measure, morphology, feature, io, segmentation, metrics
import pandas as pd
from scipy.spatial import distance
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# read beads image
beads = AICSImage(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD1\3500003331_100X_20190813_psf.czi')
bead_rescale_488_lower_thresh = 99.4
bead_rescale_638_lower_thresh = 99

channels = beads.get_channel_names()

# get gfp and cmdr channel from beads image
beads_gfp = beads.data[0, channels.index('EGFP'), :, :, :]
beads_cmdr = beads.data[0, channels.index('CMDRP'), :, :, :]

# get center slice
center_z = 0
max_intensity = 0
for z in range(0, beads_gfp.shape[0]):
    sum_intensity = np.sum(beads_gfp[z, :, :])
    if sum_intensity >= max_intensity:
        center_z = z
        max_intensity = sum_intensity

#=======================================================================================================================
# Pre-process images
# rescale intensity
ref = beads_gfp[center_z, :, :]
mov = beads_cmdr[center_z, :, :]

ref_rescaled = exp.rescale_intensity(ref,
                                     out_range=np.uint8,
                                     in_range=(np.percentile(beads_gfp[center_z, :, :], bead_rescale_488_lower_thresh),
                                               np.max(beads_gfp[center_z, :, :]))
                                     )
mov_rescaled = exp.rescale_intensity(mov,
                                     out_range=np.uint8,
                                     in_range=(np.percentile(beads_cmdr[center_z, :, :], bead_rescale_638_lower_thresh),
                                               np.max(beads_cmdr[center_z, :, :]))
                                     )

# smooth image
ref_smooth = filters.gaussian(ref_rescaled, sigma=1, preserve_range=True)
mov_smooth = filters.gaussian(mov_rescaled, sigma=1, preserve_range=True)


#=======================================================================================================================
# Process beads (segment, filter assign)

filtered, seg_mov = filter_big_beads(mov_smooth)
filtered, seg_ref = filter_big_beads(ref_smooth)

# initialize intensity-based peaks
ref_peaks = feature.peak_local_max(ref_smooth*seg_ref, min_distance=5)
ref_peak_dict, ref_labelled_seg = initialize_peaks(seg=seg_ref, peak_list=ref_peaks, show_img=False, img_shape=ref.shape)
mov_peaks = feature.peak_local_max(mov_smooth*seg_mov, min_distance=5)
mov_peak_dict, mov_labelled_seg = initialize_peaks(seg=seg_mov, peak_list=mov_peaks, show_img=False, img_shape=mov.shape)

# remove_close_peaks
ref_close_peaks = remove_close_peaks(ref_peak_dict, dist_threshold=20, show_img=True, img_shape=ref.shape)
mov_close_peaks = remove_close_peaks(mov_peak_dict, dist_threshold=20, show_img=True, img_shape=mov.shape)

# remove peaks/beads that are too big
ref_remove_overlap_peaks = remove_overlapping_beads(label_seg=ref_labelled_seg, peak_dict=ref_close_peaks, show_img=True)
mov_remove_overlap_peaks = remove_overlapping_beads(label_seg=mov_labelled_seg, peak_dict=mov_close_peaks, show_img=True)

# match peaks
updated_ref_peak_dict, updated_mov_peak_dict = match_peaks(ref_peak_dict=ref_remove_overlap_peaks,
                                                           mov_peak_dict=mov_remove_overlap_peaks,
                                                           dist_threshold=5)

# remove inconsistent intensity vs centroid beads
updated_ref_peak_dict, ref_distances, ref_centroid_dict = remove_intensity_centroid_inconsistent_beads(label_seg=ref_labelled_seg,
                                                                                                       updated_peak_dict=updated_ref_peak_dict)
updated_mov_peak_dict, mov_distances, mov_centroid_dict = remove_intensity_centroid_inconsistent_beads(label_seg=mov_labelled_seg,
                                                                                                       updated_peak_dict=updated_mov_peak_dict)

# assign updated_ref_peak_dict with updated_mov_peak_dict
# bead_peak_intensity_dict, ref_mov_num_dict, ref_mov_coor_dict = assign_ref_to_mov(updated_ref_peak_dict, updated_mov_peak_dict)

# assign centroid_dicts from ref to mov
bead_centroid_dict, ref_mov_num_dict, ref_mov_coor_dict = assign_ref_to_mov(ref_centroid_dict, mov_centroid_dict)

check_beads(ref_mov_num_dict, ref_labelled_seg, mov_labelled_seg)

# Throw a logging/warning message if there are too little number of beads
if len(bead_centroid_dict) < 10:
    # Add logging/error message
    print('number of beads seem low: ' + str(len(bead_peak_intensity_dict)))

# Change (y, x) into (x, y) for transformation matrix reading
rev_coor_dict = change_coor_system(ref_mov_coor_dict)

#=======================================================================================================================
# Initiate transform estimation
tform = tf.estimate_transform('similarity', np.asarray(list(rev_coor_dict.keys())), np.asarray(list(rev_coor_dict.values())))
mov_transformed = tf.warp(beads_cmdr[center_z, :, :], inverse_map=tform, order=3)

# Report transform parameters
transformation_parameters_dict = report_similarity_matrix_parameters(tform=tform, logging=True)

# Report intensity changes in FOV after transform
changes_fov_intensity_dictionary = report_change_fov_intensity_parameters(transformed_img=mov_transformed,
                                                                          original_img=mov,
                                                                          logging=True)

# Report changes in source and destination
coor_dist_qc, diff_sum_beads = report_changes_in_coordinates_mapping(ref_mov_coor_dict=ref_mov_coor_dict,
                                                                     tform=tform,
                                                                     logging=True)

# Report changes in nrmse in the image? NOT READY TO USE, seems to be intensity-dependent
# nrmse_qc, diff_nrmse = report_changes_in_nrmse(ref_img=ref, mov_img=mov, mov_transformed=mov_transformed, logging=True)

# making tracking map in different areas of an FOV?

# Save metrics
# Todo: Check with SW, in what format to save transform to be applied to pipeline images and saved in image metadata?
# np.savetxt(r'C:\Users\calystay\Desktop\test_transform.csv', inverse_tform, delimiter=',')

# Validate: Save beads imag (ref, before_mov, after_mov)
io.imsave(r'\\allen\aics\microscopy\Calysta\test\camera_alignment\beads_centroid\test_1_ref_gfp.tiff', beads_gfp)
io.imsave(r'\\allen\aics\microscopy\Calysta\test\camera_alignment\beads_centroid\test_1_before_cmdr.tiff', beads_cmdr)

after_cmdr = np.zeros(beads_cmdr.shape)
for z in range(0, after_cmdr.shape[0]):
    after_cmdr[z, :, :] = tf.warp(beads_cmdr[z, :, :], inverse_map=tform, order=3)
after_cmdr = (after_cmdr*65535).astype(np.uint16)
io.imsave(r'\\allen\aics\microscopy\Calysta\test\camera_alignment\beads_centroid\test_1_after_cmdr.tiff', after_cmdr)

#=======================================================================================================================
# Apply transform on testing images
load_transform_array = np.loadtxt(r'C:\Users\calystay\Desktop\test_transform.csv', delimiter=',')

argo_array = np.array([[1.0003, 0.0026, -0.599],
                       [-0.0026, 1.0003, 0.7998],
                       [0, 0, 1]])

transformed_cmdr = np.zeros(beads_cmdr.shape)
argo_transformed_cmdr = np.zeros(beads_cmdr.shape)
for z in range (0, beads_cmdr.shape[0]):
    transformed_xy = tf.warp(beads_cmdr[z, :, :], inverse_map=load_transform_array, order=3)
    transformed_cmdr[z, :, :] = transformed_xy

    transformed_xy_argo = tf.warp(beads_cmdr[z, :, :], inverse_map=argo_array, order=3)
    argo_transformed_cmdr[z, :, :] = transformed_xy_argo

transformed_cmdr = (transformed_cmdr*65535).astype(np.uint16)
io.imsave(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD3\3500003332_100X_20190813_psf_cmdr_aligned.tif',
          transformed_cmdr)

argo_transformed_cmdr = (argo_transformed_cmdr*65535).astype(np.uint16)
io.imsave(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD3\3500003332_100X_20190813_psf_cmdr_aligned_argo.tif',
          argo_transformed_cmdr)

before_gfp = beads_gfp.astype(np.uint16)
io.imsave(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD3\3500003332_100X_20190813_psf_gfp.tif',
          before_gfp)

before_cmdr = beads_cmdr.astype(np.uint16)
io.imsave(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD3\3500003332_100X_20190813_psf_cmdr.tif',
          before_cmdr)


def assign_ref_to_mov(updated_ref_peak_dict, updated_mov_peak_dict):
    """
    Assigns beads from moving image to reference image using linear_sum_assignment to reduce the distance between
    the same bead on the separate channels. In case where there is more beads in one channel than the other, this method
    will throw off the extra bead that cannot be assigned to one, single bead on the other image.
    :param updated_ref_peak_dict: A dictionary ({bead_number: (coor_y, coor_x)}) from reference beads
    :param updated_mov_peak_dict:  A dictionary ({bead_number: (coor_y, coor_x)}) from moving beads
    :return:
        bead_peak_intensity_dict: A full dictionary mapping the reference bead number and coordinates with the moving
                                  ones. ({(bead_ref_number, (bead_ref_coor_y, bead_ref_coor_x)): (bead_mov_number, (bead_mov_coor_y, bead_mov_coor_x))
        ref_mov_num_dict: A dictionary mapping the reference bead number and moving bead number
        ref_mov_coor_dict: A dictionary mapping the reference bead coordinates and moving bead coordinates
    """
    updated_ref_peak = list(OrderedDict(updated_ref_peak_dict).items())
    updated_mov_peak = list(OrderedDict(updated_mov_peak_dict).items())

    dist_tx = []
    for bead_ref, coor_ref in updated_ref_peak:
        row = []
        for bead_mov, coor_mov in updated_mov_peak:
            row.append(distance.euclidean(coor_ref, coor_mov))
        dist_tx.append(row)

    dist_tx = np.asarray(dist_tx)
    ref_ind, mov_ind = linear_sum_assignment(dist_tx)

    bead_peak_intensity_dict = {}
    ref_mov_num_dict = {}
    ref_mov_coor_dict = {}
    for num_bead in range(0, len(ref_ind)):
        bead_peak_intensity_dict.update({updated_ref_peak[ref_ind[num_bead]]: updated_mov_peak[mov_ind[num_bead]]})
        ref_mov_num_dict.update({updated_ref_peak[ref_ind[num_bead]][0]: updated_mov_peak[mov_ind[num_bead]][0]})
        ref_mov_coor_dict.update({updated_ref_peak[ref_ind[num_bead]][1]: updated_mov_peak[mov_ind[num_bead]][1]})

    return bead_peak_intensity_dict, ref_mov_num_dict, ref_mov_coor_dict


def change_coor_system(coor_dict):
    """
    Changes coordinates in a dictionary from {(y1, x1):(y2, x2)} to {(x1, y1): (x2, y2)}
    :param coor_dict: A dictionary of coordinates in the form of {(y1, x1):(y2, x2)}
    :return:
        An updated reversed coordinate dictionary that is {(x1, y1): (x2, y2)}
    """
    rev_yx_to_xy = {}
    for coor_ref, coor_mov in coor_dict.items():
        rev_yx_to_xy.update({(coor_ref[1], coor_ref[0]): (coor_mov[1], coor_mov[0])})
    return rev_yx_to_xy


def check_beads(ref_mov_num_dict, ref_labelled_seg, mov_labelled_seg):
    """
    function to check beads visually if they match after filtering on ref and mov image
    :param ref_mov_num_dict: A dictionary that maps reference bead number with moving bead number
    :param ref_labelled_seg: Labelled segmentation reference image
    :param mov_labelled_seg: Labelled segmentation moving image
    :return:
    """
    show_ref = np.zeros(ref_labelled_seg.shape)
    show_mov = np.zeros(mov_labelled_seg.shape)
    for bead_label in list(ref_mov_num_dict.keys()):
        show_ref[ref_labelled_seg==bead_label] = True
    for bead_label in list(ref_mov_num_dict.values()):
        show_mov[mov_labelled_seg==bead_label] = True

    for img in [show_ref, show_mov]:
        plt.figure()
        plt.imshow(img)
        plt.show()


def filter_big_beads(img, center=0, area=20):
    """
    Find and filter big beads from an image with mixed beads
    :param img: 3d image with big and small beads
    :param center: center slice
    :param area: area(px) cutoff of a big bead
    :return: filtered: A 3d image where big beads are masked out as 0
             seg_big_bead: A binary image showing segmentation of big beads
    """

    if len(img.shape) == 2:
        img_center = img
    elif len(img.shape) == 3:
        img_center = img[center, :, :]

    # Big beads are brighter than small beads usually
    seg_big_bead = img_center > (np.median(img_center) + 1.25 * np.std(img_center))
    label_big_bead = measure.label(seg_big_bead)

    # Size filter the labeled big beads, that could be due to bright small beads
    for obj in range(1, np.max(label_big_bead)):
        size = np.sum(label_big_bead == obj)
        if size < area:
            seg_big_bead[label_big_bead == obj] = 0

    # Save filtered beads image after removing big beads as 'filtered'
    if len(img.shape) == 3:
        mask = np.zeros(img.shape)
        for z in range(0, img.shape[0]):
            mask[z] = seg_big_bead
        filtered = img.copy()
        filtered[np.where(mask == 1)] = np.median(img)
    elif len(img.shape) == 2:
        filtered = img.copy()
        filtered[np.where(seg_big_bead>0)] = np.median(img)
    return filtered, seg_big_bead


def initialize_peaks(seg, peak_list, show_img=False, img_shape=None):
    """
    Initializes the mapping of bead label (from segmentation) and the peak intensity coordinate (from finding peaks)
    :param seg: A binary segmentation of beads
    :param peak_list: A list of peaks ([(y, x), (y2, x2), ...])
    :param show_img: A boolean to indicate if the user would like to show the peaks on the image
    :param img_shape: A tuple of the size of the image (y_dim, x_dim)
    :return:
        peak_dict: A dictionary mapping the label of the bead and the coordinates of peaks ({bead_num: (coor_y, coor_x)})
        seg_label: A labelled image from segmentation
    """
    seg_label = measure.label(seg)
    peak_dict = {}
    for peak in peak_list:
        # get label
        label = seg_label[peak[0], peak[1]]
        peak_dict.update({label: (peak[0], peak[1])})

    if show_img:
        img = np.zeros(img_shape)
        for key, coor in peak_dict.items():
            y, x = coor
            img[y - 5:y + 5, x - 5:x + 5] = True
        plt.figure()
        plt.imshow(img)
        plt.show()

    return peak_dict, seg_label

def match_peaks(ref_peak_dict, mov_peak_dict, dist_threshold=5):
    """
    Matches peaks from reference peaks and moving peaks and filter reference beads that don't have a matching moving
    beads within a distance threshold
    :param ref_peak_dict: A dictionary of reference peaks ({bead_number: (coor_y, coor_x)})
    :param mov_peak_dict: A dictionary of moving peaks ({bead_number: (coor_y, coor_x)})
    :param dist_threshold: Number of pixels as distance threshold
    :return:
        updated_ref_peak_dict: An updated dictionary after removing beads that don't have a matching peak
        updated_mov_peak_dict: An updated dictionary after removing beads that don't have a matching peak
    """
    remove_mov_peak = []
    for mov_peak_id, mov_coor in mov_peak_dict.items():
        dist_list = []
        for ref_peak_id, ref_coor in ref_peak_dict.items():
            dist = distance.euclidean(mov_coor, ref_coor)
            dist_list.append(dist)
        if np.min(dist_list) > dist_threshold:
            remove_mov_peak.append(mov_peak_id)
            # This mov_peak_id is not in ref segmentation

    remove_ref_peak = []
    for ref_peak_id, ref_coor in ref_peak_dict.items():
        dist_list = []
        for mov_peak_id, mov_coor in mov_peak_dict.items():
            dist = distance.euclidean(mov_coor, ref_coor)
            dist_list.append(dist)
        if np.min(dist_list) > dist_threshold:
            remove_ref_peak.append(ref_peak_id)
            # This ref peak id is not in mov segmentation

    updated_ref_peak_dict = remove_peaks_in_dict(full_dict=ref_peak_dict, keys=remove_ref_peak)
    updated_mov_peak_dict = remove_peaks_in_dict(full_dict=mov_peak_dict, keys=remove_mov_peak)

    return updated_ref_peak_dict, updated_mov_peak_dict


def remove_close_peaks(peak_dict, dist_threshold=20, show_img=False, img_shape=None):
    """
    Removes peaks on one image that are closer to each other than a distance threshold
    :param peak_dict: A dictionary of peaks ({peak_number: (coor_y, coor_x)})
    :param dist_threshold: Number of pixels as distance threshold
    :param show_img: A boolean to indicate if the user would like to show the peaks on the image
    :param img_shape: A tuple of the size of the image (y_dim, x_dim)
    :return:
        close_ref_peak_dict: An updated dictionary after removing peaks that are too close to each other
    """
    close_ref_peak_dict = peak_dict.copy()
    for peak_id, peak_coor in peak_dict.items():
        for compare_peak_id, compare_peak_coor in peak_dict.items():
            if peak_id != compare_peak_id:
                dist = distance.euclidean(peak_coor, compare_peak_coor)
                if dist <= dist_threshold:
                    try:
                        del close_ref_peak_dict[peak_id]
                        del close_ref_peak_dict[compare_peak_id]
                    except:
                        pass
    if show_img:
        close_peaks_img = np.zeros(img_shape)
        for peak_id, peak_coor in close_ref_peak_dict.items():
            y, x = peak_coor[0], peak_coor[1]
            close_peaks_img[y - 5:y + 5, x - 5:x + 5] = True
        plt.figure()
        plt.imshow(close_peaks_img)
        plt.show()

    return close_ref_peak_dict


def remove_intensity_centroid_inconsistent_beads(label_seg, updated_peak_dict, dist_thresh=3):
    """
    Removes beads that are inconsistent in where the peak intensity and centroid is
    :param label_seg: A labelled segmentation image of beads
    :param updated_peak_dict: A dictionary of beads and peak intensities
    :param dist_thresh: Number of pixels as distance threshold
    :return:
        remove_inconsistent_dict: An updated dictionary of beads and peak intensities are removing inconsistent beads
        distances: A list of distances of coordinates of peak intensity and centroid for each bead
    """
    props = pd.DataFrame(measure.regionprops_table(label_seg, properties=['label', 'centroid'])).set_index('label')
    distances = []
    bead_to_remove = []
    bead_centroid_dict = {}
    for label, coor_intensity in updated_peak_dict.items():
        coor_centroid = (props.loc[label, 'centroid-0'], props.loc[label, 'centroid-1'])
        bead_centroid_dict.update({label: coor_centroid})
        dist = distance.euclidean(coor_intensity, coor_centroid)

        if dist > dist_thresh:
            bead_to_remove.append(label)
        distances.append(dist)

    if len(bead_to_remove) > 0:
        remove_inconsistent_dict = remove_peaks_in_dict(full_dict=updated_peak_dict, keys=bead_to_remove)
        bead_centroid_dict = remove_peaks_in_dict(full_dict=bead_centroid_dict, keys=bead_to_remove)
    else:
        remove_inconsistent_dict = updated_peak_dict.copy()

    return remove_inconsistent_dict, distances, bead_centroid_dict


def remove_overlapping_beads(label_seg, peak_dict, area_tolerance=0.3, show_img=False):
    """
    Remove overlapping beads that give possibility to inaccurate bead registration,
    filter by size (median+area_tolerance*std)
    :param label_seg: A labelled segmentation image of beads
    :param peak_dict: A dictionary of bead number to peak coordinates
    :param area_tolerance: Arbitrary float to set threshold of size
    :param show_img: A boolean to show image
    :return:
        remove_overlapping_beads: An updated dictionary of bead number to peak coordinates after removing beads that
                                  are overlapped on one another or clustered
    """
    props = pd.DataFrame(measure.regionprops_table(label_seg, properties=['label', 'area'])).set_index('label')
    area_thresh = props['area'].median() + area_tolerance*props['area'].std()

    beads_to_remove = []
    for label, row in props.iterrows():
        if label in peak_dict.keys():
            if row['area'] > area_thresh:
                beads_to_remove.append(label)

    remove_overlapping_beads = remove_peaks_in_dict(peak_dict, beads_to_remove)

    if show_img:
        img = np.zeros(ref_labelled_seg.shape)
        for bead in list(remove_overlapping_beads.keys()):
            img[ref_labelled_seg==bead] = label

        plt.figure()
        plt.imshow(img)
        plt.show()

    return remove_overlapping_beads


def remove_peaks_in_dict(full_dict, keys):
    """
    Removes a list of keys from a dictionary
    :param full_dict: A dictionary
    :param keys: A list of keys
    :return:
        new_dict: An updated dictionary after removing the keys
    """
    new_dict = full_dict.copy()
    for key in keys:
        del new_dict[key]
    return new_dict


def report_similarity_matrix_parameters(tform, logging=True):
    """
    Reports similarity matrix and its parameters
    :param tform: A transform generated from skimage.transform.estimate_transform
    :param logging: A boolean to indicate if printing/logging statements is selected
    :return: A dictionary with the following keys and values:
        inverse_tform: An inverse transform to be applied to moving images
        scaling: Uniform scaling parameter
        shift_y: Shift in y
        shift_x: Shift in x
        rotate_angle: Rotation angle
    """
    inverse_tform = tf.SimilarityTransform(tform._inv_matrix)
    similarity_matrix_param_dict = {
        'inverse_tform': inverse_tform,
        'scaling': inverse_tform.scale,
        'shift_y': inverse_tform.translation[0],
        'shift_x': inverse_tform.translation[1],
        'rotate_angle': inverse_tform.rotation
    }

    if logging:
        for param, value in similarity_matrix_param_dict.items():
            print(param + ': ' + str(value))

    return similarity_matrix_param_dict


def report_change_fov_intensity_parameters(transformed_img, original_img, logging=True):
    """
    Reports changes in FOV intensity after transform
    :param transformed_img: Image after transformation
    :param original_img: Image before transformation
    :param logging: A boolean to indicate if printing/logging statements is selected
    :return: A dictionary with the following keys and values:
        median_intensity
        min_intensity
        max_intensity
        1_percentile: first percentile intensity
        995th_percentile: 99.5th percentile intensity
    """
    change_fov_intensity_param_dict = {
        'median_intensity': np.median(transformed_img) * 65535 - np.median(original_img),
        'min_intensity': np.min(transformed_img) * 65535 - np.min(original_img),
        'max_intensity': np.max(transformed_img) * 65535 - np.max(original_img),
        '1st_percentile': np.percentile(transformed_img, 1) * 65535 - np.percentile(original_img, 1),
        '995th_percentile': np.percentile(transformed_img, 99.5) * 65535 - np.percentile(original_img, 99.5)
    }

    if logging:
        for key, value in change_fov_intensity_param_dict.items():
            print('change in ' + key + ': ' + str(value))

    return change_fov_intensity_param_dict


def report_changes_in_coordinates_mapping(ref_mov_coor_dict, tform, logging=True):
    """
    Report changes in coordinates before and after transform. A good transform will reduce the difference in distances
    between transformed_mov_beads and ref_beads than mov_beads and ref_beads. A bad transform will increase the
    difference in distances between transformed_mov_beads and ref_beads.
    :param ref_mov_coor_dict: A dictionary mapping the reference bead coordinates and moving bead coordinates (before transform)
    :param tform: A skimage transform object
    :param logging: A boolean to indicate if printing/logging statements is selected
    :return:
    """
    transform_qc = False
    mov_coors = list(ref_mov_coor_dict.values())
    ref_coors = list(ref_mov_coor_dict.keys())
    mov_transformed_coors = tform.inverse(mov_coors)

    dist_before_list = []
    dist_after_list = []
    for bead in range(0, len(mov_coors)):
        dist_before = distance.euclidean(mov_coors[bead], ref_coors[bead])
        dist_after = distance.euclidean(mov_transformed_coors[bead], ref_coors[bead])
        dist_before_list.append(dist_before)
        dist_after_list.append(dist_after)

    sum_diff_before = sum(dist_before_list)
    sum_diff_after = sum(dist_after_list)
    diff_sum = sum_diff_after - sum_diff_before

    if logging:
        if diff_sum < 0:
            print('transform looks good - ')
            print('transform reduced the sum of distances between bead peaks on reference and moving image by ' + str(diff_sum))
        elif diff_sum == 0:
            print('no difference in distances before and after transform')
        else:
            print('transform looks bad - ')
            print('transform increased the sum of distances between bead peaks on reference and moving image by ' + str(diff_sum))

    if diff_sum < 0:
        transform_qc = True

    return transform_qc, diff_sum


def report_changes_in_nrmse(ref_img, mov_img, mov_transformed, logging=True):
    """
    Report changes in normalized root mean-squared-error value before and after transform.
    :param ref_img: Reference image
    :param mov_img: Moving image before transform
    :param mov_transformed: Moving image after transform
    :param logging: A boolean to indicate if printing/logging statements is selected
    :return:
        qc: A boolean to indicate if it passed (True) or failed (False) qc
        diff_nrmse: Difference in nrmse
    """
    qc = False

    nrmse_before = metrics.normalized_root_mse(ref_img, mov_img)
    nrmse_after = metrics.normalized_root_mse(ref_img, mov_transformed)

    print('before: ' + str(nrmse_before))
    print('after: ' + str(nrmse_after))
    diff_nrmse = nrmse_after - nrmse_before
    if diff_nrmse <= 0:
        qc = True

    if logging:
        if diff_nrmse < 0:
            print('transform is good - ')
            print('nrmse is reduced by ' + str(diff_nrmse))
        elif diff_nrmse == 0:
            print('transform did not improve or worsen - ')
            print('nrmse differences before and after is 0')
        else:
            print('transform is bad - ')
            print('nrmse is increased by ' + str(diff_nrmse))

    return qc, diff_nrmse





def verify_peaks(ref_peak_dict, mov_peak_dict, initialize_value=100):
    """
    Verifies the beads mapped on referece image are mapped on the moving image with the minimal distance
    :param ref_peak_dict: A dictionary of reference peaks ({peak_number: (coor_y, coor_x)})
    :param mov_peak_dict: A dictionary of moving peaks ({peak_number: (coor_y, coor_x)})
    :param initialize_value: An initial distance value, expected minimum distance should be less than the size of a bead
    :return:
        src_dst_dict: A dictionary mapping the reference coordinates and moving coordinates
                      ({(ref_coor_y, ref_coor_x): (mov_coor_y, mov_coor_x)})
    """
    src_dst_dict = {}
    for mov_peak_id, mov_coor in mov_peak_dict.items():
        min_dist = initialize_value
        for ref_peak_id, ref_coor in ref_peak_dict.items():
            dist = distance.euclidean(mov_coor, ref_coor)
            if dist < min_dist:
                min_dist = dist
                map_coor = ref_coor
        src_dst_dict.update({(mov_coor[1], mov_coor[0]): (map_coor[1], map_coor[0])})

    return src_dst_dict


def watershed_bead_seg(seg):
    """
    Performs watershed on a segmentation of beads to separate beads that are touching each other based on distance
    transform and morphology
    :param seg: A binary segmentation of beads
    :return:
        labels: A lablled segmentation image of beads
    """
    props = measure.regionprops(measure.label(seg))
    median_size = np.median([props[x].area for x in range(0, len(props))])
    radius = math.sqrt(median_size/math.pi)

    distance = ndimage.distance_transform_edt(seg)
    local_maxi = feature.peak_local_max(distance, indices=False, footprint=morphology.disk(radius), labels=measure.label(seg))
    markers = ndimage.label(local_maxi)[0]
    labels = segmentation.watershed(-distance, markers, mask=seg)

    return labels
