import numpy as np
import matplotlib.pyplot as plt
import os
from aicsimageio import AICSImage
from scipy import ndimage
from skimage import exposure, filters, morphology
from pipeline_qc import image_processing_methods as pm

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
    if (len(bf_z.shape) == 2) | ((len(bf_z.shape) > 2) & (bf_z.shape[0] == 1)):
        # If the input bf image is 1 plane only
        bf = bf_z
    else:
        # If the input bf image is a z-stack
        new_edge_filled, z_center = pm.find_center_z_plane(bf_z)
        bf = bf_z[z_center, :, :]

    segment_bf = segment_colony_area(bf, segment_gauss_thresh)

    if (np.sum(segment_bf)) / (bf.shape[0] * bf.shape[1]) > area_cover_thresh:
        edge = False

    return {'edge fov?': edge}

#===================================================================================
# Validate edge detection method
# folder = r'\\allen\aics\microscopy\PRODUCTION\PIPELINE_4_4\3500002823\ZSD3\100X_zstack'
# all_imgs = os.listdir(folder)
# count = 0
# for img in all_imgs:
#     if (count <= 10) & (img.split('_')[3].startswith('1c')):
#         print (img)
#         image_data = AICSImage(os.path.join(folder, img))
#         image = image_data.data
#         channel_list = np.asarray(image_data.get_channel_names())
#         for channel in channel_list:
#             if channel.startswith('Bright'):
#                 bf_index = np.where(channel in channel_list)[0][0]
#
#         bf_z = image[0, bf_index, :, :, :]
#         edge = detect_edge_position(bf_z, segment_gauss_thresh=0.045, area_cover_thresh=0.9)
#
#         print (edge)
#         count +=1
