# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:26:25 2019

@author: Calysta Yan
"""
import numpy as np
from aicsimageio import AICSImage
import ndimage
from skimage import filters, measure

def generate_images(image):
    """
    This function generates 6 images from a zstack
    :param image: an image with shape(T,C,Z,Y,X)
    :return: 6 images: highest_z, lowest_z, center_z, mip_xy, mip_xz, mip_yz
    """
    image_TL = image.data[0, 0, :, :, :]
    image_EGFP = image.data[0, 1, :, :, :]
    center_plane = find_center_z_plane(image_TL)
    # BF panels: top, bottom, center
    top_TL = image_TL[-1, :, :]
    bottom_TL = image_TL[0, :, :]
    center_TL = image_TL[center_plane, :, :]
    # EGFP panels: mip_xy, mip_xz, mip_yz
    mip_xy = np.amax(image_EGFP, axis=0)
    mip_xz = np.amax(image_EGFP, axis=1)
    mip_yz = np.amax(image_EGFP, axis=2)
    
    return top_TL, bottom_TL, center_TL, mip_xy, mip_xz, mip_yz

def create_display_setting(rows, control_column, folder_path):
    """
    This function generates a dictionary of display setting to be applied to images
    :param rows:
    :param control_column:
    :param folder_path:
    :return:
    """
    display_dict = {}
    images = os.listdir(plate_path)
    for row in rows:
        print (row)
        display_settings = []
        for img_file in images:
            if img_file.endswith(row + control_column + '.czi'):
                image = AICSImage(os.path.join(plate_path, img_file), max_workers=1)
                print (img_file)
                image_EGFP = image.data[0, 1, :, :, :]
                mip_xy = np.amax(image_EGFP, axis=0)
                display_min, display_max = np.min(mip_xy), np.max(mip_xy)
                display_settings.append((display_min, display_max))
        display_minimum = int(round(np.mean([dis_min[0] for dis_min in display_settings])))
        display_maximum = int(round(np.mean([dis_max[1] for dis_max in display_settings])))
        display_dict.update({row: (display_minimum, display_maximum)})
    return display_dict

def find_center_z_plane(image):
    """

    :param image:
    :return:
    """
    mip_yz = np.amax(image, axis=2)
    mip_gau = filters.gaussian(mip_yz, sigma=2)
    edge_slice = filters.sobel(mip_gau)
    contours = measure.find_contours(edge_slice, 0.005)
    new_edge = np.zeros(edge_slice.shape)
    for n, contour in enumerate (contours):
        new_edge[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
    
    # Fill empty spaces of contour to identify as 1 object
    new_edge_filled = ndimage.morphology.binary_fill_holes(new_edge)
    
    # Identify center of z stack by finding the center of mass of 'x' pattern
    z = []
    for i in range (100, mip_yz.shape[1]+1, 100):
        edge_slab= new_edge_filled[:, i-100:i]
        #print (i-100, i)
        z_center, x_center = ndimage.measurements.center_of_mass(edge_slab)
        z.append(z_center)
    
    z = [z_center for z_center in z if ~np.isnan(z_center)]
    z_center = int(round(np.median(z)))
    
    return (z_center)