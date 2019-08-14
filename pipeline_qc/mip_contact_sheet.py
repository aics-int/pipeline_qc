# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:20:34 2019

@author: Calysta Yan
"""
# User's input
plate_path = r'\\allen\aics\microscopy\PRODUCTION\PIPELINE_5.2\3500002174\ZSD1\100X_zstack\to_process'
output_path = r'\\allen\aics\microscopy\Calysta\test'


rows = ['B','C','D','E','F','G']
control_column = '11'

#------------------------------------------------------------------------------
import os
from aicsimageio import AICSImage, omeTifWriter
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import exposure, filters, measure


def generate_images(image):
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

    #return {'top_TL':top_TL, 'bottom_TL': bottom_TL, 'center_TL': center_TL, 'mip_xy': mip_xy, 'mip_xz': mip_xz, 'mip_yz':mip_yz}
    return top_TL, bottom_TL, center_TL, mip_xy, mip_xz, mip_yz


def create_display_setting(rows, control_column, folder_path):
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
    print (z)
    z_center = int(round(np.median(z)))
    print (z_center)
    return (z_center)


#------------------------------------------------------------------------------

display_settings_dict = create_display_setting(rows = rows, 
                                               control_column = control_column, 
                                               folder_path = plate_path)

print (display_settings_dict)
# Create folder structure
extension = plate_path.split('\\')[-1]
try:
    os.mkdir(os.path.join(output_path, extension))
except:
    pass

try:
    os.mkdir(os.path.join(output_path, extension, 'QC'))
except:
    pass

try:
    os.mkdir(os.path.join(output_path, extension, 'QC', 'qc_images'))
except:
    pass

directories = ['fuse', 'mip_xy', 'mip_xz', 'mip_yz', 'top_TL', 'bottom_TL', 'center_TL']

for row in rows:
    try: 
        os.mkdir(os.path.join(output_path, extension, row))
    except:
        pass
    
    for directory in directories:
        try:
            os.mkdir(os.path.join(output_path, extension, row, directory))
        except:
            pass
# Save display settings as a csv in folder 
display_df = pd.DataFrame.from_dict(data=display_settings_dict, orient='index', columns=['min', 'max'])
display_df.to_csv(os.path.join(output_path, extension, 'display_settings.csv'), header=True)

# Process all images
images = os.listdir(plate_path)
for img_file in images:
    well = img_file.split('-')[-1][0]
    if well in display_settings_dict:
        if img_file.endswith('.czi'):
            # get well ID, time_point_info
            wellid = img_file.split('-')[-1][0]
        
            # associate with display settings
            settings = display_settings_dict[wellid]
            img = AICSImage(os.path.join(plate_path, img_file))
            print ('read image ' + img_file)
            
            # generate 6 images 
            top_TL_0, bottom_TL_0, center_TL_0, mip_xy_0, mip_xz_0, mip_yz_0 = generate_images(img)
            img_height = top_TL_0.shape[0]
            img_width = top_TL_0.shape[1]
            z_height = mip_xz_0.shape[0]
            # set display for mip_images
            rescaled_xy = exposure.rescale_intensity(mip_xy_0, in_range=settings)
            rescaled_xz = exposure.rescale_intensity(mip_xz_0, in_range=settings)
            rescaled_yz = exposure.rescale_intensity(mip_yz_0, in_range=settings)
            # Create fuse image combining 3 mips
            fuse = np.zeros(shape = ((img_height + z_height), (img_width + z_height)))
            fuse[0:z_height, 0:img_width] = rescaled_xz
            fuse[z_height:z_height+img_height, 0:img_width] = rescaled_xy
            fuse[z_height:z_height+img_height, img_width:img_width+z_height] = np.rot90(rescaled_yz)
            
            # Create qc image combining fuse and center_TL
            qc = np.zeros(((img_height + z_height), (2*img_width + z_height)))
            qc[:, 0:img_width+z_height] = fuse
            qc[z_height:img_height+z_height, img_width+z_height:2*img_width+z_height] = center_TL_0
            
            # Save and reformat images in a dictionary
            new_images_dict = {'top_TL': np.reshape(top_TL_0, (1, img_height, img_width)),
                               'bottom_TL': np.reshape(bottom_TL_0, (1, img_height, img_width)),
                               'center_TL': np.reshape(center_TL_0, (1, img_height, img_width)),
                               'mip_xy': np.reshape(rescaled_xy, (1, img_height, img_width)), 
                               'mip_xz': np.reshape(rescaled_xz, (1, z_height, img_width)), 
                               'mip_yz': np.reshape(rescaled_yz, (1, z_height, img_height)), 
                               'fuse': np.reshape(fuse, (1, fuse.shape[0], fuse.shape[1]))}
            print ('edited ' + img_file)
            # Save image in file directory
            file_name = img_file.split('.')[0]
            
            qc_writer = omeTifWriter.OmeTifWriter(os.path.join(output_path, 'QC', 'qc_images', file_name + '-qc.tif'))
            qc_writer.save(qc.astype(np.uint16))
            
            for key, image in new_images_dict.items():
                writer = omeTifWriter.OmeTifWriter(os.path.join(output_path, extension, wellid, key, file_name + '-' + key + '.tif'), 
                                                   overwrite_file=True)
                writer.save(image.astype(np.uint16))
