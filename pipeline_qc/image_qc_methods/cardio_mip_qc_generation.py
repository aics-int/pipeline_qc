import os

import PIL
import aicsimageio
import numpy
from PIL import Image
from pipeline_qc.image_processing_methods import auto_contrast_fn, find_center_z_plane


# Creates max intensity projections for a single color channel of an image
def single_channel_processing(full_array, channel, color):

    xy = auto_contrast_fn(numpy.amax(full_array[0, 0, channel, :, :, :], axis=0))
    yz = auto_contrast_fn(numpy.transpose(numpy.amax(full_array[0, 0, channel, :, :, :], axis=2)))

    if color == 'red':
        mip_xy_array = numpy.stack((xy, numpy.zeros(xy.shape), numpy.zeros(xy.shape())), axis=2)
        mip_yz_array = numpy.stack((yz, numpy.zeros(yz.shape), numpy.zeros(yz.shape)), axis=2)
    elif color == 'green':
        mip_xy_array = numpy.stack((numpy.zeros(xy.shape), xy, numpy.zeros(xy.shape)), axis=2)
        mip_yz_array = numpy.stack((numpy.zeros(yz.shape), yz, numpy.zeros(yz.shape)), axis=2)
    elif color == 'blue':
        mip_xy_array = numpy.stack((numpy.zeros(xy.shape), numpy.zeros(xy.shape), xy), axis=2)
        mip_yz_array = numpy.stack((numpy.zeros(yz.shape), numpy.zeros(yz.shape), yz), axis=2)
    elif color == 'magenta':
        mip_xy_array = numpy.stack((xy, numpy.zeros(xy.shape), xy), axis=2)
        mip_yz_array = numpy.stack((yz, numpy.zeros(yz.shape), yz), axis=2)
    elif color == 'cyan':
        mip_xy_array = numpy.stack((numpy.zeros(xy.shape), xy, xy), axis=2)
        mip_yz_array = numpy.stack((numpy.zeros(yz.shape), yz, yz), axis=2)
    elif color == 'white':
        mip_xy_array = numpy.stack((xy, xy, xy), axis=2)
        mip_yz_array = numpy.stack((yz, yz, yz), axis=2)
    else:
        mip_xy_array = numpy.stack((numpy.zeros(xy.shape), numpy.zeros(xy.shape), numpy.zeros(xy.shape)), axis=2)
        mip_yz_array = numpy.stack((numpy.zeros(yz.shape), numpy.zeros(yz.shape), numpy.zeros(yz.shape)), axis=2)

    return mip_xy_array, mip_yz_array


# Combines all the images needed to generate a qc image (xy mip (merged for all color channels,
# yz mips (one for each color channel), and a center slice of transmitted light
# Can combine an arbitrary number of color channels
def comb_mip_image_generator(xy_mips, yz_mips, center_tl):

    xy_mip_comb = sum(xy_mips)
    thresh = xy_mip_comb > 255
    xy_mip_comb[thresh] = 255
    xy_mip_comb_img = PIL.Image.fromarray(xy_mip_comb.astype('uint8'), 'RGB')
    xy_mip_comb_enhance = numpy.array(xy_mip_comb_img)

    border_shape = list(yz_mips[0].shape)
    border_shape[1] = 1
    yz_mips_comb = numpy.ones(border_shape)*255
    for mip in yz_mips:
        yz_mips_comb = numpy.concatenate((yz_mips_comb, numpy.ones(border_shape)*255, mip), axis=1)

    yz_mips_comb_img = PIL.Image.fromarray(yz_mips_comb.astype('uint8'), 'RGB')
    yz_mips_comb_img_resize = yz_mips_comb_img.resize([yz_mips_comb_img.size[0] * 4, yz_mips_comb_img.size[1]])
    yz_mips_comb_resize = numpy.array(yz_mips_comb_img_resize)

    center_tl_n = (center_tl/center_tl.max())*255
    center_tl_rgb = numpy.stack((center_tl_n, center_tl_n, center_tl_n), axis=2)
    center_tl_img = PIL.Image.fromarray(center_tl_rgb.astype('uint8'), 'RGB')
    center_tl_array = numpy.array(center_tl_img)

    comb_array = numpy.concatenate((xy_mip_comb_enhance, yz_mips_comb_resize, center_tl_array), axis=1)
    return PIL.Image.fromarray(comb_array.astype('uint8'), 'RGB')


# Specifies which color channels are in the file, and runs each of the functions defined above to
# create a set of qc images (only one plate is done)
def create_mips(plate_folder):

    # Creates new output dir if it doesn't exist
    if os.path.isdir(plate_folder + r'/QC'):
        pass
    else:
        os.mkdir(plate_folder + r'/QC')

    output_folder = plate_folder + r'/QC/qc_images'

    if os.path.isdir(output_folder):
        pass
    else:
        os.mkdir(output_folder)

    device_folders = list()
    for sub_folder in os.listdir(plate_folder):
        if sub_folder.startswith('3i'):
            device_folders.append(sub_folder)

    for device_folder in device_folders:
        folder = plate_folder + r'/' + device_folder + r'/63X_zstacks'
        # else:
        # raise Exception('There is no 3i folder for this plate (improper folder structure)')

        # Make list of files needing to be iterated through
        dir_list = os.listdir(folder)

        for file in dir_list:
            if file.endswith('C0.tif'):
                filename = file[:-6]
                print(f'Processing {filename}')

                try:
                    im = aicsimageio.AICSImage(folder + r'/' + filename + 'C0.tif')
                    for ch in im.get_channel_names():
                        if ch[:2] == 'TL':
                            ch_tl_index = im.get_channel_names().index(ch)
                        elif ch[:3] == '405':
                            ch_405_index = im.get_channel_names().index(ch)
                        elif ch[:3] == '488':
                            ch_488_index = im.get_channel_names().index(ch)
                        elif ch[:3] == '640':
                            ch_640_index = im.get_channel_names().index(ch)

                    full_array = im.get_image_data()
                    xy_mips = []
                    yz_mips = []

                    xy,yz = single_channel_processing(full_array,
                                                      ch_405_index,
                                                      'cyan')
                    xy_mips.append(xy)
                    yz_mips.append(yz)

                    xy,yz = single_channel_processing(full_array,
                                                      ch_488_index,
                                                      'white')
                    xy_mips.append(xy)
                    yz_mips.append(yz)
                    xy,yz = single_channel_processing(full_array,
                                                      ch_640_index,
                                                      'magenta')
                    xy_mips.append(xy)
                    yz_mips.append(yz)

                    bools, center_tl_index = find_center_z_plane(full_array[0, 0, ch_tl_index, :, :, :])
                    if center_tl_index == 0:
                        center_tl = full_array[0, 0, ch_tl_index, 25, :, :]
                    else:
                        center_tl = full_array[0, 0, ch_tl_index, center_tl_index, :, :]

                    comb_img = comb_mip_image_generator(xy_mips, yz_mips, center_tl)

                    comb_img.save(output_folder + r'/' + filename + 'merge.jpg')
                    comb_img.save(output_folder + r'/' + filename + 'merge.tif')

                except AssertionError:
                    print(filename + ' was not found')
            else:
                pass


# Batches multiple plates together to run at once
def batch_cardio_qc(plates):
    all_folders = r'/allen/aics/microscopy/PRODUCTION/PIPELINE_6'
    for plate in plates:
        plate_folder = all_folders + r'/' + plate
        create_mips(plate_folder)
        print("Processed from plate #", plate)


# plates = ['5500000150','5500000151', '5500000155', '5500000156', '5500000158', '5500000159', '5500000160']
