# align dataset for RnD

# READ HERE
# Set user inputs:
optical_control_img_filepath = r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ZSD3_20200609\custombeads_100X_20200609.czi'
image_type = 'beads'  # Select between 'rings' or 'beads'
ref_channel = 'EGFP'  # Enter name of reference channel (for zsd, use 'EGFP'; for 3i, use '488/TL 50um Dual')
mov_channel = 'CMDRP'  # Enter name of moving channel (for zsd, use 'CMDRP'; for 3i, use '640/405 50um Dual')
system_type = 'zsd'  # Select between 'zsd' or '3i'

folder_to_czi = r'\\allen\aics\assay-dev\MicroscopyData\Sara\2020\20200609\to_process_100X\split'  # Input folder to czi images. (Currently don't support 3i images, check back later, sorry!)
folder_save = r'\\allen\aics\assay-dev\MicroscopyData\Sara\2020\20200609\to_process_100X\split'  # Output folder to save split scene tiffs
crop_dim = (600, 900)  # Final dimension of image after cropping in the form of (image height, image width)
#===================================
# Core script - don't change plz
import numpy as np
import os
from pipeline_qc import camera_alignment
from aicsimageio import AICSImage, writers
from skimage import io, transform as tf

def perform_similarity_matrix_transform(img, matrix):
    """
    Performs a similarity matrix geometric transform on an image
    :param img: A 2D/3D image to be transformed
    :param matrix: Similarity matrix to be applied on the image
    :param output_path: Output path to save the image
    :param filename: Name of the image to save
    :return:
    """
    after_transform = None
    if len(img.shape) == 2:
        after_transform = tf.warp(img, inverse_map=matrix, order=3)
    elif len(img.shape) == 3:
        after_transform = np.zeros(img.shape)
        for z in range(0, after_transform.shape[0]):
            after_transform[z, :, :] = tf.warp(img[z, :, :], inverse_map=matrix, order=3)
    else:
        print('dimensions invalid for img')

    if after_transform is not None:
        after_transform = (after_transform*65535).astype(np.uint16)
        # io.imsave(output_path, after_transform)

    return after_transform

print('aligning matrix')
if os.path.exists(optical_control_img_filepath.replace('.czi', '_sim_matrix.txt')) is False:
    exe = camera_alignment.Executor(
        image_path=optical_control_img_filepath,
        image_type=image_type,
        ref_channel_index=ref_channel,
        mov_channel_index=mov_channel,
        system_type=system_type,
        thresh_488=None,  # Set 'None' to use default setting
        thresh_638=None,  # Set 'None' to use default setting
        crop_center=None,  # Set 'None' to use default setting
        method_logging=True,
        align_mov_img=True,
        align_mov_img_path=optical_control_img_filepath,
        align_mov_img_file_extension='_aligned.tif',
        align_matrix_file_extension='_sim_matrix.txt')
    exe.execute()

tf_array = np.loadtxt(optical_control_img_filepath.split('.')[0] + '_sim_matrix.txt', delimiter=',')

if folder_to_czi is not None:
    imgs = os.listdir(folder_to_czi)
    for raw_split_file in imgs:
        if raw_split_file.endswith('.czi'):
            print('processing ' + raw_split_file)
            img_data = AICSImage(os.path.join(folder_to_czi, raw_split_file))
            channels = img_data.get_channel_names()
            img_stack = img_data.data
            omexml = img_data.metadata
            # process each channel
            final_img = np.zeros(img_stack.shape)
            for channel in channels:
                if channel.startswith('Bright'):
                    sub_folder = 'aligned_bf'
                elif channel.startswith('CMDR'):
                    sub_folder = 'aligned_cmdr'
                elif channel.startswith('H334'):
                    sub_folder = 'raw_nuc'
                elif channel.startswith('EGF'):
                    sub_folder = 'raw_gfp'

                img = img_stack[0, 0, channels.index(channel), :, :, :]
                if channel.startswith('Bright') or channel.startswith('CMDR'):
                    img = perform_similarity_matrix_transform(img, tf_array)

                # generate stack for data back fill
                final_img[0, 0, channels.index(channel), :, :, :] = img

            final_img = final_img.astype(np.uint16)
            s, t, c, z, y, x = final_img.shape
            upload_img = final_img[0, :, :, :, int((y-crop_dim[0])/2):int(crop_dim[0] + (y-crop_dim[0])/2), int((x-crop_dim[1])/2):int(crop_dim[1] + (x-crop_dim[1])/2)]
            upload_img = upload_img.transpose((0, 2, 1, 3, 4))

            writer = writers.OmeTiffWriter(
                os.path.join(folder_save, raw_split_file.replace('-Scene', '-alignV2-Scene').replace('.czi', '.tiff'))
            )
            writer.save(upload_img.astype(np.uint16))