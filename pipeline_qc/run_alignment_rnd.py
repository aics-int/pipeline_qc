# align dataset for RnD

# READ HERE
# Set user inputs:
optical_control_img_filepath = r'\\allen\aics\microscopy\Calysta\projects\training_emt\data\test\argo_test\argo_20210315_ZSD0_63x.czi'
image_type = 'rings'  # Select between 'rings' or 'beads'
ref_channel = 'EGFP'  # Enter name of reference channel (for zsd, use 'EGFP'; for 3i, use '488/TL 50um Dual')
mov_channel = 'CMDRP'  # Enter name of moving channel (for zsd, use 'CMDRP'; for 3i, use '640/405 50um Dual')
system_type = 'zsd'  # Select between 'zsd' or '3i'
magnification = 63

folder_to_img = r'\\allen\aics\microscopy\Calysta\projects\training_emt\data\5500000408\ZSD0\Raw_Split_Scene'  # Input folder to images
folder_save = r'\\allen\aics\microscopy\Calysta\projects\training_emt\data\test\ZSD0\alignV2'  # Output folder to save split scene tiffs
back_camera_channels = None # default is none (to set as Brightfield and CMDRP), otherwise users can listpossible channel names on back camera that need to be aligned (e.g. ['Bright', 'TaRFP'])
img_type = '.czi'  # file-extension for the images, such as '.tif', '.tiff', '.czi'
crop_dim = (1200, 1800)  # Final dimension of image after cropping in the form of (image height, image width)

#===================================
# Core script - don't change plz
import numpy as np
import os
from pipeline_qc import obtain_camera_alignment # old method: bad ring segmentation
from pipeline_qc.optical_control_qc_methods import obtain_camera_alignment_v2 as camera_alignment # updated method for accurate ring segmentation
from pipeline_qc.camera_alignment.apply_camera_alignment_utilities import perform_similarity_matrix_transform
from aicsimageio import AICSImage, writers
import pandas as pd

print('aligning matrix')
channels_to_align = {
    'ref': ref_channel,
    'mov': mov_channel
}
if os.path.exists(optical_control_img_filepath.replace(img_type, '_sim_matrix.txt')) is False:
    rings_image_data = AICSImage(optical_control_img_filepath)
    exe = camera_alignment.execute(
        image_object=rings_image_data,
        channels_to_align=channels_to_align,
        magnification=magnification,
        save_tform_path=optical_control_img_filepath.replace(img_type, '_sim_matrix.txt'),
        save_mov_transformed_path=optical_control_img_filepath.replace(img_type, '_aligned.tif'),
        alignment_method='alignV2',
        align_mov_img=True,
        method_logging=False
    )

tf_array = np.loadtxt(optical_control_img_filepath.split('.')[0] + '_sim_matrix.txt', delimiter=',')

if back_camera_channels is None:
    back_camera_channels = ['Bright', 'TL', 'CMDRP', 'CMDR', '640', 'BF']

def locate_channels_need_alignment(img_channel_names, system_type,
                                   back_camera_channels=['Bright', 'TL', 'CMDRP', 'CMDR', '640', 'BF']):
    channels_need_alignment = []
    if system_type == 'zsd':
        for channel in img_channel_names:
            if channel in back_camera_channels:
                channels_need_alignment.append(channel)
    elif system_type == '3i':
        for channel in img_channel_names:
            real_channel = channel.split('/')[0]
            if real_channel in back_camera_channels:
                channels_need_alignment.append(channel)
    return channels_need_alignment


def match_channel(img_file_name, channel, img_data):
    channel_from_file = int(img_file_name.split('_C')[-1][0])
    print(channel_from_file)
    channel_from_img = int(img_data.get_channel_names().index(channel))
    if channel_from_file == channel_from_img:
        match = True
    else:
        match = False
    return match

df = pd.DataFrame()

if folder_to_img is not None:
    imgs = os.listdir(folder_to_img)
    for raw_split_file in imgs:

        if raw_split_file.endswith(img_type):
            print('processing ' + raw_split_file)
            img_data = AICSImage(os.path.join(folder_to_img, raw_split_file))
            channels = img_data.get_channel_names()

            img_data.dask_data
            s, t, c, z, y, x = img_data.shape

            if s > 1:
                print("please split scenes in your data")
                pass

            for time_point in range(0, t):
                img_data.get_image_dask_data()

                img_stack = img_data.get_image_dask_data("CZYX", S=0, T=time_point).compute()
                omexml = img_data.metadata
                # process each channel
                final_img = np.zeros(img_stack.shape)

                channels_need_alignment = locate_channels_need_alignment(img_channel_names=channels, system_type=system_type,
                                                                         back_camera_channels=back_camera_channels)
                for channel in channels:
                    img = img_stack[channels.index(channel), :, :, :]
                    if system_type == 'zsd':
                        if channel in channels_need_alignment:
                            img = perform_similarity_matrix_transform(img, tf_array)
                    if system_type == '3i':
                        if match_channel(img_file_name=raw_split_file, channel=channel, img_data = img_data):
                            if channel in channels_need_alignment:
                                img = perform_similarity_matrix_transform(img, tf_array)
                    # generate stack for data back fill
                    final_img[channels.index(channel), :, :, :] = img

                final_img = final_img.astype(np.uint16)

                upload_img = final_img[:, :, int((y-crop_dim[0])/2):int(crop_dim[0] + (y-crop_dim[0])/2), int((x-crop_dim[1])/2):int(crop_dim[1] + (x-crop_dim[1])/2)]
                upload_img = upload_img.transpose((1, 0, 2, 3))

                row = {}
                row['raw_split_scene_file_name'] = raw_split_file
                row['path_to_raw_file_name'] = os.path.join(folder_to_img, raw_split_file)

                if system_type == 'zsd':
                    new_file_name = raw_split_file.replace('-Scene', '-alignV2-Scene').replace('.czi', '.tiff')
                    if t > 1:
                        new_file_name = new_file_name.replace('-P', '-T' + str(time_point) + '-P')
                    if not os.path.exists(os.path.join(folder_save, new_file_name)):
                        writer = writers.OmeTiffWriter(
                            os.path.join(folder_save, new_file_name),
                        )

                        row['aligned_file_name'] = new_file_name
                        row['path_to_aligned_file'] = os.path.join(folder_save, new_file_name)
                    else:
                        writer = None

                elif system_type == '3i':
                    if not os.path.exists(os.path.join(folder_save, raw_split_file)):
                        writer = writers.OmeTiffWriter(
                            os.path.join(folder_save, raw_split_file)
                        )
                        row['aligned_file_name'] = raw_split_file
                        row['path_to_aligned_file'] = os.path.join(folder_save, raw_split_file)
                    else:
                        writer = None

                if writer is not None:
                    writer.save(
                        upload_img.astype(np.uint16),
                        channel_names=channels
                    )
                row['align_version'] = 'alignV2'

                df = df.append(row, ignore_index=True)

df.to_csv(os.path.join(folder_save, 'aligned_ref.csv'))
