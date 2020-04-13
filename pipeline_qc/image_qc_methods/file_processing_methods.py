from pipeline_qc import image_processing_methods
from lkaccess import LabKey, contexts
from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
import numpy as np
import pandas as pd
import os
from skimage import exposure
from labkey.utils import create_server_context


def split_image_into_channels(im_path, source_image_file_id):
    # Splits image data into all requisite channels, as 3D images (405, 488, 561, 638, bf are normal channels)
    # Uses the context table in labkey to find the channel number and then splits the aicsimage loaded file accordingly
    # This allows us to only load an image once and then use it for as many qc steps as we want to later

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
                if np_im.shape[0] > row['channelnumber']:
                    channel_number = row['channelnumber']
                    exec('ch' + channel + "= np_im[channel_number, :, :, :]")
                    exec("split_channels.update({channel: ch" + channel + "})")
                else:
                    pass
            else:
                pass

    return split_channels

def generate_images(image):
    new_edge_filled, center_plane = image_processing_methods.find_center_z_plane(image)
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


def insert_qc_data_labkey(fovid, stat_dict, env):
    if env == 'prod':
        context = create_server_context(
            'aics.corp.alleninstitute.org',
            'AICS/Microscopy',
            'labkey',
            use_ssl=False
        )
    elif env == 'stg':
        context = create_server_context(
            'stg-aics.corp.alleninstitute.org',
            'AICS/Microscopy',
            'labkey',
            use_ssl=False
        )

    lk = LabKey(server_context=context)

    new_row = {key:(str(value) if value else None) for (key, value) in stat_dict.items()}
    new_row['FovId'] = fovid
    lk.insert_rows(
        schema_name='lists',
        query_name='FOV QC Metrics',
        rows=[new_row]
    )
