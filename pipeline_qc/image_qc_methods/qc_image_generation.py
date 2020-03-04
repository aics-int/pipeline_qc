from aicsimageio.writers import ome_tiff_writer
import numpy as np
import os
from skimage import exposure
from pipeline_qc.image_utils import find_center_z_plane

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

    # Generates 6 imagesg
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
