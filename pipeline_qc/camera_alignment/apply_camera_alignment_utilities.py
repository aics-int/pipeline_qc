# apply alignment utilities

from skimage import io, transform as tf
import os
import numpy as np


def perform_similarity_matrix_transform(img, matrix):
    """
    Performs a similarity matrix geometric transform on an image
    Parameters
    ----------
    img         a 2D/3D image to be transformed
    matrix      similarity matrix to be applied on the image

    Returns
    -------
    after_transform a 2D/3D image after transformation
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
        after_transform = (after_transform * 65535).astype(np.uint16)

    return after_transform


def generate_augments(img, path, img_size=(360, 536)):
    """
    Generates and saves image augmentations for label-free training images
    Parameters
    ----------
    img         a 3D image
    path        path to save augmented image
    img_size    size of augmented image in (y, x)

    Returns
    -------
    row         a dictionary of {type_of_augment: path_to_augmented_img}
    """
    y_size, x_size = img_size
    flippedlr = np.zeros(img.shape)
    flippedud = np.zeros(img.shape)
    rot180 = np.zeros(img.shape)

    for z in range(0, img.shape[0]):
        z_slice = img[z, :, :]
        flippedlr[z, :, :] = np.fliplr(z_slice).astype(np.uint16)
        flippedud[z, :, :] = np.flipud(z_slice).astype(np.uint16)
        rot180[z, :, :] = tf.rotate(z_slice, angle=180., order=3, preserve_range=True).astype(np.uint16)
    row = {}
    for key, crop_4x_img in {'_cropped': img,
                             '_flippedlr': flippedlr,
                             '_flippedud': flippedud,
                             '_rot180': rot180}.items():
        final_crop = crop_4x_img[:,
                     int(crop_4x_img.shape[1] / 2 - y_size / 2):int(crop_4x_img.shape[1] / 2 + y_size / 2),
                     int(crop_4x_img.shape[2] / 2 - x_size / 2):int(crop_4x_img.shape[2] / 2 + x_size / 2)]

        io.imsave(path.replace('.ome', key + '.ome'), final_crop)
        row.update({key[1:]: path.replace('.ome', key + '.ome')})
    return row


def get_matrix_from_file(image_date=None, system=None,
                         optical_control_img_file_path=None,
                         optical_control_folder='/allen/aics/microscopy/PRODUCTION/OpticalControl',
                         aligned_matrix_endstring='sim_matrix.txt'):
    """
    Reads a .txt file to get similarity transform matrix
    Parameters
    ----------
    image_date          Date of imaging
    system              System that the images were collected on
    optical_control_img_file_path   Optional file path to get the matrix
    optical_control_folder          Pipeline production optical control folder
    aligned_matrix_endstring        End string of matrix file

    Returns
    -------
    tf_array            an array of similarity matrix for transformation
    """
    tf_array = None

    if optical_control_img_file_path is not None:
        tf_array = np.loadtxt(optical_control_img_file_path, delimiter=',')
    elif (image_date is not None) & (system is not None):
        optical_control_path = os.path.join(optical_control_folder, system + '_' + image_date)
        path_exists = os.path.isdir(optical_control_path)
        if path_exists:
            optical_control_files = os.listdir(optical_control_path)
            for file in optical_control_files:
                if file.endswith(aligned_matrix_endstring):
                    align_file = file
                    break

            if align_file is not None:
                tf_array = np.loadtxt(os.path.join(optical_control_path, align_file), delimiter=',')
            else:
                print('cannot find file in plate: ' + system + '_' + image_date)

        if align_file is None:
            # Find it in argolight
            argo_path = os.path.join(optical_control_folder, 'ARGO-POWER', system, 'split_scenes', image_date)
            argo_exists = os.path.isdir(argo_path)
            if argo_exists:
                optical_control_files = os.listdir(argo_path)
                for file in optical_control_files:
                    if file.endswith('sim_matrix.txt'):
                        align_file = file
                        break
                if align_file is not None:
                    tf_array = np.loadtxt(os.path.join(argo_path, align_file), delimiter=',')
                else:
                    print('cannot find file in argo: ' + system + '_' + image_date)
    else:
        print('missing image date and system inputs')

    return tf_array
