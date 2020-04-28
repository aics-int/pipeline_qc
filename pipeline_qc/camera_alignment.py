import sys
import logging
import argparse
import traceback

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


log = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s')

# Initialize variables
# image_path = r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD1\3500003331_100X_20190813_psf.czi'
# image_type = 'rings'
# ref_channel_index = 'EGFP'
# mov_channel_index = 'CMDRP'
# bead_488_lower_thresh = 99.4
# bead_638_lower_thresh = 99
# method_logging = True
# align_mov_img = True
# align_mov_img_path = r''
# align_mov_img_file_extension = '_aligned.tif'
# align_matrix_file_extension = '_sim_matrix.txt'


class Args(object):
    def __init__(self, log_cmdline=True):

        self.__parse()

        if self.debug:
            log.setLevel(logging.DEBUG)
            log.debug("-" * 80)
            self.show_info()
            log.debug("-" * 80)

    def __parse(self):
        # Todo: Add --help
        p = argparse.ArgumentParser()
        # Add arguments
        p.add_argument('--img-path', required=True, dest='image_path')
        p.add_argument('--img-type', required=True, dest='image_type')
        p.add_argument('--ref-ch-index', required=True, dest='ref_channel_index')
        p.add_argument('--mov-ch-index', required=True, dest='mov_channel_index')
        p.add_argument('--align-mov-img', required=True, default=False, dest='align_mov_img')
        p.add_argument('--align-mov-img-path', required=True, dest='align_mov_img_path')
        p.add_argument('--align-mov-img-file_extension', required=True, default='_aligned.tif',
                       dest='align_mov_img_file_extension')
        p.add_argument('--align-matrix-file-extension', required=True, default='_sim_matrix.txt',
                       dest='align_matrix_file_extension')
        p.add_argument('--thresh_488', required=False, default=99.4, dest='thresh_488')
        p.add_argument('--thresh_638', required=False, default=99, dest='thresh_638')
        p.add_argument('--method-debug', required=False, default=True, dest='method_logging')


class Executor(object):
    def __init__(self, image_path, image_type, ref_channel_index, mov_channel_index, thresh_488,
                 thresh_638, method_logging, align_mov_img, align_mov_img_path, align_mov_img_file_extension,
                 align_matrix_file_extension, crop_center):
        self.image_path = image_path
        self.image_type = image_type
        self.ref_channel_index = ref_channel_index
        self.mov_channel_index = mov_channel_index
        self.thresh_488 = thresh_488
        self.thresh_638 = thresh_638
        self.method_logging = method_logging
        self.align_mov_img = align_mov_img
        self.align_mov_img_path = align_mov_img_path
        self.align_mov_img_file_extension = align_mov_img_file_extension
        self.align_matrix_file_extension = align_matrix_file_extension
        self.crop_center = crop_center

    def append_file_name_with_ext(self, image_path, align_mov_img_file_extension):
        """
        # Todo: Only works on Windows, need to change to linux-compatible in production!
        Appends the file name with extension
        :param image_path: Image path (windows)
        :param align_mov_img_file_extension: String of the extension
        :return:
            Path to image, appended with file extension
        """
        file_name = image_path.split('\\')[-1]
        new_name = file_name.split('.')[0] + align_mov_img_file_extension
        return os.path.join(image_path.split(file_name)[0], new_name)

    def report_number_beads(self, bead_dict, method_logging=True):
        """
        Reports the number of beads used to estimate transform
        :param bead_dict: A dictionary that each key is a bead
        :param method_logging: A boolean to indicate if user wants print statements
        :return:
            bead_num_qc: Boolean indicates if number of beads passed QC (>=10) or failed (<10)
            num_beads: An integer of number of beads used
        """
        bead_num_qc = False
        num_beads = len(bead_dict)
        if num_beads >= 10:
            bead_num_qc = True
        if method_logging:
            print('number of beads used to estimate transform: ' + str(num_beads))
        return bead_num_qc, num_beads

    def rescale_ref_mov(self, ref, mov, ref_thresh, mov_thresh, image_type):
        """
        Rescales reference and moving image
        :param ref: 2D reference image
        :param mov: 2D moving image
        :param ref_lower_thresh: Lower threshold to rescale on reference image
        :param mov_lower_thresh: Lower threshold to rescale on moving image
        :param image_type: 'beads' or 'rings'
        :return:
            ref_rescaled: rescaled reference image
            mov_rescaled: rescaled moving image
        """
        if image_type == 'beads':
            ref_rescaled = exp.rescale_intensity(ref,
                                                 out_range=np.uint8,
                                                 in_range=(np.percentile(ref, ref_thresh[0]),
                                                           np.percentile(ref, ref_thresh[1]))
                                                 )
            mov_rescaled = exp.rescale_intensity(mov,
                                                 out_range=np.uint8,
                                                 in_range=(np.percentile(mov, mov_thresh[0]),
                                                           np.percentile(mov, mov_thresh[1]))
                                                 )

        elif image_type == 'rings':
            ref_rescaled = exp.rescale_intensity(ref, in_range=(np.percentile(ref, ref_thresh[0]),
                                                                np.percentile(ref, ref_thresh[1])))
            mov_rescaled = exp.rescale_intensity(mov, in_range=(np.percentile(mov, mov_thresh[0]),
                                                                np.percentile(mov, mov_thresh[1])))
        else:
            print('invalid image type')
        plt.figure()
        plt.imshow(ref_rescaled)
        plt.show()
        plt.figure()
        plt.imshow(mov_rescaled)
        plt.show()
        return ref_rescaled, mov_rescaled

    def perform_similarity_matrix_transform(self, img, matrix, output_path):
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
            io.imsave(output_path, after_transform)

    def get_ref_mov_img(self, ref_stack, mov_stack, crop_center=None):
        """
        Gets reference and moving (2D) image from a stack
        :param ref_stack: A 3D reference stack
        :param mov_stack: A 3D moving stack
        :return:
            ref: A 2D slice from reference stack
            mov: A 2D slice from moving stack
        """
        ref = None
        mov = None

        if len(ref_stack.shape) == 2:
            ref = ref_stack
        elif len(ref_stack.shape) == 3:
            ref_center_z, ref_max_intensity = Executor.get_center_slice(self, ref_stack)
            ref = ref_stack[ref_center_z, :, :]
        else:
            print('dimension of ref_stack does not fit')

        if len(mov_stack.shape) == 2:
            mov = mov_stack
        elif len(mov_stack.shape) == 3:
            if ref is not None:
                mov = mov_stack[ref_center_z, :, :]
            else:
                print('cannot find ref center')
        else:
            print('dimension of mov_stack does not fit')

        if self.crop_center is not None:
            ref = ref[self.crop_center[0]:(ref.shape[0] - self.crop_center[1]),
                  self.crop_center[0]:(ref.shape[1] - self.crop_center[1])]
            mov = mov[self.crop_center[0]:(mov.shape[0] - self.crop_center[1]),
                  self.crop_center[0]:(mov.shape[1] - self.crop_center[1])]

        print(ref.shape)
        print(mov.shape)
        return ref, mov

    def get_center_slice(self, stack):
        """
        Gets index of center z slice by finding the slice with max. sum intensity
        :param stack: A 3D (or 2D) image
        :return:
            center_z: index of center z slice
            max_intensity: the sum of intensity of that slice
        """
        center_z = 0
        max_intensity = 0
        for z in range(0, stack.shape[0]):
            sum_intensity = np.sum(stack[z, :, :])
            if sum_intensity >= max_intensity:
                center_z = z
                max_intensity = sum_intensity
        return center_z, max_intensity

    def assign_ref_to_mov(self, updated_ref_peak_dict, updated_mov_peak_dict):
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

    def change_coor_system(self, coor_dict):
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

    def check_beads(self, ref_mov_num_dict, ref_labelled_seg, mov_labelled_seg):
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

    def filter_big_beads(self, img, center=0, area=20):
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

    def filter_center_cross(self, label_seg, show_img=False):
        """
        from a labelled rings image, filter out where the center cross is (the biggest segmented object)
        :param label_seg: A labelled image
        :param show_img: A boolean to indicate if the user would like to show the peaks on the image
        :return:
            filter_label: A labelled image after filtering the center cross (center cross = 0)
            props_df: A dataframe from regionprops_table with columns ['label', 'centroid-0', 'centroid-y', 'area']
            cross_label: The integer label of center cross
        """
        props = measure.regionprops_table(label_seg, properties=['label', 'area', 'centroid'])
        props_df = pd.DataFrame(props)
        cross_label = props_df.loc[(props_df['area'] == props_df['area'].max()), 'label'].values.tolist()[0]

        filter_label = label_seg.copy()
        filter_label[label_seg==cross_label] = 0

        if show_img:
            plt.figure()
            plt.imshow(filter_label)
            plt.show()

        return filter_label, props_df, cross_label

    def initialize_peaks(self, seg, peak_list, show_img=False, img_shape=None):
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

    def match_peaks(self, ref_peak_dict, mov_peak_dict, dist_threshold=5):
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

        updated_ref_peak_dict = Executor.remove_peaks_in_dict(self, full_dict=ref_peak_dict, keys=remove_ref_peak)
        updated_mov_peak_dict = Executor.remove_peaks_in_dict(self, full_dict=mov_peak_dict, keys=remove_mov_peak)

        return updated_ref_peak_dict, updated_mov_peak_dict

    def process_beads(self, ref_smooth, mov_smooth):
        """
        Carry out the processes to generate coordinate dictionaries from beads image
        :param ref_smooth: A reference beads image that was smoothed
        :param mov_smooth:A moving beads image that was smoothed
        :return:
            updated_ref_peak_dict: A dictionary of reference peak labels and coordinates ({label: (coor_y, coor_x)})
            ref_distances: A list of distances of coordinates of reference peak intensity and centroid for each bead
            ref_centroid_dict: A dictionary of reference centroid labels and coordinates ({label: (coor_y, coor_x)})
            updated_mov_peak_dict: A dictionary of moving peak labels and coordinates ({label: (coor_y, coor_x)})
            mov_distances: A list of distances of coordinates of moving peak intensity and centroid for each bead
            mov_centroid_dict: A dictionary of moving centroid labels and coordinates ({label: (coor_y, coor_x)})
            ref_labelled_seg: An image of labelled reference beads
            mov_labelled_seg: An image of labelled moving beads
        """
        filtered, seg_mov = Executor.filter_big_beads(self, mov_smooth)
        filtered, seg_ref = Executor.filter_big_beads(self, ref_smooth)

        # initialize intensity-based peaks
        ref_peaks = feature.peak_local_max(ref_smooth * seg_ref, min_distance=5)
        ref_peak_dict, ref_labelled_seg = Executor.initialize_peaks(self, seg=seg_ref, peak_list=ref_peaks,
                                                                    show_img=False, img_shape=seg_ref.shape)
        mov_peaks = feature.peak_local_max(mov_smooth * seg_mov, min_distance=5)
        mov_peak_dict, mov_labelled_seg = Executor.initialize_peaks(self, seg=seg_mov, peak_list=mov_peaks,
                                                                    show_img=False, img_shape=seg_mov.shape)

        # remove_close_peaks
        ref_close_peaks = Executor.remove_close_peaks(self, ref_peak_dict, dist_threshold=20, show_img=False,
                                                      img_shape=seg_ref.shape)
        mov_close_peaks = Executor.remove_close_peaks(self, mov_peak_dict, dist_threshold=20, show_img=False,
                                                      img_shape=seg_mov.shape)

        # remove peaks/beads that are too big
        ref_remove_overlap_peaks = Executor.remove_overlapping_beads(self, label_seg=ref_labelled_seg,
                                                                     peak_dict=ref_close_peaks,
                                                                     show_img=False)
        mov_remove_overlap_peaks = Executor.remove_overlapping_beads(self, label_seg=mov_labelled_seg,
                                                                     peak_dict=mov_close_peaks,
                                                                     show_img=False)

        # match peaks
        updated_ref_peak_dict, updated_mov_peak_dict = Executor.match_peaks(self,
                                                                            ref_peak_dict=ref_remove_overlap_peaks,
                                                                            mov_peak_dict=mov_remove_overlap_peaks,
                                                                            dist_threshold=5)

        # remove inconsistent intensity vs centroid beads
        updated_ref_peak_dict, ref_distances, ref_centroid_dict = Executor.remove_intensity_centroid_inconsistent_beads(
            self, label_seg=ref_labelled_seg, updated_peak_dict=updated_ref_peak_dict)
        updated_mov_peak_dict, mov_distances, mov_centroid_dict = Executor.remove_intensity_centroid_inconsistent_beads(
            self, label_seg=mov_labelled_seg, updated_peak_dict=updated_mov_peak_dict)

        # assign updated_ref_peak_dict with updated_mov_peak_dict
        # bead_peak_intensity_dict, ref_mov_num_dict, ref_mov_coor_dict = assign_ref_to_mov(updated_ref_peak_dict, updated_mov_peak_dict)

        return updated_ref_peak_dict, ref_distances, ref_centroid_dict, updated_mov_peak_dict, mov_distances, \
               mov_centroid_dict, ref_labelled_seg, mov_labelled_seg

    def process_rings(self, ref_smooth, mov_smooth):
        """
        Carry out the processes to generate coordinate dictionaries from rings image
        :param ref_smooth: A reference rings image that was smoothed
        :param mov_smooth: A moving rings image that was smoothed
        :return:
            ref_centroid_dict: A dictionary of reference centroid labels and coordinates ({label: (coor_y, coor_x)})
            mov_centroid_dict:A dictionary of moving centroid labels and coordinates ({label: (coor_y, coor_x)})
            filtered_label_ref: An image of labelled reference rings
            filtered_label_mov: An image of labelled moving rings
        """
        seg_ref, label_ref = Executor.segment_rings(self, ref_smooth, mult_factor=2.5, show_seg=True)
        seg_mov, label_mov = Executor.segment_rings(self, mov_smooth, mult_factor=2.5, show_seg=True)
        plt.figure()
        plt.imshow(seg_mov)
        plt.show()

        # filter center cross
        filtered_label_ref, props_ref, cross_label_ref = Executor.filter_center_cross(self, label_ref, show_img=False)
        filtered_label_mov, props_mov, cross_label_mov = Executor.filter_center_cross(self, label_mov, show_img=False)

        # make dictionary
        ref_centroid_dict = Executor.rings_coor_dict(self, props_ref, cross_label_ref)
        mov_centroid_dict = Executor.rings_coor_dict(self, props_mov, cross_label_mov)

        return ref_centroid_dict, mov_centroid_dict, filtered_label_ref, filtered_label_mov

    def remove_close_peaks(self, peak_dict, dist_threshold=20, show_img=False, img_shape=None):
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

    def remove_intensity_centroid_inconsistent_beads(self, label_seg, updated_peak_dict, dist_thresh=3):
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
            remove_inconsistent_dict = Executor.remove_peaks_in_dict(self, full_dict=updated_peak_dict,
                                                                     keys=bead_to_remove)
            bead_centroid_dict = Executor.remove_peaks_in_dict(self,
                                                               full_dict=bead_centroid_dict, keys=bead_to_remove)
        else:
            remove_inconsistent_dict = updated_peak_dict.copy()

        return remove_inconsistent_dict, distances, bead_centroid_dict

    def remove_overlapping_beads(self, label_seg, peak_dict, area_tolerance=0.3, show_img=False):
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

        remove_overlapping_beads = Executor.remove_peaks_in_dict(self, peak_dict, beads_to_remove)

        if show_img:
            img = np.zeros(label_seg.shape)
            for bead in list(remove_overlapping_beads.keys()):
                img[label_seg == bead] = label

            plt.figure()
            plt.imshow(img)
            plt.show()

        return remove_overlapping_beads

    def remove_peaks_in_dict(self, full_dict, keys):
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

    def remove_small_objects(self, label_img, filter_px_size=100):
        filtered_seg = np.zeros(label_img.shape)
        for obj in range(1, np.max(label_img)+1):
            obj_size = np.sum(label_img == obj)
            if obj_size > filter_px_size:
                filtered_seg[label_img == obj] = True
        filtered_label = measure.label(filtered_seg)

        return filtered_seg, filtered_label

    def report_similarity_matrix_parameters(self, tform, method_logging=True):
        """
        Reports similarity matrix and its parameters
        :param tform: A transform generated from skimage.transform.estimate_transform
        :param method_logging: A boolean to indicate if printing/logging statements is selected
        :return: A dictionary with the following keys and values:
            transform: A transform to be applied to moving images
            scaling: Uniform scaling parameter
            shift_y: Shift in y
            shift_x: Shift in x
            rotate_angle: Rotation angle
        """
        similarity_matrix_param_dict = {
            'transform': tform,
            'scaling': tform.scale,
            'shift_y': tform.translation[0],
            'shift_x': tform.translation[1],
            'rotate_angle': tform.rotation
        }

        if method_logging:
            for param, value in similarity_matrix_param_dict.items():
                print(param + ': ' + str(value))

        return similarity_matrix_param_dict

    def report_change_fov_intensity_parameters(self, transformed_img, original_img, method_logging=True):
        """
        Reports changes in FOV intensity after transform
        :param transformed_img: Image after transformation
        :param original_img: Image before transformation
        :param method_logging: A boolean to indicate if printing/logging statements is selected
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

        if method_logging:
            for key, value in change_fov_intensity_param_dict.items():
                print('change in ' + key + ': ' + str(value))

        return change_fov_intensity_param_dict

    def report_changes_in_coordinates_mapping(self, ref_mov_coor_dict, tform, ref, method_logging=True):
        """
        Report changes in beads (center of FOV) centroid coordinates before and after transform. A good transform will
        reduce the difference in distances, or at least not increase too much (thresh=5), between transformed_mov_beads and
        ref_beads than mov_beads and ref_beads. A bad transform will increase the difference in distances between
        transformed_mov_beads and ref_beads.
        :param ref_mov_coor_dict: A dictionary mapping the reference bead coordinates and moving bead coordinates (before transform)
        :param tform: A skimage transform object
        :param method_logging: A boolean to indicate if printing/logging statements is selected
        :return:
        """
        transform_qc = False
        mov_coors = list(ref_mov_coor_dict.values())
        ref_coors = list(ref_mov_coor_dict.keys())
        mov_transformed_coors = tform(mov_coors)

        dist_before_list = []
        dist_after_list = []
        for bead in range(0, len(mov_coors)):
            dist_before = distance.euclidean(mov_coors[bead], ref_coors[bead])
            dist_after = distance.euclidean(mov_transformed_coors[bead], ref_coors[bead])
            dist_before_list.append(dist_before)
            dist_after_list.append(dist_after)

        # filter center beads only
        y_size = 360
        x_size = 536

        y_lim = (int(ref.shape[0] / 2 - y_size / 2), int(ref.shape[0] / 2 + y_size / 2))
        x_lim = (int(ref.shape[1] / 2 - x_size / 2), int(ref.shape[1] / 2 + x_size / 2))

        dist_before_center = []
        dist_after_center = []
        for bead in range(0, len(mov_coors)):
            if (y_lim[1] > mov_coors[bead][0]) & (mov_coors[bead][0] > y_lim[0]):
                if (x_lim[1] > mov_coors[bead][1]) & (mov_coors[bead][1] > x_lim[0]):
                    dist_before_center.append(distance.euclidean(mov_coors[bead], ref_coors[bead]))
                    dist_after_center.append(distance.euclidean(mov_transformed_coors[bead], ref_coors[bead]))
        average_before_center = sum(dist_before_center) / len(dist_before_center)
        average_after_center = sum(dist_after_center) / len(dist_after_center)

        if method_logging:
            # print('average distance in center beads before: ' + str(average_before_center))
            # print('average distance in center beads after: ' + str(average_after_center))
            if (average_after_center - average_before_center) < 5:
                print('transform looks good - ')
                print('diff. in distance before and after transform: ' + str(average_after_center - average_before_center))
            elif (average_after_center - average_before_center) >= 5:
                print('no difference in distances before and after transform')
            else:
                print('transform looks bad - ')
                print('diff. in distance before and after transform: ' + str(average_after_center - average_before_center))

        if (average_after_center - average_before_center) < 5:
            transform_qc = True

        return transform_qc, (average_after_center - average_before_center)

    def report_changes_in_mse(self, ref_smooth, mov_smooth, mov_transformed, image_type, rescale_thresh_mov=None,
                              method_logging=True):
        """
        Report changes in normalized root mean-squared-error value before and after transform, post-segmentation.
        :param image_type: 'rings' or 'beads
        :param ref_smooth: Reference image, after smoothing
        :param mov_smooth: Moving image, after smoothing, before transform
        :param mov_transformed: Moving image after transform
        :param rescale_thresh_mov: A tuple to rescale moving image before segmentation. No need to rescale for rings image
        :param method_logging: A boolean to indicate if printing/logging statements is selected
        :return:
            qc: A boolean to indicate if it passed (True) or failed (False) qc
            diff_mse: Difference in mean squared error
        """
        qc = False

        if image_type == 'rings':
            seg_ref, label_ref = Executor.segment_rings(self, ref_smooth)
            seg_mov, label_mov = Executor.segment_rings(self, mov_smooth)
            seg_transformed, label_transform = Executor.segment_rings(self, filters.gaussian(mov_transformed, sigma=1))
        elif image_type == 'beads':
            mov_transformed_rescaled = exp.rescale_intensity(mov_transformed, out_range=np.uint8,
                                                             in_range=(np.percentile(mov_transformed, rescale_thresh_mov[0]),
                                                                       np.percentile(mov_transformed, rescale_thresh_mov[1])
                                                                       ))
            filtered, seg_ref = Executor.filter_big_beads(self, ref_smooth)
            filtered, seg_mov = Executor.filter_big_beads(self, mov_smooth)
            filtered, seg_transformed = Executor.filter_big_beads(self,
                                                                  filters.gaussian(mov_transformed_rescaled, sigma=1))

        mse_before = metrics.mean_squared_error(seg_ref, seg_mov)
        mse_after = metrics.mean_squared_error(seg_ref, seg_transformed)

        print('before: ' + str(mse_before))
        print('after: ' + str(mse_after))
        diff_mse = mse_after - mse_before
        if diff_mse <= 0:
            qc = True

        if method_logging:
            if diff_mse < 0:
                print('transform is good - ')
                print('mse is reduced by ' + str(diff_mse))
            elif diff_mse == 0:
                print('transform did not improve or worsen - ')
                print('mse differences before and after is 0')
            else:
                print('transform is bad - ')
                print('mse is increased by ' + str(diff_mse))

        return qc, diff_mse

    def rings_coor_dict(self, props, cross_label):
        """
        Generate a dictionary from regionprops_table in the form of {label: (coor_y, coor_x)} for rings image
        :param props: a dataframe containing regionprops_table output
        :param cross_label: Integer value representing where the center cross is in the rings image
        :return:
            img_dict: A dictionary of label to coordinates
        """
        img_dict = {}
        for index, row in props.iterrows():
            if row['label'] is not cross_label:
                img_dict.update({row['label']: (row['centroid-0'], row['centroid-1'])})

        return img_dict

    def verify_peaks(self, ref_peak_dict, mov_peak_dict, initialize_value=100):
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

    def segment_rings(self, smooth_img, mult_factor=2.5, show_seg=False):
        """
        segment rings using a cutoff at median+filter*std
        :param mult_factor: A float to adjust threshold to segment (multiplication factor to standard deviation)
        :param smooth_img: An image after smoothing
        :param show_seg: A boolean to show image
        :return:
            seg: segmentation of rings
            labelled_seg: labelled segmentation of rings
        """
        # thresh = filters.threshold_li(smooth_img)
        thresh = np.median(smooth_img) + mult_factor*np.std(smooth_img)
        seg = np.zeros(smooth_img.shape)
        seg[smooth_img >= thresh] = True

        labelled_seg = measure.label(seg)

        filtered_seg, filtered_label = Executor.remove_small_objects(self, labelled_seg, filter_px_size=50)
        if show_seg:
            plt.figure()
            plt.imshow(filtered_seg)
            plt.show()
        return filtered_seg, filtered_label


    def watershed_bead_seg(self, seg):
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
        local_maxi = feature.peak_local_max(distance, indices=False, footprint=morphology.disk(radius),
                                            labels=measure.label(seg))
        markers = ndimage.label(local_maxi)[0]
        labels = segmentation.watershed(-distance, markers, mask=seg)

        return labels

    def execute(self):

        # image_path = args.image_path
        # image_type = str(args.image_type)
        # ref_channel_index = str(args.ref_channel_index)
        # mov_channel_index = str(args.mov_channel_index)
        # bead_488_lower_thresh = float(args.bead_488_lower_thresh)
        # bead_638_lower_thresh = float(args.bead_638_lower_thresh)
        # method_logging = bool(args.method_logging)
        # align_mov_img = bool(args.align_mov_img)
        # align_mov_img_path = args.align_mov_img_path
        # align_mov_img_file_extension = str(args.align_mov_img_file_extension)
        # align_matrix_file_extension = str(args.align_matrix_file_extension)

        # read image
        img = AICSImage(self.image_path)
        channels = img.get_channel_names()

        # split ref and move channels from image
        ref_stack = img.data[0, 0, channels.index(self.ref_channel_index), :, :, :]
        mov_stack = img.data[0, 0, channels.index(self.mov_channel_index), :, :, :]

        #=======================================================================================================================
        # Pre-process images
        # rescale intensity
        ref, mov = Executor.get_ref_mov_img(self, ref_stack=ref_stack, mov_stack=mov_stack)
        ref_rescaled, mov_rescaled = Executor.rescale_ref_mov(self, ref=ref, mov=mov,
                                                              ref_thresh=self.thresh_488,
                                                              mov_thresh=self.thresh_638,
                                                              image_type=self.image_type)

        # smooth image
        ref_smooth = filters.gaussian(ref_rescaled, sigma=1, preserve_range=True)
        mov_smooth = filters.gaussian(mov_rescaled, sigma=1, preserve_range=True)

        #=======================================================================================================================
        # Process structures (segment, filter, assign)
        if self.image_type == 'beads':
            updated_ref_peak_dict, ref_distances, ref_centroid_dict, updated_mov_peak_dict, mov_distances, mov_centroid_dict, \
            labelled_ref, labelled_mov = Executor.process_beads(self, ref_smooth, mov_smooth)
        elif self.image_type == 'rings':
            ref_centroid_dict, mov_centroid_dict, labelled_ref, labelled_mov = Executor.process_rings(self,
                                                                                                      ref_smooth,
                                                                                                      mov_smooth)
        else:
            print('invalid image type')

        # assign centroid_dicts from ref to mov
        bead_centroid_dict, ref_mov_num_dict, ref_mov_coor_dict = Executor.assign_ref_to_mov(self, ref_centroid_dict,
                                                                                             mov_centroid_dict)

        debug_mode = False
        if debug_mode:
            Executor.check_beads(self, ref_mov_num_dict, labelled_ref, labelled_ref)

        # Change (y, x) into (x, y) for transformation matrix reading
        rev_coor_dict = Executor.change_coor_system(self, ref_mov_coor_dict)

        #=======================================================================================================================
        # Initiate transform estimation
        tform = tf.estimate_transform('similarity',
                                      np.asarray(list(rev_coor_dict.keys())), np.asarray(list(rev_coor_dict.values())))
        mov_transformed = tf.warp(mov, inverse_map=tform, order=3)

        # Report number of beads used to estimate transform
        bead_num_qc, num_beads = Executor.report_number_beads(self, bead_centroid_dict,
                                                              method_logging=self.method_logging)

        # Report transform parameters
        transformation_parameters_dict = Executor.report_similarity_matrix_parameters(self, tform=tform,
                                                                                      method_logging=self.method_logging)

        # Report intensity changes in FOV after transform
        changes_fov_intensity_dictionary = Executor.report_change_fov_intensity_parameters(self,
                                                                                           transformed_img=mov_transformed,
                                                                                           original_img=mov,
                                                                                           method_logging=self.method_logging)

        # Report changes in source and destination
        coor_dist_qc, diff_sum_beads = Executor.report_changes_in_coordinates_mapping(self,
                                                                                      ref_mov_coor_dict=ref_mov_coor_dict,
                                                                                      tform=tform, ref=ref,
                                                                                      method_logging=self.method_logging)

        # Report changes in nrmse in the image (after segmentation)
        mse_qc, diff_mse = Executor.report_changes_in_mse(self,
                                                          ref_smooth=ref_smooth, mov_smooth=mov_smooth,
                                                          mov_transformed=mov_transformed,
                                                          rescale_thresh_mov=self.thresh_638,
                                                          image_type=self.image_type, method_logging=self.method_logging)

        # Save metrics
        # Todo: Check with SW, in what format to save transform to be applied to pipeline images and saved in image metadata?
        np.savetxt(Executor.append_file_name_with_ext(self, image_path=self.align_mov_img_path,
                                                      align_mov_img_file_extension=self.align_matrix_file_extension),
                   tform.params, delimiter=',')
        print ('here')
        if self.align_mov_img:
            Executor.perform_similarity_matrix_transform(self, img=mov_stack, matrix=tform,
                                                         output_path=Executor.append_file_name_with_ext(
                                                             self, image_path=self.align_mov_img_path,
                                                             align_mov_img_file_extension=self.align_mov_img_file_extension))

        return transformation_parameters_dict, bead_num_qc, num_beads, changes_fov_intensity_dictionary, coor_dist_qc, diff_sum_beads, mse_qc, diff_mse

def main():
    dbg = False
    try:
        # args = Args()
        # dbg = args.debug
        exe = Executor(image_path=r'\\allen\aics\microscopy\Calysta\test\argo_3i\20200312\Capture 1 - Position 2_XY1584023869_Z00_T0_C0.tiff',
                       image_type='rings',
                       ref_channel_index='488/TL 50um Dual',
                       mov_channel_index='640/405 50um Dual',
                       # for beads
                       # thresh_488=(99.4, 100)
                       # thresh_638=(99, 100)
                       # crop_center=None
                       # for zsd rings
                       # thresh_488=(0.2, 99.8)
                       # thresh_638=(0.2, 99.8)
                       # crop_center=None
                       # for 3i rings
                       thresh_488=(0.2, 99.8),
                       thresh_638=(0.2, 99.8),
                       crop_center=(60, 60),
                       method_logging=True,
                       align_mov_img=True,
                       align_mov_img_path=r'\\allen\aics\microscopy\Calysta\test\argo_3i\20200312\Capture 1 - Position 2_XY1584023869_Z00_T0_C0.tiff',
                       align_mov_img_file_extension='_aligned.tif',
                       align_matrix_file_extension='_sim_matrix.txt')
        exe.execute()

    except Exception as e:
        log.error("===============")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("===============")
        log.error("\n\n" + str(e) + "\n")
        log.error("===============")
        sys.exit(1)


# if __name__ == "__main__":
#    main()
