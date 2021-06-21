import matplotlib.pyplot as plt
import numpy as np

from skimage import exposure as exp, filters, measure, feature
import pandas as pd
from scipy.spatial import distance

class Executor(object):
    def __init__(self, img, thresh, filter_beads=True):
        self.img = img

        if thresh is not None:
            self.thresh = thresh
        else:
            self.thresh = (99, 100)

        self.filter_beads = filter_beads

    def preprocess_img(self):
        """
        Pre-process image with raw-intensity with rescaling and smoothing using pre-defined parameters from image
        magnification information
        Returns
        -------
        smooth: smooth image
        """
        rescale = exp.rescale_intensity(self.img, in_range=(np.percentile(self.img, self.thresh[0]),
                                                            np.percentile(self.img, self.thresh[1])))
        smooth = filters.gaussian(rescale, sigma=1, preserve_range=False)

        return smooth

    def filter_big_beads(self, img, center=0, area=20):
        """
        Find and filter big beads from an image with mixed beads
        Parameters
        ----------
        img: 3d image with big and small beads
        center: center slice
        area: area in pixel cutoff of a big bead

        Returns
        -------
        filtered: A 3d image where big beads are masked out as 0
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

    def initialize_peaks(self, seg, peak_list, show_img=False, img_shape=None):
        """
        Initializes the mapping of bead label (from segmentation) and the peak intensity coordinate (from finding peaks)
        Parameters
        ----------
        seg: A binary segmentation of beads
        peak_list: A list of peaks ([(y, x), (y2, x2), ...])
        show_img: A boolean to indicate if the user would like to show the peaks on the image
        img_shape: A tuple of the size of the image (y_dim, x_dim)

        Returns
        -------
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

    def remove_close_peaks(self, peak_dict, dist_threshold=20, show_img=False, img_shape=None):
        """
        Removes peaks on one image that are closer to each other than a distance threshold
        Parameters
        ----------
        peak_dict: A dictionary of peaks ({peak_number: (coor_y, coor_x)})
        dist_threshold: Number of pixels as distance threshold
        show_img: A boolean to indicate if the user would like to show the peaks on the image
        img_shape: A tuple of the size of the image (y_dim, x_dim)

        Returns
        -------
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
        Parameters
        ----------
        label_seg: A labelled segmentation image of beads
        updated_peak_dict: A dictionary of beads and peak intensities
        dist_thresh: Number of pixels as distance threshold

        Returns
        -------
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

        Parameters
        ----------
        label_seg: A labelled segmentation image of beads
        peak_dict: A dictionary of bead number to peak coordinates
        area_tolerance: Arbitrary float to set threshold of size
        show_img: A boolean to show image

        Returns
        -------
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

    def execute(self):
        smooth = Executor.preprocess_img(self)

        # filter out big beads from small beads
        filtered, seg = Executor.filter_big_beads(self, smooth)

        # initialize peaks
        peaks = feature.peak_local_max(smooth * seg, min_distance=5)
        peak_dict, labelled_seg = Executor.initialize_peaks(self, seg=seg, peak_list=peaks,
                                                                    show_img=False, img_shape=seg.shape)

        if self.filter_beads:
            # remove_close_peaks
            remove_close_peaks = Executor.remove_close_peaks(self, peak_dict, dist_threshold=20, show_img=False,
                                                          img_shape=seg.shape)

            remove_overlap_peaks = Executor.remove_overlapping_beads(self, label_seg=labelled_seg,
                                                                         peak_dict=remove_close_peaks,
                                                                         show_img=False)
            return labelled_seg, remove_overlap_peaks
        else:
            return labelled_seg, peak_dict

