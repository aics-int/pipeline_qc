import math
import matplotlib.pyplot as plt
import numpy as np
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from skimage import transform as tf, exposure as exp, filters, measure, morphology, feature, io, segmentation, metrics
from skimage.morphology import remove_small_objects
import pandas as pd


class Executor(object):
    def __init__(self, img, pixel_size, magnification, thresh=None):
        self.img = img
        self.pixel_size = pixel_size
        self.magnification = magnification

        if thresh is not None:
            self.thresh = thresh
        elif self.magnification in [40, 63, 100]:
            self.thresh = (0.2, 99.8)
        else:
            self.thresh = (0.5, 99.5)

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

    def segment_cross_with_iteration(self, img, cross_size_px, mult_factor_range=(1, 5)):
        """
        Segments the center cross in the image through iterating the intensity-threshold parameter until one object
        greater than the expected cross size (in pixel) is segmented
        Parameters
        ----------
        img: image (intensity after smoothing)
        cross_size_px: expected size of center cross in pixels
        mult_factor_range: range of multiplication factor to determine threshold for segmentation

        Returns
        -------
        seg_cross: binary image of segmented cross
        props: dataframe describing the centroid location and size of the segmented cross
        """
        for mult_factor in np.linspace(mult_factor_range[1], mult_factor_range[0], 50):
            seg_for_cross, label_for_cross = self.segment_rings_intensity_threshold(self, img, mult_factor=mult_factor, show_seg=False)
            if (np.max(label_for_cross) >= 1) & (np.sum(label_for_cross) > cross_size_px):
                break

        filtered_label, props, cross_label = self.filter_center_cross(self, label_for_cross, show_img=False)
        seg_cross = label_for_cross == cross_label

        return seg_cross, props

    def segment_rings_intensity_threshold(self, img, filter_px_size=50, mult_factor=2.5, show_seg=False):
        """
        Segments rings using intensity-thresholded method
        Parameters
        ----------
        img: rings image (after smoothing)
        filter_px_size: any segmented below this size will be filtered out
        mult_factor: parameter to adjust threshold
        show_seg: boolean to display segmentation

        Returns
        -------
        filtered_seg: binary mask of ring segmentation
        filtered_label: labelled mask of ring segmentation
        """

        thresh = np.median(img) + mult_factor * np.std(img)
        seg = np.zeros(img.shape)
        seg[img >= thresh] = True

        labelled_seg = measure.label(seg)

        filtered_seg, filtered_label = Executor.remove_small_objects(self, labelled_seg, filter_px_size=filter_px_size)
        if show_seg:
            plt.figure()
            plt.imshow(filtered_seg)
            plt.show()
        return filtered_seg, filtered_label


    def segment_rings_dot_filter(self, img, seg_cross, num_beads, minArea, search_range=(0, 0.75), size_param=2.5, show_seg=False):
        """
        Segments rings using 2D dot filter from aics-segmenter. The method loops through a possible range of parameters
        and automatically detects the optimal filter parameter when it segments the number of expected rings objects
        Parameters
        ----------
        img: rings image (after smoothing)
        seg_cross: binary mask of the center cross object
        num_beads: expected number of beads
        minArea: minimum area of rings, any segmented object below this size will be filtered out
        search_range: initial search range of filter parameter
        size_param: size parameter of dot filter
        show_seg: boolean to show segmentation

        Returns
        -------
        seg: binary mask of ring segmentation
        label: labelled mask of ring segmentation
        thresh: filter parameter after optimization

        """
        thresh = None
        for seg_param in np.linspace(search_range[1], search_range[0], 500):
            s2_param = [[size_param, seg_param]]
            seg = dot_2d_slice_by_slice_wrapper(img, s2_param)[0, :, :]

            remove_small = remove_small_objects(seg > 0, min_size=minArea, connectivity=1, in_place=False)

            dilate = morphology.binary_dilation(remove_small, selem=morphology.disk(2))
            seg_rings = morphology.binary_erosion(dilate, selem=morphology.disk(2))

            seg = np.logical_or(seg_cross, seg_rings)
            label = measure.label(seg)

            if np.max(label) >= num_beads:
                thresh = seg_param
                break

            if show_seg:
                plt.figure()
                plt.imshow(seg)
                plt.show()

        return seg, label, thresh

    def filter_center_cross(self, label_seg, show_img=False):
        """
        filters out where the center cross (the biggest segmented object) is in a labelled rings image

        Parameters
        ----------
        label_seg: A labelled image
        show_img: A boolean to indicate if the user would like to show the peaks on the image

        Returns
        -------
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


    def get_number_rings(self, img, bead_dist_px, mult_factor=5):
        """
        Estimates the number of rings in a rings object using the location of the center cross
        Parameters
        ----------
        img: input image (after smoothing)
        bead_dist_px: calculated distance between rings in pixels
        mult_factor: parameter to segment cross with

        Returns
        -------
        num_beads: number of beads after estimation
        """
        # update cross info
        seg_cross, label_for_cross, props = self.segment_cross(self.img, mult_factor=mult_factor)


        # get number of beads from the location of center of cross
        cross_y, cross_x = props.loc[props['area'] == props['area'].max(), 'centroid-0'].values.tolist()[0], \
                           props.loc[props['area'] == props['area'].max(), 'centroid-1'].values.tolist()[0]

        num_beads = (math.floor(cross_y / bead_dist_px) + math.floor((img.shape[0] - cross_y) / bead_dist_px) + 1) * \
                    (math.floor(cross_x / bead_dist_px) + math.floor((img.shape[1] - cross_x) / bead_dist_px) + 1)

        return num_beads


    def execute(self):

        img_preprocessed = self.preprocess_img(self)

        num_beads = self.get_number_rings(self, mult_factor=5)

        cross_size_px = 7.5 * 6 * 10 ** -6 / self.pixel_size
        ring_size_px = math.pi * (0.7 * 10 ** -6 / self.pixel_size) ** 2
        minArea = ring_size_px * 0.8

        if self.magnification in [40, 63, 100]:
            seg_rings, ring_label = self.segment_rings_intensity_threshold(self, img=img_preprocessed)
        else:
            seg_cross, props = self.segment_cross_with_iteration(
                self, img=img_preprocessed, minArea=cross_size_px
            )

            seg_rings, ring_label, thresh = self.segment_rings_dot_filter(
                self, img=img_preprocessed, seg_cross=seg_cross, num_beads=num_beads, minArea=minArea)

        return seg_rings, ring_label

