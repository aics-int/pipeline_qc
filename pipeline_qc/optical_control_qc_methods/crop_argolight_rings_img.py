import math
import numpy as np
from pipeline_qc.optical_control_qc_methods import segment_argolight_rings
from skimage import measure
import pandas as pd


class Executor(object):
    def __init__(self, img, pixel_size, magnification, filter_px_size=50):
        self.img = img
        self.bead_dist_px = 15 / (pixel_size / 10 ** -6)
        self.filter_px_size = filter_px_size
        self.magnification = magnification

        self.show_seg = False

    def get_crop_dimensions(self, img, cross_y, cross_x, bead_dist_px, crop_param=0.5):
        """
        Calculates the crop dimension from the location of the cross to capture complete rings in the image
        Parameters
        ----------
        img: mxn nd-array image of rings
        cross_y: y location of center cross
        cross_x: x location of center cross
        bead_dist_px: distance between rings in pixels
        crop_param: a float between 0 and 1 that indicates a factor of distance between rings that should be left behind
            after cropping

        Returns
        -------
        crop_top: top pixels to keep
        crop_bottom: bottom pixels to keep
        crop_left: left pixels to keep
        crop_right: right pixels to keep
        """
        if cross_y % bead_dist_px > (bead_dist_px * crop_param):
            crop_top = 0
        else:
            crop_top = round(cross_y - (math.floor(cross_y / bead_dist_px) - (1 - crop_param)) * bead_dist_px)

        if (img.shape[0] - cross_y) % bead_dist_px > (bead_dist_px * crop_param):
            crop_bottom = img.shape[0]
        else:
            crop_bottom = img.shape[0] - \
                          round(img.shape[0] - (cross_y + (
                                  math.floor((img.shape[0] - cross_y) / bead_dist_px) - (1 - crop_param)) * bead_dist_px))

        if cross_x % bead_dist_px > (bead_dist_px * crop_param):
            crop_left = 0
        else:
            crop_left = round(cross_x - (math.floor(cross_x / bead_dist_px) - (1 - crop_param)) * bead_dist_px)

        if (img.shape[1] - cross_x) % bead_dist_px > (bead_dist_px * crop_param):
            crop_right = img.shape[1]
        else:
            crop_right = img.shape[1] - round(
                img.shape[1] - (
                        cross_x + (math.floor((img.shape[1] - cross_x) / bead_dist_px) - (1 - crop_param)) * bead_dist_px))

        return crop_top, crop_bottom, crop_left, crop_right

    def make_grid(self, img, cross_y, cross_x, bead_dist_px):
        grid = np.zeros(img.shape)
        for y in np.arange(cross_y, 0, -bead_dist_px):
            for x in np.arange(cross_x, 0, -bead_dist_px):
                grid[int(y), int(x)] = True
            for x in np.arange(cross_x, img.shape[1], bead_dist_px):
                grid[int(y), int(x)] = True
        for y in np.arange(cross_y, img.shape[0], bead_dist_px):
            for x in np.arange(cross_x, 0, -bead_dist_px):
                grid[int(y), int(x)] = True
            for x in np.arange(cross_x, img.shape[1], bead_dist_px):
                grid[int(y), int(x)] = True

        return grid

    def execute(self):
        seg_cross, props = segment_argolight_rings.Executor.segment_cross(
            self, self.img, input_mult_factor=2.5
        )

        cross_y, cross_x = props.loc[props['area'] == props['area'].max(), 'centroid-0'].values.tolist()[0], \
                           props.loc[props['area'] == props['area'].max(), 'centroid-1'].values.tolist()[0]

        if self.magnification < 63:
            crop_top, crop_bottom, crop_left, crop_right = Executor.get_crop_dimensions(
                self, self.img, int(cross_y), int(cross_x), self.bead_dist_px
            )
        else:
            crop_top = 0
            crop_left = 0
            crop_bottom = self.img.shape[0]
            crop_right = self.img.shape[1]

        crop_dimensions = (crop_top, crop_bottom, crop_left, crop_right)

        img_out = self.img[crop_top:crop_bottom, crop_left:crop_right]

        updated_cross_y = cross_y - crop_bottom
        updated_cross_x = cross_x - crop_left

        grid = Executor.make_grid(self, img_out, int(updated_cross_y), int(updated_cross_x), self.bead_dist_px)

        labelled_grid = measure.label(grid)
        props = measure.regionprops_table(labelled_grid, properties=['label', 'area', 'centroid'])
        props_grid = pd.DataFrame(props)
        center_cross_label = labelled_grid[int(updated_cross_y), int(updated_cross_x)]

        number_of_rings = len(props)

        return img_out, crop_dimensions, labelled_grid, props_grid, center_cross_label, number_of_rings