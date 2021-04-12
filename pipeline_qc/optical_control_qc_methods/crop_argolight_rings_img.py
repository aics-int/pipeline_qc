import math
from pipeline_qc.optical_control_qc_methods import segment_argolight_rings

class Executor(object):
    def __init__(self, img, bead_dist_px, filter_pixel_size=50):
        self.img = img
        self.bead_dist_px = bead_dist_px
        self.filter_pixel_size = filter_pixel_size

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

    def execute(self):
        seg_cross, props = segment_argolight_rings.Executor.segment_rings_intensity_threshold(
            self.img, self.filter_px_size, show_seg=False
        )

        cross_y, cross_x = props.loc[props['area'] == props['area'].max(), 'centroid-0'].values.tolist()[0], \
                           props.loc[props['area'] == props['area'].max(), 'centroid-1'].values.tolist()[0]

        crop_top, crop_bottom, crop_left, crop_right = self.get_crop_dimensions(
            self, self.img, cross_y, cross_x, self.bead_dist_px
        )

        crop_dimensions = (crop_top, crop_bottom, crop_left, crop_right)
        img_out = self.img[crop_top:crop_bottom, crop_left:crop_right]

        return img_out, crop_dimensions