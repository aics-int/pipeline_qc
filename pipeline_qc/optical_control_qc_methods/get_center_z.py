import numpy as np
import matplotlib.pyplot as plt

class Executor(object):
    def __init__(self, img_stack, thresh=(0.2, 99.8), plot_contrast=False):
        self.img_stack = img_stack
        self.thresh = thresh
        self.plot_contrast = plot_contrast

    def get_center_z(self):
        """
        Getx index of center z slice by finding the slice with max. contrast value
        Parameters
        ----------
        stack           a 3D (or 2D) image

        Returns
        -------
        center_z        index of center z-slice
        max_contrast    contrast of that slice
        """
        center_z = 0
        max_contrast = 0
        all_contrast = []
        for z in range(0, self.img_stack.shape[0]):
            contrast = (np.percentile(self.img_stack[z, :, :], self.thresh[1]) - np.percentile(self.img_stack[z, :, :], self.thresh[0])) / (
                np.max(self.img_stack[z, :, :]))
            all_contrast.append(contrast)
            if contrast > max_contrast:
                center_z = z
                max_contrast = contrast

        if self.plot_contrast:
            plt.figure()
            plt.plot(all_contrast)
            plt.show()

        return center_z, max_contrast

    def execute(self):
        center_z, max_contrast = Executor.get_center_z(self)
        return center_z, max_contrast