import numpy as np
from scipy import ndimage
from skimage import exposure, filters, morphology, measure


def segment_colony_area(bf, gaussian_thresh):
    """
    From a 2D bright field image (preferably the center slice), determine foreground vs background to segment
    colony area coverage
    :param bf: a 2D bright field image
    :param gaussian_thresh: a threshold cutoff for gaussian to separate foreground from background
    :return: a 2D segmented image
    """
    p2, p98 = np.percentile(bf, (2, 98))
    rescale = exposure.rescale_intensity(bf, in_range=(p2, p98))
    dist_trans = filters.sobel(rescale)
    gaussian_2 = filters.gaussian(dist_trans, sigma=15)

    mask = np.zeros(bf.shape, dtype=bool)
    mask[gaussian_2 <= gaussian_thresh] = True

    mask_erode = morphology.erosion(mask, selem=morphology.disk(3))
    remove_small = filter_small_objects(mask_erode, 750)
    dilate = morphology.dilation(remove_small, selem=morphology.disk(10))

    new_mask = np.ones(bf.shape, dtype=bool)
    new_mask[dilate == 1] = False
    return new_mask


def filter_small_objects(bw_img, area):
    """
    From a segmented image, filter out segmented objects smaller than a certain area threshold
    :param bw_img: a 2D segmented image
    :param area: an integer of object area threshold (objects with size smaller than that will be dropped)
    :return: a 2D segmented, binary image with small objects dropped
    """
    label_objects, nb_labels = ndimage.label(bw_img)
    sizes = np.bincount(label_objects.ravel())
    max_area = max(sizes)
    # Selecting objects above a certain size threshold
    # size_mask = (sizes > area) & (sizes < max_area)
    size_mask = (sizes > area)
    size_mask[0] = 0
    filtered = label_objects.copy()
    filtered_image = size_mask[filtered]

    int_img = np.zeros(filtered_image.shape)
    int_img[filtered_image == True] = 1
    int_img = int_img.astype(int)
    return int_img


def find_center_z_plane(image):
    mip_yz = np.amax(image, axis=2)
    mip_gau = filters.gaussian(mip_yz, sigma=2)
    edge_slice = filters.sobel(mip_gau)
    if edge_slice.shape[0] < 2:
        # This piece of code is due to error when edge_slice doesn't make an array at least 2x2 (1xX is what is made)
        print("Cannot find z_center plane")
        return 0
    else:
        contours = measure.find_contours(edge_slice, 0.005)
    new_edge = np.zeros(edge_slice.shape)
    for n, contour in enumerate(contours):
        new_edge[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1

    # Fill empty spaces of contour to identify as 1 object
    new_edge_filled = ndimage.morphology.binary_fill_holes(new_edge)

    # Identify center of z stack by finding the center of mass of 'x' pattern
    z = []
    for i in range(100, mip_yz.shape[1] + 1, 100):
        edge_slab = new_edge_filled[:, i - 100:i]
        # print (i-100, i)
        z_center, x_center = ndimage.measurements.center_of_mass(edge_slab)
        z.append(z_center)

    z = [z_center for z_center in z if ~np.isnan(z_center)]
    if np.isnan(z_center):
        print("Cannot find z_center plane")
        z_center = 0
    else:
        z_center = int(round(np.median(z)))
    return z_center
