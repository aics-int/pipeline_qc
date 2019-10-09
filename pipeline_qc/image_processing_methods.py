import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
from scipy import interpolate, ndimage, optimize
from scipy.optimize import curve_fit
from skimage import filters, measure
import xml.etree.ElementTree as ET


def get_img_info(img, data):
    """

    :param img: a hxw image array
    :param data: original data read from AICSImage
    :return: A dictionary containing descriptions of the image (intensity, focus position in um)
    """
    meta = data.metadata
    settings = meta.find("Metadata").getchildren()
    hw_setting = settings[1]

    for param_coll in hw_setting.getchildren():
        if param_coll.attrib == {'Id': 'MTBFocus'}:
            for info in param_coll.getchildren():
                if info.tag == 'Position':
                    position = info.text

    max = np.max(img)
    min = np.min(img)
    median = np.median(img)
    mean = np.average(img)
    std = np.std(img)

    return {'img_max': max, 'img_min': min, 'img_median': median, 'img_mean': mean, 'img_std': std,
            'z_position': position
            }

def plot_profile(norm_img, px_crop=0, plot=True, fit=False):
    """

    :param norm_img: A normalized image (intensity ranges from 0-1)
    :param px_crop: An integer to crop intensity profile from pixels
    :param fit: A boolean to provide the option of curve fitting over intensity profile
    :return: A plot showing the intensity profile
    """
    positive_profile = measure.profile_line(image=norm_img, src=(norm_img.shape[0], 0),
                                            dst=(0, norm_img.shape[1]))
    negative_profile = measure.profile_line(image=norm_img, src=(norm_img.shape[0], norm_img.shape[1]),
                                            dst=(0, 0))

    if px_crop==0:
        positive_crop = positive_profile
        negative_crop = negative_profile
    else:
        positive_crop = positive_profile[px_crop:-px_crop]
        negative_crop = negative_profile[px_crop:-px_crop]

    roll_off_pos = find_roll_off(positive_crop[5:-5])
    roll_off_neg = find_roll_off(negative_crop[5:-5])
    x_data = np.linspace(0, 1, len(negative_crop))

    if plot:
        plt.figure()
        plt.ylim((0, 1))
        if px_crop != 0:
            plt.xlim(px_crop, len(negative_profile)-px_crop)
        plt.plot(negative_crop, 'r')
        plt.plot(positive_crop, 'b')
        plt.title('roll-off: ' + str(np.min([roll_off_neg, roll_off_pos])))

        if fit:
            popt_neg, pcov_neg = curve_fit(f=fit_1d_parabola, xdata=x_data, ydata=negative_crop)
            popt_pos, pcov_pos = curve_fit(f=fit_1d_parabola, xdata=x_data, ydata=positive_crop)
            plt.plot(fit_1d_parabola(x_data, *popt_neg), 'r-')
            plt.plot(fit_1d_parabola(x_data, *popt_pos), 'b-')

    return positive_profile, negative_profile, roll_off_pos, roll_off_neg


def generate_homogeneity_ref(label_ref, img_raw, mode):
    """

    :param label_ref: An image with objects labelled (mostly used when ff is sampled, and only want to take
                      measurements over the centroid of object location
    :param img_raw: An image with intensities to map over with
    :param mode: Method to set intensity of the object (mean, median, max)
    :return: A field non homogeneity raw map, z = intensity values of points used as data for interpolation,
             coors = set of coordinates used as data for interpolation
    """
    props = measure.regionprops(label_ref)
    all_coors = []
    for x in range(0, label_ref.shape[1]):
        for y in range(0, label_ref.shape[0]):
            coor = (int(y), int(x))
            all_coors.append(coor)
    z = []
    coors = []
    for prop in props:
        centroid = prop.centroid
        obj_label = prop.label
        intensities = img_raw[np.where(label_ref == obj_label)]
        mean_int = np.average(intensities)
        max_int = np.max(intensities)
        if mode == 'mean':
            z.append(mean_int)
        elif mode == 'max':
            z.append(max_int)
        elif mode == 'median':
            z.append(np.median(intensities))

        coors.append((int(centroid[0]), int(centroid[1])))
    print (coors)
    grid_0 = interpolate.griddata(points=coors, values=z, xi=all_coors, method='linear', fill_value=False)

    field_non_uni_raw = np.zeros((label_ref.shape))
    for x in range(0, len(grid_0)):
        point = all_coors[x]
        value = grid_0[x]
        field_non_uni_raw[point] = value

    return field_non_uni_raw, z, coors


def find_roll_off(profile):
    """

    :param profile: A list of intensity values over a profile
    :return: A roll off value (0-1) across the line profile
    """
    roll_off = (np.max(profile) - np.min(profile))/np.max(profile)
    return roll_off


def fit_1d_parabola(x, a, b, c):
    """

    :param x: x data points
    :param a: weight to
    :param b: shift of location of maximum intensity
    :param c: constant, expected to be 1 if the input image was normalized
    :return: A 1d fit function for an intensity profile
    """
    return c - a * ((x - b) ** 2)


def correct_img(img_to_corr, br, img_homogeneity_ref):
    """

    :param img_to_corr: An image to be corrected
    :param br: A black reference image
    :param img_homogeneity_ref: A field non homogeneity image
    :return: A corrected image
    """
    corr = (img_to_corr - br)/(img_homogeneity_ref - br)
    return corr


def fit_2d_paraboloid(xdata_tuple, a, b, c, d, e, f):
    """

    :param xdata_tuple: a tuple of (x,y) data points
    :param a: weight
    :param b: shift in x
    :param c: weight
    :param d: shift in y
    :param e: weight
    :param f: constant
    :return: A 2d paraboloid fit function for a field non homogeneity image
    """
    (y, x) = xdata_tuple
    g = a*(x-b)**2 + c*(y-d)**2 + e*x*y + f
    return g.ravel()


def gaussian(height, center_x, center_y, width_x, width_y):
    """
    Referenced from https://scipy-cookbook.readthedocs.io/items/FittingData.html
    :param height: amplitude
    :param center_x: xo
    :param center_y: yo
    :param width_x: x
    :param width_y: y
    :return: a gaussian function with the given parameters
    """

    width_x = float(width_x)
    width_y = float(width_y)

    return lambda x, y: height * np.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)


def moments(data):
    """
    Referenced from https://scipy-cookbook.readthedocs.io/items/FittingData.html
    :param data: image to fit over (e.g. ff_norm)
    :return: the gaussian parameters of a 2D distribution by calculating its
        moments (height, x, y, width_x, width_y)
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):
    """
    Referenced from https://scipy-cookbook.readthedocs.io/items/FittingData.html
    :param data: image to fit over (e.g. ff_norm)
    :return: gaussian parameters of a 2D distribution found by a fit (height, x, y, width_x, width_y)
    """
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def report_metric(homogeneity_map, roll_off_range):
    """
    Metric: 1) pos_roll_off, 2) neg_roll_off, 3) img_roll_off, 4) range_y_x_0.1_roll_off, 5) centroid_position,
     6) angle/magnitude from center (x_axis), 7_ centering accuracy (daybook)
    :param homogeneity_map: A field homogeneity map, could be un-normalized
    :param roll_off_range: A float of roll off of interest, e.g. 0.1 roll off from max
    :return: A dictionary of metrics
    """
    norm_map = homogeneity_map / np.max(homogeneity_map)
    pos_prof, neg_prof, pos_roll_off, neg_roll_off = plot_profile(norm_img=norm_map, px_crop=0, plot=False, fit=False)
    img_roll_off = (np.max(homogeneity_map) - np.min(homogeneity_map)) / np.max(homogeneity_map)

    roll_off_area = (1.-roll_off_range)*np.max(homogeneity_map)
    y_ro, x_ro = np.where(homogeneity_map > roll_off_area)
    box_range = [np.min(y_ro), np.max(y_ro), np.min(x_ro), np.max(x_ro)]

    hot_spot = homogeneity_map > roll_off_area
    hot_spot_coverage = float(np.sum(hot_spot))/(homogeneity_map.shape[0]*homogeneity_map.shape[1])

    hot_spot = hot_spot.astype(int)  # TODO: peanut-shaped centroid?
    props = measure.regionprops(hot_spot)

    centroid_position = props[0].centroid
    y_dist_from_center = centroid_position[0] - int(np.shape(homogeneity_map)[0] / 2)
    x_dist_from_center = centroid_position[1] - int(np.shape(homogeneity_map)[1] / 2)

    mag = math.sqrt(y_dist_from_center**2 + x_dist_from_center**2)
    angle = math.degrees(math.atan2(x_dist_from_center, y_dist_from_center))

    centering_accuracy = 1. - ((2. * mag) / (math.sqrt(np.shape(homogeneity_map)[0]**2 + np.shape(homogeneity_map)[1]**2)))

    return {'pos_roll_off': pos_roll_off,
            'neg_roll_off': neg_roll_off,
            'img_roll_off': img_roll_off,
            'roll_off_intensity': roll_off_area,
            'px_range': box_range,
            'hot_spot_coverage': hot_spot_coverage,
            'hot_spot_center': centroid_position,
            'hot_spot_deviation': (y_dist_from_center, x_dist_from_center),
            'hot_spot_magnitude': mag,
            'hot_spot_angle': angle,
            'centering_accuracy': centering_accuracy
            }


def generate_images(image):
    """
    This function generates 6 images from a zstack
    :param image: an image with shape(T,C,Z,Y,X)
    :return: 6 images: highest_z, lowest_z, center_z, mip_xy, mip_xz, mip_yz
    """
    image_TL = image.data[0, 0, :, :, :]
    image_EGFP = image.data[0, 1, :, :, :]
    center_plane = find_center_z_plane(image_TL)
    # BF panels: top, bottom, center
    top_TL = image_TL[-1, :, :]
    bottom_TL = image_TL[0, :, :]
    center_TL = image_TL[center_plane, :, :]
    # EGFP panels: mip_xy, mip_xz, mip_yz
    mip_xy = np.amax(image_EGFP, axis=0)
    mip_xz = np.amax(image_EGFP, axis=1)
    mip_yz = np.amax(image_EGFP, axis=2)

    return top_TL, bottom_TL, center_TL, mip_xy, mip_xz, mip_yz

def create_display_setting(rows, control_column, folder_path):
    """
    This function generates a dictionary of display setting to be applied to images
    :param rows:
    :param control_column:
    :param folder_path:
    :return:
    """
    display_dict = {}
    images = os.listdir(plate_path)
    for row in rows:
        print (row)
        display_settings = []
        for img_file in images:
            if img_file.endswith(row + control_column + '.czi'):
                image = AICSImage(os.path.join(plate_path, img_file), max_workers=1)
                print (img_file)
                image_EGFP = image.data[0, 1, :, :, :]
                mip_xy = np.amax(image_EGFP, axis=0)
                display_min, display_max = np.min(mip_xy), np.max(mip_xy)
                display_settings.append((display_min, display_max))
        display_minimum = int(round(np.mean([dis_min[0] for dis_min in display_settings])))
        display_maximum = int(round(np.mean([dis_max[1] for dis_max in display_settings])))
        display_dict.update({row: (display_minimum, display_maximum)})
    return display_dict

def find_center_z_plane(image):
    """

    :param image:
    :return:
    """
    mip_yz = np.amax(image, axis=2)
    mip_gau = filters.gaussian(mip_yz, sigma=2)
    edge_slice = filters.sobel(mip_gau)
    contours = measure.find_contours(edge_slice, 0.005)
    new_edge = np.zeros(edge_slice.shape)
    for n, contour in enumerate (contours):
        new_edge[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1

    # Fill empty spaces of contour to identify as 1 object
    new_edge_filled = ndimage.morphology.binary_fill_holes(new_edge)

    # Identify center of z stack by finding the center of mass of 'x' pattern
    z = []
    for i in range (100, mip_yz.shape[1]+1, 100):
        edge_slab= new_edge_filled[:, i-100:i]
        #print (i-100, i)
        z_center, x_center = ndimage.measurements.center_of_mass(edge_slab)
        z.append(z_center)

    z = [z_center for z_center in z if ~np.isnan(z_center)]
    z_center = int(round(np.median(z)))

    return (z_center)
