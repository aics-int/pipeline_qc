import numpy as np
import matplotlib.pyplot as plt
import math
from aicsimageio import AICSImage
from scipy import interpolate, optimize
from scipy.optimize import curve_fit
from skimage import filters, measure, io
import xml.etree.ElementTree as ET

channel = '405'

# Read images (flat field, black reference, argolight)
ff_f_data = AICSImage(r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ZSD1_20180925\3500002315_100X_20180925_' + channel + '.czi')
ff_f = ff_f_data.data[0, 0, 0, :, :]
br_data = AICSImage(r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ZSD1_20180925\3500002315_100X_20180925_BR.czi')
br = br_data.data[0, 0, 0, :, :]
argo_data = AICSImage(r'C:\Users\calystay\Desktop\argo_488_test.czi')
argo = argo_data.data[0, 0, 0, : ,:]

# Get image information
img_dict = get_img_info(img=ff_f, data=ff_f_data)

# Pre-process flat field images
ff_smooth = filters.gaussian(image=ff_f, sigma=1, preserve_range=True)
ff_norm = ff_smooth/np.max(ff_smooth)

# Plot profiles for flat field images
plot_profile(ff_f, px_crop=0, fit=False)  # Intensity profile of raw dye ff image
plot_profile(norm_img=ff_norm, px_crop=0, fit=False)  # Intensity profile of normalized dye ff

# ======================================================================================================================
# Generate simulated homogeneity map with dye ff (sample across the image)

# Sample a dye-ff image with points
img_mask = np.zeros((ff_norm.shape), dtype=bool)
for x in range (50, img_mask.shape[1], 100):
    for y in range (50, img_mask.shape[0], 100):
        img_mask[y:y+25, x:x+25] = True
masked_ff = ff_smooth*img_mask
label_ref = measure.label(img_mask)

field_non_uni_raw, z, coors = generate_homogeneity_ref(label_ref=label_ref, img_raw=ff_smooth, mode='mean')
norm_corr = field_non_uni_raw/np.max(field_non_uni_raw)
plot_profile(norm_corr, px_crop=100, fit=False)  # Intensity profile of normalized simulated sampled ff

# ======================================================================================================================
# Use a simulated homogeneity map to correct images (requires black reference, homogeneity map)
smooth_br = filters.gaussian(br, sigma=1, preserve_range=True)

# Use sampled FF to correct full FF
corr_ff = correct_img(ff_f[100:-100, 100:-100], smooth_br[100:-100, 100:-100], field_non_uni_raw[100:-100, 100:-100])
plt.figure()
plt.imshow(corr_ff)
norm_corr_ff = corr_ff/np.max(corr_ff)
plot_profile(norm_img=norm_corr_ff, px_crop=12, fit=False)

# Use sampled FF to correct for argo
corr_argo = correct_img(argo[100:-100, 100:-100], smooth_br[100:-100, 100:-100], field_non_uni_raw[100:-100, 100:-100])
plt.figure()
plt.imshow(corr_argo)

# ======================================================================================================================
# Generate homogeneity maps from argolight
# Option 1: homogeneity_raw_map.png output from daybook
compare = io.imread(r'\\allen\aics\microscopy\Calysta\argolight\zsd1_20190813\homogeneity_raw_map.png')
compare_norm = compare/np.max(compare)
plot_profile(compare_norm, 100, fit=False)

# Option 2: From rings image, segment just the rings as samples
argo_smooth = filters.gaussian(argo, sigma=3, preserve_range=True) # TODO: change filter, don't fill holes in argolight!
thresh = filters.threshold_local(argo_smooth, block_size=11, offset=10)
ff_argo_segment = (argo_smooth>thresh) & (argo_smooth>500)

show = argo_smooth*ff_argo_segment
plt.figure()
plt.imshow(show)
label_ref = measure.label(ff_argo_segment)

field_non_uni_raw, z, coors = generate_homogeneity_ref(label_ref=label_ref, img_raw=argo_smooth, mode='median')
norm_f = field_non_uni_raw/np.max(field_non_uni_raw)
plot_profile(norm_f, px_crop=100)  # Intensity profile of normalized simulated rings ff

# Option 3: From rings image, segment rings with the centered cross
update = label_ref.copy()
update[label_ref == 15] = 0

masked_update = argo_smooth*update
field_non_uni_raw, z, coors = generate_homogeneity_ref(label_ref=update, img_raw=argo_smooth, mode='median')
norm_f = field_non_uni_raw/np.max(field_non_uni_raw)
plot_profile(norm_f, px_crop=100)  # Intensity profile of normalized simulated rings+cross ff

# ======================================================================================================================
# 2D fit over (sampled) flat field images
# Option 1: Fit the sampled image with a 2D paraboloid function
sampled_img = masked_ff

# Gather inputs for curve fit: x, y, z
label_ref = measure.label(img_mask)
field_non_uni_raw, z, coors = generate_homogeneity_ref(label_ref=label_ref, img_raw=ff_smooth, mode='median')
x = []
y = []
for coor in coors:
    x.append(coor[1])
    y.append(coor[0])

# Generate mesh grid to fit over
all_x = np.arange(0, np.shape(sampled_img)[1])
all_y = np.arange(0, np.shape(sampled_img)[0])
xx, yy = np.meshgrid(all_x, all_y, sparse=True)

# Fit (x,y), z to a 2d paraboloid function
params_paraboloid, cov_paraboloid = curve_fit(f=fit_2d_paraboloid, xdata=(y, x), ydata=z)

fit = fit_2d_paraboloid((yy, xx), *params_paraboloid)
fit_paraboloid_field_non_uni_raw = fit.reshape(624, 924)
norm = fit_paraboloid_field_non_uni_raw/np.max(fit_paraboloid_field_non_uni_raw)
plot_profile(norm, px_crop=0)

# Option 2: Fit the normalized image (more data points) with a 2D gaussian function
# 2D gaussian fit cannot find parameters for small sample size (tested with image with only 20 data points).
# Only use 2D gaussian fit for more data points (e.g. normalized ff)

params_gaussian = fitgaussian(ff_norm)
fit_function = gaussian(*params_gaussian)
fit_gaussian_field_non_uni_raw = fit_function(*np.indices(ff_norm.shape))
norm = fit_gaussian_field_non_uni_raw/np.max(fit_gaussian_field_non_uni_raw)
plot_profile(norm, px_crop=0)

# ======================================================================================================================
# Extract metrics from homogeneity map:
# 1) pos_roll_off, 2) neg_roll_off, 3) img_roll_off, 4) range_y_x_0.1_roll_off, 5) centroid_position

homogeneity_map = fit_gaussian_field_non_uni_raw  # Select which method to use as homogeneity reference
metric_dict = report_metric(homogeneity_map=homogeneity_map, roll_off_range=0.1)
metric_dict_2 = report_metric(homogeneity_map=ff_norm, roll_off_range=0.1)

# Test for correction
corr = correct_img(ff_norm, smooth_br, homogeneity_map)
plt.figure()
plt.imshow(corr, cmap='gray')
corr_norm = corr/np.max(corr)
plot_profile(corr_norm)
metric_corr = report_metric(homogeneity_map=corr, roll_off_range=0.1)

# ======================================================================================================================
# Functions developed


def get_img_info (img, data):
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
        plt.title('roll-off for ' + channel + ': ' + str(np.min([roll_off_neg, roll_off_pos])))

        if fit:
            popt_neg, pcov_neg = curve_fit(f=fit_func, xdata=x_data, ydata=negative_crop)
            popt_pos, pcov_pos = curve_fit(f=fit_func, xdata=x_data, ydata=positive_crop)
            plt.plot(fit_func(x_data, *popt_neg), 'r-')
            plt.plot(fit_func(x_data, *popt_pos), 'b-')

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


def fit_func(x, a, b, c):
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
