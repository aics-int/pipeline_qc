from aicsimageio import AICSImage, omeTifWriter
from scipy import optimize, ndimage, spatial
from skimage import measure, morphology
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

pxl_xy = 0.108
pxl_z = 0.29

window_xy = 6
window_z = 6

def calculate_fwhm(sigma_x, pxl_size, sigma_y=0):
    """
    Calculate full-width-half-max from sigma (for gaussian curves only)
    :param sigma_x: sigma_x from a gaussian fit
    :param pxl_size: pixel size (um/px)
    :param sigma_y: (optional sigma for 2d gaussian fit)
    :return: full-width-half-max of a gaussian fit
    """
    if sigma_y is not 0:
        sigma_psf = (sigma_x + sigma_y) / 2 * pxl_size
    else:
        sigma_psf = sigma_x * pxl_size
    fwhm = sigma_psf * 2 * math.sqrt(2 * math.log(2))
    return fwhm


def calculate_psf_z(bead_img_3d, window_xy, window_z, sigma_x, sigma_y):
    """
    Calculate 1D psf in the z direction
    :param bead_img_3d: A 3d bead image
    :param window_xy: size of window_xy (1/2 of bead img height)
    :param window_z: size of window_z (1/2 of bead img height)
    :param sigma_x: sigma_x from gaussian 2d fit over plane xy
    :param sigma_y: sigma y from gaussian 2d fit over plane xy
    :return: A dictionary with 'fit' as the values over the fit, and 'par' as parameters output from fit
    """
    psf_z = bead_img_3d[:, window_xy, window_xy]
    z = np.arange(bead_img_3d.shape[0])

    par_init_psf_z = np.max(psf_z), np.min(psf_z), window_z, (sigma_x + sigma_y) / 2 * 0.108 * 3
    par_psf_z = create_gauss_1d_fit(z, psf_z, par_init_psf_z)
    # sigma_psf_z = par_psf_z[0, 3] * 0.29
    amp, offset, z0, sigma_z = par_psf_z[0]

    fit_z = gauss_1d(z, amp, offset, z0, sigma_z)

    return {'fit': fit_z, 'par': {'amp': amp, 'offset': offset, 'z0': z0, 'sigma_z': sigma_z}}


def gauss_1d(x, amp, offset, x0, sigma_x):
    """
    1d gaussian function
    :param x:
    :param amp:
    :param offset:
    :param x0:
    :param sigma_x:
    :return: Z values over a gaussian fitted curve
    """
    z = offset + amp * np.exp(-((x - x0) / sigma_x) ** 2)
    return z


def create_gauss_1d_fit(x, profile, par_init):
    """
    fit 1D rotated Gaussian to curve
    :param x: x-values
    :param profile: profile to be fitted over with
    :param par_init: initializing parameters
    :return: Parameters to fit over a 1d gaussian
    """
    popt, pcov = optimize.curve_fit(gauss_1d, x, profile, p0=par_init)
    return popt.reshape(1, 4)


def get_fit_parameters(grid, image):
    """
    Get fit parameters from an image for a gaussian fit
    :param grid: An empty xy grid
    :param image: An image to be fitted over with
    :return: Rough estimate of parameters to initialize a gaussian fit
    """
    x, y = grid
    # calculate initial fit parameters
    amp = image.max()
    offset = image.min()

    # calculate center of mass and covariance
    cov = np.zeros((2, 2))
    image_corr = image - offset;
    m00 = np.sum(image_corr)  # 0 th raw moment=total I of plane k
    m10 = np.sum(np.sum(image_corr, axis=0) * x[0, :])  # 1st order raw moment
    m01 = np.sum(np.sum(image_corr, axis=1) * y[:, 0])  # 1st order raw moment
    x0 = m10 / m00  # center of mass x0
    y0 = m01 / m00  # center of mass y0
    m11 = np.sum(image_corr * x * y)  # 1st order raw moment
    m20 = np.sum(np.sum(image_corr, axis=0) * x[0, :] * x[0, :])  # 2nd order raw moment
    m02 = np.sum(np.sum(image_corr, axis=1) * y[:, 0] * y[:, 0])  # 2nd order raw moment
    mu11 = m11 / m00 - x0 * y0
    mu20 = m20 / m00 - x0 ** 2
    mu02 = m02 / m00 - y0 ** 2
    cov[:, :] = np.array([[mu20, mu11], [mu11, mu02]])  # covariance matrix

    # calculate eigenvectors and eigenvalues of covariance matrix
    eigval, eigvec = np.linalg.eigh(cov)

    # calculate theta, sigma_x, sigma_y
    theta = np.arctan(eigvec[0, 0] / eigvec[1, 0])  # theta [rad]
    sigma_x = np.sqrt(eigval[0])
    sigma_y = np.sqrt(eigval[1])
    return amp, offset, x0, y0, theta, sigma_x, sigma_y


def gauss_2d_rot(grid, amp, offset, x0, y0, theta, sigma_x, sigma_y):
    """
    A rotational gaussian 2d function
    :param grid: xy grid
    :param amp:
    :param offset:
    :param x0:
    :param y0:
    :param theta:
    :param sigma_x:
    :param sigma_y:
    :return: z values (intensity) fitted over a 2d gaussian function
    """
    x, y = grid
    # 2D rotated Gaussian
    a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
    b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
    c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
    z = offset + amp*np.exp( - (a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    return np.ravel(z)


def create_gauss_2d_rot_fit(grid, image, par_init=None):
    """
    Perform a rotational 2d gaussian curve fit
    :param grid: xy grid
    :param image: image to be fitted over with
    :param par_init: initializing parameters
    :return: Parameters to fit over a rotational 2d gaussian
    """
    # fit 2D rotated Gaussian to image
    popt, pcov = optimize.curve_fit(gauss_2d_rot, grid, np.ravel(image), p0=par_init)
    return popt.reshape(1,7)


def find_z_center (img):
    """
    Find center slice, at focal plane with maximum std in intensity
    :param img: a 3d image
    :return: z-index of image slice in focus
    """
    std = 0
    center = 0
    for z in range (0, img.shape[0]):
        slice_std = np.std(img[z, :, :])
        if slice_std > std:
            std = slice_std
            center = z
    return center


def filter_big_beads(img, center, area):
    """
    Find and filter big beads from an image with mixed beads
    :param img: 3d image with big and small beads
    :param center: center slice
    :param area: area(px) cutoff of a big bead
    :return: filtered: A 3d image where big beads are masked out as 0
             seg_big_bead: A binary image showing segmentation of big beads
    """
    # Big beads are brighter than small beads usually
    seg_big_bead = img[center, :, :] > (np.median(img[center, :, :]) + 1.25 * np.std(img[center, :, :]))
    label_big_bead = measure.label(seg_big_bead)

    # Size filter the labeled big beads, that could be due to bright small beads
    for obj in range(1, np.max(label_big_bead)):
        size = np.sum(label_big_bead == obj)
        if size < area:
            seg_big_bead[label_big_bead == obj] = 0

    # dilate seg big beads to minimize effects of rings from big beads
    seg_big_bead = morphology.dilation(seg_big_bead, selem=morphology.disk(3))

    # Save filtered beads image after removing big beads as 'filtered'
    mask = np.zeros(img.shape)
    for z in range(0, img.shape[0]):
        mask[z] = seg_big_bead
    filtered = img.copy()
    filtered[np.where(mask == 1)] = np.median(img)
    return filtered, seg_big_bead


def find_small_beads(img_center, thresh_std=2):
    """
    Segment and find all small beads
    :param img_center: 2D image of focal plane
    :param thresh_std: threshold intensity cutoff to segment small beads
    :return: seg_small_bead: binary segmentation of small beads
             label_small_bead: labelled small beads
    """
    # Segment all small beads
    seg_small_bead = img_center > (np.median(img_center) + thresh_std * np.std(img_center))
    label_small_bead = measure.label(seg_small_bead)
    return seg_small_bead, label_small_bead

def measure_small_beads (label_img, intensity_img):
    """
    Measure properties of small beads
    :param label_img: Labelled image of small beads
    :param intensity_img: Intensity image of small beads
    :return: Labelled image will provide the id number of each bead, for bead_reg (CoM), bead_area (size),
    bead_center (location at brightest intensity), bead_intensity (max intensity of bead)
    """
    props = measure.regionprops(label_img, intensity_image=intensity_img)
    bead_reg = {}
    bead_area = {}
    bead_intensity = {}
    bead_center = {}
    for obj in props:
        bead_reg.update({int(obj.label): obj.centroid})
        bead_area.update({obj.label: obj.area})
        bead_intensity.update({obj.label: obj.max_intensity})

        loc_info = np.where((intensity_img == obj.max_intensity) & (label_img == obj.label))
        bead_center.update({obj.label: (loc_info[0][0], loc_info[1][0])})

    return bead_reg, bead_area, bead_center, bead_intensity


def filter_bead_by_size(label_img, binary_img, bead_area, bead_center):
    """
    Filter beads by size, remove beads that are 1.2> or 0.8< median of bead size
    :param label_img: Labelled image of small beads
    :param binary_img: Segmentation of small beads
    :param bead_area: Dictionary of bead_area {bead_id:size}
    :param bead_center: Dictionary of bead_center {bead_id:center_location}
    :return: Updated bead registration dictionary
    """
    update_bead_reg = bead_center.copy()
    for bead, area in bead_area.items():
        if (area > 1.2*np.median(list(bead_area.values()))) or (area < 0.8*np.median(list(bead_area.values()))):
            binary_img[label_img==bead] = 0
            del update_bead_reg[bead]
    return update_bead_reg


def filter_beads_close_to_big(seg_big_bead, update_bead_reg):
    """
    Filter beads that are close to big beads
    :param seg_big_bead: Segmentation of big beads
    :param update_bead_reg: Bead registration dictionary
    :return: Updated bead registration after removing beads
    """
    big_bead_loc = np.where(seg_big_bead == 1)
    big_bead_coor = []
    for px in range (0, len(big_bead_loc[0])):
        big_bead_coor.append((big_bead_loc[0][px], big_bead_loc[1][px]))

    # remove bead that is 10px from big beads
    bead_remove_size = update_bead_reg.copy()
    for bead, coor in bead_remove_size.items():
        dist = spatial.distance.cdist(XA=[coor], XB=big_bead_coor)
        min_dist = np.min(dist)
        if min_dist <= 10:
            del update_bead_reg[bead]
    return update_bead_reg


def filter_beads_close_to_small(update_bead_reg, bead_reg):
    """
    Filter beads that are clsoe to other small beads
    :param update_bead_reg: Bead registration dictionary
    :param bead_reg: Original bead registration with all beads
    :return: Updated bead registration after removing beads
    """
    # calculate distance with the nearest neighbor bead
    # remove beads that are 10px from each other
    bead_dist = {}
    bead_remove_close_beads = update_bead_reg.copy()
    for bead, coor in bead_remove_close_beads.items():
        dist = spatial.distance.cdist(XA=[coor], XB=list(bead_reg.values()))
        min_dist = np.min(dist[dist>0])
        bead_dist.update({bead: min_dist})
        if min_dist <= 10:
            del update_bead_reg[bead]
    return update_bead_reg


def filter_beads_intensity(bead_intensity, update_bead_reg):
    """
    Filter beads that are too dim or too bright
    :param bead_intensity: dictionary of {bead_id:max_intensity}
    :param update_bead_reg: Bead registration dictionary
    :return: Updated bead registration after removing beads
    """
    bead_remove_dim_bright = update_bead_reg.copy()
    lower_thresh = np.median(list(bead_intensity.values())) - 1*np.std(list(bead_intensity.values()))
    upper_thresh = np.median(list(bead_intensity.values())) + 3*np.std(list(bead_intensity.values()))
    for bead, coor in bead_remove_dim_bright.items():
        if (bead_intensity[bead] < lower_thresh) or (bead_intensity[bead] > upper_thresh):
            del update_bead_reg[bead]
    return update_bead_reg


def filter_beads_edge(img_h, img_w, crop, update_bead_reg):
    """
    Filter beads too close to the edges of FOV
    :param img_h: height of fov
    :param img_w: width of fov
    :param crop: px to crop
    :param update_bead_reg: Bead registration dictionary
    :return: Updated bead registration dictionary after removing beads
    """
    bead_remove_edge = update_bead_reg.copy()
    for bead, coor in bead_remove_edge.items():
        if (coor[0] < crop) or (coor[0] > img_h-crop) or (coor[1]<crop) or (coor[1]>img_w-crop):
            del update_bead_reg[bead]
    return update_bead_reg


def show_beads (seg, label, update_bead_reg):
    """

    :param seg: Binary segmentation of beads
    :param label: Labelled image of beads
    :param update_bead_reg: Bead registration
    :return: Showing a plot of final beads
    """
    # finalize bead selection
    final_beads = np.zeros(seg.shape)
    label_num = 1
    for bead in list(update_bead_reg.keys()):
        final_beads[label == bead] = label_num
        label_num += 1
    plt.figure()
    plt.imshow(final_beads)


def center_bead_3d_window (update_bead_reg, intensity_img, center, window_z=6, window_xy=6):
    """
    Create an array of all beads that are centered with shape (#_beads, z, y, x)
    :param update_bead_reg: bead registration dictionary
    :param intensity_img: bead intensity image
    :param window_z: window size of beads, slightly bigger than radius of bead (px)
    :param window_xy: window size of beads, slightly bigger than radius of bead (px)
    :return: an array of all beads that are centered with shape (#_beads, z, y, x)
    """
    bead_3d_window = {}
    for bead, coor in update_bead_reg.items():
        bead_3d_window.update({bead: intensity_img[center-window_z:center+window_z, (int(coor[0])-window_xy):(int(coor[0])+window_xy),
                                     (int(coor[1])-window_xy):(int(coor[1])+window_xy)]})
        std = 0
        center_cut = 0
        for slice in range (0, bead_3d_window[bead].shape[0]):
            slice_std = np.std(bead_3d_window[bead][slice, :, :])
            if slice_std > std:
                std = slice_std
                center_cut = slice
        if center_cut!=6:
            bead_3d_window[bead] = intensity_img[
                                   center + (center_cut - window_z) - window_z:center + (center_cut - window_z) + window_z,
                                   (int(coor[0]) - window_xy):(int(coor[0]) + window_xy),
                                   (int(coor[1]) - window_xy):(int(coor[1]) + window_xy)]
    return bead_3d_window


def get_psf_xy (img_2d):
    """
    Get PSF in xy
    :param img_2d: a 2d image with a bead
    :return: a dictionary with 'img' as the gaussian-fitted image, and 'par' for parameters to perform the fit
    """
    y, x = np.mgrid[0:img_2d.shape[0], 0:img_2d.shape[1]]
    par_init = get_fit_parameters((x, y), img_2d)
    try:
        par = create_gauss_2d_rot_fit((x, y), img_2d, par_init=par_init)
        amp, offset, x0, y0, theta, sigma_x, sigma_y = par[0]
        fit = gauss_2d_rot((x, y), amp, offset, x0, y0, theta, sigma_x, sigma_y)
        fit_img = fit.reshape(img_2d.shape[0], img_2d.shape[1])
    except:
        fit_img = amp = offset = x0 = y0 = theta = sigma_x = sigma_y = None
        pass
    return {'img': fit_img, 'par': {'amp':amp, 'offset':offset, 'x0':x0,
                                        'y0':y0, 'theta':theta, 'sigma_x':sigma_x, 'sigma_y':sigma_y}}


def calculate_average_psf (psf_xy_all):
    """
    Get psf across many beads with shape (#_beads, y, x)
    :param psf_xy_all: An array with many beads with shape (#_beads, y, x)
    :return: a dictionary with 'img' as the gaussian-fitted image, and 'par' for parameters to perform the fit
    """
    psf = np.sum(psf_xy_all, axis=0)/psf_xy_all.shape[0]
    fit_psf_avg_dict = get_psf_xy(psf)
    return fit_psf_avg_dict


prod_optical_control = '/allen/aics/microscopy/PRODUCTION/OpticalControl'
beads_output_folder = '/allen/aics/microscopy/PRODUCTION/OpticalControl/output/psf_output'
df = pd.DataFrame()
optical_control_folders = os.listdir(prod_optical_control)
failed_beads = pd.DataFrame()
failed_beads_fit = pd.DataFrame()

folder_ct = 0
total = len(optical_control_folders)
for optical_control_folder in optical_control_folders:
    #if optical_control_folder not in processed_folders:
    #if optical_control_folder in reprocess:
    print ('processing folder: ' + str(folder_ct) + ' out of ' + str(total))
    folder_ct += 1
    if optical_control_folder.startswith('ZSD1') or optical_control_folder.startswith('ZSD2') or optical_control_folder.startswith('ZSD3'):
        folder = os.path.join(prod_optical_control, optical_control_folder)
        print ('in folder: ' + optical_control_folder)
        files = os.listdir(folder)
        system = optical_control_folder.split('_')[0]

        file_name = None
        date = optical_control_folder.split('_')[1]

        for file in files:
            if file.endswith('psf.czi'):
                file_name = file

        if file_name is not None:
            file = file_name.split('.czi')[0]
            os.mkdir(os.path.join(beads_output_folder, system + '_' + date))
        else:
            beads = None

        try:
            print ('reading ' + folder + ' ' + file_name)
            beads = AICSImage(os.path.join(folder, file_name))
        except:
            beads = None
            failed_beads = failed_beads.append(row, ignore_index=True)
            print ('failed to read img in ' + folder)
            pass

        if beads is not None:
            beads_img_save_folder = os.path.join(beads_output_folder, system + '_' + date)

            channels = beads.get_channel_names()
            for channel in channels:
                if (channel.startswith('Bright')) or (channel.startswith('TL')):
                    channels.remove(channel)

            for channel in channels:
                print (channel)
                row = {'date': date,
                       'system': system,
                       'file_name': file_name,
                       'channel': channel}

                par_output = row.copy()
                img = beads.data[0, channels.index(channel), :, :, :]
                img_h = img.shape[1]
                img_w = img.shape[2]
                # Pre-processing beads, save bead in bead_3d_window
                center = find_z_center(img)
                img_center = img[center, :, :]

                # Separate big bead from small beads
                filtered, seg_big_bead = filter_big_beads(img=img, center=center, area=20)
                # Use small beads to find image center
                center = find_z_center(filtered)

                # Segment and measure small beads
                seg_small_beads, label_small_beads = find_small_beads(img_center=filtered[center, :, :], thresh_std=2)
                bead_reg, bead_area, bead_center, bead_intensity = measure_small_beads(label_img=label_small_beads,
                                                                                       intensity_img=filtered[center, :, :])
                update_bead_reg = {}
                # Filter small beads by properties to use good beads to calculate psf
                update_bead_reg = filter_bead_by_size(label_small_beads, seg_small_beads, bead_area, bead_center=bead_center)
                update_bead_reg = filter_beads_close_to_big(seg_big_bead=seg_big_bead, update_bead_reg=update_bead_reg)
                update_bead_reg = filter_beads_close_to_small(update_bead_reg=update_bead_reg, bead_reg=bead_center)
                update_bead_reg = filter_beads_intensity(bead_intensity=bead_intensity, update_bead_reg=update_bead_reg)
                update_bead_reg = filter_beads_edge(img_h=seg_small_beads.shape[0], img_w=seg_small_beads.shape[1], crop=12,
                                                    update_bead_reg=update_bead_reg)

                row.update({'total_num_beads': len(update_bead_reg)})

                if len(update_bead_reg) > 0:
                    # show_beads(seg=seg_small_beads, label=label_small_beads, update_bead_reg=update_bead_reg)
                    bead_3d_window = center_bead_3d_window(update_bead_reg=update_bead_reg, intensity_img=filtered,
                                                           window_z=window_z, window_xy=window_xy, center=center)
                else:
                    bead_3d_window = None
                    failed_beads = failed_beads.append(row, ignore_index=True)

                # ======================================================================================================================
                if bead_3d_window is not None:
                    try:
                        # Gather psf_xy for all beads, save parameters as json
                        psf_xy_all = np.zeros(shape=(len(bead_3d_window), window_xy*2, window_xy*2))
                        count=0
                        bead_psf_xy = {}
                        for bead, bead_img in bead_3d_window.items():
                            psf_dict = get_psf_xy(bead_img[window_z, :, :])
                            if psf_dict['img'] is not None:
                                psf_xy_all[count, :, :] = psf_dict['img']
                                bead_psf_xy.update({bead: psf_dict['par']})
                                count+=1
                        if count < len(bead_3d_window):
                            empty_slices = np.arange(count+1, len(bead_3d_window))
                            psf_xy_all = np.delete(psf_xy_all, obj=empty_slices, axis=0)

                        # Calculate average psf_xy
                        avg_psf_xy_dict = calculate_average_psf(psf_xy_all=psf_xy_all)
                        avg_fwhm_xy = calculate_fwhm(sigma_x=avg_psf_xy_dict['par']['sigma_x'], sigma_y=avg_psf_xy_dict['par']['sigma_y'],
                                                     pxl_size=pxl_xy)

                        row.update({'avg_fwhm_xy': avg_fwhm_xy})
                        par_output.update({'avg_par_xy': avg_psf_xy_dict['par']})
                        writer_psf_xy = omeTifWriter.OmeTifWriter(os.path.join(beads_img_save_folder, file + '_' + channel + '_psf_xy.tif'))
                        writer_psf_xy.save(np.reshape(avg_psf_xy_dict['img'].astype(np.uint16), (1, window_xy*2, window_xy*2)))

                        # Calculate goodness of fit: Is the psf gaussian?
                        diff = np.mean((np.mean(psf_xy_all, axis=0) - np.mean(np.mean(psf_xy_all, axis=0)) * (avg_psf_xy_dict['img'] - np.mean(avg_psf_xy_dict['img']))) / (
                                    np.std(np.mean(psf_xy_all, axis=0)) * np.std(avg_psf_xy_dict['img'])))

                        # Measure angle deviation from center with pearson correlation
                        angle_deviate = avg_psf_xy_dict['par']['theta']

                        row.update({'avg_fwhm_xy_gof': diff,
                                    'avg_fwhm_xy_tilt': angle_deviate})

                        # ======================================================================================================================
                        # Gather psf_3d for all beads, save parameters as json
                        psf_3d_all = np.zeros(shape=(len(bead_3d_window), window_z*2, window_xy*2, window_xy*2))
                        count_3d=0
                        for bead, img in bead_3d_window.items():
                            psf_3d_all[count_3d, :, :, :] = img
                            count_3d += 1

                        psf_3d = np.sum(psf_3d_all, axis=0)/psf_3d_all.shape[0]

                        max_intensity = np.max(psf_3d)
                        row.update({'avg_max_intensity': max_intensity})

                        avg_psf_z_dict = calculate_psf_z(psf_3d, window_xy, window_z, avg_psf_xy_dict['par']['sigma_x'],
                                                         avg_psf_xy_dict['par']['sigma_y'])
                        writer_psf_3d = omeTifWriter.OmeTifWriter(os.path.join(beads_img_save_folder, file + '_' + channel + '_psf3d.tif'))
                        writer_psf_3d.save(psf_3d.astype(np.uint16))

                        avg_fwhm_z = calculate_fwhm (sigma_x=avg_psf_z_dict['par']['sigma_z'], pxl_size=pxl_z)

                        par_output.update({'avg_par_3d': avg_psf_z_dict['par']})
                        row.update({'avg_fwhm_z': avg_fwhm_z})

                        # ======================================================================================================================
                        # Get psf for each bead, save parameters as json file
                        bead_psf = {}
                        # Gather psf_z for each bead
                        for bead, par in bead_psf_xy.items():
                            bead_img = bead_3d_window[bead]
                            bead_psf_z_dict = calculate_psf_z(bead_img, window_xy, window_z, par['sigma_x'],
                                                              par['sigma_y'])
                            bead_fwhm_xy = calculate_fwhm(sigma_x=par['sigma_x'], sigma_y=par['sigma_y'], pxl_size=pxl_xy)
                            bead_fwhm_z = calculate_fwhm(sigma_x=bead_psf_z_dict['par']['sigma_z'], pxl_size=pxl_z)

                            # Calculate asymmetry?
                            bead_psf.update({bead: {'par_xy':bead_psf_xy[bead], 'par_z':bead_psf_z_dict['par'],
                                                    'fwhm_xy':bead_fwhm_xy, 'fwhm_z':bead_fwhm_z}
                                             })

                        par_output.update({'per_bead_par': bead_psf})

                        # ======================================================================================================================
                        # Get average psf for each region, save parameters as json
                        box=3
                        flag = -1
                        org_y = 0
                        box_id = 1
                        box_id_name = {1:'topL', 3:'topR', 5:'center', 7:'botL', 9:'botR'}
                        axis_dict = {0:'yx', 1:'zx', 2:'zy'}
                        px_size_dict = {'yx':pxl_xy, 'zx':pxl_z, 'zy':pxl_z}
                        for y in range (int(img_h/box), img_h+1, int(img_h/box)):
                            org_x = 0
                            for x in range (int(img_w/box), img_w+1, int(img_w/box)):
                                upper_left = (org_y, org_x)
                                corner = (y, x)
                                flag = flag*-1
                                beads_in_box = []
                                if flag >0 :
                                    for bead, coor in update_bead_reg.items():
                                        if (coor[0] > upper_left[0]) & (coor[0] < corner[0]) & (coor[1] > upper_left[1]) & (coor[1]< corner[1]):
                                            beads_in_box.append(bead)
                                    psf_3d_box = np.zeros(shape=(len(beads_in_box), window_z * 2, window_xy * 2, window_xy * 2))
                                    count_bx=0
                                    for bead in beads_in_box:
                                        # Calculate overall psf
                                        psf_3d_box[count_bx, :, :, :] = bead_3d_window[bead]
                                        count_bx+=1

                                    psf_3d = np.sum(psf_3d_box, axis=0) / psf_3d_box.shape[0]
                                    psf_3d = psf_3d.astype(np.uint16)

                                    max_intensity = np.max(psf_3d)
                                    row.update({box_id_name[box_id] + '_avg_max_intensity': max_intensity})

                                    avg_psf_xy_dict = get_psf_xy(img_2d=psf_3d[window_z, :, :])

                                    avg_fwhm_xy = calculate_fwhm(sigma_x=avg_psf_xy_dict['par']['sigma_x'],
                                                                 sigma_y=avg_psf_xy_dict['par']['sigma_y'],
                                                                 pxl_size=pxl_xy)
                                    avg_psf_z_dict = calculate_psf_z(psf_3d, window_xy, window_z, avg_psf_xy_dict['par']['sigma_x'],
                                                                     avg_psf_xy_dict['par']['sigma_y'])
                                    par_output.update({box_id_name[box_id] + '_par_z': avg_psf_z_dict['par'],
                                                       box_id_name[box_id] + '_par_xy': avg_psf_xy_dict['par']})

                                    # Save psf_3d_box
                                    writer_psf_3d_box = omeTifWriter.OmeTifWriter(os.path.join(beads_img_save_folder, file + '_' + channel + '_3d_box_' + box_id_name[box_id] + '.tif'))
                                    writer_psf_3d_box.save(np.reshape(psf_3d, (window_z*2, window_xy*2, window_xy*2)))

                                    # For 3 axis, get fwhm, asymmetry
                                    for axis in range (0, 3):
                                        # axis=0
                                        max_proj = np.amax(psf_3d, axis=axis)
                                        psf_fit = get_psf_xy(img_2d=max_proj)
                                        fwhm = calculate_fwhm(sigma_x= psf_fit['par']['sigma_x'], pxl_size=px_size_dict[axis_dict[axis]], sigma_y=psf_fit['par']['sigma_y'])

                                        writer_axis = omeTifWriter.OmeTifWriter(os.path.join(beads_img_save_folder, file + '_' + channel + '_box_' + box_id_name[box_id] + '_' + axis_dict[axis] + '.tif'))
                                        writer_axis.save(np.reshape(max_proj, (1, window_xy*2, window_xy*2)))

                                        row.update({box_id_name[box_id] + '_' + axis_dict[axis] + '_fwhm': fwhm})
                                        par_output.update({box_id_name[box_id] + '_' + axis_dict[axis] + '_par': psf_fit['par']})

                                        # Calculate goodness of fit: Is the psf gaussian?
                                        diff = np.mean((max_proj-np.mean(max_proj) * (psf_fit['img'] - np.mean(psf_fit['img']))) / (np.std(max_proj) * np.std(psf_fit['img'])))

                                        # Measure angle deviation from center with pearson correlation
                                        angle_deviate = psf_fit['par']['theta']

                                        row.update({box_id_name[box_id] + '_' + axis_dict[axis] + '_gof': diff,
                                                    box_id_name[box_id] + '_' + axis_dict[axis] + '_tilt': angle_deviate})


                                box_id += 1
                                org_x += int(img_w/box)
                            org_y += int(img_h/box)

                        with open (os.path.join(beads_img_save_folder, file_name + '_parameters_test.json'), 'w') as f:
                            json.dump(par_output, f)

                        df = df.append(row, ignore_index=True)

                    except:
                        print ('failed to process beads in img' + file_name + ' for system ' + system + ' in C' + channel)
                        failed_beads_fit = failed_beads_fit.append(row, ignore_index=True)

                        df.to_csv(os.path.join(beads_output_folder, 'psf_data.csv'))
                        failed_beads.to_csv(os.path.join(beads_output_folder, 'failed_beads.csv'))
                        failed_beads_fit.to_csv(os.path.join(beads_output_folder, 'failed_beads_fit.csv'))
                        pass


df.to_csv(os.path.join(beads_output_folder, 'psf_data.csv'))
failed_beads.to_csv(os.path.join(beads_output_folder, 'failed_beads.csv'))
failed_beads_fit.to_csv(os.path.join(beads_output_folder, 'failed_beads_fit.csv'))
