import numpy as np
from skimage import metrics
from scipy.spatial import distance
from pipeline_qc.optical_control_qc_methods import get_center_z


def calculate_crop_image_size(image_shape, crop_threshold=0.5, size_threshold=(365, 560)):
    image_size_y, image_size_x = image_shape
    if image_size_y * crop_threshold > size_threshold[0]:
        crop_size_y = int(image_size_y * crop_threshold)
    else:
        crop_size_y = size_threshold[0]

    if image_size_x * crop_threshold > size_threshold[1]:
        crop_size_x = int(image_size_x * crop_threshold)
    else:
        crop_size_x = size_threshold[1]

    return (crop_size_y, crop_size_x)


def check_z_offest_between_ref_mov(ref_stack, mov_stack, method_logging):
    org_ref_center, org_ref_max_i = get_center_z.Executor(stack=ref_stack)
    org_mov_center, org_mov_max_i = get_center_z.Executor(stack=mov_stack)

    if method_logging:
        print('z offset between ref and mov images: ' + str(org_ref_center - org_mov_center))
    return org_ref_center - org_mov_center, org_ref_center, org_mov_center


def get_image_snr(seg, img_intensity):
    signal = np.median(img_intensity[seg.astype(bool)])
    noise = np.median(img_intensity[~seg.astype(bool)])

    return signal, noise

def report_number_beads(bead_dict, method_logging=True):
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


def report_ref_mov_image_snr(ref, mov, ref_seg, mov_seg, method_logging):
    ref_signal, ref_noise = get_image_snr(seg=ref_seg, img_intensity=ref)
    mov_signal, mov_noise = get_image_snr(seg=mov_seg, img_intensity=mov)

    if method_logging:
        print('ref img snr: ' + str(ref_signal / ref_noise))
        print('mov img snr: ' + str(mov_signal / mov_noise))

    return ref_signal, ref_noise, mov_signal, mov_noise


def report_change_fov_intensity_parameters(image_a, image_b, method_logging=True):
    """
    Reports changes in FOV intensity after transform
    :param image_a: image to compare (a)
    :param image_b: image to compare (b)
    :param method_logging: A boolean to indicate if printing/logging statements is selected
    :return: A dictionary with the following keys and values:
        median_intensity
        min_intensity
        max_intensity
        1_percentile: first percentile intensity
        995th_percentile: 99.5th percentile intensity
    """
    change_fov_intensity_param_dict = {
        'median_intensity': np.median(image_a) - np.median(image_b),
        'min_intensity': np.min(image_a) - np.min(image_b),
        'max_intensity': np.max(image_a) - np.max(image_b),
        '1st_percentile': np.percentile(image_a, 1) - np.percentile(image_b, 1),
        '995th_percentile': np.percentile(image_a, 99.5) - np.percentile(image_b, 99.5)
    }

    if method_logging:
        for key, value in change_fov_intensity_param_dict.items():
            print('change in ' + key + ': ' + str(value))

    return change_fov_intensity_param_dict


def report_changes_in_mse(image_ref, image_mov, image_transformed, method_logging=True):
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

    mse_before = metrics.mean_squared_error(image_ref, image_mov)
    mse_after = metrics.mean_squared_error(image_ref, image_transformed)

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


def report_changes_in_coordinates_mapping(ref_mov_coor_dict, tform, image_shape, method_logging=True):
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
    crop_dim = calculate_crop_image_size(image_shape)

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

    y_lim = (int(image_shape[0] / 2 - crop_dim[0] / 2), int(image_shape[0] / 2 + crop_dim[0] / 2))
    x_lim = (int(image_shape[1] / 2 - crop_dim[1] / 2), int(image_shape[1] / 2 + crop_dim[1] / 2))

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


