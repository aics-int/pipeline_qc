import os
import matplotlib.pyplot as plt
import numpy as np

from aicsimageio import AICSImage
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from skimage import transform as tf, exposure as exp, filters, measure, morphology, feature, io
import SimpleITK as sitk
from scipy.spatial import distance

# read beads image
beads = AICSImage(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD3\3500003332_100X_20190813_psf.czi')
beads_gfp = beads.data[0, 1, :, :, :]
beads_cmdr = beads.data[0, 3, :, :, :]

center_z=0
max_intensity=0
for z in range (0, beads_gfp.shape[0]):
    sum_intensity = np.sum(beads_gfp[z, :, :])
    if sum_intensity >= max_intensity:
        center_z = z
        max_intensity = sum_intensity


# pre-processing data, more processing to make the two similar
# rescale intensity
ref = exp.rescale_intensity(beads_gfp[center_z, :, :],
                            out_range=np.uint8,
                            in_range=(np.percentile(beads_gfp[center_z, :, :], 99.4), np.percentile(beads_gfp[center_z, :, :], 100))
                            )
mov = exp.rescale_intensity(beads_cmdr[center_z, :, :],
                            out_range=np.uint8,
                            in_range=(np.percentile(beads_cmdr[center_z, :, :], 99), np.percentile(beads_cmdr[center_z, :, :], 100))
                            )

# smooth image
ref_smooth = filters.gaussian(ref, sigma=1, preserve_range=True)
mov_smooth = filters.gaussian(mov, sigma=1, preserve_range=True)


#==========================================================
# segment beads image
# TODO: segment mov image until seg area and seg count is the same?
# TODO: use 3d segmentation?`
filtered, seg_mov = filter_big_beads(mov_smooth)
filtered, seg_ref = filter_big_beads(ref_smooth)

# initialize peaks
ref_peaks = feature.peak_local_max(ref_smooth*seg_ref, min_distance=5)
ref_peak_dict = initialize_peaks(peak_list=ref_peaks, show_img=False, img_shape=(624, 924))
mov_peaks = feature.peak_local_max(mov_smooth*seg_mov, min_distance=5)
mov_peak_dict = initialize_peaks(peak_list=mov_peaks, show_img=False, img_shape=(624, 924))

# remove_close_peaks
ref_close_peaks = remove_close_peaks(ref_peak_dict, dist_threshold=20, show_img=False, img_shape=(624, 924))
mov_close_peaks = remove_close_peaks(mov_peak_dict, dist_threshold=20, show_img=False, img_shape=(624, 924))

peak_img = np.zeros(seg_ref.shape)
for bead_id, coors in ref_close_peaks.items():
    peak_img[coors] = 1
plt.figure()

# match peaks
updated_ref_peak_dict, updated_mov_peak_dict = match_peaks(ref_peak_dict=ref_close_peaks, mov_peak_dict=mov_close_peaks,
                                                           dist_threshold=5)

# After remove keys, rearrange source and destination coordinates
src_dst_dict = verify_peaks(ref_peak_dict=updated_ref_peak_dict, mov_peak_dict=updated_mov_peak_dict,
                            initialize_value=100)

# Apply transform
tform = tf.estimate_transform('similarity', np.asarray(list(src_dst_dict.keys())), np.asarray(list(src_dst_dict.values())))
cmdr_transformed = tf.warp(beads_cmdr[center_z, :, :], inverse_map=tform._inv_matrix, order=3)


np.savetxt(r'C:\Users\calystay\Desktop\test_transform.csv', tform._inv_matrix, delimiter=',')
load_transform_array = np.loadtxt(r'C:\Users\calystay\Desktop\test_transform.csv', delimiter=',')

argo_array = np.array([[1.0003, 0.0026, -0.599],
                       [-0.0026, 1.0003, 0.7998],
                       [0, 0, 1]])

transformed_cmdr = np.zeros(beads_cmdr.shape)
argo_transformed_cmdr = np.zeros(beads_cmdr.shape)
for z in range (0, beads_cmdr.shape[0]):
    transformed_xy = tf.warp(beads_cmdr[z, :, :], inverse_map=load_transform_array, order=3)
    transformed_cmdr[z, :, :] = transformed_xy

    transformed_xy_argo = tf.warp(beads_cmdr[z, :, :], inverse_map=argo_array, order=3)
    argo_transformed_cmdr[z, :, :] = transformed_xy_argo

transformed_cmdr = (transformed_cmdr*65535).astype(np.uint16)
io.imsave(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD3\3500003332_100X_20190813_psf_cmdr_aligned.tif',
          transformed_cmdr)

argo_transformed_cmdr = (argo_transformed_cmdr*65535).astype(np.uint16)
io.imsave(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD3\3500003332_100X_20190813_psf_cmdr_aligned_argo.tif',
          argo_transformed_cmdr)

before_gfp = beads_gfp.astype(np.uint16)
io.imsave(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD3\3500003332_100X_20190813_psf_gfp.tif',
          before_gfp)

before_cmdr = beads_cmdr.astype(np.uint16)
io.imsave(r'\\allen\aics\microscopy\Calysta\argolight\data_set_to_share\ZSD3\3500003332_100X_20190813_psf_cmdr.tif',
          before_cmdr)


def verify_peaks(ref_peak_dict, mov_peak_dict, initialize_value=100):
    src_dst_dict = {}
    for mov_peak_id, mov_coor in mov_peak_dict.items():
        min_dist = initialize_value
        for ref_peak_id, ref_coor in ref_peak_dict.items():
            dist = distance.euclidean(mov_coor, ref_coor)
            if dist < min_dist:
                min_dist = dist
                map_coor = ref_coor
        src_dst_dict.update({(mov_coor[1], mov_coor[0]): (map_coor[1], map_coor[0])})

    return src_dst_dict


def initialize_peaks(peak_list, show_img=False, img_shape=None):
    peak_dict = {}
    count = 0
    for peak in peak_list:
        count += 1
        y, x = peak[0], peak[1]
        peak_dict.update({count: (y, x)})

    if show_img:
        img = np.zeros(img_shape)
        for key, coor in peak_dict.items():
            y, x = coor
            img[y - 5:y + 5, x - 5:x + 5] = True
        plt.figure()
        plt.imshow(img)
        plt.show()

    return peak_dict

def match_peaks(ref_peak_dict, mov_peak_dict, dist_threshold=5):
    remove_mov_peak = []
    for mov_peak_id, mov_coor in mov_peak_dict.items():
        dist_list = []
        for ref_peak_id, ref_coor in ref_peak_dict.items():
            dist = distance.euclidean(mov_coor, ref_coor)
            dist_list.append(dist)
        if np.min(dist_list) > dist_threshold:
            remove_mov_peak.append(mov_peak_id)
            # This mov_peak_id is not in ref segmentation

    remove_ref_peak = []
    for ref_peak_id, ref_coor in ref_peak_dict.items():
        dist_list = []
        for mov_peak_id, mov_coor in mov_peak_dict.items():
            dist = distance.euclidean(mov_coor, ref_coor)
            dist_list.append(dist)
        if np.min(dist_list) > dist_threshold:
            remove_ref_peak.append(ref_peak_id)
            # This ref peak id is not in mov segmentation

    updated_ref_peak_dict = remove_peaks_in_dict(full_dict=ref_peak_dict, keys=remove_ref_peak)
    updated_mov_peak_dict = remove_peaks_in_dict(full_dict=mov_peak_dict, keys=remove_mov_peak)

    return updated_ref_peak_dict, updated_mov_peak_dict


def remove_peaks_in_dict(full_dict, keys):
    new_dict = full_dict.copy()
    for key in keys:
        del new_dict[key]
    return new_dict


def remove_close_peaks(peak_dict, dist_threshold=20, show_img=False, img_shape=None):
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
        for peak_id, peak_coor in peak_dict.items():
            y, x = peak_coor[0], peak_coor[1]
            close_peaks_img[y - 5:y + 5, x - 5:x + 5] = True
        plt.figure()
        plt.imshow(close_peaks_img)
        plt.show()

    return close_ref_peak_dict


def filter_big_beads(img, center=0, area=20):
    """
    Find and filter big beads from an image with mixed beads
    :param img: 3d image with big and small beads
    :param center: center slice
    :param area: area(px) cutoff of a big bead
    :return: filtered: A 3d image where big beads are masked out as 0
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


#=================================================================
# Test method
ref = np.zeros((100, 100))
mov = np.zeros((100, 100))

ref[25:50, 25:50] = True
mov[27:52, 25:50] = True


#==================================================================
# Use ITK

# set itk image objects
ref_itk = sitk.GetImageFromArray(ref_smooth)
ref_itk = sitk.Cast(ref_itk, sitk.sitkFloat32)

mov_itk = sitk.GetImageFromArray(mov_smooth)
mov_itk = sitk.Cast(mov_itk, sitk.sitkFloat32)

# set itk optimizer
R = sitk.ImageRegistrationMethod()
# R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
#                                            numberOfIterations=10000,
#                                            convergenceMinimumValue=1e-5,
#                                            convergenceWindowSize=5)

#R.SetOptimizerAsRegularStepGradientDescent(5.0, .01, 50)
R.SetOptimizerAsGradientDescent(learningRate=1.0,
                                numberOfIterations=30,
                                convergenceMinimumValue=1,
                                convergenceWindowSize=20)
R.SetInitialTransform(sitk.Similarity2DTransform(mov_itk.GetDimension()))
R.SetInterpolator(sitk.sitkLinear)
R.SetMetricAsCorrelation()
# R.SetMetricAsJointHistogramMutualInformation()
# R.SetMetricAsMattesMutualInformation(100)
# R. SetMetricAsMeanSquares
# R.SetMetricSamplingStrategy(R.RANDOM)
# R.SetMetricSamplingPercentage(0.75)


outTx = R.Execute(ref_itk, mov_itk)
# rings image

outTx.GetParameters()
print (outTx)

transform_func = sitk.ResampleImageFilter()
transform_func.SetReferenceImage(ref_itk)
transform_func.SetInterpolator(sitk.sitkLinear)
transform_func.SetDefaultPixelValue(1)
transform_func.SetTransform(outTx)
mov_transformed = transform_func.Execute(mov_itk)

new_img = sitk.GetArrayFromImage(mov_transformed)

model = tf.AffineTransform()
model.estimate(np.array(ref_itk), np.array(mov_itk))

plt.figure()
plt.imshow(ref_smooth)
plt.show()

plt.figure()
plt.imshow(mov_smooth)
plt.show()

plt.figure()
plt.imshow(sitk.GetArrayFromImage(ref_itk))
plt.show()
