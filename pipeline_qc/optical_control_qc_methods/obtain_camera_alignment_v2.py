from skimage import transform as tf
from pipeline_qc.optical_control_qc_methods import \
    segment_argolight_rings as segment, \
    get_center_z, \
    crop_argolight_rings_img as crop, \
    estimate_alignment
from pipeline_qc.optical_control_qc_methods import ring_qc_utils as qc


def execute(image_object, channels_to_align, magnification, method_logging=False):
    px_size_x, px_size_y, px_size_z = image_object.get_physical_pixel_size()

    # detect center z-slice on reference channel
    ref_center_z, contrast = get_center_z.Executor(
        img_stack=image_object.data[0, 0, image_object.get_channel_names().index(channels_to_align['ref']), :, :, :]
    ).execute()

    # Crop method 1: Use all available rings
    ref_crop, crop_dims, ref_labelled_grid, ref_props_grid, ref_center_cross_label, ref_number_of_rings = crop.Executor(
        img=image_object.data[0, 0, image_object.get_channel_names().index(channels_to_align['ref']), ref_center_z, :,
            :],
        pixel_size=px_size_x, filter_px_size=50
    ).execute()

    mov_crop = image_object.data[
               0, 0,
               image_object.get_channel_names().index(channels_to_align['mov']),
               ref_center_z,
               crop_dims[0]:crop_dims[1], crop_dims[2]:crop_dims[3]
               ]

    # segment rings on reference image
    ref_seg_rings, ref_seg_rings_label, ref_props_df, ref_cross_label = segment.Executor(
        ref_crop, px_size_x, magnification, debug_mode=True
    ).execute()

    # segment rings on moving image
    mov_seg_rings, mov_seg_rings_label, mov_props_df, mov_cross_label = segment.Executor(
        mov_crop, px_size_x, magnification, debug_mode=True
    ).execute()

    # estimate alignment from segmentation
    tform, ref_coor_dict, transformation_parameters_dict, num_beads_for_estimation = estimate_alignment.Executor(
        ref_seg_rings, ref_seg_rings_label, ref_props_df, ref_cross_label,
        mov_seg_rings, mov_seg_rings_label, mov_props_df, mov_cross_label
    ).execute()

    # apply alignment on moving image
    tf_mov = tf.warp(
        image_object.data[0, 0, image_object.get_channel_names().index(channels_to_align['mov']), ref_center_z, :, :],
        inverse_map=tform, order=3, preserve_range=True
    )

    # crop aligned/transformed moving image
    tf_mov_crop = tf_mov[crop_dims[0]:crop_dims[1], crop_dims[2]:crop_dims[3]]
    tf_seg_rings, tf_seg_rings_label, tf_props_df, tf_cross_label = segment.Executor(
        tf_mov_crop, px_size_x, magnification, debug_mode=True
    ).execute()

    # get qc metric 1: change in fov intensity
    changes_fov_intensity_dictionary = qc.report_change_fov_intensity_parameters(
        mov_crop, tf_mov_crop
    )
    # get qc metric 2: change in mse in segmentation between reference and moving image
    mse_qc, diff_mse = qc.report_changes_in_mse(ref_seg_rings, mov_seg_rings, tf_seg_rings)

    # get qc metric 3: change in distances between centroid locations between reference and moving image
    coor_dist_qc, diff_sum_beads = qc.report_changes_in_coordinates_mapping(ref_coor_dict, tform, tf_mov_crop.shape)

    # get qc metric 4: expected number of beads
    bead_num_qc, num_beads = qc.report_number_beads(ref_coor_dict)

    # Report z offset for QC
    z_offset, org_ref_center, org_mov_center = qc.check_z_offest_between_ref_mov(
        ref_stack=image_object.data[0, 0, image_object.get_channel_names().index(channels_to_align['ref']), :, :, :],
        mov_stack=image_object.data[0, 0, image_object.get_channel_names().index(channels_to_align['mov']), :, :, :],
        method_logging=method_logging
    )

    # Report image SNR
    ref_signal, ref_noise, mov_signal, mov_noise = qc.report_ref_mov_image_snr(
        ref_crop, mov_crop,
        ref_seg=ref_seg_rings,
        mov_seg=mov_seg_rings > 0,
        method_logging=method_logging
    )
    print(ref_signal, ref_noise)

    return transformation_parameters_dict, bead_num_qc, num_beads, changes_fov_intensity_dictionary, coor_dist_qc, \
           diff_sum_beads, mse_qc, diff_mse, z_offset, ref_signal, ref_noise, mov_signal, mov_noise
