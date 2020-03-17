import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skimage import io

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, tools
from bokeh.transform import jitter
from bokeh.layouts import gridplot

csv = pd.read_csv(r'\\allen\aics\microscopy\Aditya\image_qc_outputs\All.csv')
csv = csv.set_index(keys='fovid')

# Exclude AICS-0
# val_csv = val_csv.loc[val_csv['cellline'] != 'AICS-0']
# for index, row in val_csv.iterrows():
#     windowspath = '\\' + row['localfilepath'].replace('/', '\\')
#     val_csv.loc[index, 'windowspath'] = windowspath
val_csv = csv.copy()
val_csv = val_csv.loc[val_csv['workflow'] == "['Pipeline 4.4']"]
# val_csv = val_csv.loc[val_csv['cellline'] == 'AICS-61']
df_crop_top = val_csv[val_csv['638nm crop_top-false clip'] == True]
df_crop_bot = val_csv[val_csv['638nm crop_bottom-false clip'] == True]

# Sort by pipeline, cell line
# labels = sorted(set(val_csv['cellline'].values.tolist()))
# new_labels = sorted(set(df_crop_top['cellline'].values.tolist()))
# fig = plt.figure()
# ax = pd.value_counts(val_csv['cellline']).loc[new_labels].plot.bar(sort_columns=True, color='b')
# ax = pd.value_counts(df_crop_top['cellline']).loc[new_labels].plot.bar(sort_columns=True, color='r')
# title = 'Crop top proportion in pipeline 4_2'
# plt.title(title)
# plt.ylim((0, 3000))
# plt.show()
# fig.savefig(os.path.join(r'\\allen\aics\microscopy\Calysta\test\crop_top_bottom\validation', title + '.png'))


# Pick 20 images to check
normal = val_csv.loc[((val_csv['638nm crop_top-false clip'] == False) & (val_csv['638nm crop_bottom-false clip'] == False))]
test_normal = normal.sample(frac=0.015)
test = df_crop_top.sample(frac=0.038)

for check_fov, row in test.iterrows():
    # check_fov = row['fovid']
    fuse_638 = io.imread(os.path.join(r'\\allen\aics\microscopy\Aditya\image_qc_outputs\top', str(check_fov) + '_638nm-top.tif'))
    center = io.imread(os.path.join(r'\\allen\aics\microscopy\Aditya\image_qc_outputs\center', str(check_fov) + '_638nm-center.tif'))
    # local_path_name = val_csv.loc[val_csv['fovid'] == check_fov, 'windowspath'].values.tolist()[0]
    # print(local_path_name)
    plt.figure()
    plt.title(str(check_fov) + ' top')
    plt.imshow(fuse_638, cmap='gray')
    plt.show()
    plt.figure()
    plt.title(str(check_fov) + ' center')
    plt.imshow(center, cmap='gray')
    plt.show()

# Check fov contrast, intensity profiles
# check_fov = 12499
check_fov = 7251
intensity = literal_eval(val_csv.loc[check_fov, '638nm z_aggregate-false clip'])
contrast = literal_eval(val_csv.loc[check_fov, '638nm contrast_99_percentile-false clip'])
plt.figure()
for profile in [intensity, contrast]:
    plt.scatter(y=profile, x=np.linspace(0, len(profile) - 1, len(profile)))
plt.show()

# Check intensity
fig = plt.figure()
ax = sns.swarmplot(x='638nm crop_top-false clip', y='638nm mean-intensity', data=normal, color='k', alpha=0.6, s=3)
ax = sns.swarmplot(x='638nm crop_top-false clip', y='638nm mean-intensity', data=df_crop_top, color='r', alpha=0.6, s=3)
plt.title('mean intensity')
plt.ylim((0, 8000))
plt.show()

fig = plt.figure()
ax = sns.swarmplot(x='638nm crop_top-false clip', y='638nm max-intensity', data=normal, color='k', alpha=0.6, s=3)
ax = sns.swarmplot(x='638nm crop_top-false clip', y='638nm max-intensity', data=df_crop_top, color='r', alpha=0.6, s=3)
plt.title('max intensity')
plt.ylim((0, 66600))
plt.show()

# Try to improve algorithm
# fovs for AICS-11
false_pos_fov = [3877,
                 16084,
                 1358,
                 16687,
                 3262,
                 15834]
#
true_pos_fov = [17233,
                835,
                976,
                3553,
                14782]
# float_fov = [2800,
#              3590,
#              1151,
#              14324]

# fovs for AICS-10
false_pos_fov = [4570,
                 16928,
                 16322,
                 ]
true_pos_fov = [16920,
                4610,
                4580,
                5467,
                5026,
                16863,
                4677,
                ]
# false_neg_fov = []

# fovs for AICS-61
false_pos_fov = [64878,
                 87803,
                 89408]
true_pos_fov = [79532,
                79533,
                74401,
                86999,
                65615,
                76144,
                87117,
                ]
# false_neg_fov = []

# fovs for AICS-57
false_pos_fov = [12623,
                 12341,
                 12767,
                 12772,
                 13150,
                 12476,
                 16886,
                 12677]
true_pos_fov = [13083,
                12285,
                12513,
                16541,
                12763,
                12764,
                ]
# float = [12954,
#          ]

# fovs for AICS-24
false_pos_fov = [7518,
                 6387,
                 7559,
                 7075]
true_pos_fov = [6435,
                6345,
                7251,
                6729,
                6678,
                6495,
                7135,
                7083]
float = [6493,
         6333]

false_pos_fov_all = [3877,
                     16084,
                     1358,
                     16687,
                     3262,
                     15834,
                     4570,
                     16928,
                     16322,
                     64878,
                     87803,
                     89408,
                     12623,
                     12341,
                     12767,
                     12772,
                     13150,
                     12476,
                     16886,
                     12677,
                     7518,
                     6387,
                     7559,
                     7075]

true_pos_fov_all = [17233,
                    835,
                    976,
                    3553,
                    14782,
                    16920,
                    4610,
                    4580,
                    5467,
                    5026,
                    16863,
                    4677,
                    79532,
                    79533,
                    74401,
                    86999,
                    65615,
                    76144,
                    13083,
                    12285,
                    12513,
                    16541,
                    12763,
                    12764,
                    6435,
                    6345,
                    7251,
                    6729,
                    6678,
                    6495,
                    7135,
                    7083]

# Tune/tweak method a little bit
def detect_false_clip_cmdr(z_aggregate, contrast_99_percentile, contrast_threshold=(0.2, 0.19)):
    """
    Detects top/bottom clipping in a z-stack. (The method will fail if you have small debris/floating cells on top. )
    :param cmdr: a (z, y, x) cmdr image
    :param contrast_threshold: a tuple of contrast threshold (threshold for finding bottom, threshold for finding top)
    :return:
        real_bottom: an integer of index of bottom-z-stack or None
        real_top: an integer of index of top-z-stack or None
        crop_top: a boolean if crop top is True or False
        crop_bottom: a boolean if crop bottom is True or False
        flag_top: a boolean if the top should be flagged
        flag_bottom: a boolean if the bottom should be flagged
        contrast_99_percentile: contrast profile in z
        z_aggregate: median intensity profile in z
    """

    # Initialize values
    crop_top = True
    crop_bottom = True
    real_top = None
    real_bottom = None
    flag_bottom = False
    flag_top = False

    # # Rescale image
    # cmdr = exposure.rescale_intensity(cmdr, in_range='image')
    #
    # # Generate contrast and median intensity curve along z
    # z_aggregate = []
    # contrast_99_percentile = []
    # for z in range(0, cmdr.shape[0]):
    #     z_aggregate.append(np.median(cmdr[z, :, :]) / np.max(cmdr[:, :, :]))
    #     contrast_99_percentile.append(
    #         (np.percentile(cmdr[z, :, :], 99.9) - np.min(cmdr[z, :, :])) / np.percentile(cmdr[:, :, :], 99.9))

    # Find intensity peaks in bottom and top of z-stack. A perfect z-stack should return 2 peaks,
    # the peak at lower index is the bottom of z-stack in focus, and the peak at higher index is the top of z-stack
    # in focus
    try:
        all_peaks = signal.argrelmax(np.array(z_aggregate))[0]
    except:
        all_peaks = []

    print (all_peaks)
    # Initialize top and bottom peaks from all_peaks
    if len(all_peaks) == 2:
        bottom_peak = all_peaks[0]
        top_peak = all_peaks[1]
    elif len(all_peaks) > 2:
        # Get the peak with highest intensity and initialize top/bottom with the same peak
        indexed = stats.rankdata(all_peaks, method='ordinal')
        refined_z = []
        for index in all_peaks:
            refined_z.append(z_aggregate[index])
        top_peak = bottom_peak = all_peaks[np.where(refined_z==np.max(refined_z))][0]
        print('more than 2 peaks')
    elif len(all_peaks) == 1:
        # Set bottom peak and top peak as the same peak
        bottom_peak = all_peaks[0]
        top_peak = all_peaks[0]
    else:
        # Report cannot find peak
        bottom_peak = None
        top_peak = None
        print('cannot find peak')

    if bottom_peak is not None:
        # From bottom and top peak, find the z plane at contrast threshold to the bottom and top of z-stack
        bottom_range = contrast_99_percentile[0: bottom_peak]

        # Iterate from bottom peak to the bottom of z-stack, find the closest z-stack that reaches lower contrast threshold
        for z in reversed(range(0, bottom_peak)):
            contrast_value = contrast_99_percentile[z]
            if contrast_value <= contrast_threshold[0]:
                real_bottom = z
                crop_bottom = False
                break
            else:
                real_bottom = np.where(bottom_range == np.min(bottom_range))[-1][-1]

    if top_peak is not None:
        # Iterate from top peak to the top of z-stack, find the closest z-stack that reaches upper contrast threshold
        top_range = contrast_99_percentile[top_peak:len(z_aggregate)]
        for z in range(top_peak, len(z_aggregate)):
            contrast_value = contrast_99_percentile[z]
            if contrast_value <= contrast_threshold[1]:
                real_top = z
                crop_top = False
                break

    # For logging purposes only
    # if real_top is None:
    #     print('crop top')
    # if real_bottom is None:
    #     print('crop bottom')

    # Refine crop bottom identification with the slope and fit of the contrast curve
    # Linear fit for first five z-stacks from bottom
    if real_bottom is not None:
        bottom_range = np.linspace(0, real_bottom - 1, real_bottom)
        if len(bottom_range) >= 5:
            # Get linear regression
            slope, y_int, r, p, err = stats.linregress(x=list(range(0, 5)), y=contrast_99_percentile[0:5])
            # Set criteria with slope and r-value to determine if the bottom is cropped
            if slope <= -0.005:
                real_bottom = real_bottom
                crop_bottom = False
            elif (slope <= 0) & (math.fabs(r) > 0.8):
                real_bottom = real_bottom
                crop_bottom = False
        else:
            # The z-stack might not be cropped, but should require validation
            print('flag bottom, too short')
            flag_bottom = True
            crop_bottom = False

    # Refine crop top identification with the slope and fit of the contrast curve
    # Linear fit for first five z-stacks from top
    if real_top is not None:
        top_range = np.linspace(real_top, len(z_aggregate) - 1, len(z_aggregate) - real_top)
        if len(top_range) >= 5:
            # get linear regression
            slope, y_int, r, p, err = stats.linregress(x=list(range(real_top, real_top + 5)),
                                                       y=contrast_99_percentile[real_top:real_top + 5])
            # Set criteria with slope and r-value to determine if the top is cropped
            if slope <= -0.005:
                real_top = real_top
                crop_top = False
            elif (slope <= 0) & (math.fabs(r) > 0.8):
                real_top = real_top
                crop_top = False
        else:
            # The z-stack might not be cropped, but should require validation
            print('flag top, too short')
            flag_top = True

    if real_top is None:
        top_range = np.linspace(len(z_aggregate)-5, len(z_aggregate)-1, 5)
        slope, y_int, r, p, err = stats.linregress(x=top_range,
                                                   y=np.take(contrast_99_percentile, indices=top_range.astype(int)))

        # print(slope)
        if (slope > -0.01) & (slope <= 0) & (math.fabs(r) > 0.8):
            real_top = real_top
            crop_top = False

    return real_bottom, real_top, crop_top, crop_bottom, flag_top, flag_bottom

from scipy import stats, signal
import numpy as np

for fovid in false_pos_fov:
    # fovid = 3329
    contrast_99_percentile = literal_eval(csv.loc[fovid, '638nm contrast_99_percentile-false clip'])
    z_aggregate = literal_eval(csv.loc[fovid, '638nm z_aggregate-false clip'])
    real_bottom, real_top, crop_top, crop_bottom, flag_top, flag_bottom = detect_false_clip_cmdr(z_aggregate, contrast_99_percentile)
    print (str(fovid) + ' ' + str(csv.loc[fovid, '638nm max-intensity']) + ' crop_top is ' + str(crop_top))
    plt.figure()
    for profile in [z_aggregate, contrast_99_percentile]:
        plt.scatter(y=profile, x=np.linspace(0, len(profile) - 1, len(profile)))
    plt.show()

# Make interactive plots
TOOLS = "box_select, lasso_select, box_zoom, pan, wheel_zoom, reset"
hover = tools.HoverTool()
base_list = [('crop_top', '@638nm real_top-false clip'),
             ('crop_bottom', '@638nm real_bottom-false clip'),
             ('barcode', '@barcode'),
             ('edge', '@638nm edge fov?'),
             ('fovid', '@fovid'),
             ('localfilepath', '@localfilepath')]


# Data exploration
count = 0
for fov, row in csv.iterrows():
    count += 1
    contrast_99_percentile = literal_eval(row['638nm contrast_99_percentile-false clip'])
    if len(contrast_99_percentile) >= 5:
        top_range = np.linspace(len(contrast_99_percentile) - 5, len(contrast_99_percentile) - 1, 5)
        slope, y_int, r, p, err = stats.linregress(x=top_range,
                                                   y=np.take(contrast_99_percentile, indices=top_range.astype(int)))

        csv.loc[fov, 'slope'] = slope
        csv.loc[fov, 'r'] = r
    print(count)

val_csv = csv.copy()
val_csv = val_csv.loc[val_csv['cellline'] == 'AICS-61']

fig = plt.figure()
ax = sns.swarmplot(x='638nm crop_top-false clip', y='slope', data=normal, color='k', alpha=0.6, s=3)
ax = sns.swarmplot(x='638nm crop_top-false clip', y='slope', data=df_crop_top, color='r', alpha=0.6, s=3)
# plt.title('mean intensity')
# plt.ylim((0, 8000))
plt.show()

false_pos_df = csv.loc[false_pos_fov_all]
true_pos_df = csv.loc[true_pos_fov_all]
fig = plt.figure()
ax = sns.swarmplot(x='638nm crop_top-false clip', y='638nm max-intensity', data=true_pos_df, color='k', alpha=0.6, s=3)
ax = sns.swarmplot(x='638nm crop_top-false clip', y='638nm max-intensity', data=false_pos_df, color='r', alpha=0.6, s=3)
# plt.title('mean intensity')
# plt.ylim((0, 8000))
plt.show()

# Rerun crop vs not crop with new method to find slope
