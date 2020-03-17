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
# val_csv = val_csv.loc[val_csv['workflow'] == "['Pipeline 4.2']"]
val_csv = val_csv.loc[val_csv['cellline'] == 'AICS-11']
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
test_normal = normal.sample(frac=0.02)
test = df_crop_top.sample(frac=0.04)

for check_fov, row in test.iterrows():
    # check_fov = row['fovid']
    fuse_638 = io.imread(os.path.join(r'\\allen\aics\microscopy\Aditya\image_qc_outputs\top', str(check_fov) + '_638nm-top.tif'))
    center = io.imread(os.path.join(r'\\allen\aics\microscopy\Aditya\image_qc_outputs\center', str(check_fov) + '_638nm-center.tif'))
    # local_path_name = val_csv.loc[val_csv['fovid'] == check_fov, 'windowspath'].values.tolist()[0]
    # print(local_path_name)
    plt.figure()
    plt.title(str(check_fov))
    plt.imshow(fuse_638, cmap='gray')
    plt.show()
    plt.figure()
    plt.title(str(check_fov) + ' center')
    plt.imshow(center, cmap='gray')
    plt.show()

# check_fov = 12499
check_fov = 4599
profile = literal_eval(val_csv.loc[check_fov, '638nm z_aggregate-false clip'])
# profile = literal_eval(val_csv.loc[check_fov, '638nm contrast_99_percentile-false clip'])
plt.figure()
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
false_pos_fov = [1166,
                 3329,
                 3517,
                 886,
                 1483,
                 1328,
                 3243]

true_pos_fov = [1434,
                14267,
                1242,
                14322,
                14314,
                15207]

for fovid in false_pos_fov:




# Make interactive plots
TOOLS = "box_select, lasso_select, box_zoom, pan, wheel_zoom, reset"
hover = tools.HoverTool()
base_list = [('crop_top', '@638nm real_top-false clip'),
             ('crop_bottom', '@638nm real_bottom-false clip'),
             ('barcode', '@barcode'),
             ('edge', '@638nm edge fov?'),
             ('fovid', '@fovid'),
             ('localfilepath', '@localfilepath')]

