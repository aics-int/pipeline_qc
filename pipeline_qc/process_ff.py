from pipeline_qc import image_processing_methods as pm
from aicsimageio import AICSImage
from skimage import filters
import numpy as np
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import math
from aicsimageio import AICSImage
from scipy import interpolate, ndimage, optimize
from scipy.optimize import curve_fit
from skimage import filters, measure
import xml.etree.ElementTree as ET

df = pd.DataFrame()

prod_optical_control = r'\\allen\aics\microscopy\PRODUCTION\OpticalControl'
optical_control_folders = os.listdir(prod_optical_control)
count = 0
for folder in optical_control_folders:
    count += 1
    print('processing ' + str(count) + ' out of ' + str(len(optical_control_folders)))
    if (folder.startswith('ZSD')) & (folder[3] is not '0'):
        system = folder[0:4]
        date = folder[5:]

        files = os.listdir(os.path.join(prod_optical_control, folder))
        for file in files:
            if (file.endswith('405.czi')) or (file.endswith('488.czi')) or (file.endswith('561.czi')) or (file.endswith('638.czi')):
                print(file)
                channel = file[-7:-4]
                row = {'system': system,
                       'date': date,
                       'channel': channel,
                       'img_path': os.path.join(prod_optical_control, folder, file)
                       }

                ff_data = AICSImage(os.path.join(prod_optical_control, folder, file))
                ff_f = ff_data.data[0, 0, 0, :, :]

                img_info = pm.get_img_info(img=ff_f, data=ff_data)
                row.update(img_info)

                metric = process_image(ff_f)
                row.update(metric)
                print(row)
                df = df.append(row, ignore_index=True)

df.to_csv(r'C:\Users\calystay\Desktop\ff_output_20191010_2.csv')


def process_image(ff_f):

    # Pre-process flat field images
    ff_smooth = filters.gaussian(image=ff_f, sigma=1, preserve_range=True)
    ff_norm = ff_smooth/np.max(ff_smooth)

    # Fit flat field image with 2D gaussian function
    params_gaussian = pm.fitgaussian(ff_norm)
    fit_function = pm.gaussian(*params_gaussian)
    fit_gaussian_field_non_uni_raw = fit_function(*np.indices(ff_norm.shape))

    homogeneity_map = fit_gaussian_field_non_uni_raw

    # Extract metric from fitted flat field image
    metric_dict = pm.report_metric(homogeneity_map=homogeneity_map, roll_off_range=0.1)

    return metric_dict
