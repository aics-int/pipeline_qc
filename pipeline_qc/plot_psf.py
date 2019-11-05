import os
data_folder = r'\\allen\aics\microscopy\Calysta\test\psf'
folders = os.listdir(data_folder)
reprocess = []
for folder in folders:
    if folder.startswith('ZSD'):
        files = os.listdir(os.path.join(data_folder, folder))
        if len(files) == 0:
            reprocess.append(folder)
            os.removedirs(os.path.join(data_folder, folder))

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import os

psf_df = pd.read_csv(r'\\allen\aics\microscopy\Calysta\test\psf\psf_data_2.csv', parse_dates=['date'])

systems = ['ZSD1', 'ZSD2', 'ZSD3']
plot_save_folder = r'\\allen\aics\microscopy\Calysta\test\psf\plots'
df_analyze = psf_df.loc[(psf_df['avg_fwhm_xy_gof'] < 0.1) & (psf_df['center_yx_gof'] < 0.1) & (psf_df['center_zx_gof'] < 0.1) & (psf_df['center_zx_gof'] < 0.1)]

for column in df_analyze.columns.values.tolist():
    if column.endswith('tilt'):
        print (column)
        column = 'center_zy_tilt'
        for index, row in df_analyze.iterrows():
            degree = divmod((row[column]*180./math.pi), 360)[1]
            df_analyze.loc[index, column + '_degree'] = degree
            #print (df_analyze.loc[index, column + '_degree'])
for system in systems:

    metric_range = {'avg_fwhm_xy': (0.2, 0.3),
                    'avg_fwhm_xy_gof': (0, 0.1),
                    'avg_fwhm_xy_tilt': (0, 3),
                    'avg_fwhm_z': (0.7, 1.5),
                    'avg_max_intensity': (np.min(df_analyze['avg_max_intensity']), np.max(df_analyze['avg_max_intensity'])),
                    'center_avg_max_intensity': (np.min(df_analyze['center_avg_max_intensity']), np.max(df_analyze['center_avg_max_intensity'])),
                    'center_yx_fwhm': (0.2, 0.3),
                    'center_zy_fwhm': (0.7, 1.5)}

    metrics = ['avg_fwhm_xy', 'avg_fwhm_xy_gof', 'avg_fwhm_z', 'avg_max_intensity',
               'center_avg_max_intensity', 'center_yx_fwhm', 'center_zy_fwhm', 'center_zx_fwhm', 'center_zy_tilt_degree', 'center_zx_tilt_degree', 'center_zy_gof', 'center_zx_gof']
    #metrics = ['center_zy_tilt', 'center_zx_tilt']
    df_system = df_analyze.loc[(psf_df['system'] == system)]
    for metric in metrics:
        print (metric)
        title = metric + ' vs date'
        fig = plt.figure()
        df_system.plot(x='date', y=metric,
                            style=['k.'],
                            ylim=(np.min(df_analyze[metric]), np.max(df_analyze[metric])),
                            xlim=(df_analyze['date'].min(), df_analyze['date'].max()),
                            grid=True,
                            title=title)
        plt.savefig(os.path.join(plot_save_folder, system + '_'+ title + '.png'))
        plt.close()

