import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plot_save_folder = r'\\allen\aics\microscopy\Calysta\test\optical_control_plots\ff\20191121'

df = pd.read_csv(r'C:\Users\calystay\Desktop\ff_output_20191010_2.csv', parse_dates=['date'])

channels = [405, 488, 561, 638]
systems = ['ZSD1', 'ZSD2', 'ZSD3']

metric_range = {'img_max': (0, 66000),
                'img_min': (0, 66000),
                'img_mean': (0, 66000),
                'img_median': (0, 66000),
                'img_std': (0, 66000),
                'intensity_range': (0, 66000),
                'z_position': (int(np.min(df['z_position']))-100, int(np.max(df['z_position']))+100),
                'centering_accuracy': (0, 1),
                'hot_spot_magnitude': (int(np.min(df['hot_spot_magnitude']))-100, int(np.max(df['hot_spot_magnitude']))+100)
                }


all_columns = df.columns.values.tolist()
year = '2019'
for system in systems:
    for channel in channels:
        df_channel = df.loc[(df['channel'] == channel) & (df['date']>(year + '-01-01')) & (df['system']==system)]

        # Plot roll off changes over time
        title = 'roll off values for ' + system + ' (' + year + ') in ' + str(channel)
        fig, ax = plt.subplots()
        ax = df_channel.plot(x='date', y=['img_roll_off'],
                             style=['k.'],
                             ax=ax,
                             title=title,
                             ylim=(0, 1),
                             grid=True)
        fig.savefig(os.path.join(plot_save_folder, title + '.png'))
        plt.close()
        img_metric = ['img_max', 'img_min', 'img_median', 'intensity_range', 'z_position', 'centering_accuracy',
                      'hot_spot_magnitude']

        for metric in img_metric:
            # plot roll off against metric
            title = 'roll off vs ' + metric + ' for ' + system + ' (' + year + ') in ' + str(channel)
            fig, ax = plt.subplots()
            ax = df_channel.plot(x=metric, y='img_roll_off',
                                 style='k.',
                                 ax=ax,
                                 title=title,
                                 ylim=(0, 1),
                                 xlim=metric_range[metric],
                                 grid=True)
            fig.savefig(os.path.join(plot_save_folder, title + '.png'))
            plt.close()
            # plot metric over date
            title = metric + ' vs date for '  + system + ' (' + year + ') in ' + str(channel)
            fig, ax = plt.subplots()
            ax = df_channel.plot(x='date', y=metric,
                                 style='k.',
                                 ax=ax,
                                 title=title,
                                 ylim=metric_range[metric],
                                 grid=True)
            fig.savefig(os.path.join(plot_save_folder, title + '.png'))
            plt.close()


metric = 'img_roll_off'

zsd_colors = ["130122163", ""]

for channel in channels:
    df_plot = df.loc[(df['date'] > (year + '-01-01')) & (df['channel'] == channel)]
    fig = plt.figure()
    ax = sns.scatterplot(x="date", y="img_roll_off", data=df_plot, hue="system")
    plt.xlim((df_plot['date'].min(), df_plot['date'].max()))
    plt.ylim()
    plt.title('intensity roll off in ' + str(channel))
    plt.show()























