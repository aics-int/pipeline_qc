import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import shutil
import seaborn as sns

df = pd.read_csv('/allen/aics/microscopy/Aditya/image_qc_outputs/20200410_metrics/fov_qc_metrics.csv')
aics61 = df[df['cellline'] == 'AICS-61']
aics61_4 = aics61[aics61['workflow'] == "['Pipeline 4.4']"]
aics61_4_aligned = aics61_4[aics61_4['localfilepath'].str.contains('aligned_cropped')]
aics61_4_aligned = aics61_4

for index, row in aics61_4_aligned.iterrows():
    file_name = row['localfilepath'].replace('-', '_').lower().split('/')[-1]
    [*prefixes] = file_name.split('_')
    for prefix in prefixes:
        if prefix == 'scene':
            aics61_4_aligned.at[index, 'Scene'] = prefixes[prefixes.index('scene') + 1]
        if prefix[0] == 'p':
            aics61_4_aligned.at[index, 'Position'] = prefix[1:]
    for prefix in prefixes:
        if (len(prefix)==2):
            if (prefix[1] in ['e', 'r', 'c']):
                # df.at[index, 'Mode'] = 'C'
                if prefix[1] == 'e':
                    df.at[index, 'Colony Position'] = 'Edge'
                    break
                elif prefix[1] == 'r':
                    df.at[index, 'Colony Position'] = 'Ridge'
                    break
                elif prefix[1] == 'c':
                    df.at[index, 'Colony Position'] = 'Center'
                    break
        else:
            # df.at[index, 'Mode'] = 'A'
            df.at[index, 'Colony Position'] = 'Center'

for wavelength in ['405nm', '488nm', '561nm', '638nm']:
    if f'{wavelength} median-intensity' not in aics61_4_aligned.columns:
        continue
    else:
        for sort in ['cellline', 'barcode', 'instrument']:
            for unique in aics61_4_aligned[sort].unique():
                mean = np.mean(aics61_4_aligned[aics61_4_aligned[sort] == unique][f'{wavelength} median-intensity'])
                std = np.std(aics61_4_aligned[aics61_4_aligned[sort] == unique][f'{wavelength} median-intensity'])
                for index, row in aics61_4_aligned[aics61_4_aligned[sort] == unique].iterrows():
                    aics61_4_aligned.at[aics61_4_aligned.index[aics61_4_aligned['fovid'] == row['fovid']], f'{wavelength} {sort}_z-score'] = (row[f'{wavelength} median-intensity'] - mean) / std

plate_stat_list = list()
for plate in aics61_4_aligned.barcode.unique():
    plate_dict = dict()
    plate_dict.update({'Barcode': plate})
    for wavelength in ['405nm', '488nm', '561nm', '638nm']:
        if f'{wavelength} median-intensity' not in aics61_4_aligned.columns:
            continue
        else:
            sub_df = aics61_4_aligned[aics61_4_aligned['barcode'] == plate]
            scenes = np.array(list(map(int, list(sub_df['Scene'])))).reshape((-1, 1))
            positions = np.array(list(map(int, list(sub_df['Position'])))).reshape((-1, 1))
            y = np.array(sub_df[f'{wavelength} median-intensity'])
            if np.isnan(np.sum(y)):
                coef = 0
                intercept = 0
                r_sq = 0
            else:
                if np.isnan(np.sum(scenes)):
                    if np.isnan(np.sum(positions)):
                        coef = 0
                        intercept = 0
                        r_sq = 0
                    else:
                        model = LinearRegression().fit(positions, np.array(sub_df[f'{wavelength} median-intensity']))
                        coef = model.coef_
                        intercept = model.intercept_
                        r_sq = model.score(positions, np.array(sub_df[f'{wavelength} median-intensity']))
                else:
                    model = LinearRegression().fit(scenes, np.array(sub_df[f'{wavelength} median-intensity']))
                    coef = model.coef_
                    intercept = model.intercept_
                    r_sq = model.score(positions, np.array(sub_df[f'{wavelength} median-intensity']))

                plate_dict.update({f'{wavelength} linear_reg_r^2': r_sq})
                plate_dict.update({f'{wavelength} linear_reg_coef': coef[0]})
                plate_dict.update({f'{wavelength} linear_reg_intercept': intercept})
    plate_stat_list.append(plate_dict)

plate_df = pd.DataFrame(plate_stat_list)


for index, row in aics61_4_aligned.iterrows():
    for wavelength in ['405nm', '488nm', '638nm']:
        if wavelength == '405nm':
            if 400 < row[f'{wavelength} median-intensity'] < 430:
                aics61_4_aligned.at[index, f'{wavelength} raw_pass'] = True
            else:
                aics61_4_aligned.at[index, f'{wavelength} raw_pass'] = False
        if wavelength == '488nm':
            if 400 < row[f'{wavelength} median-intensity'] < 900:
                aics61_4_aligned.at[index, f'{wavelength} raw_pass'] = True
            else:
                aics61_4_aligned.at[index, f'{wavelength} raw_pass'] = False
        if wavelength == '638nm':
            if 400 < row[f'{wavelength} median-intensity'] < 8000:
                aics61_4_aligned.at[index, f'{wavelength} raw_pass'] = True
            else:
                aics61_4_aligned.at[index, f'{wavelength} raw_pass'] = False

        if row[f'{wavelength} barcode_z-score'] <= -2.58:
            aics61_4_aligned.at[index, f'{wavelength} barcode_pass'] = False
        else:
            aics61_4_aligned.at[index, f'{wavelength} barcode_pass'] = True

        if row[f'{wavelength} cellline_z-score'] <= -2.58:
            aics61_4_aligned.at[index, f'{wavelength} cellline_pass'] = False
        else:
            aics61_4_aligned.at[index, f'{wavelength} cellline_pass'] = True

for index, row in aics61_4_aligned.iterrows():
    if ((row['405nm barcode_pass'] == False) or
            (row['405nm cellline_pass'] == False) or
            (row['488nm barcode_pass'] == False) or
            (row['488nm cellline_pass'] == False) or
            (row['638nm barcode_pass'] == False) or
            (row['638nm cellline_pass'] == False) or
            (row['405nm raw_pass'] == False) or
            (row['488nm raw_pass'] == False) or
            (row['638nm raw_pass'] == False)):
        aics61_4_aligned.at[index, 'Pass'] = False
    else:
        aics61_4_aligned.at[index, 'Pass'] = True

for index, row in aics61_4_aligned.iterrows():
    aics61_4_aligned.at[index, 'filename'] = str(row['localfilepath'])[80:]
    if ((row['405nm cellline_pass'] == False) or
            (row['488nm cellline_pass'] == False) or
            (row['638nm cellline_pass'] == False) or
            (row['405nm raw_pass'] == False) or
            (row['488nm raw_pass'] == False) or
            (row['638nm raw_pass'] == False)):
        aics61_4_aligned.at[index, 'Pass_cellline'] = False
    else:
        aics61_4_aligned.at[index, 'Pass_cellline'] = True

aics61_failed = aics61_4_aligned[(aics61_4_aligned['Pass_cellline'] == False)]
aics61_passed = aics61_4_aligned[(aics61_4_aligned['Pass_cellline'] == True)]

# target = '/allen/aics/microscopy/Aditya/image_qc_outputs/failed_examples_aics61'
# for filepath in list(aics61_failed['localfilepath']):
#     shutil.copy(filepath, target + '/' + filepath.split('/')[-1])

sns.scatterplot(aics61_4_aligned['405nm median-intensity'], aics61_4_aligned['488nm median-intensity'], hue=aics61_4_aligned['Pass_cellline'])
sns.scatterplot(aics61_4_aligned['405nm median-intensity'], aics61_4_aligned['638nm median-intensity'], hue=aics61_4_aligned['Pass_cellline'])
sns.scatterplot(aics61_4_aligned['488nm median-intensity'], aics61_4_aligned['638nm median-intensity'], hue=aics61_4_aligned['Pass_cellline'])