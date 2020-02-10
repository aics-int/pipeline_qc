import pandas as pd
import os
import numpy

output_df = pd.DataFrame()

systems = ['ZSD1', 'ZSD2', 'ZSD3']
for system in systems:
    argo_folder = os.path.join(r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ARGO-POWER', system, 'power_meter')

    files = os.listdir(argo_folder)
    for file in files:
        if file.endswith('.csv') and (not file.startswith('test')) and file.startswith(system):
            #file = 'POWER_ZSD1_2019_04_03 at 11_37_15.csv'
            date = file.split('2019_')[1]
            month = date.split('_')[0].split('0')[-1]
            day = date.split('_')[1][:2].split('0')[-1]
            final_date = month + '/' + day + '/2019'

            argo_csv = pd.read_csv(os.path.join(argo_folder, file))

            begin = None
            end = None
            for index, row in argo_csv.iterrows():
                #row = argo_csv.loc[10]
                content = row.values.tolist()[0]
                argo_csv.loc[index] = content.replace(';', ', ')
                if 'power_instruction' in content:
                    begin = index
                if 'measured_optical_power_average' in content:
                    end = index

            if (begin is not None) & (end is not None):
                columns_str = argo_csv.loc[begin].values.tolist()[0]
                col_list = columns_str.split(', ')

                power_df = pd.DataFrame(argo_csv.loc[begin+1:end-1])

                edit_df = pd.DataFrame(columns=col_list)
                for index, row in power_df.iterrows():
                    content = row.values.tolist()[0].split(', ')
                    for i in range (0, len(col_list)):
                        edit_df.loc[index, col_list[i]] = content[i]

                edit_df = edit_df.set_index(keys='power_instruction')


                channels_power_dict = {'405': 280,
                                       '488': 2300,
                                       '561': 2400,
                                       '638': 2400}

                channels = []
                for item in edit_df.columns.values.tolist():
                    if item.split('_')[0] not in channels:
                        channels.append(item.split('_')[0])

                for power_instruction, row in edit_df.iterrows():
                    if power_instruction == '100.0':
                        for channel in channels:
                            power = row[channel + '_power']
                            new_row = {'date': final_date,
                                       'MeasurementMadeBy': 'CY',
                                       'System': system,
                                       'Objective': '10X',
                                       'Laser Line': channel + 'nm',
                                       'Power at the objective (mW)': float(power)/ 1000.,
                                       'Laser Power': power_instruction,
                                       'measured_power': power
                                       }
                            output_df = output_df.append(new_row, ignore_index=True)
                    else:
                        for channel in channels:
                            power = channels_power_dict[channel]
                            if (power > (float(row[channel+'_power']) - float(row[channel+ '_error']))) & (power < (float(row[channel+'_power']) + float(row[channel+ '_error']))):
                                print (channel + '_' + power_instruction)
                                new_row = {'date': final_date,
                                           'MeasurementMadeBy': 'CY',
                                           'System': system,
                                           'Objective': '10X',
                                           'Laser Line': channel + 'nm',
                                           'Power at the objective (mW)': float(power)/1000.,
                                           'Laser Power': power_instruction,
                                           'measured_power': float(row[channel+'_power'])/1000.
                                           }
                                output_df = output_df.append(new_row, ignore_index=True)
                            if ((power*2) > (float(row[channel+'_power']) - float(row[channel+ '_error']))) & ((power*2) < (float(row[channel+'_power']) + float(row[channel+ '_error']))):
                                print (channel + '_2x_' + power_instruction)
                                new_row = {'date': final_date,
                                           'MeasurementMadeBy': 'CY',
                                           'System': system,
                                           'Objective': '10X',
                                           'Laser Line': channel + 'nm',
                                           'Power at the objective (mW)': float(power)*2/1000.,
                                           'Laser Power': power_instruction,
                                           'measured_power': float(row[channel+'_power'])/1000.
                                           }
                                output_df = output_df.append(new_row, ignore_index=True)

output_df = output_df[['date', 'MeasurementMadeBy', 'System', 'Objective', 'Laser Line', 'Power at the objective (mW)', 'Laser Power', 'measured_power']]

test_df = output_df.copy()
check_df = output_df[['date', 'System', 'Laser Line', 'Power at the objective (mW)', 'measured_power']]
duplicated_check = check_df[['date', 'System', 'Laser Line', 'Power at the objective (mW)']].duplicated()
check_index = duplicated_check.loc[duplicated_check==True].index

for row_index in check_index:
    initiate_row = check_df.loc[row_index]

    same_rows = check_df.loc[(check_df['date']==initiate_row['date']) & (check_df['System'] == initiate_row['System']) & (check_df['Laser Line'] == initiate_row['Laser Line']) & (check_df['Power at the objective (mW)'] == initiate_row['Power at the objective (mW)'])]
    target_power = initiate_row['Power at the objective (mW)']
    keep_idx = (same_rows['measured_power'] - target_power).astype(float).idxmin()
    idx_rmv = [idx for idx in same_rows.index if idx != keep_idx]
    print (idx_rmv)
    for idx in idx_rmv:
        test_df = test_df.drop(idx)
