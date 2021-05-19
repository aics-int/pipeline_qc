import pandas as pd
import os
import numpy


def find_beginning_end_index(argo_csv):
    begin = None
    end = None
    for index, row in argo_csv.iterrows():
        # row = argo_csv.loc[10]
        content = row.values.tolist()[0]
        argo_csv.loc[index] = content.replace(';', ', ')
        if 'power_instruction' in content:
            begin = index
        if 'measured_optical_power_average' in content:
            end = index

    return begin, end


def transform_power_instruction_with_measurement(extract_df, col_list):
    edit_df = pd.DataFrame()
    for col in col_list:
        edit_df[col] = None

    for index, row in extract_df.iterrows():
        content = row.values.tolist()[0].split(', ')
        for i in range(0, len(col_list)):
            edit_df.loc[index, col_list[i]] = content[i]

    edit_df = edit_df.set_index(keys='power_instruction')

    return edit_df


def match_measurement_with_spotfire_table(channels_power_dict, df, system, date, objective, measurement_made_by_initial):

    output_df = pd.DataFrame()
    channels = []
    for item in df.columns.values.tolist():
        if item.split('_')[0] not in channels:
            channels.append(item.split('_')[0])

    for power_instruction, row in df.iterrows():
        if power_instruction == '100.0':
            for channel in channels:
                power = row[channel + '_power']
                print('laser power for ' + channel + ' at 100%: ' + str(float(power)/1000) + 'mW')
                new_row = {'date': date,
                           'MeasurementMadeBy': measurement_made_by_initial,
                           'System': system,
                           'Objective': objective,
                           'Laser Line': channel + 'nm',
                           'Power at the objective (mW)': float(power) / 1000.,
                           'Laser Power': power_instruction,
                           'measured_power': power
                           }
                output_df = output_df.append(new_row, ignore_index=True)
        else:
            for channel in channels:
                power = channels_power_dict[channel]
                if (power > (float(row[channel + '_power']) - float(row[channel + '_error']))) & (
                        power < (float(row[channel + '_power']) + float(row[channel + '_error']))):
                    print('laser % for ' + channel + ' at ' + str(power/1000) + 'mW: ' + power_instruction + '%')
                    new_row = {'date': date,
                               'MeasurementMadeBy': measurement_made_by_initial,
                               'System': system,
                               'Objective': objective,
                               'Laser Line': channel + 'nm',
                               'Power at the objective (mW)': float(power) / 1000.,
                               'Laser Power': power_instruction,
                               'measured_power': float(row[channel + '_power']) / 1000.
                               }
                    output_df = output_df.append(new_row, ignore_index=True)
                if ((power * 2) > (float(row[channel + '_power']) - float(row[channel + '_error']))) & (
                        (power * 2) < (float(row[channel + '_power']) + float(row[channel + '_error']))):
                    print('laser % for ' + channel + ' at ' + str(2*power/1000) + 'mW: ' + power_instruction + '%')
                    new_row = {'date': date,
                               'MeasurementMadeBy': measurement_made_by_initial,
                               'System': system,
                               'Objective': objective,
                               'Laser Line': channel + 'nm',
                               'Power at the objective (mW)': float(power) * 2 / 1000.,
                               'Laser Power': power_instruction,
                               'measured_power': float(row[channel + '_power']) / 1000.
                               }
                    output_df = output_df.append(new_row, ignore_index=True)

    return output_df


def drop_duplicate_laser_power_rows(df):
    duplicated_check = df[['date', 'System', 'Laser Line', 'Power at the objective (mW)']].duplicated()
    check_index = duplicated_check.loc[duplicated_check == True].index

    for row_index in check_index:
        initiate_row = df.loc[row_index]

        same_rows = df.loc[
            (df['date'] == initiate_row['date']) & (df['System'] == initiate_row['System']) & (
                    df['Laser Line'] == initiate_row['Laser Line']) & (
                    df['Power at the objective (mW)'] == initiate_row['Power at the objective (mW)'])]
        target_power = initiate_row['Power at the objective (mW)']
        keep_idx = (same_rows['measured_power'] - target_power).astype(float).idxmin()
        idx_rmv = [idx for idx in same_rows.index if idx != keep_idx]
        for idx in idx_rmv:
            df = df.drop(idx)

    return df


def process_argo_csv(csv_path, system, date, objective, measurement_made_by_initial, channels_power_dict):
    argo_csv = pd.read_csv(csv_path)
    begin, end = find_beginning_end_index(argo_csv)

    if (begin is not None) & (end is not None):
        columns_str = argo_csv.loc[begin].values.tolist()[0]
        col_list = columns_str.split(', ')

        extract_df = pd.DataFrame(argo_csv.loc[begin + 1:end - 1])

        edit_df = transform_power_instruction_with_measurement(extract_df, col_list)

        output_df = match_measurement_with_spotfire_table(
            channels_power_dict,
            edit_df,
            system=system,
            date=date,
            objective=objective,
            measurement_made_by_initial=measurement_made_by_initial
        )

        output_df = drop_duplicate_laser_power_rows(output_df)










