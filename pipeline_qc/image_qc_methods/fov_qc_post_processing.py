import numpy as np
import math

import pandas as pd
from pipeline_qc.image_qc_methods import query_fovs
from labkey.utils import create_server_context
from lkaccess import LabKey

import labkey

RAW_CUTOFFS = {'405nm':[400, 430],
               '488nm':[400, 1600],
               '561nm':[400, 700],
               '638nm':[400, 8000],
               'brightfield':[0, 50000]}

def create_full_dataset():

    server_context = labkey.utils.create_server_context('aics.corp.alleninstitute.org', 'AICS/Microscopy', 'labkey',
                                                        use_ssl=False)
    my_results = labkey.query.select_rows(
        server_context=server_context,
        schema_name='lists',
        query_name='FOV QC Metrics',
        sort='FovId/FOVId'
    )

    data_df = pd.DataFrame(my_results['rows']).rename(columns={'FovId':'fovid'})
    metadata_df = query_fovs.query_fovs()

    return pd.merge(data_df, metadata_df, how='left', on='fovid')

def z_score_stat_generation():

    full_df = create_full_dataset()
    for wavelength in ['405nm', '488nm', '561nm', '638nm', 'brightfield']:
        mean_all = np.mean(full_df[f'_{wavelength} median_intensity'])
        std_all = np.std(full_df[f'_{wavelength} median_intensity'])
        for i, row in full_df.iterrows():
            if row[f'_{wavelength} median_intensity'] is not None:
                if RAW_CUTOFFS[wavelength][0] < row[f'_{wavelength} median_intensity'] < RAW_CUTOFFS[wavelength][1]:
                    full_df.at[i, f'{wavelength}_raw_pass'] = True
                else:
                    full_df.at[i, f'{wavelength}_raw_pass'] = False
                full_df.at[i, f'{wavelength}_all_z_score'] = (row[f'_{wavelength} median_intensity'] - mean_all) / std_all
            else:
                continue

    for filter in ['instrument', 'cellline', 'barcode', 'workflow', 'imaging_mode']:
        # exec(filter + '_df = pd.DataFrame()')
        uniques = list(full_df[filter].astype(str).unique())
        for unique in uniques:
            filtered_df = full_df[full_df[filter].astype(str) == unique]
            for wavelength in ['405nm', '488nm', '561nm', '638nm', 'brightfield']:
                pass_filtered_df = filtered_df[filtered_df[f'{wavelength}_raw_pass'] == True]
                mean = np.mean(pass_filtered_df[f'_{wavelength} median_intensity'])
                std = np.std(pass_filtered_df[f'_{wavelength} median_intensity'])
                for i, row in filtered_df.iterrows():
                    if row[f'_{wavelength} median_intensity'] is not None:
                        full_df.at[full_df.index[full_df['fovid'] == row['fovid']], f'{wavelength}_{filter}_z_score'] =\
                            (row[f'_{wavelength} median_intensity'] - mean) / std


    for i, row in full_df.iterrows():
        if ~(math.isnan(row['_488nm median_intensity'])) and (math.isnan(row['_561nm median_intensity'])):
            if ((row['405nm_all_z_score'] < -2.58) or
                    (row['488nm_cellline_z_score'] < -2.58) or
                    (row['638nm_all_z_score'] < -2.58) or
                    (row['405nm_raw_pass'] == False) or
                    (row['488nm_raw_pass'] == False) or
                    (row['638nm_raw_pass'] == False) or
                    (row['_405nm max_intensity'] <= 500) or
                    (row['_638nm max_intensity'] <= 635)
            ):
                full_df.at[i, 'Pass_intensity'] = False
            else:
                full_df.at[i, 'Pass_intensity'] = True
        elif (math.isnan(row['_488nm median_intensity'])) and ~(math.isnan(row['_561nm median_intensity'])):
            if ((row['405nm_all_z_score'] < -2.58) or
                    (row['561nm_cellline_z_score'] < -2.58) or
                    (row['638nm_all_z_score'] < -2.58) or
                    (row['405nm_raw_pass'] == False) or
                    (row['561nm_raw_pass'] == False) or
                    (row['638nm_raw_pass'] == False) or
                    (row['_405nm max_intensity'] <= 500) or
                    (row['_638nm max_intensity'] <= 635)
            ):
                full_df.at[i, 'Pass_intensity'] = False
            else:
                full_df.at[i, 'Pass_intensity'] = True
        else:
            if ((row['405nm_all_z_score'] < -2.58) or
                    (row['488nm_cellline_z_score'] < -2.58) or
                    (row['561nm_cellline_z_score'] < -2.58) or
                    (row['638nm_all_z_score'] < -2.58) or
                    (row['405nm_raw_pass'] == False) or
                    (row['488nm_raw_pass'] == False) or
                    (row['561nm_raw_pass'] == False) or
                    (row['638nm_raw_pass'] == False) or
                    (row['_405nm max_intensity'] <= 500) or
                    (row['_638nm max_intensity'] <= 635)
            ):
                full_df.at[i, 'Pass_intensity'] = False
            else:
                full_df.at[i, 'Pass_intensity'] = True

    return full_df

def update_qc_data_labkey(df, env):

    if env == 'prod':
        context = create_server_context(
            'aics.corp.alleninstitute.org',
            'AICS/Microscopy',
            'labkey',
            use_ssl=False
        )
    elif env == 'stg':
        context = create_server_context(
            'stg-aics.corp.alleninstitute.org',
            'AICS/Microscopy',
            'labkey',
            use_ssl=False
        )

    lk = LabKey(server_context=context)

    for i, row in df.iterrows():
        subset_row = row[['Key', 'fovid', '405nm_raw_pass', '405nm_all_z_score', '405nm_instrument_z_score', '405nm_cellline_z_score',
        '405nm_barcode_z_score', '405nm_workflow_z_score', '405nm_imaging_mode_z_score', '488nm_raw_pass',
        '488nm_all_z_score', '488nm_instrument_z_score', '488nm_cellline_z_score', '488nm_barcode_z_score',
        '488nm_workflow_z_score', '488nm_imaging_mode_z_score', '561nm_raw_pass', '561nm_all_z_score',
        '561nm_instrument_z_score', '561nm_cellline_z_score', '561nm_barcode_z_score', '561nm_workflow_z_score',
        '561nm_imaging_mode_z_score','638nm_raw_pass', '638nm_all_z_score', '638nm_instrument_z_score',
        '638nm_cellline_z_score', '638nm_barcode_z_score', '638nm_workflow_z_score', '638nm_imaging_mode_z_score',
        'brightfield_raw_pass', 'brightfield_all_z_score', 'brightfield_instrument_z_score',
        'brightfield_cellline_z_score', 'brightfield_barcode_z_score', 'brightfield_workflow_z_score',
        'brightfield_imaging_mode_z_score', 'Pass_intensity']].fillna(0)
        row_dict = subset_row.drop(['fovid']).to_dict()
        upload_row = {key: value for (key, value) in row_dict.items()}
        print(f"Processing {row['fovid']}")
        lk.update_rows(
            schema_name='lists',
            query_name='FOV QC Metrics',
            rows=[upload_row]
        )
        lk.update_rows(
            schema_name='microscopy',
            query_name='FOV',
            rows=[{
                'FOVId': row['fovid'],
                'QCPassIntensity': row['Pass_intensity'],
                'QC638nmCropBottomFalseClip': row['_638nm crop_bottom_false clip'],
                'QC638nmCropTopFalseClip': row['_638nm crop_top_false clip'],
                'QCBrightfieldZStacksOutOfOrder': row['_brightfield Z-stacks out of order']
            }]
        )


