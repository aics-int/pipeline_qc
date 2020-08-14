import numpy as np

import pandas as pd
from pipeline_qc.image_qc_methods import (file_processing_methods, intensity,
                                          query_fovs, z_stack_check)
from labkey.utils import create_server_context
from lkaccess import LabKey

import labkey

RAW_CUTOFFS = {'405nm':[400, 430],
               '488nm':[400, 900],
               '561nm':[400, 900],
               '638nm':[400, 8000],
               'brightfield':[0, 20000]}

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
        if ((row['405nm_all_z_score'] < -2.58) or
                (row['488nm_cellline_z_score'] < -2.58) or
                (row['638nm_all_z_score'] < -2.58) or
                (row['405nm_raw_pass'] == False) or
                (row['488nm_raw_pass'] == False) or
                (row['638nm_raw_pass'] == False)):
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

    upload_df = df.drop(['_638nm contrast_99_percentile_false clip','_brightfield mean_intensity',
                         '_638nm flag_top_false clip','_brightfield flag_bottom_false clip', '_405nm mean_intensity',
                         '_488nm 0_5_intensity', '_638nm max_intensity', '_561nm mean_intensity',
                         '_488nm max_intensity', '_brightfield detect_top_false clip',
                         '_638nm median_intensity', '_488nm 99_5_intensity',
                         '_638nm flag_bottom_false clip', '_561nm 0_5_intensity',
                         '_405nm max_intensity', '_488nm min_intensity', '_638nm edge fov',
                         '_488nm std_intensity', '_561nm max_intensity',
                         '_405nm min_intensity', '_405nm 0_5_intensity',
                         '_638nm z_aggregate_false clip', '_405nm median_intensity',
                         '_638nm 0_5_intensity', '_638nm 99_5_intensity', '_561nm min_intensity',
                         '_638nm real_bottom_false clip', '_638nm min_intensity',
                         '_638nm crop_top_false clip', 'Key', '_brightfield 0_5_intensity',
                         '_brightfield min_intensity', '_561nm std_intensity',
                         '_brightfield max_intensity', '_561nm 99_5_intensity',
                         '_405nm std_intensity', '_488nm median_intensity', '_labkeyurl_FovId',
                         '_638nm real_top_false clip', '_brightfield laplace_range_false clip',
                         '_brightfield 99_5_intensity', '_638nm crop_bottom_false clip',
                         '_brightfield crop_top_false clip',
                         '_brightfield crop_bottom_false clip',
                         '_brightfield flag_top_false clip', '_638nm mean_intensity',
                         '_brightfield std_intensity', '_561nm median_intensity',
                         '_638nm std_intensity', '_brightfield detect_bottom_false clip',
                         '_brightfield median_intensity', '_488nm mean_intensity',
                         '_405nm 99_5_intensity', 'sourceimagefileid', 'fovimagedate',
                         'instrument', 'localfilepath', 'wellname', 'barcode', 'cellline',
                         'workflow', 'imaging_mode', 'gene', 'latest_segmentation_filename',
                         'latest_segmentation_readpath', 'latest_segmentation_metadata',
                         '_405nm_all_z_score', '_405nm_barcode_z_score', '_405nm_cellline_z_score',
                         '_405nm_imaging_mode_z_score', '_405nm_instrument_z_score', '_405nm_workflow_z_score',
                         '_405nm_raw_pass', '_488nm_all_z_score', '_488nm_barcode_z_score', '_488nm_cellline_z_score',
                         '_488nm_imaging_mode_z_score', '_488nm_instrument_z_score', '_488nm_workflow_z_score',
                         '_488nm_raw_pass', '_561nm_all_z_score', '_561nm_barcode_z_score', '_561nm_cellline_z_score',
                         '_561nm_imaging_mode_z_score', '_561nm_instrument_z_score', '_561nm_workflow_z_score',
                         '_561nm_raw_pass', '_638nm_all_z_score', '_638nm_barcode_z_score', '_638nm_cellline_z_score',
                         '_638nm_imaging_mode_z_score', '_638nm_instrument_z_score', '_638nm_workflow_z_score',
                         '_638nm_raw_pass', '_brightfield_all_z_score', '_brightfield_barcode_z_score',
                         '_brightfield_cellline_z_score', '_brightfield_imaging_mode_z_score',
                         '_brightfield_instrument_z_score', '_brightfield_workflow_z_score','_brightfield_raw_pass',
                         '_Pass_intensity'], axis=1)

    for i, row in upload_df.iterrows():
        row_dict = row.drop(['fovid']).to_dict()
        upload_row = {key:(str(value) if value else None) for (key, value) in row_dict.items()}
        upload_row['FovId'] = row['fovid']
        lk.update_rows(
            schema_name='lists',
            query_name='FOV QC Metrics',
            rows=[upload_row]
        )

