from lkaccess import LabKey, contexts
import os
import pandas as pd


def query_fovs_from_fms(workflows = None, cell_lines = None, plates = None, fovids = None):
    # Queries FMS (only using cell line right now) for image files that we would QC
    # Inputs all need to be lists of strings

    # This logic allows us to optionally leave any inout blank and still be able to
    if not workflows:
        workflow_query = ''
    else:
        workflow_query = f"AND fov.wellid.plateid.workflow.name IN {str(workflows).replace('[','(').replace(']',')')}"
    if not cell_lines:
        cell_line_query = ""
    else:
        cell_line_query = f"AND fcl.celllineid.name IN {str(cell_lines).replace('[','(').replace(']',')')}"
    if not plates:
        plate_query = ""
    else:
        plate_query = f"AND plate.barcode IN {str(plates).replace('[','(').replace(']',')')}"
    if not fovids:
        fovid_query = ""
    else:
        fovid_query = f"AND fov.fovid IN {str(fovids).replace('[','(').replace(']',')')}"
    server_context = LabKey(contexts.PROD)

    sql = f'''
     SELECT fov.fovid, fov.sourceimagefileid, well.wellname.name as wellname, plate.barcode,
        instrument.name as instrument, fcl.celllineid.name as cellline, fov.fovimagedate, file.localfilepath,
        fov.wellid.plateid.workflow.name as workflow
        FROM microscopy.fov as fov
        INNER JOIN microscopy.well as well on fov.wellid = well.wellid
        INNER JOIN microscopy.plate as plate on well.plateid = plate.plateid
        INNER JOIN microscopy.instrument as instrument on fov.instrumentid = instrument.instrumentid
        INNER JOIN celllines.filecellline as fcl on fov.sourceimagefileid = fcl.fileid
        INNER JOIN fms.file as file on fov.sourceimagefileid = file.fileid
        WHERE fov.objective = 100
        AND file.localfilepath NOT LIKE '%aligned_cropped%'
        {workflow_query}
        {cell_line_query}
        {plate_query}
        {fovid_query}
    '''
    result = server_context.execute_sql('microscopy', sql)
    df = pd.DataFrame(result['rows'])

    sql2 = f'''
      SELECT filefov.fovid, file.localfilepath as alignedfilepath, filefov.fileid as alignedimagefileid
         FROM microscopy.filefovpath as filefov
         INNER JOIN fms.file as file on filefov.fileid = file.fileid
         WHERE filefov.fileid.filename LIKE '%alignV2%'
    '''

    result2 = server_context.execute_sql('microscopy', sql2)
    df2 = pd.DataFrame(result2['rows'])

    df = pd.merge(df, df2, how='left', on='fovid')

    for i in range(len(df)):
        if pd.isna(df.iloc[i, df.columns.get_loc('alignedfilepath')]):
            df.at[i, 'alignedimagefileid'] = df.iloc[i, df.columns.get_loc('sourceimagefileid')]
            df.at[i, 'alignedfilepath'] = df.iloc[i, df.columns.get_loc('localfilepath')]

    if df.empty:
        print("Query from FMS returned no fovids")
        return pd.DataFrame(columns=['sourceimagefileid', 'fovimagedate', 'fovid', 'instrument', 'localfilepath',
                                     'wellname', 'barcode', 'cellline', 'workflow',
                                     'alignedimagefileid', 'alignedfilepath'])
    else:
        return df[['sourceimagefileid', 'fovimagedate', 'fovid', 'instrument', 'localfilepath', 'wellname', 'barcode',
                    'cellline', 'workflow', 'alignedimagefileid', 'alignedfilepath']]


def query_fovs_from_filesystem(plates, workflows = ['PIPELINE_4_4', 'PIPELINE_4_5', 'PIPELINE_4_6', 'PIPELINE_4_7', 'PIPELINE_5.2', 'PIPELINE_6', 'PIPELINE_7', 'RnD_Sandbox']):
    # Querying the filesystem for plates, and creating a list of all filepaths needing to be processed
    # Inputs all need to be lists of strings
    # Need to work on

    prod_dir = '/allen/aics/microscopy/'
    pipeline_dirs = ['PIPELINE_4_4', 'PIPELINE_4_5', 'PIPELINE_4_6', 'PIPELINE_4_7', 'PIPELINE_5.2', 'PIPELINE_6', 'PIPELINE_7']
    data_dirs = ['RnD_Sandbox']
    paths = list()
    # Iterates through all dirs in PRODUCTION directory that are relevant, and finds all plate directories that match
    for dir in pipeline_dirs:
        if dir not in workflows:
            pass
        else:
            for subdir in os.listdir(prod_dir + 'PRODUCTION/' + dir):
                if subdir in plates:
                    paths.append({dir:prod_dir + 'PRODUCTION/' + dir + '/' + subdir})
                else:
                    pass

    # Iterates through all dirs in Data directory that are relevant, and finds all plate directories that match
    for rnd_dir in data_dirs:
        if dir not in workflows:
            pass
        else:
            for rnd_subdir in os.listdir(prod_dir + 'Data/' + rnd_dir):
                if rnd_subdir in plates:
                    paths.append({'RnD':prod_dir + 'Data/' + rnd_dir + '/' + rnd_subdir})

    supported_folders = ['100X_zstack', '100XB_zstack']

    # For all folders of plate that match query, finds all files that exist in those folders
    # associated with a single scene
    image_metadata_list = list()
    for row in paths:
        for workflow, path in row.items():
            for instrument in os.listdir(path):
                for folder in os.listdir(path + '/' + instrument):
                    if folder in supported_folders:
                        image_dir = path + '/' + instrument + '/' + folder
                        for image in os.listdir(image_dir):
                            if image.endswith('czi'):
                                image_metadata_dict = dict()
                                image_metadata_dict.update({'workflow': workflow})
                                image_metadata_dict.update({'barcode': path[-10:]})
                                image_metadata_dict.update({'instrument': instrument})
                                image_metadata_dict.update({'localfilepath': image_dir + '/' + image})
                                image_metadata_list.append(image_metadata_dict)
                            else:
                                pass

    return pd.DataFrame(image_metadata_list)


def query_fovs_cell_seg(workflows=None, cell_lines=None, plates=None, fovids=None):
    # Queries FMS (only using cell line right now) for image files that we would QC
    # Inputs all need to be lists of strings

    # This logic allows us to optionally leave any inout blank and still be able to
    if not workflows:
        workflow_query = ''
    else:
        workflow_query = f"AND fov.wellid.plateid.workflow.name IN {str(workflows).replace('[','(').replace(']',')')}"
    if not cell_lines:
        cell_line_query = ""
    else:
        cell_line_query = f"AND fcl.celllineid.name IN {str(cell_lines).replace('[','(').replace(']',')')}"
    if not plates:
        plate_query = ""
    else:
        plate_query = f"AND plate.barcode IN {str(plates).replace('[','(').replace(']',')')}"
    if not fovids:
        fovid_query = ""
    else:
        fovid_query = f"AND fov.fovid IN {str(fovids).replace('[','(').replace(']',')')}"
    server_context = LabKey(contexts.PROD)



    return

def query_fovs(workflows=None, cell_lines=None, plates=None, fovids=None, only_from_fms=True):
    # Script that can query multiple parameters and join those tables into one query dataframe
    # workflows, cell_lines, plates, and fovs are all lists of strings
    # options: only_from_fms means you can only query fms. If false, will call the filesystem query as well
    df = query_fovs_from_fms(workflows, cell_lines, plates, fovids)
    if only_from_fms == False:
        df_2 = query_fovs_from_filesystem(plates)
        df = pd.concat([df, df_2], axis=0, ignore_index=True)

    return df
