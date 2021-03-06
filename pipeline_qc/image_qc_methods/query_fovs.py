from lkaccess import LabKey
import os
import pandas as pd

DEFAULT_LK_HOST = "aics.corp.alleninstitute.org"
DEFAULT_LK_PORT = 80


def query_fovs_from_fms(workflows=None, cell_lines=None, plates=None, fovids=None, labkey_host: str = DEFAULT_LK_HOST,
                        labkey_port: int = DEFAULT_LK_PORT):
    # Queries FMS (only using cell line right now) for image files that we would QC
    # Inputs all need to be lists of strings

    # This logic allows us to optionally leave any inout blank and still be able to
    if not workflows:
        workflow_query = ""
    else:
        workflow_query = f"AND fov.wellid.plateid.workflow.name IN {str(workflows).replace('[', '(').replace(']', ')')}"
    if not cell_lines:
        cell_line_query = ""
    else:
        cell_line_query = f"AND fcl.celllineid.name IN {str(cell_lines).replace('[', '(').replace(']', ')')}"
    if not plates:
        plate_query = ""
    else:
        plate_query = f"AND plate.barcode IN {str(plates).replace('[', '(').replace(']', ')')}"
    if not fovids:
        fovid_query = ""
    else:
        fovid_query = f"AND fov.fovid IN {str(fovids).replace('[', '(').replace(']', ')')}"

    labkey_client = LabKey(host=labkey_host, port=labkey_port)

    sql = f'''
    WITH CellNucSegFiles AS (
        SELECT f.fileid,
               ff.fovid,
               -- LabKey returns weirdly inconsistent date strings - sometimes they're 25 characters long, sometimes
               --  they're 26. If we shorten them arbitrarily we can maintain the same functionality on a more
               --  consistent basis.
               SUBSTRING(CAST(f.created AS VARCHAR), 0, 22)||' '||f.filename AS filename,
               SUBSTRING(CAST(f.created AS VARCHAR), 0, 22)||' '||f.localfilepath AS readpath,
               SUBSTRING(CAST(f.created AS VARCHAR), 0, 22)||' '||f.metadata AS metadata
        FROM fms.file f
            JOIN microscopy.filefov ff ON ff.fileid = f.fileid
        WHERE f.filename LIKE '%CellNucSegCombined%'
    )
    SELECT fov.fovid,
           fov.sourceimagefileid,
           well.wellname.name as wellname,
           plate.barcode,
           instrument.name as instrument,
           fcl.celllineid.name as cellline,
           fov.fovimagedate,
           file.localfilepath,
           fov.wellid.plateid.workflow.name as workflow,
           welljn.imagingmodeid.name as imaging_mode,
           cldef.geneid.name as gene,
           -- Here we use SUBSTRING() to chop off the added "created" dates from above, which are used to grab the
           -- most recent segmentation info. Dates returned from LabKey are of the format '2000-00-00 00:00:00.00000'
           SUBSTRING(MAX(CellNucSegFiles.filename), 23) AS latest_segmentation_filename,
           SUBSTRING(MAX(CellNucSegFiles.readpath), 23) AS latest_segmentation_readpath,
           SUBSTRING(MAX(CellNucSegFiles.metadata), 23) AS latest_segmentation_metadata
    FROM microscopy.fov as fov
        INNER JOIN microscopy.well as well on fov.wellid = well.wellid
        INNER JOIN microscopy.plate as plate on well.plateid = plate.plateid
        INNER JOIN microscopy.instrument as instrument on fov.instrumentid = instrument.instrumentid
        INNER JOIN celllines.filecellline as fcl on fov.sourceimagefileid = fcl.fileid
        INNER JOIN fms.file as file on fov.sourceimagefileid = file.fileid
        INNER JOIN microscopy.wellimagingmodejunction as welljn on well.wellid = welljn.wellid
        INNER JOIN celllines.celllinedefinition as cldef on fcl.celllineid = cldef.celllineid
        LEFT JOIN CellNucSegFiles ON CellNucSegFiles.fovid = fov.fovid
    WHERE fov.objective = 100
        AND file.filename NOT LIKE '%aligned_cropped%'
        AND fov.qcstatusid.name = 'Passed'
        {workflow_query}
        {cell_line_query}
        {plate_query}
        {fovid_query}
    GROUP BY fov.fovid,
             fov.sourceimagefileid,
             well.wellname.name,
             plate.barcode,
             instrument.name,
             fcl.celllineid.name,
             fov.fovimagedate,
             file.localfilepath,
             fov.wellid.plateid.workflow.name,
             welljn.imagingmodeid.name,
             cldef.geneid.name
    '''
    result = labkey_client.execute_sql('microscopy', sql)
    df = pd.DataFrame(result['rows'])

    sql2 = f'''
      SELECT filefov.fovid, file.localfilepath as alignedfilepath, filefov.fileid as alignedimagefileid
         FROM microscopy.filefovpath as filefov
         INNER JOIN fms.file as file on filefov.fileid = file.fileid
         WHERE filefov.fileid.filename LIKE '%alignV2%'
    '''

    result2 = labkey_client.execute_sql('microscopy', sql2)
    df2 = pd.DataFrame(result2['rows'])

    df = pd.merge(df, df2, how='left', on='fovid')



    for i in range(len(df)):
        if not pd.isna(df.iloc[i, df.columns.get_loc('alignedfilepath')]):
            df.at[i, 'sourceimagefileid'] = df.iloc[i, df.columns.get_loc('alignedimagefileid')]
            df.at[i, 'localfilepath'] = df.iloc[i, df.columns.get_loc('alignedfilepath')]

    if df.empty:
        print("Query from FMS returned no fovids")
        return pd.DataFrame(columns=['sourceimagefileid', 'fovimagedate', 'fovid', 'instrument', 'localfilepath',
                                     'wellname', 'barcode', 'cellline', 'workflow', 'imaging_mode', 'gene'])
    else:
        df = df.drop(
            df[(df['workflow'].astype('str') == "['Pipeline 4.4']") &
               (df['localfilepath'].str.contains('alignV2') == False)].index)
        return df[['sourceimagefileid', 'fovimagedate', 'fovid', 'instrument', 'localfilepath', 'wellname', 'barcode',
                   'cellline', 'workflow', 'imaging_mode', 'gene',
                   'latest_segmentation_filename', 'latest_segmentation_readpath', 'latest_segmentation_metadata']]


def query_fovs_from_filesystem(plates, workflows=['PIPELINE_4_4', 'PIPELINE_4_5', 'PIPELINE_4_6', 'PIPELINE_4_7',
                                                  'PIPELINE_5.2', 'PIPELINE_6', 'PIPELINE_7', 'RnD_Sandbox']):
    # Querying the filesystem for plates, and creating a list of all filepaths needing to be processed
    # Inputs all need to be lists of strings
    # Need to work on

    prod_dir = '/allen/aics/microscopy/'
    pipeline_dirs = ['PIPELINE_4_4', 'PIPELINE_4_5', 'PIPELINE_4_6', 'PIPELINE_4_7', 'PIPELINE_5.2', 'PIPELINE_6',
                     'PIPELINE_7']
    data_dirs = ['RnD_Sandbox']
    paths = list()
    # Iterates through all dirs in PRODUCTION directory that are relevant, and finds all plate directories that match
    for dir in pipeline_dirs:
        if dir not in workflows:
            pass
        else:
            for subdir in os.listdir(prod_dir + 'PRODUCTION/' + dir):
                if subdir in plates:
                    paths.append({dir: prod_dir + 'PRODUCTION/' + dir + '/' + subdir})
                else:
                    pass

    # Iterates through all dirs in Data directory that are relevant, and finds all plate directories that match
    for rnd_dir in data_dirs:
        if dir not in workflows:
            pass
        else:
            for rnd_subdir in os.listdir(prod_dir + 'Data/' + rnd_dir):
                if rnd_subdir in plates:
                    paths.append({'RnD': prod_dir + 'Data/' + rnd_dir + '/' + rnd_subdir})

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


def query_fovs(workflows=None, cell_lines=None, plates=None, fovids=None, only_from_fms=True,
               labkey_host: str = DEFAULT_LK_HOST, labkey_port: int = DEFAULT_LK_PORT):
    # Script that can query multiple parameters and join those tables into one query dataframe
    # workflows, cell_lines, plates, and fovs are all lists of strings
    # options: only_from_fms means you can only query fms. If false, will call the filesystem query as well
    df = query_fovs_from_fms(workflows, cell_lines, plates, fovids, labkey_host=labkey_host, labkey_port=labkey_port)
    if only_from_fms == False:
        df_2 = query_fovs_from_filesystem(plates)
        df = pd.concat([df, df_2], axis=0, ignore_index=True)

    return df
