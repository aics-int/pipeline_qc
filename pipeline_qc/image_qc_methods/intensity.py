from lkaccess import LabKey, contexts
import math
from aicsimageio import AICSImage
import numpy as np
import pandas as pd
from scipy import signal, stats
from skimage import exposure, filters


def intensity_stats_single_channel(single_channel_im):

    result = dict()
    result.update({'mean': single_channel_im.mean()})
    result.update({'max': single_channel_im.max()})
    result.update({'min': single_channel_im.min()})
    result.update({'std': single_channel_im.std()})

    return result

def query_from_fms(cellline):
    server_context = LabKey(contexts.PROD)

    sql = f'''
     SELECT fov.fovid, fov.sourceimagefileid, well.wellname.name as wellname, plate.barcode,
        instrument.name as instrument, fcl.celllineid.name as cellline, fov.created, file.localfilepath
        FROM microscopy.fov as fov
        INNER JOIN microscopy.well as well on fov.wellid = well.wellid
        INNER JOIN microscopy.plate as plate on well.plateid = plate.plateid
        INNER JOIN microscopy.instrument as instrument on fov.instrumentid = instrument.instrumentid
        INNER JOIN celllines.filecellline as fcl on fov.sourceimagefileid = fcl.fileid
        INNER JOIN fms.file as file on fov.sourceimagefileid = file.fileid
        WHERE fov.objective = 100
        AND fov.qcstatusid.name = 'Passed'
        AND fcl.celllineid.name = '{cellline}'
    '''

    result = server_context.execute_sql('microscopy', sql)
    df = pd.DataFrame(result['rows'])
    return df[['sourceimagefileid', 'created', 'fovid', 'instrument', 'localfilepath', 'wellname', 'barcode',
                    'cellline']]


def split_image_into_channels(im_path, sourceimagefileid):

    server_context = LabKey(contexts.PROD)
    im = AICSImage(im_path)
    np_im = im.data[0, 0, :, :, :, :]
    sql = f'''
      SELECT content.contenttypeid.name, content.channelnumber
        FROM processing.content as content
        WHERE content.fileid = '{sourceimagefileid}'
    '''

    result = server_context.execute_sql('microscopy', sql)
    df = pd.DataFrame(result['rows'])

    split_channels = dict()

    for index, row in df.iterrows():
        for channel in ['405nm', '488nm', '561nm', '638nm', 'brightfield']:
            if row['name'] == 'Raw ' + channel:
                channel_number = row['channelnumber']
                exec('ch' + channel + "= np_im[channel_number, :, :, :]")
                exec("split_channels.update({channel: ch" + channel + "})")
            else:
                pass

    return split_channels


def batch_qc(query_df):

    pd.options.mode.chained_assignment = None
    stat_list = list()
    for index, row in query_df.iterrows():
        channel_dict = split_image_into_channels(row['localfilepath'], row['sourceimagefileid'])
        stat_dict = dict()
        for key, value in channel_dict.items():
            intensity_dict = intensity_stats_single_channel(value)
            for key2, value2, in intensity_dict.items():
                stat_dict.update({key + ' ' + key2 + ' ' + 'intensity': value2})
        stat_list.append(stat_dict)

    return pd.concat([query_df, pd.DataFrame(stat_list)], axis=1)
