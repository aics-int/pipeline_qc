import numpy as np
from aicsimageio import AICSImage
import matplotlib.pyplot as plt
import pandas as pd
from lkaccess import LabKey, contexts


def intensity_stats_per_image(im_path, plots=False):

    im = AICSImage(im_path) # this is a 6D STCZYX data structure
    data = im.data
    np_im = data[0, 0, :, :, :, :]
    shape = im.shape # index of dimensions of the image
    mean = dict()
    max = dict()
    min = dict()
    std = dict()

    for i in range(shape[2]):
        ch_im = np_im[i,:,:,:]
        mean.update({i:ch_im.mean()})
        max.update({i+'_max':ch_im.max()})
        min.update({i+'_min':ch_im.min()})
        std.update({i+'std':ch_im.std()})

    if plots:
        for stat in [mean, max, min, std]:
            list = sorted(stat.items)
            x, y = zip(*list)
            plt.plot(x, y)
            plt.show()

    return mean, max, min, std


def run_qc(im_path_list, intensity_check=True):

    stat_df = pd.DataFrame(im_path_list, columns=['im_path'])
    if intensity_check:
        for stat_str in ['mean', 'max', 'min', 'std']:
            for j in range(5):
                stat_df[stat_str + 'ch' + str(j)] = ''

            for index, row in stat_df.iterrows():
                mean, max, min, std = intensity_stats_per_image(row('im_path'))

                for stat in [mean, max, min, std]:
                    for item in stat.items():
                        stat_df[str(stat) + str(item[0])] = item[1]

    return stat_df


def query_from_fms(cellline):

    server_context = LabKey(contexts.PROD)

    cellline = 'AICS-7'

    sql = f'''
     SELECT
        fov.fovid, fov.sourceimagefileid, well.wellname.name as wellname, plate.barcode,
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
    result_df = pd.DataFrame(result['rows'])

