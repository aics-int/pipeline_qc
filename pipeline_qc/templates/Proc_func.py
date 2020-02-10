import os 
import pathlib
import numpy as np
import pandas as pd
from aicsimageio import omeTifWriter, AICSImage

def simple_zstack_Proc(all_stacks, proc_config):
    for fi, fn in enumerate(all_stacks):
        print(fn)
        reader = AICSImage(fn)
        data = reader.data
        assert not(len(data.shape)==5 and data.shape[0]>1), print(f'check data type of {fn}, time-dimension is found, but suppose to be non-timelapse')

        for proc_step in proc_config:

            if proc_step['do'] == 'extract-single-channel':
                im = data[0, int(proc_step['channel']), :, :, :]
                writer = omeTifWriter.OmeTifWriter(proc_step['save'] + os.sep + pathlib.PurePosixPath(fn).stem + '.tif')
                writer.save(im)
            elif proc_step['do'] == 'generate_csv':
                csv_name = proc_step['csv']
                df = pd.DataFrame({'index':fi,'filename':fn}, columns=['index', 'filename'], index=[1])
                if not os.path.exists(csv_name):
                    with open(csv_name, 'a') as f: 
                        df.to_csv(f, header=True, index=False)
                else:
                    with open(csv_name, 'a') as f: 
                        df.to_csv(f, header=False, index=False)
