import numpy as np 
import os
from aicsimageio import AICSImage
from glob import glob
from skimage.util import montage
from skimage.io import imsave
import math
import pandas as pd 

def simple_zstack_QC(all_stacks, qc_config):

    # peek which qc will be done and get prepared accordingly
    for qc_step in qc_config:
        if qc_step['check'] == 'glance':
            montage_list = []
        elif qc_step['check'] == 'intensity_stats':
            column_names = ['filename']
            for cid, ch in enumerate(qc_step['channel']):
                column_names.append(str(ch+1)+'_mean')
                column_names.append(str(ch+1)+'_max')
                column_names.append(str(ch+1)+'_min')
                column_names.append(str(ch+1)+'_std')
            df = pd.DataFrame(columns=column_names)


    for fi, fn in enumerate(all_stacks):
        print(fn)
        reader = AICSImage(fn)
        data = reader.data
        assert not(len(data.shape)==5 and data.shape[0]>1), print(f'check data type of {fn}, time-dimension is found, but suppose to be non-timelapse')
  
        for qc_step in qc_config:
            if qc_step['check'] == 'glance':
                im = data[0, int(qc_step['channel']), :, :, :]
                if qc_step['contrast'] == 'min-max':
                    im = (im - im.min())/(im.max() - im.min())
                elif qc_step['contrast'] == 'auto':
                    strech_min = np.percentile(im, 0.5)
                    strech_max = np.percentile(im, 99.5)
                    im[im<strech_min]=strech_min
                    im[im>strech_max]=strech_max
                    im = (im - strech_min) / (strech_max - strech_min)

                if qc_step['view'] == 'mid_z':
                    montage_list.append(im[im.shape[0]//2, :,:])
                elif qc_step['view'] == 'mip':
                    montage_list.append(np.amax(im,axis=0))

                # after reaching the last image
                if fi == len(all_stacks)-1:
                    if qc_step['type'] == 'in-montage':
                        # find max size
                        max_y, max_x = 0, 0
                        for img_sample in montage_list:
                            max_y = max((max_y, img_sample.shape[0]))
                            max_x = max((max_x, img_sample.shape[1]))
                        montage_file = np.zeros((len(montage_list), max_y, max_x), dtype=np.float32)
                        for index_sample in range(len(montage_list)):
                            montage_file[index_sample,:montage_list[index_sample].shape[0],:montage_list[index_sample].shape[1]] = montage_list[index_sample]
                        # generate montage
                        grid_x = 3000//max_x
                        grid_y = int(math.ceil(len(montage_list)/grid_x))
                        montage_out = montage(montage_file, grid_shape=(grid_y, grid_x),padding_width=1)

                        imsave(qc_step['save'], montage_out)

                    #elif qc_step['type'] == 'in-sequence':
            elif qc_step['check'] == 'intensity_stats':
                new_stats = {'filename':fn}
                for cid, ch in enumerate(qc_step['channel']):
                    im = data[0,ch,:,:,:]
                    new_stats.update({str(ch+1)+'_mean':im.mean()})
                    new_stats.update({str(ch+1)+'_max':im.max()})
                    new_stats.update({str(ch+1)+'_min':im.min()})
                    new_stats.update({str(ch+1)+'_std':im.std()})
                df = df.append(new_stats,ignore_index=True)
                df.to_csv(qc_step['save'],mode='w')




