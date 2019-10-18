# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:51:35 2015

@author: olgag
"""
# Calculates PSF
# Calculates FOV flattness including overall tilt and curvature based on 
# Gaussian fit

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import white_tophat, disk
import tifffile
import psftools

# specify file name and path, number of channels, and channel to process
# can be function's imput parameters
filepath = 'C:/Users/olgag/Documents/data/PSF_measure040215/'
filename = 'PSF_measure032015_3_MMStack_1-Pos_000_001.ome.tif'
fullfilename = filepath + filename
total_channel = 4
num_channel = 1
pxl_xy = 0.1 #pxl size in um
pxl_z = 0.1
window_xy = window_z = 10

# read image data from tiff file
tiffimg = tifffile.TiffFile(fullfilename)
img = tiffimg.asarray() # whole stack is extracted
total_frame, img_height, img_width = img.shape
idx = np.arange(num_channel-1, total_frame, total_channel)
img_channel = img[idx,:,:].astype(float)

# remove background by subtracting morphological opening with disk-shaped
# structuring element; white_tophat(image)=image-opening(image)
selem = disk(15)
for k in range(len(idx)):
    img_channel[k,:,:] = white_tophat(img_channel[k,:,:], selem)
    
# find frame with max intensity    
k_select, i, j = np.unravel_index(img_channel.argmax(), img_channel.shape)
img_select = img_channel[k_select,:,:]
 
# find beads as local maxima and remove the ones that are too close to each
# other, near the edges; cut 3d window around each bead, remove too dim/bright   
peaks1 = psftools.find_peaks(img_select)
peaks2 = psftools.remove_close_peaks(img_select, peaks1, min_dist=30)
peaks3 = psftools.remove_near_edge(img_select, peaks2, window_xy=10)
peaks4 = psftools.find_3d_max(img_select, img_channel, peaks3, window_xy=10, 
                              window_z=10)
peaks5, bead_3d_window = psftools.remove_dim_bright(img_select, img_channel,
                                                    peaks4, pxl_z, window_xy=10, 
                                                    window_z=10)
# plot beads on image
plt.subplots(figsize=(10,10))
plt.imshow(img_select, cmap='jet', vmin=0, vmax=np.median(img_select)
           +np.std(img_select)*10, interpolation='nearest') # cmap = 'gray'
plt.colorbar()
plt.scatter(peaks1[:,1], peaks1[:,0], c='y', marker='x')
plt.scatter(peaks2[:,1], peaks2[:,0], c='m', marker='x')
plt.scatter(peaks3[:,1], peaks3[:,0], c='g', marker='x')
plt.scatter(peaks4[:,2], peaks4[:,1], c='c', marker='x')
plt.scatter(peaks5[:,2], peaks5[:,1], c='r', marker='x')
plt.axis([0, img_width, 0, img_height])   
plt.show  

# fit 2D rotated Gaussian to each plane in the stack for each bead
y, x = np.mgrid[0:window_xy*2+1,0:window_xy*2+1] 
par = np.zeros((window_z*2+1,7,len(peaks5)))
for i in range(len(peaks5)): 
    for k in range(window_z*2+1):        
        # calculate fit initial parameters
        par_init = psftools.get_fit_parameters((x,y), bead_3d_window[k,:,:,i])
       
        # fit 2D rotated Gaussian
        par[k,:,i] = psftools.create_gauss_2d_rot_fit((x, y), 
                                                      bead_3d_window[k,:,:,i], 
                                                      par_init)

# for each bead calculate absolute z-pozition of the tightest focus and all
# parameters for that z
sigma_av = np.zeros((window_z*2+1,len(peaks5)))
sigma_min = np.zeros((len(peaks5),1))
k_min = np.zeros(len(peaks5), dtype=np.int)
sigma_x_min = np.zeros((len(peaks5),1))
sigma_y_min = np.zeros((len(peaks5),1))
theta_min = np.zeros((len(peaks5),1))
x0_min = np.zeros((len(peaks5),1))
y0_min = np.zeros((len(peaks5),1))
for i in range(len(peaks5)):
    sigma_av[:,i] = (np.abs(par[:,5,i]) + np.abs(par[:,6,i]))/2
    sigma_min[i] = np.min(sigma_av[:,i])
    k_min[i] = np.argmin(sigma_av[:,i])
    sigma_x_min[i] = par[k_min[i],5,i]
    sigma_y_min[i] = par[k_min[i],6,i]
    theta_min[i] = par[k_min[i],4,i]
    x0_min[i] = par[k_min[i],2,i]
    y0_min[i] = par[k_min[i],3,i]
z_tight = peaks5[:,0] - (window_z - k_min[:])      

#plot sigma_av and z_tight
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(peaks5[:,2], peaks5[:,1], c=z_tight*pxl_z, cmap='jet', 
            marker='o') # cmap='Reds'
plt.axis('scaled')    
plt.axis([0, img_width, 0, img_height])   
plt.colorbar()    
plt.title('z$_{tight}$ [$\mu$m]')
plt.subplot(1,2,2)
plt.scatter(peaks5[:,2], peaks5[:,1], c=sigma_min*pxl_xy, cmap='jet', 
            marker='o') # cmap='Reds'
plt.axis('scaled')    
plt.axis([0, img_width, 0, img_height])   
plt.colorbar()    
plt.title('$\sigma_{av}$ [$\mu$m]')    
plt.show 

plt.figure(figsize=(10,3))
plt.subplot(1,2,1) 
plt.hist(z_tight*pxl_z, bins=10)
plt.xlabel('z$_{tight}$ [$\mu$m]')
plt.ylabel('# of beads')
plt.title('z$_{tight}$ [$\mu$m]')   
plt.subplot(1,2,2)
plt.hist(sigma_min*pxl_xy, bins=20)
plt.xlabel('$\sigma_{av}$ [$\mu$m]')
plt.ylabel('# of beads')
plt.title('$\sigma_{av}$ [$\mu$m]')   
plt.show

# calculate PSF by averaging beads shifted to the pixel center
thresholds = [np.min(sigma_min*pxl_xy), np.mean(sigma_min*pxl_xy)]
h = plt.hist(sigma_min*pxl_xy, bins=20, color='b')
plt.vlines([thresholds[0], thresholds[1]], 0, max(h[0]), colors='r')      
plt.xlabel('$\sigma_{av}$ [$\mu$m]')
plt.ylabel('# of beads')
plt.title('$\sigma_{av}$ [$\mu$m]')   
plt.show
select = np.nonzero(sigma_min < np.mean(sigma_min))
psf = psftools.calculate_psf(bead_3d_window, k_min, x0_min, y0_min, window_xy, 
                             select=0) # average all beads if select=0
# use nonzero select to average beads with sigma < mean(sigma)
                                
# fit 2D rotated Gaussian to PSF
par_init_psf = psftools.get_fit_parameters((x,y), psf)
par_psf = psftools.create_gauss_2d_rot_fit((x, y), psf, par_init_psf)
sigma_psf = (par_psf[0,5] + par_psf[0,6])/2*pxl_xy
print 'PSF_xy FWHM: %.2f [nm]'% (sigma_psf*2.355e3)
fit_psf = psftools.gauss_2d_rot((x, y), par_psf[0,0], par_psf[0,1], par_psf[0,2], 
                       par_psf[0,3], par_psf[0,4], par_psf[0,5], par_psf[0,6])
fit_psf = fit_psf.reshape(window_xy*2+1, window_xy*2+1)

# plot PSF and Gaussian fit
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(psf, cmap='jet', vmin=0, vmax=np.max(psf), interpolation='nearest')
plt.colorbar()
plt.title('PSF')   
plt.subplot(1,2,2)
plt.imshow(fit_psf, cmap='jet', vmin=0, vmax=np.max(fit_psf),
                     interpolation='nearest')
plt.colorbar()
plt.title('PSF fit')   
plt.show 

# plot x- and y- profiles of PSF and Gaussian fit
plt.subplots(figsize=(5,4)) 
plt.plot(x[0,:], psf[window_xy,:], 'kx')
plt.plot(y[:,0], psf[:,window_xy], 'ro')
plt.plot(x[0,:], fit_psf[window_xy,:], 'k-')
plt.plot(y[:,0], fit_psf[:,window_xy], 'r-')
plt.xlabel('Distance [pxl]')
plt.ylabel('I [a.u.]')
plt.legend(['x', 'y', 'fit x', 'fit y'])
plt.title('PSF_xy Gaussian fit')                                           
plt.show

# save PSF as TIFF file
outpath = filepath[:-1] +'_output/'
if not os.path.isdir(outpath):
    os.mkdir(outpath)
filename_psf = 'ch' + str(num_channel) + '_psf.tif'
tifffile.imsave(outpath + filename_psf, np.uint16(psf))

# calculate 3D PSF
psf_3d, peaks6 = psftools.calculate_3d_psf(img_channel, z_tight, peaks5, 
                    window_z, window_xy, x0_min, y0_min, sigma_min, select=0)
# average all beads if select=0
# use nonzero select to average beads with sigma < mean(sigma)

# plot xy-, xz-, yz- projections of 3D PSF
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(psf_3d[window_z,:,:], cmap='jet', vmin=0, vmax=np.max(psf_3d),
                     interpolation='nearest')
plt.colorbar()
plt.xlabel('x [pxl]')
plt.ylabel('y [pxl]')
plt.title('PSF_xy')
plt.subplot(2,2,2)
plt.imshow(psf_3d[:,window_xy,:], cmap='jet', vmin=0, vmax=np.max(psf_3d),
                     interpolation='nearest')
plt.colorbar()
plt.xlabel('x [pxl]')
plt.ylabel('z [pxl]')
plt.title('PSF_yz')  
plt.subplot(2,2,3) 
plt.imshow(psf_3d[:,:,window_xy], cmap='jet', vmin=0, vmax=np.max(psf_3d),
                     interpolation='nearest')
plt.colorbar()
plt.xlabel('x [pxl]')
plt.ylabel('z [pxl]')
plt.title('PSF_xz')  
plt.show 
 
psf_z = psf_3d[:,window_xy,window_xy]

#fit 1D Gaussian to z-profile of 3D PSF
z = np.arange(window_z*2+1)
par_init_psf_z = np.max(psf_z), np.min(psf_z), window_z, sigma_psf*3 
par_psf_z = psftools.create_gauss_1d_fit(z, psf_z, par_init_psf_z)
sigma_psf_z = par_psf_z[0,3]*pxl_z
print 'PSF_z FWHM: %.2f [nm]'% (sigma_psf_z*2.355e3)
fit_psf_z = psftools.gauss_1d(z, par_psf_z[0,0], par_psf_z[0,1], par_psf_z[0,2], 
                       par_psf_z[0,3])

# plot z-profile of 3D PSF
plt.subplots(figsize=(5,4)) 
plt.plot(z, psf_z, 'kx')
plt.plot(z, fit_psf_z, 'k-')
plt.xlabel('Distance z [pxl]')
plt.ylabel('I [a.u.]')
plt.legend(['z', 'fit z'])
plt.title('PSF_z Gaussian fit')
plt.show

# calculate FOV tilt, flatness by fitting bead center zyx positions 
# peaks5 (based on max I) or peaks6 (based on 2D Gaussian fit) to a plane   
residuals, ax_field, ay_field, field_flatness = psftools.calculate_fov(peaks5,
                                                                pxl_xy, pxl_z)
# ax_field - z difference in nm across 200um distance
# ay_field - z difference in nm across 200um distance
# field_flatness - field flatness = max z - min z in nm (after tilt subtraction)                                                            

print 'ax_field: %.2f [nm/200um]'% ax_field
print 'ay_field: %.2f [nm/200um]'% ay_field
print 'field_flatness: %.2f [nm]'% field_flatness

# plot residuals
plt.subplots(figsize=(5,5))
plt.scatter(peaks5[:,2], peaks5[:,1], c=residuals*pxl_z, cmap='jet', 
            marker='o') # cmap='Reds'
plt.axis('scaled')    
plt.axis([0, img_width, 0, img_height])    
plt.colorbar()    
plt.title('residuals [$\mu$m]')    
plt.show 

plt.subplots(figsize=(5,3))
plt.hist(residuals*pxl_z, bins=10)
plt.xlabel('residuals [$\mu$m]')
plt.ylabel('# of beads')
plt.title('residuals [$\mu$m]')   
plt.show      