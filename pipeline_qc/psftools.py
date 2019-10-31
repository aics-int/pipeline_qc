# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:19:42 2016

@author: olgag
"""

#psftools.py
#this module contains all functions used to calculate psf and fov

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.optimize as opt

def find_peaks(image):
    # find peaks by locating local maxima
    threshold = np.median(image) + np.std(image)*5
    image_max = ndimage.maximum_filter(image,size = 3)
    peaks1 = np.nonzero((image == image_max) & (image> threshold))
    peaks1 = np.transpose(peaks1)

    # plot peaks on image (optional)
    plt.subplots(figsize=(10,10))
    plt.imshow(image, cmap='jet', vmin=0, vmax=np.median(image)+np.std(image)*5,
               interpolation='nearest') # cmap = 'gray'
    plt.colorbar()
    plt.scatter(peaks1[:,1], peaks1[:,0], c='y', marker='x')
    plt.axis([0, image.shape[1], 0, image.shape[0]])
    plt.show
    return peaks1 

def remove_close_peaks(image, peaks1, min_dist=30):
    # remove peaks that are too close to each other
    select = np.ones((len(peaks1),1), dtype=np.int)    
    for i in range(len(peaks1)):
        for j in range(len(peaks1)):
            dist = np.sqrt(np.sum((peaks1[i,:] - peaks1[j,:])**2)) 
            if dist < min_dist + 1 and i!=j:
                select[i] = 0
    peaks2 = peaks1[np.nonzero(select[:,0]==1)]
    
    # plot peaks on image (optional)
#    plt.subplots(figsize=(10,10))
#    plt.imshow(image, cmap='jet', vmin=0, vmax=np.median(image)+np.std(image)*5,
#               interpolation='nearest') # cmap = 'gray'
#    plt.colorbar()
#    plt.scatter(peaks2[:,1], peaks2[:,0], c='m', marker='x')
#    plt.axis([0, image.shape[1], 0, image.shape[0]])
#    plt.show
    return peaks2
    
def remove_near_edge(image, peaks2, window_xy=10):
    # remove peaks that are near the edges
    select = np.nonzero((peaks2[:,0] > window_xy - 1) 
                        & (peaks2[:,0] < image.shape[0] - window_xy)
                        & (peaks2[:,1] > window_xy - 1)
                        & (peaks2[:,1] < image.shape[1] - window_xy))
    peaks3 = peaks2[select]
    
    # plot peaks on image (optional)
#    plt.subplots(figsize=(10,10))
#    plt.imshow(image, cmap='jet', vmin=0, vmax=np.median(image)+np.std(image)*5,
#               interpolation='nearest') # cmap = 'gray'
#    plt.colorbar()
#    plt.scatter(peaks3[:,1], peaks3[:,0], c='g', marker='x')
#    plt.axis([0, image.shape[1], 0, image.shape[0]])
#    plt.show
    return peaks3
    
def find_3d_max(image, img_channel, peaks3, window_xy=10, window_z=10):
    # for each bead find xyz of the max intensity pixel
    int_max = np.zeros((len(peaks3),1), dtype=np.int)    
    peaks4 = np.zeros((len(peaks3),3), dtype=np.int)
    for i in range(len(peaks3)):
        y, x = peaks3[i,:]
        beadstack = img_channel[:, y-window_xy:y+window_xy+1, 
                                x-window_xy:x+window_xy+1]
        int_max[i] = np.max(beadstack)        
        z_max, y_max, x_max = np.unravel_index(beadstack.argmax(),
                                                        beadstack.shape)
        peaks4[i,:] = [z_max, y_max+y-window_xy, x_max+x-window_xy]  
   
    # remove peaks that are near the top/bottom of the stack
    select = np.nonzero((peaks4[:,0] > window_z-1) 
                        & (peaks4[:,0] < img_channel.shape[0] - window_z))
    peaks4 = peaks4[select]
    
    # plot peaks on image (optional)
#    plt.subplots(figsize=(10,10))
#    plt.imshow(image, cmap='jet', vmin=0, vmax=np.median(image)+np.std(image)*5,
#               interpolation='nearest') # cmap = 'gray'
#    plt.colorbar()
#    plt.scatter(peaks4[:,2], peaks4[:,1], c='c', marker='x')
#    plt.axis([0, image.shape[1], 0, image.shape[0]])
#    plt.show
    return peaks4
   
def remove_dim_bright(image, img_channel, peaks4, pxl_z, window_xy=10, 
                      window_z=10):
    # remove peaks that are too dim/bright
    bead_3d_window = np.zeros((window_z*2+1, window_xy*2+1, window_xy*2+1, 
                               len(peaks4)), dtype=np.int)    
    for i in range(len(peaks4)):
        z, y, x = peaks4[i,:]
        bead_3d_window[:,:,:,i] = img_channel[z-window_z:z+window_z+1, 
                 y-window_xy:y+window_xy+1, x-window_xy:x+window_xy+1]
    int_total = bead_3d_window.sum(axis=(0,1,2))
    
    f1 = 1.2 #added this 02/12/18 (vs 1)
    f2 = 1.2
    thresholds = [np.median(int_total)-f1*np.std(int_total), 
                  np.median(int_total)+f2*np.std(int_total)]
    
    # plot histogram of total intensity   
    plt.subplots(figsize=(5,3))    
    h = plt.hist(int_total, bins=50)
    plt.vlines([thresholds[0], thresholds[1]], 0, max(h[0]), colors='r')      
    plt.xlabel('Total I')
    plt.ylabel('# of beads')
    plt.title('Total I distribution')
    plt.show
    select = np.nonzero((int_total > thresholds[0]) & 
                        (int_total < thresholds[1]))     
    peaks5 = peaks4[select]
    bead_3d_window = np.squeeze(bead_3d_window[:,:,:,select])
    
    # plot peaks on image (optional)
#    plt.subplots(figsize=(10,10))
#    plt.imshow(image, cmap='jet', vmin=0, vmax=np.median(image)+np.std(image)*5,
#               interpolation='nearest') # cmap = 'gray'
#    plt.colorbar()
#    plt.scatter(peaks5[:,2], peaks5[:,1], c='r', marker='x')
#    plt.axis([0, image.shape[1], 0, image.shape[0]])
#    plt.show
    
    #plot xyz position of selected beads based on int_max
    plt.subplots(figsize=(5,5))
    plt.scatter(peaks5[:,2], peaks5[:,1], c=peaks5[:,0]*pxl_z, cmap='jet', 
                marker='o') # cmap='Reds'
    plt.axis('scaled')    
    plt.axis([0, image.shape[1], image.shape[0], 0])
    plt.colorbar()    
    plt.title('z [$\mu$m]')    
    plt.show
    return peaks5, bead_3d_window   
    
def get_fit_parameters((x,y), image):
    # calculate initial fit parameters
    amp = image.max()
    offset = image.min()
    
    # calculate center of mass and covariance
    cov = np.zeros((2,2))
    image_corr = image-offset;
    m00 = np.sum(image_corr) #0 th raw moment=total I of plane k
    m10 = np.sum(np.sum(image_corr, axis=0)*x[0,:]) # 1st order raw moment
    m01 = np.sum(np.sum(image_corr, axis=1)*y[:,0]) # 1st order raw moment
    x0 = m10/m00 # center of mass x0
    y0 = m01/m00 # center of mass y0
    m11 = np.sum(image_corr*x*y) # 1st order raw moment
    m20 = np.sum(np.sum(image_corr, axis=0)*x[0,:]*x[0,:]) # 2nd order raw moment
    m02 = np.sum(np.sum(image_corr, axis=1)*y[:,0]*y[:,0]) # 2nd order raw moment
    mu11 = m11/m00 - x0*y0
    mu20 = m20/m00 - x0**2
    mu02 = m02/m00 - y0**2
    cov[:,:] = np.array([[mu20,mu11], [mu11,mu02]]) # covariance matrix
    
    # calculate eigenvectors and eigenvalues of covariance matrix
    eigval, eigvec = np.linalg.eigh(cov)
    
    # calculate theta, sigma_x, sigma_y 
    theta = np.arctan(eigvec[0,0]/eigvec[1,0]) # theta [rad]
    sigma_x = np.sqrt(eigval[0])
    sigma_y = np.sqrt(eigval[1])
    return amp, offset, x0, y0, theta, sigma_x, sigma_y 
 
def gauss_2d_rot((x, y), amp, offset, x0, y0, theta, sigma_x, sigma_y):
    # 2D rotated Gaussian   
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    z = offset + amp*np.exp( - (a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    return np.ravel(z)
    
def create_gauss_2d_rot_fit((x,y),image,par_init):
    # fit 2D rotated Gaussian to image
    popt, pcov = opt.curve_fit(gauss_2d_rot, (x, y), np.ravel(image), p0=par_init)
    return popt.reshape(1,7)   

def calculate_psf(bead_3d_window, k_min, x0_min, y0_min, window_xy, select):   
    beads_tight_shift = np.zeros(bead_3d_window.shape[1:4])
    for i in range(bead_3d_window.shape[3]): #range(len(peaks5)):
        beads_tight = bead_3d_window[k_min[i],:,:,i]
        shift = [window_xy-float(y0_min[i]), window_xy-float(x0_min[i])]   
        beads_tight_shift[:,:,i] = ndimage.interpolation.shift(beads_tight,
                                    shift, mode='nearest')
    if select==0:
        psf = np.sum(beads_tight_shift, axis=2)/bead_3d_window.shape[3]
        # average of all shifted beads
        print 'Use all beads to calculate PSF'    
    else:
        psf = np.sum(beads_tight_shift[:,:,select[0]], axis=2)/len(select[0]) 
        # average of shifted beads with sigma<mean sigma   
        #print 'Use beads with sigma < mean(sigma) to calculate PSF'
    return psf

def calculate_3d_psf(img_channel, z_tight, peaks5, window_z, window_xy, x0_min,
                     y0_min, sigma_min, select):
    # remove beads with z_tight too close to the top/bottom of the image stack    
    select1 = np.nonzero((z_tight > window_z-1) 
                        & (z_tight < img_channel.shape[0] - window_z))
    peaks6 = peaks5[select1]
    peaks6[:,0] = z_tight[select1]    
    sigma_min = sigma_min[select1]
    
    bead_stack_shift = np.zeros((window_z*2+1, window_xy*2+1, window_xy*2+1, 
                               len(peaks6)))    
    #bead_stack = np.zeros((window_z*2+1, window_xy*2+1, window_xy*2+1))   
    for i in range(len(peaks6)):
        z, y, x = peaks6[i,:]
        bead_stack = img_channel[z-window_z:z+window_z+1, 
                 y-window_xy:y+window_xy+1, x-window_xy:x+window_xy+1]
        shift = [window_xy-float(y0_min[i]), window_xy-float(x0_min[i])]          
        for k in range(window_z*2+1):
            bead_stack_shift[k,:,:,i] = ndimage.interpolation.shift(bead_stack[k,:,:],
                                    shift, mode='nearest')
    if select==0:
        psf_3d = np.sum(bead_stack_shift, axis=3)/len(peaks6) 
        #average of all shifted beads
        print 'Use all beads to calculate 3D PSF' 
    else:    
        #select = np.nonzero(sigma_min < np.mean(sigma_min)) #comment 02/13/18
        psf_3d = np.sum(bead_stack_shift[:,:,:,select[0]], axis=3)/len(select[0]) 
        #average of shifted beads with sigma<mean sigma
        #print 'Use beads with sigma < mean(sigma) to calculate 3D PSF'
    return psf_3d, peaks6

def calculate_3d_psf1(img_channel, z_tight, peaks5, window_z, window_xy, x0_min,
                     y0_min, sigma_min, select):
    # remove beads with z_tight too close to the top/bottom of the image stack    
#    select1 = np.nonzero((z_tight > window_z-1) 
#                        & (z_tight < img_channel.shape[0] - window_z))
#    peaks6 = peaks5[select1]
#    peaks6[:,0] = z_tight[select1]    
#    sigma_min = sigma_min[select1]
    peaks6 = np.copy(peaks5)
    peaks6[:,0] = z_tight[:]
    z_shift = z_tight - np.min(z_tight)
    bead_stack_shift = np.zeros((img_channel.shape[0], window_xy*2+1, window_xy*2+1, 
                               len(peaks6)))    
    #bead_stack = np.zeros((window_z*2+1, window_xy*2+1, window_xy*2+1))   
    for i in range(len(peaks6)):
        z, y, x = peaks6[i,:]
        #bead_stack = img_channel[:, y-window_xy:y+window_xy+1, x-window_xy:x+window_xy+1]
        bead_stack = img_channel[0+z_shift[i]:, y-window_xy:y+window_xy+1, x-window_xy:x+window_xy+1]
        shift = [window_xy-float(y0_min[i]), window_xy-float(x0_min[i])]          
        for k in range(bead_stack.shape[0]):
            bead_stack_shift[k,:,:,i] = ndimage.interpolation.shift(bead_stack[k,:,:],
                                    shift, mode='nearest')
    if select==0:
        psf_3d = np.sum(bead_stack_shift, axis=3)/len(peaks6) 
        #average of all shifted beads
        print 'Use all beads to calculate 3D PSF' 
    else:    
        #select = np.nonzero(sigma_min < np.mean(sigma_min)) #comment 02/13/18
        psf_3d = np.sum(bead_stack_shift[:,:,:,select[0]], axis=3)/len(select[0]) 
        #average of shifted beads with sigma<mean sigma
        #print 'Use beads with sigma < mean(sigma) to calculate 3D PSF'
    return psf_3d, peaks6

def gauss_1d(x, amp, offset, x0, sigma_x):
    #1D Gaussian
    z = offset + amp*np.exp(-((x-x0)/sigma_x)**2)
    return z
    
def create_gauss_1d_fit(x, profile, par_init):
    # fit 1D rotated Gaussian to curve
    popt, pcov = opt.curve_fit(gauss_1d, x, profile, p0=par_init)
    return popt.reshape(1,4)
    
def fitpoly((x, y), a, b, c):
    # polynomial    
    z = a*x + b*y + c
    return z    
    
def create_plane_fit((x, y), z):
    #fit xyz surface to a plane    
    popt, pcov = opt.curve_fit(fitpoly, (x, y), z)
    return popt
    
def calculate_fov(peaks, pxl_xy, pxl_z):
    x_data = peaks[:,2]
    y_data = peaks[:,1] 
    z_data = peaks[:,0]
    par1 = create_plane_fit((x_data, y_data), z_data)
    fit_plane = x_data*par1[0] + y_data*par1[1] + par1[2]
    residuals = fit_plane - z_data  
    ax_field = par1[0]*2e5*pxl_z/pxl_xy #z difference in nm across 200um distance
    ay_field = par1[1]*2e5*pxl_z/pxl_xy #z difference in nm across 200um distance
    field_flatness = (residuals.max() - residuals.min())*pxl_z*1000 #field 
    # flatness in nm (after tilt subtraction)
    return residuals, ax_field, ay_field, field_flatness