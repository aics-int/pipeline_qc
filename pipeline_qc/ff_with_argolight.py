import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
from scipy import interpolate
from scipy.optimize import curve_fit
from skimage import filters, measure, io

channel = '488'

ff_f_data = AICSImage(r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ZSD1_20190813\3500003331_100X_20190813_' + channel + '.czi')
ff_f = ff_f_data.data[0, 0, 0, :, :]
br_data = AICSImage(r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ZSD1_20190813\3500003331_100X_20190813_BR.czi')
br = br_data.data[0, 0, 0, :, :]
argo_data = AICSImage(r'C:\Users\calystay\Desktop\argo_488_test.czi')
argo = argo_data.data[0, 0, 0, : ,:]

ff_smooth = filters.gaussian(image=ff_f, sigma=3, preserve_range=True)
ff_norm = ff_smooth/np.max(ff_smooth)

img_mask = np.zeros((ff_norm.shape), dtype=bool)
for x in range (50, img_mask.shape[1], 200):
    for y in range (25, img_mask.shape[0], 180):
        img_mask[y:y+25, x:x+25] = True

plot_profile(ff_f, px_crop=100, fit=False) # Intensity profile of raw dye ff image
plot_profile(norm_img=ff_norm, px_crop=100, fit=False) # Intensity profile of normalized dye ff

#=======================================================================================================================
# Generate simulated homogeneity map with dye ff (sample across the image)
masked_ff = ff_smooth*img_mask
label_ref = measure.label(img_mask)

field_non_uni_raw, z, coors = generate_homogeneity_ref(label_ref=label_ref, img_raw=ff_smooth, mode='mean')
norm_corr = field_non_uni_raw/np.max(field_non_uni_raw)
plot_profile(norm_corr, px_crop=100, fit=False) # Intensity profile of normalized simulated sampled ff

#=======================================================================================================================
# Use a simulated homogeneity map to correct images (requires black reference, homogeneity map)
smooth_br = filters.gaussian(br, sigma=3, preserve_range=True)

# Use sampled FF to correct full FF
corr_ff = correct_img(ff_f[100:-100, 100:-100], smooth_br[100:-100, 100:-100], field_non_uni_raw[100:-100, 100:-100])
plt.figure()
plt.imshow(corr_ff)
norm_corr_ff = corr_ff/np.max(corr_ff)
plot_profile(norm_img=norm_corr_ff, px_crop=12, fit=False)

# Use sampled FF to correct for argo
corr_argo = correct_img(argo[100:-100, 100:-100], smooth_br[100:-100, 100:-100], field_non_uni_raw[100:-100, 100:-100])
plt.figure()
plt.imshow(corr_argo)

#=======================================================================================================================
# Generate homogeneity maps from argolight
# Option 1: homogeneity_raw_map.png output from daybook
compare = io.imread(r'\\allen\aics\microscopy\Calysta\argolight\zsd1_20190813\homogeneity_raw_map.png')
compare_norm = compare/np.max(compare)
plot_profile(compare_norm, 100, fit=False)

# Option 2: From rings image, segment just the rings as samples
argo_smooth = filters.gaussian(argo, sigma=3, preserve_range=True)
thresh = filters.threshold_local(argo_smooth, block_size=11, offset=10)
ff_argo_segment = (argo_smooth>thresh) & (argo_smooth>500)

show = argo_smooth*ff_argo_segment
plt.figure()
plt.imshow(show)
label_ref = measure.label(ff_argo_segment)

field_non_uni_raw, z, coors = generate_homogeneity_ref(label_ref=label_ref, img_raw=argo_smooth, mode='median')
norm_f = field_non_uni_raw/np.max(field_non_uni_raw)
plot_profile(norm_f, px_crop=100) # Intensity profile of normalized simulated rings ff

# Option 3: From rings image, segment rings with the centered cross
update = label_ref.copy()
update[label_ref==15] = 0

masked_update = argo_smooth*update
field_non_uni_raw, z, coors = generate_homogeneity_ref(label_ref=update, img_raw=argo_smooth, mode='median')
norm_f = field_non_uni_raw/np.max(field_non_uni_raw)
plot_profile(norm_f, px_crop=100) # Intensity profile of normalized simulated rings+cross ff

#=======================================================================================================================
# TODO: Curve fitting method does not extend to edge of image. Need to add functionality!!!
params, cov = curve_fit(f=fit_2d_parabaloid, xdata=np.meshgrid(np.arange(624), np.arange(924)), ydata= )
def fit_2d_parabaloid(x, a, b, c, d, e, f):
    return a*(x-b)**2 + c*(y-d)**2 + e*x*y + f

#=======================================================================================================================
# Functions developed

def plot_profile (norm_img, px_crop, fit=False):
    positive_profile = measure.profile_line(image=norm_img, src=(norm_img.shape[0], 0),
                                            dst=(0, norm_img.shape[1]))
    negative_profile = measure.profile_line(image=norm_img, src=(norm_img.shape[0], norm_img.shape[1]),
                                            dst=(0, 0))
    roll_off_pos = find_roll_off(positive_profile[px_crop:-px_crop])
    roll_off_neg = find_roll_off(negative_profile[px_crop:-px_crop])
    x_data = np.linspace(0, 1, len(negative_profile[px_crop:-px_crop]))

    plt.figure()
    #plt.ylim((0,1))
    plt.xlim(px_crop, len(negative_profile)-px_crop)
    plt.plot(negative_profile[5:-5], 'r')
    plt.plot(positive_profile[5:-5], 'b')
    plt.title('roll-off for ' + channel + ': ' + str(np.min([roll_off_neg, roll_off_pos])))

    if fit:
        popt_neg, pcov_neg = curve_fit(f=fit_func, xdata=x_data, ydata=negative_profile[px_crop:-px_crop])
        popt_pos, pcov_pos = curve_fit(f=fit_func, xdata=x_data, ydata=positive_profile[px_crop:-px_crop])
        plt.plot(fit_func(x_data, *popt_neg), 'r-')
        plt.plot(fit_func(x_data, *popt_pos), 'b-')


def generate_homogeneity_ref(label_ref, img_raw, mode):
    props = measure.regionprops(label_ref)
    all_coors = []
    for x in range(0, label_ref.shape[1]):
        for y in range(0, label_ref.shape[0]):
            coor = (int(y), int(x))
            all_coors.append(coor)
    z = []
    coors = []
    for prop in props:
        centroid = prop.centroid
        obj_label = prop.label
        intensities = img_raw[np.where(label_ref == obj_label)]
        mean_int = np.average(intensities)
        max_int = np.max(intensities)
        if mode == 'mean':
            z.append(mean_int)
        elif mode == 'max':
            z.append(max_int)
        elif mode == 'median':
            z.append(np.median(intensities))

        coors.append((int(centroid[0]), int(centroid[1])))
    print (coors)
    grid_0 = interpolate.griddata(points=coors, values=z, xi=all_coors, method='cubic', fill_value=False)
    field_non_uni_raw = np.zeros((label_ref.shape))
    for x in range(0, len(grid_0)):
        point = all_coors[x]
        value = grid_0[x]
        field_non_uni_raw[point] = value
    return field_non_uni_raw, z, coors


def find_roll_off (profile):
    roll_off = (np.max(profile) - np.min(profile))/np.max(profile)
    return roll_off


def fit_func(x, a, b, c):
    return c - a * ((x - b) ** 2)


def correct_img(img_to_corr, br, img_homogeneity_ref):
    corr = (img_to_corr - br)/img_homogeneity_ref
    return corr


