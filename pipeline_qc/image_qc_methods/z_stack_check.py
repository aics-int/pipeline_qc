import numpy as np


def z_stack_order_check(image):
    # Takes a single channel image and assesses whether the maximum average intensity of the image is
    # at the bottom zstack. Also outputs what zstack is the brightest

    result = dict()
    im = np.array(image)
    means = []
    for i in range(im.shape[0]):
        means.append(im[i].mean())

    if np.argmax(means) == 0:
        out_of_order = True
    else:
        out_of_order = False

    result.append({"Max Intensity z-stack": np.argmax(means)})
    result.append({"Z-stacks out of order": out_of_order})

    return result

