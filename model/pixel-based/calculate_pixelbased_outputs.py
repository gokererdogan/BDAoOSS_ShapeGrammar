# coding=utf-8
"""
Big data analysis of object shape representations
Calculate outputs of the pixel-based model, which are the input images themselves!

Created on Feb 9, 2016

@author: goker erdogan
gokererdogan@gmail.com
https://github.com/gokererdogan/
"""

import numpy as np
import scipy.misc as misc

import gmllib.helpers as hlp

if __name__ == "__main__":
    stimuli_folder = '../stimuli/stimuli20150624_144833/single_view'

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']
    variations = [o + '_' + t for t in transformations for o in objects]

    img_size = (100, 100)
    output = np.zeros((90, np.prod(img_size)))
    row_labels = []

    i = 0
    for obj in objects:
        print(obj)

        target_file = "{0:s}/{1:s}.png".format(stimuli_folder, obj)
        target_img = misc.imread(target_file)
        target_img = hlp.rgb2gray(target_img)
        target_img = misc.imresize(target_img, img_size)

        output[i, :] = np.ravel(target_img)

        row_labels.append(obj)

        i += 1

    for obj in variations:
        print(obj)

        target_file = "{0:s}/{1:s}.png".format(stimuli_folder, obj)
        target_img = misc.imread(target_file)
        target_img = hlp.rgb2gray(target_img)
        target_img = misc.imresize(target_img, img_size)

        output[i, :] = np.ravel(target_img)

        row_labels.append(obj)

        i += 1

    print("\tNumber of columns: {0:d}".format(output.shape[1]))
    # get rid of columns with constant values
    output_mean = np.mean(output, axis=0)
    output -= output_mean
    nonzero_columns = np.logical_not(np.isclose(np.sum(np.abs(output), axis=0), b=0.0))
    output = output[:, nonzero_columns]
    output_mean = output_mean[nonzero_columns]
    output += output_mean
    print("\tNumber of non-zero columns: {0:d}".format(output.shape[1]))
    np.save('outputs/pb_.npy', output)

    open('outputs/pb_row_labels.txt', 'w').write(repr(row_labels))
