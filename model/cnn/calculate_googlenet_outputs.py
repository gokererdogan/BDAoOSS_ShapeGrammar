# coding=utf-8
"""
Big data analysis of object shape representations
Calculate outputs of the deep CNN GoogLeNet.

Created on Feb 9, 2016

@author: goker erdogan
gokererdogan@gmail.com
https://github.com/gokererdogan/
"""

import numpy as np

from calculate_googlenet_predictions import *

if __name__ == "__main__":
    stimuli_folder = '../stimuli/stimuli20150624_144833/single_view'
    caffe_root = "/home/goker/Programs/caffe/"

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']
    variations = [o + '_' + t for t in transformations for o in objects]

    net, transformer, labels = load_googlenet(caffe_root)

    layers = ['pool1/norm1', 'conv2/norm2', 'inception_3a/output', 'inception_3b/output', 'pool3/3x3_s2',
              'inception_4a/output', 'inception_4b/output', 'inception_4c/output', 'inception_4d/output',
              'inception_4e/output', 'pool4/3x3_s2', 'inception_5a/output', 'inception_5b/output',
              'pool5/7x7_s1', 'loss3/classifier', 'prob']

    output_mats = {}
    for layer in layers:
        shape = net.blobs[layer].data.shape
        output_mats[layer] = np.zeros((90, np.prod(shape) / shape[0]))

    row_labels = []

    i = 0
    for obj in objects:
        print(obj)

        target_file = "{0:s}/{1:s}.png".format(stimuli_folder, obj)
        target_img = misc.imread(target_file)
        target_img = preprocess_image(target_img)

        target_net_outputs, target_class_labels = calculate_googlenet_prediction(net, transformer, labels, target_img)

        for layer in layers:
            output_mats[layer][i, :] = np.ravel(target_net_outputs[layer])

        row_labels.append(obj)

        i += 1

    for obj in variations:
        print(obj)

        target_file = "{0:s}/{1:s}.png".format(stimuli_folder, obj)
        target_img = misc.imread(target_file)
        target_img = preprocess_image(target_img)

        target_net_outputs, target_class_labels = calculate_googlenet_prediction(net, transformer, labels, target_img)

        for layer in layers:
            output_mats[layer][i, :] = np.ravel(target_net_outputs[layer])

        row_labels.append(obj)

        i += 1

    for layer in layers:
        print(layer)
        print("\tNumber of columns: {0:d}".format(output_mats[layer].shape[1]))
        # get rid of columns with constant values
        output_mean = np.mean(output_mats[layer], axis=0)
        output_mats[layer] -= output_mean
        nonzero_columns = np.logical_not(np.isclose(np.sum(np.abs(output_mats[layer]), axis=0), b=0.0))
        output_mats[layer] = output_mats[layer][:, nonzero_columns]
        output_mean = output_mean[nonzero_columns]
        output_mats[layer] += output_mean
        print("\tNumber of non-zero columns: {0:d}".format(output_mats[layer].shape[1]))
        np.save('outputs/googlenet_{0:s}.npy'.format(layer.replace('/', '_')), output_mats[layer])

    open('outputs/googlenet_row_labels.txt', 'w').write(repr(row_labels))
