# coding=utf-8
"""
Big data analysis of object shape representations
Calculate predictions of deep CNN AlexNet finetuned on our stimuli.

Created on May 23, 2016

@author: goker erdogan
gokererdogan@gmail.com
https://github.com/gokererdogan/
"""

import numpy as np
from scipy import misc
import pandas as pd
import pandasql as psql

import caffe

INPUT_SIZE = (227, 227, 3)
LAYER_COUNT = 14  # except input layer


def preprocess_image(img):
    # image we get has a single channel, make it RGB
    cimg = np.zeros(img.shape + (3,))
    cimg[:, :, 0] = img
    cimg[:, :, 1] = img
    cimg[:, :, 2] = img
    # resize the image to AlexNet input size
    cimg = misc.imresize(cimg, INPUT_SIZE)
    # our stimuli is input upside down to alexnet, so flipud the image
    cimg = np.flipud(cimg)
    return cimg


def load_net():
    caffe.set_mode_cpu()
    net = caffe.Net('alexnet_blocks_deploy.prototxt',
                    'snapshot/alexnet_blocks_iter_1800.caffemodel',
                    caffe.TEST)
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load('ilsvrc_2012_mean.npy').mean(1).mean(1))  # mean pixel
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    # set batch size to 1
    net.blobs['data'].reshape(1, 3, 227, 227)

    return net, transformer


def calculate_prediction(net, transformer, input):
    net.blobs['data'].data[...] = transformer.preprocess('data', input)
    out = net.forward()
    layer_outputs = {}
    for k, v in net.blobs.iteritems():
        if k != 'data':
            layer_outputs[k] = v.data[0].copy()

    return layer_outputs

if __name__ == "__main__":
    # network is trained on grayscale images rendered by VTK, so use them (instead of blender renders)
    stimuli_folder = '../../../stimuli/stimuli20150624_144833/single_view_vtk'

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']
    comparisons = {o: [o + '_' + t for t in transformations] for o in objects}

    df = pd.DataFrame(index=np.arange(0, len(objects) * len(comparisons['o1']) * LAYER_COUNT),
                      columns=['Target', 'Comparison', 'Layer', 'Distance'], dtype=float)

    net, transformer = load_net()

    i = 0
    for obj in objects:
        print(obj)

        target_file = "{0:s}/{1:s}_0.png".format(stimuli_folder, obj)
        target_img = misc.imread(target_file)
        target_img = preprocess_image(target_img)

        target_net_outputs = calculate_prediction(net, transformer, target_img)

        for comparison in comparisons[obj]:
            print('.'),
            comparison_file = "{0:s}/{1:s}_0.png".format(stimuli_folder, comparison)
            comparison_img = misc.imread(comparison_file)
            comparison_img = preprocess_image(comparison_img)

            comparison_net_outputs = calculate_prediction(net, transformer, comparison_img)

            # compare the output of each layer
            for layer in target_net_outputs.keys():
                mse = np.mean(np.square(target_net_outputs[layer] - comparison_net_outputs[layer]))
                df.loc[i] = [obj, comparison, "AlexNet_finetuned_" + layer, mse]
                i += 1

    # calculate model predictions
    predictions = psql.sqldf("select d1.Target as Target, d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                             "d1.Layer as Layer, d1.Distance<d2.Distance as AlexNet_FT_Prediction "
                             "from df as d1, df as d2 "
                             "where d1.Target = d2.Target and d1.Comparison<d2.Comparison and d1.Layer = d2.Layer",
                             env=locals())

    # Put prediction from each layer to a separate column
    predictions = pd.pivot_table(predictions, index=['Target', 'Comparison1', 'Comparison2'], columns='Layer',
                                 values='AlexNet_FT_Prediction')
    # convert MultiIndex to columns
    predictions.reset_index(inplace=True)
    del predictions.columns.name # get rid of name

    # write to disk
    open('AlexNet_FT_ModelPredictions.txt', 'w').write(predictions.to_string())
    # open('../../../R/BDAoOSS_Synthetic/AlexNet_ModelPredictions.txt', 'w').write(predictions.to_string())
    open('AlexNet_FT_ModelDistances.txt', 'w').write(df.to_string())

