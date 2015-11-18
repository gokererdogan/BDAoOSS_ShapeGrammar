# coding=utf-8
"""
Big data analysis of object shape representations
Calculate predictions of the deep CNN AlexNet.

Code for loading/using AlexNet is taken from
    http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

Created on Nov 3, 2015

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
LAYER_COUNT = 14 # except input layer

def preprocess_image(img):
    # get rid of alpha channel
    img = img[:, :, 0:3]
    # resize the image to AlexNet input size
    img = misc.imresize(img, INPUT_SIZE)
    return img

def load_alexnet(caffe_root):
    caffe.set_mode_cpu()
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    # set batch size to 1
    net.blobs['data'].reshape(1, 3, 227, 227)

    imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

    return net, transformer, labels

def calculate_alexnet_prediction(net, transformer, labels, input):
    net.blobs['data'].data[...] = transformer.preprocess('data', input)
    out = net.forward()
    layer_outputs = {}
    for k, v in net.blobs.iteritems():
        if k != 'data':
            layer_outputs[k] = v.data[0].copy()

    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    class_labels = labels[top_k]

    return layer_outputs, class_labels

if __name__ == "__main__":
    stimuli_folder = '../stimuli/stimuli20150624_144833/single_view'
    caffe_root = "/home/goker/Programs/caffe/"

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']

    df = pd.DataFrame(index=np.arange(0, 80 * LAYER_COUNT), columns=['Target', 'Comparison', 'Layer', 'Distance'],
                      dtype=float)

    net, transformer, labels = load_alexnet(caffe_root)

    i = 0
    for obj in objects:
        print(obj)

        target_file = "{0:s}/{1:s}.png".format(stimuli_folder, obj)
        target_img = misc.imread(target_file)
        target_img = preprocess_image(target_img)

        target_net_outputs, target_class_labels = calculate_alexnet_prediction(net, transformer, labels, target_img)

        for transformation in transformations:
            comparison = "{0:s}_{1:s}".format(obj, transformation)
            comparison_file = "{0:s}/{1:s}.png".format(stimuli_folder, comparison)
            comparison_img = misc.imread(comparison_file)
            comparison_img = preprocess_image(comparison_img)

            comparison_net_outputs, comparison_class_labels = calculate_alexnet_prediction(net, transformer, labels,
                                                                                           comparison_img)

            # compare the output of each layer
            for layer in target_net_outputs.keys():
                mse = np.mean(np.square(target_net_outputs[layer] - comparison_net_outputs[layer]))
                df.loc[i] = [obj, comparison, "AlexNet_" + layer, mse]
                i += 1

    # calculate model predictions
    predictions = psql.sqldf("select d1.Comparison as Comparison1, d2.Comparison as Comparison2, d1.Layer as Layer,"
                             "d1.Distance<d2.Distance as AlexNet_Prediction "
                             "from df as d1, df as d2 "
                             "where d1.Target = d2.Target and d1.Comparison<d2.Comparison and d1.Layer = d2.Layer",
                             env=locals())

    # Put prediction from each layer to a separate column
    predictions = pd.pivot_table(predictions, index=['Comparison1', 'Comparison2'], columns='Layer',
                                 values='AlexNet_Prediction')
    # convert MultiIndex to columns
    predictions.reset_index(inplace=True)
    del predictions.columns.name # get rid of name

    # write to disk
    open('AlexNet_ModelPredictions.txt', 'w').write(predictions.to_string())
    open('../../../R/BDAoOSS_Synthetic/AlexNet_ModelPredictions.txt', 'w').write(predictions.to_string())
    open('AlexNet_ModelDistances.txt', 'w').write(df.to_string())

