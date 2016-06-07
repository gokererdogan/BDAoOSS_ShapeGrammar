"""
Fine tune alexnet using our block stimuli.

21 May 2016
goker erdogan
https://github.com/gokererdogan
"""

import numpy as np
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

# caffe.set_mode_cpu()

# solver_fixed learns only the last fully connected layer, solver learns all layers.
fix_weights = False
append_str = "_fixed" if fix_weights else ""
solver = caffe.get_solver("solver{0:s}.prototxt".format(append_str))
solver.net.copy_from("pretrained_alexnet/bvlc_reference_caffenet.caffemodel")

# train and test
train_iter = 1800
test_interval = 18
test_iter = 18

epoch_count = int(train_iter / test_interval)

train_loss = np.zeros((epoch_count, test_interval))
test_loss = np.zeros((epoch_count, test_iter))
test_accuracy = np.zeros((epoch_count, test_iter))
for e in range(epoch_count):
    tot_loss = 0.0
    for i in range(test_interval):
        solver.step(1)
        loss = solver.net.blobs['loss'].data
        train_loss[e, i] = loss
        tot_loss += loss
    print "Epoch {0:d}, Training loss: {1:.4f}\n".format(e, tot_loss / test_interval)

    tot_loss = 0.0
    tot_accuracy = 0.0
    for i in range(test_iter):
        solver.test_nets[0].forward()
        acc = solver.test_nets[0].blobs['accuracy'].data
        test_accuracy[e, i] = acc
        tot_accuracy += acc
        loss = solver.test_nets[0].blobs['loss'].data
        test_loss[e, i] = loss
        tot_loss += loss
    print "Epoch {0:d}, Test loss: {1:.4f}, Test accuracy: {2:.4f}\n".format(e, tot_loss / test_iter, tot_accuracy / test_iter)

np.save('train_loss{0:s}.npy'.format(append_str), train_loss)
np.save('test_loss{0:s}.npy'.format(append_str), test_loss)
np.save('test_accuracy{0:s}.npy'.format(append_str), test_accuracy)
