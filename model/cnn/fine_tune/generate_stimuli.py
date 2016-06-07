"""
Generate random stimuli for fine-tuning AlexNet.

19 May 2016
goker erdogan
https://github.com/gokererdogan
"""

import numpy as np
from scipy import misc
import lmdb
import caffe

import Infer3DShape.bdaooss_shape_maxd as bdaooss
import Infer3DShape.vision_forward_model as vfm

RENDER_SIZE = (256, 256)
RENDER_WIDTH = RENDER_SIZE[0]
RENDER_HEIGHT = RENDER_SIZE[1]

ANGLE_INCREMENT = 10
POSITION_BOUND = 2.0
STIMULI_COUNT = 100
IMAGE_PER_STIMULI = int(360/ANGLE_INCREMENT)
TRAIN_RATIO = 0.75
VAL_RATIO = 1 - TRAIN_RATIO


def _check_bounds(ss):
    for k, v in ss.iteritems():
        if np.sqrt(np.sum(np.square(v.position))) + np.sqrt(np.sum(np.square(v.size))) > POSITION_BOUND:
            return False
    return True


def generate_random_stimulus():
    while True:
        # note the viewpoint distance to origin. since alexnet is trained on 227x227 crops of
        # a 256x256 image, we take the camera a bit farther out to make the object fit in the
        # crop
        s = bdaooss.BDAoOSSShapeMaxD(forward_model=fwm, max_depth=4,
                                     viewpoint=[[np.sqrt(8.0)*(256/227.), 45.0, 45.0]])

        part_count = len(s.shape.spatial_model.spatial_states)
        # check shape
        if 3 < part_count < 14 and _check_bounds(s.shape.spatial_model.spatial_states):
            return s


def render_stimuli(s):
    imgs = np.zeros((IMAGE_PER_STIMULI, 3) + (RENDER_HEIGHT, RENDER_WIDTH), dtype=np.uint8)
    for i in range(IMAGE_PER_STIMULI):
        r = s.forward_model.render(s, rgb=True)
        r = r[0]
        # render returns image in c x w x h order, caffe wants it in c x h x w order.
        imgs[i] = r.transpose([0, 2, 1])
        # rotate around z
        s.viewpoint[0][1] += ANGLE_INCREMENT

    return imgs


def split_train_val(data, train_ratio=TRAIN_RATIO):
    N = data.shape[0]
    train_N = int(N * train_ratio)
    val_N = N - train_N
    rand_perm = np.random.permutation(N)
    trainx = data[rand_perm[0:train_N]]
    valx = data[rand_perm[train_N:]]
    return trainx, valx


def write_data_to_db(txn, data, label, key_offset, keys):
    for i in range(data.shape[0]):
        di = data[i]
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = di.shape[0]
        datum.height = di.shape[1]
        datum.width = di.shape[2]
        datum.data = di.tobytes()
        datum.label = label
        key_id = "{:08}".format(keys[key_offset + i])
        txn.put(key_id.encode("ascii"), datum.SerializeToString())

    return data.shape[0]


if __name__ == "__main__":
    fwm = vfm.VisionForwardModel(render_size=RENDER_SIZE, offscreen_rendering=True, custom_lighting=True)
    fwm_view = vfm.VisionForwardModel(render_size=(300, 300), offscreen_rendering=False, custom_lighting=True)

    img_count = STIMULI_COUNT * IMAGE_PER_STIMULI
    train_img_count = int(img_count * TRAIN_RATIO)
    val_img_count = img_count - train_img_count

    train_keys = np.random.permutation(train_img_count)
    val_keys = np.random.permutation(val_img_count)

    train_db = lmdb.open("train_lmdb", map_size=RENDER_WIDTH*RENDER_HEIGHT*3*STIMULI_COUNT*200)
    val_db = lmdb.open("val_lmdb", map_size=RENDER_WIDTH*RENDER_HEIGHT*3*STIMULI_COUNT*100)
    train_txn = train_db.begin(write=True)
    val_txn = val_db.begin(write=True)

    train_key_offset = 0
    val_key_offset = 0

    for object_id in range(STIMULI_COUNT):
        print object_id
        s = generate_random_stimulus()
        """
        while True:
            s = generate_random_stimulus()
            fwm_view._view(s)
            ok = raw_input("Accept object? (y/n) ")
            if ok == "y":
                break
        """

        x = render_stimuli(s)
        misc.imsave("stimuli/{0:d}.png".format(object_id), x[0].transpose([1, 2, 0]))
        tx, vx = split_train_val(x)

        key_inc = write_data_to_db(train_txn, tx, label=object_id + 1, key_offset=train_key_offset, keys=train_keys)
        train_key_offset += key_inc

        key_inc = write_data_to_db(val_txn, vx, label=object_id + 1, key_offset=val_key_offset, keys=val_keys)
        val_key_offset += key_inc

    train_txn.commit()
    train_db.close()
    val_txn.commit()
    val_db.close()
