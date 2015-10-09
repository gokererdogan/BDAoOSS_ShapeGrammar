"""
This file contains the script for converting
BDAoOSS stimuli to Infer3DShape Shape format.

Created on Sep 17, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import cPickle
import numpy as np
import Infer3DShape.hypothesis as hyp

# BDAoOSSStimuli are larger than Infer3DShape.Shape objects.
# Maximum size for Infer3DShape.Shape objects are 1.0; it is
# 1.5 for BDAoOSSStimuli. Therefore, we scale the positions
# and sizes to shrink them.
SCALE_FACTOR = 1.0 / 1.5

def create_shape_from_stimuli(stim):
    """
    This function creates a Infer3DShape.Shape instance
    from the BDAoOSSStimuli object
    :param stim: BDAoOSS stimulus object
    :return: Returns Infer3DShape.Shape instance
    """
    sm = stim.spatial_model
    parts = []
    for ss in sm.spatial_states.itervalues():
        pos = ss.position * SCALE_FACTOR
        size = ss.size * SCALE_FACTOR
        parts.append(hyp.CuboidPrimitive(position=pos, size=size))

    h = hyp.Shape(forward_model=None, parts=parts)
    return h

if __name__ == "__main__":
    fwm = hyp.vfm.VisionForwardModel()

    stimuli_folder = 'stimuli20150624_144833'
    stimuli_file = '{0:s}/stimuli_set.pkl'.format(stimuli_folder)
    save_folder = '../../Infer3DShape/data/{0:s}/'.format(stimuli_folder)

    # read stimuli
    stim_set = cPickle.load(open(stimuli_file))

    shapes = {}
    for sname, stim in stim_set.stimuli_objects.iteritems():
        print(sname)
        h = create_shape_from_stimuli(stim)
        shapes[sname] = h
        print(h)
        # save render images and data
        fwm.save_render("{0:s}/png/{1:s}.png".format(save_folder, sname), h)
        img = fwm.render(h)
        np.save("{0:s}/{1:s}.npy".format(save_folder, sname), img)

    # save shapes to disk
    cPickle.dump(shapes, open("{0:s}/shapes.pkl".format(save_folder), 'wb'), protocol=2)


