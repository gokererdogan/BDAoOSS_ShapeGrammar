"""
This file contains the script for converting
BDAoOSS stimuli to Infer3DShape Shape format
for single_view stimuli (objects rendered from
a single viewpoint).

Created on Nov 17, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import cPickle
import numpy as np
import Infer3DShape.shape as hyp
import Infer3DShape.vision_forward_model as vfm

# BDAoOSSStimuli are larger than Infer3DShape.Shape objects.
# Maximum size for Infer3DShape.Shape objects are 1.0; it is
# 1.5 for BDAoOSSStimuli. Also, the camera is located at (0.0, 2.0, 2.0) in Infer3DShape while it is at
# (0.0, 4*sqrt(2.0), 4*sqrt(2.0)) in BDAoOSS.
# Therefore, we scale the positions and sizes to shrink them.
SCALE_FACTOR = 2.0 / (4.0 * np.sqrt(2.0))

# camera is rotated around z axis with a fixed distance from origin.
# this distance is equal to the height (z) of the camera.
d = 2.0
z = 2.0


def create_shape_from_stimuli(stim, view_angle):
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
    view_x = d * np.cos(view_angle * np.pi / 180.0)
    view_y = d * np.sin(view_angle * np.pi / 180.0)
    view_z = z
    h.viewpoint = [(view_x, view_y, view_z)]
    return h

if __name__ == "__main__":
    fwm = vfm.VisionForwardModel(render_size=(200, 200))

    stimuli_folder = 'stimuli20150624_144833'
    stimuli_file = '{0:s}/stimuli_set.pkl'.format(stimuli_folder)
    save_folder = '../../Infer3DShape/data/{0:s}'.format(stimuli_folder)
    # read viewpoints for stimuli
    view_angles = eval(open('{0:s}/viewpoints.txt'.format(stimuli_folder)).read())

    # read stimuli
    stim_set = cPickle.load(open(stimuli_file))

    shapes = {}
    for sname, stim in stim_set.stimuli_objects.iteritems():
        print(sname)
        h = create_shape_from_stimuli(stim, view_angles[sname])
        shapes[sname] = h
        print(h)
        # save render images and data
        fwm.save_render("{0:s}/png_single_view/{1:s}.png".format(save_folder, sname), h)
        img = fwm.render(h)
        np.save("{0:s}/{1:s}_single_view.npy".format(save_folder, sname), img)

    # save shapes to disk
    cPickle.dump(shapes, open("{0:s}/shapes_single_view.pkl".format(save_folder), 'wb'), protocol=2)

