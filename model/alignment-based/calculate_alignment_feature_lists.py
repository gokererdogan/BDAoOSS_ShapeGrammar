"""
Big data analysis of object shape representations

Calculate features lists for each image to use in 2D alignment model.
For each object, we find the positions of eight features (corners of the root part).


Created on Apr 8, 2015

@author: goker erdogan
gokererdogan@gmail.com
https://github.com/gokererdogan/
"""

import cPickle as pkl
import numpy as np

import Infer3DShape.geometry_3d as geom_3d

# root part is a block centered at the origin with below width, height, and depth
w, h, d = 0.53025, 0.3535, 0.3535
x, y, z = w/2.0, h/2.0, d/2.0
points = [(x, y, z), (-x, y, z), (x, -y, z), (x, y, -z), (-x, -y, z), (-x, y, -z), (x, -y, -z), (-x, -y, -z)]

# note that feature positions depend on image size. If image size changes, alignment features need to be recalculated.
IMG_SIZE = (300, 300)


def get_feature_positions(forward_model, shape):
    # fix missing and ill-formed data
    shape.primitive_type = 'CUBE'
    shape.viewpoint[0] = geom_3d.cartesian_to_spherical(shape.viewpoint[0])

    # find position of each feature
    feature_list = np.zeros((2, 8))
    for i, p in enumerate(points):
        px, py = forward_model.convert_world_to_display(shape.viewpoint[0], p[0], p[1], p[2])
        # in vtk's coordinate system, left bottom corner is (0, 0)
        feature_list[:, i] = [px, IMG_SIZE[1] - py]

    return feature_list

if __name__ == "__main__":
    import Infer3DShape.vision_forward_model as vfm
    fwm = vfm.VisionForwardModel(render_size=IMG_SIZE, offscreen_rendering=True)

    shapes = pkl.load(open('../../../Infer3DShape/data/stimuli20150624_144833/shapes_single_view.pkl'))

    feature_lists = {}
    for name, shape in shapes.iteritems():
        print '.',
        feature_list = get_feature_positions(fwm, shape)
        feature_lists[name] = feature_list

    pkl.dump(feature_lists, open("AlignmentFeatureLists.pkl", "w"))


