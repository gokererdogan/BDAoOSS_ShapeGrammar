# Run with 
#   "blender -b RenderScene.blend -P render_stimuli.py obj_filename rotx roty rotz
# Expects an obj file with the 3D model. Specify filename without the 
# extension obj.
# rotx, roty, rotz are the camera rotation settings. Expected to be integers 
# in [0, 360]
#
# 8 Jun 2015
# Goker Erdogan

import bpy
import random as rnd
from math import pi
import sys

# get obj filename
fname = sys.argv[6]
# import the 3D model. NOTE the axis mappings.
bpy.ops.import_scene.obj(filepath=fname+'.obj', axis_forward='-X', axis_up='Z')

# get rotation settings 
rx = int(sys.argv[7])
ry = int(sys.argv[8])
rz = int(sys.argv[9])
rx = rx * pi / 180.0
ry = ry * pi / 180.0
rz = rz * pi / 180.0
bpy.data.objects['Empty'].rotation_euler = (rx, ry, rz)

bpy.context.scene.render.filepath = fname + '.png' 
bpy.ops.render.render(write_still=True)

