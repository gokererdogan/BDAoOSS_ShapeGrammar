# Run with 
#   "blender -b RenderScene.blend -P render_stimuli_single_view.py stimuli_path obj_filename
# Expects an obj file with the 3D model. Specify filename without the 
# extension obj.
# Stimuli path is the folder containing obj files.
#   We assume that obj files are under stimuli_path/obj folder.
#
# 15 Oct 2015
# Goker Erdogan

import bpy
from math import pi
import sys

# get stimuli path
spath = sys.argv[6]
# get obj filename
fname = sys.argv[7]
objname = "{0:s}/obj/{1:s}.obj".format(spath, fname)
# get view orientation
rotz = int(sys.argv[8])

# import the 3D model. NOTE the axis mappings.
bpy.ops.import_scene.obj(filepath=objname, axis_forward='-X', axis_up='Z')

# rotate object and render
# rotation around x and y are fixed. we rotate viewpoint around z axis.
radx = 45.0 * pi / 180.0
rady = 0
print('.'),
radz = rotz * pi / 180.0
bpy.data.objects['Empty'].rotation_euler = (radx, rady, radz)
imgname = "{0:s}/single_view/{1:s}.png"
bpy.context.scene.render.filepath = imgname.format(spath, fname)
bpy.ops.render.render(write_still=True)

