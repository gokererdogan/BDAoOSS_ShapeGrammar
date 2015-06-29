# Run with 
#   "blender -b RenderScene.blend -P render_stimuli.py stimuli_path obj_filename 
# Expects an obj file with the 3D model. Specify filename without the 
# extension obj.
# Stimuli path is the folder containing obj and rendered png files.
#   We assume that obj files are under stimuli_path/obj folder.
#   Similarly, we assume that png files are under stimuli_path/png folder.
#
# 8 Jun 2015
# Goker Erdogan

import bpy
import random as rnd
from math import pi
import sys
import subprocess

# get stimuli path
spath = sys.argv[6]
# get obj filename
fname = sys.argv[7]
objname = "{0:s}/obj/{1:s}.obj".format(spath, fname)
# import the 3D model. NOTE the axis mappings.
bpy.ops.import_scene.obj(filepath=objname, axis_forward='-X', axis_up='Z')

# rotate object and render
# rotation around x and y are fixed. we rotate viewpoint around z axis.
radx = 45.0 * pi / 180.0
rady = 0
# start from a random viewpoint
start_z = rnd.randint(1, 360);

i = 0
for rz in range(0, 360, 4):
    print('.'),
    i = i + 1
    radz = (rz + start_z) * pi / 180.0
    bpy.data.objects['Empty'].rotation_euler = (radx, rady, radz)
    imgname = "{0:s}/png/{1:s}_{2:03d}.png"
    bpy.context.scene.render.filepath = imgname.format(spath, fname, i) 
    bpy.ops.render.render(write_still=True)

print()

# convert all pngs to a single animated gif
# 4/100 = 40msecs. each image is shown for 20ms, 
# hence, 90*40ms = 3.6secs. 
# Each animation is 3.6secs long.
cmd_template = "convert -delay 4/100 -loop 0 {0:s}/png/{1:s}*.png {0:s}/gif/{1:s}.gif"
cmd = cmd_template.format(spath, fname)
return_code = subprocess.call(cmd, shell=True)
print("Gif file {0:s} created.".format(fname))
