# Run with "blender -b StimuliBase.blend -P generate_stimuli.py"
# 12 May 2015
# Goker Erdogan

import bpy
import random as rnd

verts = bpy.data.meshes['Circle'].vertices
nverts = len(verts)
for i, v in enumerate(verts[1:nverts-1]):
    # v.co[0] = rnd.expovariate(10)
    v.co[0] = rnd.gauss(verts[i].co[0], .2)

bpy.context.scene.render.filepath = './test.png'
bpy.ops.render.render(write_still=True)


