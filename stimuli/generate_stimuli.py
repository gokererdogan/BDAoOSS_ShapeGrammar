"""
Generate stimuli for BDAoOSS experiment
@author: goker erdogan
@email: gokererdogan@gmail.com
1 Jun 2015
"""


from bdaooss_grammar import *
from copy import deepcopy
import pickle
import subprocess
import time

class StimuliSet:
    def __init__(self, stimuli_names, stimuli_objects, stimuli_rotations):
        self.stimuli_names = stimuli_names
        self.stimuli_objects = stimuli_objects
        self.stimuli_rotations = stimuli_rotations
        self.created_on = time.time()

fwdm = BDAoOSSVisionForwardModel()
data = np.zeros((600, 600))
params = {'b': 1}
OBJ_PER_TYPE = 3

stimuli_names = []
stimuli = {}
stimuli_rotations = []

# stimulus naming:
#   we denote the object number with ox. Therefore, stimulus
#   name starts with ox. e.g., o3 (3rd object)
#
#   then comes the variation related info. If no variations
#   are applied to the object (i.e., it is the original),
#   stimulus name is simply ox.
#
#   variation (i.e., manipulation) types
#   t1: changing part properties
#       cs: change size of part
#   t2: changing spatial structure
#       mf: move subtree to another face of the same parent
#       ms: move subtree to a random place in tree
#       as: add/remove subtree
#   for all t2 type variations, we can add a depth variable as well.
#       e.g., d1 means the change is at depth 1 in the tree
#
# for example, stimulus name o3_t2_mf_d1 means that the stimulus is 
# made by applying the move face variation to a node at depth 1 for
# object 3.

sname_template = 'o{0:d}_{1:s}_{2:s}_d{3:d}'
cmd_template = "blender -b RenderScene.blend -P render_stimuli.py -- /home/goker/Dropbox/Code/Python/BDAoOSS/{0:s} {1:d} {2:d} {3:d}"

object_id = 1

# generate the original
sname = 'o{0:d}'.format(object_id)
sm = BDAoOSSSpatialModel()
ss = BDAoOSSShapeState(forward_model=fwdm, spatial_model=sm, data=data, ll_params=params)
# save it to obj file
fwdm._save_obj(sname, ss)
stimuli_names.append(sname)
stimuli[sname] = ss

# render stimuli
rotx = 40
roty = 0
rotz = 50
cmd = cmd_template.format(sname, rotx, roty, rotz)
return_code = subprocess.call(cmd, shell=True)

# control render
polar = 40.0
azimuth = 50.0
viewpoint = [np.sqrt(100.0), polar*np.pi/180.0, azimuth*np.pi/180.0]
ss.viewpoint = viewpoint
fwdm.save_render('render.png', ss)

# generate the variation
var_type = 't1'
var_op = 'mf'
depth = 1
sname = sname_template.format(object_id, var_type, var_op, depth)
var_ss = deepcopy(ss)
var_ss._stimuli_vary_dock_face(depth=depth)
# save it to obj file
fwdm._save_obj(sname, var_ss)
stimuli_names.append(sname)
stimuli[sname] =  var_ss

# render stimuli
cmd = cmd_template.format(sname, rotx, roty, rotz)
return_code = subprocess.call(cmd, shell=True)

# save all stimuli to disk
stimuli_set = StimuliSet(stimuli_names, stimuli, [])
pickle.dump(stimuli_set, open('stimuli_set.pkl', 'wb'), protocol=2)

"""
# generate stimuli of type 1: varying the dock_face of a single part
o = 0
while o < OBJ_PER_TYPE:
    o = o + 1
    print(o)
    sm = BDAoOSSSpatialModel()
    ss = BDAoOSSShapeState(forward_model=fwdm, spatial_model=sm, data=data, ll_params=params)

    fname = "Stimuli/c_t1_o{0:d}_s{1:d}_d{2:d}.png"
    fnamev = "Stimuli/t1_o{0:d}_s{1:d}_d{2:d}_v{3:d}.png"
    # save image
    render = fwdm.render(ss)
    save_grayscale(fnamev.format(o, 1, 0, 1), render[0,:,:])
    save_grayscale(fnamev.format(o, 1, 0, 2), render[1,:,:])
    save_grayscale(fnamev.format(o, 1, 0, 3), render[2,:,:])
    fwdm.save_image(fname.format(o, 1, 0), ss)
    # if we can't find a node to modify, we skip this object
    failed = False
    # samples per object
    for s, d in it.product(range(2, 4), [1,2]):
        try: # the tree might be too small to have enough nodes
            ss._stimuli_vary_dock_face(depth=d)
        except:
            failed = True
            print('failed')
            break
        # save image
        render = fwdm.render(ss)
        save_grayscale(fnamev.format(o, s, d, 1), render[0,:,:])
        save_grayscale(fnamev.format(o, s, d, 2), render[1,:,:])
        save_grayscale(fnamev.format(o, s, d, 3), render[2,:,:])
        fwdm.save_image(fname.format(o, s, d), ss)

    if failed: # we failed, redo this iteration
        o = o - 1
        continue

# generate stimuli of type 2: varying the size of a random part 
o = 0
while o < OBJ_PER_TYPE:
    o = o + 1
    print(o)
    sm = BDAoOSSSpatialModel()
    ss = BDAoOSSShapeState(forward_model=fwdm, spatial_model=sm, data=data, ll_params=params)

    fname = "Stimuli/c_t2_o{0:d}_s{1:d}.png"
    fnamev = "Stimuli/t2_o{0:d}_s{1:d}_v{2:d}.png"
    # save image
    render = fwdm.render(ss)
    save_grayscale(fnamev.format(o, 1, 1), render[0,:,:])
    save_grayscale(fnamev.format(o, 1, 2), render[1,:,:])
    save_grayscale(fnamev.format(o, 1, 3), render[2,:,:])
    fwdm.save_image(fname.format(o, 1), ss)
    # if we can't find a node to modify, we skip this object
    failed = False
    # samples per object
    for s in range(2, 6):
        try: 
            ss._stimuli_vary_part_size()
        except:
            print('failed')
            failed = True
            break
        # save image
        render = fwdm.render(ss)
        save_grayscale(fnamev.format(o, s, 1), render[0,:,:])
        save_grayscale(fnamev.format(o, s, 2), render[1,:,:])
        save_grayscale(fnamev.format(o, s, 3), render[2,:,:])
        fwdm.save_image(fname.format(o, s), ss)

    if failed: # we failed, redo this iteration
        o = o - 1
        continue


o = 0
# generate stimuli of type 3: add a new part
while o < OBJ_PER_TYPE:
    o = o + 1
    print(o)
    sm = BDAoOSSSpatialModel()
    ss = BDAoOSSShapeState(forward_model=fwdm, spatial_model=sm, data=data, ll_params=params)

    fname = "Stimuli/c_t3_o{0:d}_s{1:d}_d{2:d}.png"
    fnamev = "Stimuli/t3_o{0:d}_s{1:d}_d{2:d}_v{3:d}.png"
    # save image
    render = fwdm.render(ss)
    save_grayscale(fnamev.format(o, 1, 0, 1), render[0,:,:])
    save_grayscale(fnamev.format(o, 1, 0, 2), render[1,:,:])
    save_grayscale(fnamev.format(o, 1, 0, 3), render[2,:,:])
    fwdm.save_image(fname.format(o, 1, 0), ss)
    # if we can't find a node to modify, we skip this object
    failed = False
    # samples per object
    for s, d in it.product(range(2, 4), [1,2]):
        css = deepcopy(ss)
        try: # the tree might be too small to have enough nodes
            css._stimuli_add_part(depth=d)
        except:
            print('failed')
            failed = True
            break
        # save image
        render = fwdm.render(css)
        save_grayscale(fnamev.format(o, s, d, 1), render[0,:,:])
        save_grayscale(fnamev.format(o, s, d, 2), render[1,:,:])
        save_grayscale(fnamev.format(o, s, d, 3), render[2,:,:])
        fwdm.save_image(fname.format(o, s, d), css)

    if failed: # we failed, redo this iteration
        o = o - 1
        continue

"""
