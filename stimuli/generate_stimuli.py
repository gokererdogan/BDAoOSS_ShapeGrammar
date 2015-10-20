"""
Generate stimuli for BDAoOSS experiment
@author: goker erdogan
@email: gokererdogan@gmail.com
1 Jun 2015
"""


from BDAoOSS.bdaooss_grammar import *
from bdaooss_stimuli import *
from copy import deepcopy
import pickle
import json
import subprocess
import datetime
import os

# stimulus naming:
#   we denote the object number with oX. Therefore, stimulus
#   name starts with oX. e.g., o3 (3rd object)
#
#   then comes the variation related info. If no variations
#   are applied to the object (i.e., it is the original),
#   stimulus name is simply oX_orig.
#
#   variation (i.e., manipulation) types
#   t1: changing part properties
#       cs: change size of part
#   t2: changing spatial structure
#       mf: move subtree to another face of the same parent
#       ms: move subtree to a random place in tree
#       as: add/remove subtree
#       ap: add (a single) part
#       rp: remove (a single) part
#       mp: move one part from depth x to depth y
#   for all variations, we can add a depth variable as well.
#       e.g., d1 means the change is at depth 1 in the tree
#
#   the last part of the stimulus name denotes the viewpoint in
#   the format rxDDryDDrzDD where DD is an integer in the range
#   [0, 360]. rx denotes rotation around x, ry around y, so on.
#
# for example, stimulus name o3_t2_mf_d1_rx10ry0rz50 means that the 
# stimulus is  made by applying the move face variation to a node at 
# depth 1 for object 3, and camera is rotated 10 degrees around x
# and 50 degrees around z.

# render stimuli from different viewpoints 
def render_stimuli(name):
    """
    Renders the stimuli with name `name`. 
    Note this function assumes that the stimulus object is 
    already generated and written to disk as an OBJ file.
    This function simply calls render_stimuli.py script.
    """
    cmd = cmd_template.format(name)
    return_code = subprocess.call(cmd, shell=True)
    return return_code

def generate_object_and_variations(object_id, var_types, var_ops, op_handles, depths):
    """
    Generate an object and apply given manipulations to it to create a set of
    objects.
    object_id: An integer denoting the id of the object
    var_types: A list containing the possible types of manipulations
    var_ops: A dictionary containing the list of possible operations
        for each manipulation type
    op_handle: Function handle for each operation. These functions are implemented 
        in BDAoOSSShapeState.
    depths: The set of depths at which we apply each operation, 
        i.e., manipulation.

    """
    # generate object until you can generate all its variations without any errors
    stim_names = []
    stim_instances = {}
    success = False 
    
    while not success:
        # generate the original
        obj_name = 'o{0:d}'.format(object_id)
        sm = BDAoOSSSpatialModel()
        ss = BDAoOSSShapeState(forward_model=fwdm, spatial_model=sm, data=data, ll_params=params)
        
        # clear the variables
        stim_names[:] = []
        stim_instances.clear()
        # add the object
        stim_names.append(obj_name + '_orig')
        stim_instances[obj_name] = ss 

        success = True
        # generate the variations
        for var_type in var_types:
            if not success:
                break

            for var_op in var_ops[var_type]:
                if not success:
                    break

                for depth in depths:
                    # generate the variation
                    sname = sname_template.format(object_id, var_type, var_op, depth)
                    var_ss = deepcopy(ss)

                    # apply operation
                    op_handle = op_handles[var_op]
                    try:
                        op_handle(var_ss, depth=depth)
                    except Exception as e:
                        success = False 
                        print('------------FAILED when generating object--------------')
                        # print(type(e))
                        print(e)
                        # import traceback
                        # traceback.print_exc()
                        break

                    stim_names.append(sname)
                    stim_instances[sname] = var_ss

    return obj_name, stim_names, stim_instances



# shared objects and variables
fwdm = BDAoOSSVisionForwardModel()
data = np.zeros((600, 600))
params = {'b': 1}

# number of distinct objects in stimulus set. from each object
# we generate multiple stimuli by applying various manipulations
OBJ_COUNT = 10

# create a unique identifier for this stimulus set
# we use the current time
stimuli_created_on = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

# we create a folder with the identifier and use that folder and its subfolders 
# for saving obj, png, gif and other files.
stimuli_file_path = "/home/gerdogan/Dropbox/Code/Python/BDAoOSS/stimuli/stimuli" + stimuli_created_on
obj_file_path = stimuli_file_path + "/obj"
png_file_path = stimuli_file_path + "/png"
gif_file_path = stimuli_file_path + "/gif"
os.mkdir(stimuli_file_path)
os.mkdir(obj_file_path)
os.mkdir(png_file_path)
os.mkdir(gif_file_path)

# render scene path (blender file containing the scene setup)
render_scene_file = "/home/gerdogan/Dropbox/Code/Python/BDAoOSS/stimuli/RenderScene.blend"
# render script
render_script_file = "/home/gerdogan/Dropbox/Code/Python/BDAoOSS/stimuli/render_stimuli.py"

# stimulus name template
sname_template = 'o{0:d}_{1:s}_{2:s}_d{3:d}'

# blender rendering command template
# dev/null is for discarding the output from blender. 
cmd_template = "blender -b " + render_scene_file + " -P " + render_script_file + " -- " + stimuli_file_path + " {0:s} > /dev/null"

# variation types and operations
"""
var_types = ['t1', 't2']
var_ops = {'t1' : ['cs'], 't2' : ['mf', 'ap', 'rp', 'mp']}
op_handles = {'cs' : BDAoOSSShapeState._stimuli_vary_part_size,
              'mf' : BDAoOSSShapeState._stimuli_vary_dock_face,
              'ap' : BDAoOSSShapeState._stimuli_add_part,
              'rp' : BDAoOSSShapeState._stimuli_remove_part,
              'mp' : BDAoOSSShapeState._stimuli_move_part}
"""
# we are not using the move_part manipulation for now.
var_types = ['t1', 't2']
var_ops = {'t1': ['cs'], 't2': ['mf', 'ap', 'rp']}
op_handles = {'cs': BDAoOSSShapeState._stimuli_vary_part_size,
              'mf': BDAoOSSShapeState._stimuli_vary_dock_face,
              'ap': BDAoOSSShapeState._stimuli_add_part,
              'rp': BDAoOSSShapeState._stimuli_remove_part}


# variables for storing the generated stimuli
object_names = []
stimuli_names = {} 
# these are the BDAoOOSStimuli instances used for storing the stimuli to disk
stimuli = {}
# these are the BDAoOSSShapeState instances used for rendering
stimuli_instances = {}

for object_id in range(1, OBJ_COUNT+1):
    regenerate = True
    while regenerate:
        print('Generating object {0:d}'.format(object_id))
        # we generate the object and all its variations first to see if
        # we run into any problems. in some cases, some operations cannot
        # be applied to an object (i.e., we can't modify the node at depth
        # 2 if the tree has less than 2 levels).
        obj_name, ostim_names, ostim_instances = generate_object_and_variations(object_id, var_types, var_ops, op_handles, depths=[1,2])

        # ask the user if he/she wants to keep this object
        keep_object = raw_input('Do you want to [K]eep this object or [R]egenerate it? ')
        if keep_object == 'K':
            object_names.append(obj_name)
            stimuli_names[obj_name] = ostim_names
            stimuli_instances[obj_name] = ostim_instances

            for sname in ostim_names:
                ss = ostim_instances[sname]
                stimuli[sname] = BDAoOSSStimuli(name=sname, tree=ss.tree, spatial_model=ss.spatial_model)
            regenerate = False

# print(object_names)
# print(stimuli_names)
# print(stimuli)
# print(stimuli_instances)

# loop over generated stimuli and render them
for obj_name in object_names:
    for sname, ss in stimuli_instances[obj_name].iteritems():
        print('Rendering {0:s}'.format(sname))

        # save it to obj file
        fwdm._save_obj(obj_file_path + "/" + sname, ss)

        # render stimuli 
        # (generate a gif movie of object rotating around the vertical axis)
        render_stimuli(sname)
    

# save all stimuli to disk
stimuli_set = BDAoOSSStimuliSet(bdaooss_shape_pcfg, stimuli_created_on, OBJ_COUNT, object_names, stimuli_names, stimuli) 
pickle.dump(stimuli_set, open(stimuli_file_path + '/stimuli_set.pkl', 'wb'), protocol=2)
# save stimuli information to a text file for use in the experiment code.
# this file contains [creation_time, number_of_objects, [list of stimuli names]] in JSON format
json.dump([stimuli_set.created_on, OBJ_COUNT, object_names, stimuli_names], open(stimuli_file_path + '/stimuli.txt', 'w'))

