"""
Generate static stimuli for BDAoOSS experiment

This script reads in the stimuli already generated and renders the
objects from a single view. We will use these to run a new experiment.

@author: goker erdogan
@email: gokererdogan@gmail.com
15 Oct 2015
"""


import cPickle as pkl
import numpy as np
import subprocess

def render_stimuli_single_view(name, viewpoint):
    """
    Renders the stimuli with name `name`. 
    Note this function assumes that the stimulus object is 
    already generated and written to disk as an OBJ file.
    This function simply calls render_stimuli.py script.
    """
    cmd = cmd_template.format(name, viewpoint)
    return_code = subprocess.call(cmd, shell=True)
    return return_code


stimuli_folder = 'stimuli20150624_144833'
# render scene path (blender file containing the scene setup)
render_scene_file = "/home/goker/Dropbox/Code/Python/BDAoOSS/stimuli/RenderScene.blend"
# render script
render_script_file = "/home/goker/Dropbox/Code/Python/BDAoOSS/stimuli/render_stimuli_single_view.py"
# blender rendering command template
# dev/null is for discarding the output from blender.
cmd_template = "blender -b " + render_scene_file + " -P " + render_script_file + " -- " + stimuli_folder + " {0:s} {1:d}"

stimuli_file = '{0:s}/stimuli_set.pkl'.format(stimuli_folder)

# read stimuli
stim_set = pkl.load(open(stimuli_file))

viewpoints = {}
for sname in stim_set.stimuli_objects.keys():
    print(sname)
    rotz = np.random.randint(0, 360)
    render_stimuli_single_view(sname, rotz)
    viewpoints[sname] = rotz

# write viewpoints to disk
f = open("{0:s}/viewpoints.txt".format(stimuli_folder), "w")
f.write(str(viewpoints))
f.close()


