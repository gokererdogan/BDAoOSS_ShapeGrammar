"""
Generate inverted static stimuli for BDAoOSS experiment.

This script reads in the stimuli already generated and renders the
objects from a single view, upside down (i.e., camera rotated 180 
degrees around the line of sight). 

@author: goker erdogan
@email: gokererdogan@gmail.com
26 May 2016
"""


import cPickle as pkl
import numpy as np
import subprocess

def render_stimuli_single_view(name, viewpoint):
    cmd = cmd_template.format(name, viewpoint)
    return_code = subprocess.call(cmd, shell=True)
    return return_code


stimuli_folder = 'stimuli20150624_144833'
# render scene path (blender file containing the scene setup)
render_scene_file = "/home/goker/Dropbox/Code/Python/BDAoOSS/stimuli/RenderScene.blend"
# render script
render_script_file = "/home/goker/Dropbox/Code/Python/BDAoOSS/stimuli/render_stimuli_single_view_inverted.py"
# blender rendering command template
# dev/null is for discarding the output from blender.
cmd_template = "blender -b " + render_scene_file + " -P " + render_script_file + " -- " + stimuli_folder + " {0:s} {1:d}"

stimuli_file = '{0:s}/stimuli_set.pkl'.format(stimuli_folder)
viewpoints_file = '{0:s}/viewpoints.txt'.format(stimuli_folder)
viewpoints = eval(open(viewpoints_file, 'r').read())

# read stimuli
stim_set = pkl.load(open(stimuli_file))
snames = stim_set.stimuli_objects.keys()
for sname in snames:
    print(sname)
    rotz = viewpoints[sname]
    render_stimuli_single_view(sname, rotz)


