import bpy
import math, glob, os, time, re, time
from mathutils import Vector
from collections import namedtuple

# Time to play animation in seconds
animation_time_secs = 2.2
animation_time_frames = math.ceil(animation_time_secs * 24)
animation_time_ticks = math.ceil(animation_time_secs * 60)

mesh_path = '/home/moverlan/bcs/shape_concept/model/stl/'
output_path = '/home/moverlan/bcs/shape_concept/model/renders/'

#
# Read bounding box dimensions
#
BoundingBox = namedtuple("BoundingBox", "Xmin Xmax Ymin Ymax Zmin Zmax")
with open(mesh_path + 'stl_dims.txt') as f:
    lines = [line.rstrip('\n') for line in f]
bounding_boxes = {}
i = 0
while i < len(lines):
    shapeline = lines[i]
    dimsline = lines[i + 1]
    shape = re.match( r'stl/([A-Za-z]+)\.stl', shapeline).group(1)
    group = re.match( r'\((.+), (.+), (.+), (.+), (.+), (.+)\)', dimsline).groups()
    bounding_box = BoundingBox(float(group[0]), float(group[1]), float(group[2]), float(group[3]), float(group[4]), float(group[5]))
    bounding_boxes[shape] = bounding_box
    i += 2

mesh_files = glob.glob(mesh_path + "*.stl")

bpy.ops.screen.frame_jump(end=False)
ground = bpy.data.scenes['Scene'].objects['Ground']

#
# Set up logic to exit game mode after animation play time
#
bpy.ops.object.select_pattern(pattern=ground.name)
# Remove any existing logic blocks
bpy.ops.logic.sensor_remove(sensor='delay_sensor', object='Ground')
bpy.ops.logic.controller_remove(controller='and_controller', object='Ground')
bpy.ops.logic.actuator_remove(actuator='game_actuator', object='Ground')

# Add logic blocks
bpy.ops.logic.sensor_add(type='DELAY', object=ground.name, name='delay_sensor')
sensor = ground.game.sensors[-1]
sensor.delay = animation_time_ticks

bpy.ops.logic.controller_add(type='LOGIC_AND', object=ground.name, name='and_controller')
controller = ground.game.controllers[-1]

bpy.ops.logic.actuator_add(type='GAME', object=ground.name, name='game_actuator')
actuator = ground.game.actuators[-1]
actuator.mode = 'QUIT'

# Link blocks
sensor.link(controller)
actuator.link(controller)

objects_to_keep = ['Light','Hemi.001','Ground','Camera','World','RenderLayers']

#
# Load objects and render
#
start = time.time()
index = 1

def render_next_mesh():
    global index
    if index == len(mesh_files):
        print("Done, returning")
        return
    print(index)
    render_mesh_file(mesh_files[index])
    index += 1

def render_mesh_file(mesh_file):

    #mesh_file = mesh_files[i]

    print("Rendering mesh file " + mesh_file)

    bpy.data.scenes['Scene'].render.engine = 'CYCLES'
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            override = {'area': area, 'region': area.regions[-1]}
            bpy.ops.object.mode_set(override, mode='OBJECT')

    shape_name = os.path.basename(mesh_file)[:-4]

    # Delete old object
    scn = bpy.context.scene
    for ob in scn.objects:
        if ob.name not in objects_to_keep:
            print("deleting " + ob.name)
            scn.objects.unlink(ob)

    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    # Import object with fix for rotation axis
    bpy.ops.import_mesh.stl(filepath=mesh_file, filter_glob="*.stl", files=[], directory="", axis_forward='X', axis_up='Y', global_scale=1)

    # Get object just imported
    for ob in scn.objects:
        if ob.type == 'MESH' and ob.name != 'Ground':
            imported_object = ob
    # Set material of object
    for material in bpy.data.materials:
        if material.name == 'Material.001':
            imported_object.active_material = material
    # Select object
    bpy.ops.object.select_pattern(pattern=imported_object.name)
    # Smooth object
    bpy.ops.object.shade_smooth()
    # Subdivision surface
    bpy.ops.object.modifier_add(type='SUBSURF')

    # UV project (doesn't quite work)
    #for area in bpy.context.screen.areas:
        #if area.type == 'VIEW_3D':
            #override = {'area': area, 'region': area.regions[-1]}
            #bpy.ops.object.mode_set(override, mode='EDIT')
            #bpy.ops.view3d.viewnumpad(override, type='CAMERA')
            #bpy.ops.uv.project_from_view(override)
            #bpy.ops.object.mode_set(override, mode='OBJECT')

    # Make object rest on surface
    bounding_box = bounding_boxes[shape_name]
    Zmin = bounding_box.Zmin
    Zmax = bounding_box.Zmax

    #imported_object.location = (0.0, 0.0, ((Zmax - Zmin) / 2.0) - 0.25 + ground.location[2])
    imported_object.location = (0.0, 0.0, ((Zmax - Zmin) / 2.0) - 0. + ground.location[2])

    # Adjust camera
    camera = bpy.data.scenes['Scene'].objects['Camera']
    if len(shape_name) > 5:
        camera.data.shift_y = 0.20
        bpy.data.scenes['Scene'].render.resolution_y = 800
        camera.data.lens = 35.0
    else:
        camera.data.shift_y = 0.0
        bpy.data.scenes['Scene'].render.resolution_y = 600
        camera.data.lens = 35.0

    # Physics simulation onto surface
    bpy.data.scenes['Scene'].render.engine = 'BLENDER_GAME'
    ground.game.use_collision_bounds = True
    ground.game.collision_margin = 0.040

    imported_object.game.physics_type = 'RIGID_BODY'
    imported_object.game.use_collision_bounds = True
    imported_object.game.collision_bounds_type = 'CONVEX_HULL'
    imported_object.game.lock_location_x = True
    imported_object.game.lock_location_y = True
    imported_object.game.lock_rotation_y = True
    imported_object.game.lock_rotation_z = True
    # Prevent rotation for ring structures
    #if len(os.path.basename(mesh_file)[:-4]) > 5:
    imported_object.game.lock_rotation_x = True

    bpy.data.scenes['Scene'].frame_end = animation_time_frames
    bpy.data.scenes['Scene'].game_settings.use_animation_record = True
    bpy.data.scenes['Scene'].frame_current = 0

    def stop_playback_and_render(scene):
        if scene.frame_current == animation_time_frames - 1:
            # Pause animation
            bpy.ops.screen.animation_cancel(restore_frame=False)

            scene.frame_current = animation_time_frames

            # Render settings (change based on your machine)
            scene.render.tile_x = 32
            scene.render.tile_y = 32

            bpy.data.scenes['Scene'].render.engine = 'CYCLES'

            # Render
            outfile = shape_name + '.png'
            print('saving', outfile) 
            scene.render.filepath = output_path + outfile
            bpy.ops.render.render( write_still=True )

            bpy.app.handlers.frame_change_pre.remove(stop_playback_and_render)

            render_next_mesh()

    # Must have an actuator set up in Blender which exits game after a 2s delay
    if bpy.ops.view3d.game_start.poll():
        bpy.ops.view3d.game_start()
        bpy.app.handlers.frame_change_pre.append(stop_playback_and_render)
        bpy.ops.screen.animation_play()
    else:
        print('Context fail')

render_next_mesh()
