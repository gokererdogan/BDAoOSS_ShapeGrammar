'''
Created on May 31, 2015
Vision forward model for BDAoOSS Shape grammar
Creates 3D scene according to given shape representation 
and uses VTK to render 3D scene to 2D image
@author: goker erdogan
@email: gokererdogan@gmail.com
'''
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import scipy.misc
from gmllib.helpers import rgb2gray


class BDAoOSSVisionForwardModel():
    """
    Vision forward model for BDAoOSS Shape grammar
    Creates 3D scene according to given shape representation 
    and uses VTK to render 3D scene to 2D image
    Each part is assumed to be a rectangular prism.
    Forward model expects the size and position of each such
    part along with the viewpoint (i.e., orientation of the
    camera)
    """
    camera_pos = (5.0, -5.0, 5.0) # canonical view
    camera_up = (0, 0, 1)
    render_size = (600, 600)
    save_image_size = (600, 600)
    def __init__(self):
        """
        Initializes VTK objects for rendering.
        """
        # vtk objects for rendering
        self.vtkrenderer = vtk.vtkRenderer()
        
        self.vtkcamera = vtk.vtkCamera()
        self.vtkcamera.SetPosition(self.camera_pos)
        self.vtkcamera.SetFocalPoint(0, 0, 0)
        self.vtkcamera.SetViewUp(self.camera_up)
 
        # lighting
        self.light1 = vtk.vtkLight()
        self.light1.SetIntensity(.4)
        self.light1.SetPosition(10, -10, 10)
        self.light1.SetDiffuseColor(1, 1, 1)
        self.light2 = vtk.vtkLight()
        self.light2.SetIntensity(.4)
        self.light2.SetPosition(-10, -10, 10)
        self.light2.SetDiffuseColor(1, 1, 1)
        self.light3 = vtk.vtkLight()
        self.light3.SetIntensity(.4)
        self.light3.SetPosition(10, -10, -10)
        self.light3.SetDiffuseColor(1, 1, 1)
        self.light4 = vtk.vtkLight()
        self.light4.SetIntensity(.4)
        self.light4.SetPosition(-10, -10, -10)
        self.light4.SetDiffuseColor(1, 1, 1)
        self.vtkrenderer.AddLight(self.light1)
        self.vtkrenderer.AddLight(self.light2)
        self.vtkrenderer.AddLight(self.light3)
        self.vtkrenderer.AddLight(self.light4)
        self.vtkrenderer.SetBackground(0.1, 0.1, 0.1) # Background color

        self.vtkrender_window = vtk.vtkRenderWindow()
        self.vtkrender_window.AddRenderer(self.vtkrenderer)
        self.vtkrender_window.SetSize(self.render_size)
        self.vtkrender_window_interactor = vtk.vtkRenderWindowInteractor()
        self.vtkrender_window_interactor.SetRenderWindow(self.vtkrender_window)

        # vtk objects for reading, and rendering object parts
        self.part_source = vtk.vtkCubeSource()
        self.part_output = self.part_source.GetOutput()
        self.part_mapper = vtk.vtkPolyDataMapper()
        self.part_mapper.SetInput(self.part_output)

        # exporters
        self.vtkvrml_exporter = vtk.vtkVRMLExporter()
        self.vtkobj_exporter = vtk.vtkOBJExporter()
        self.stl_writer = vtk.vtkSTLWriter()

    def render(self, *args):
        """
        Construct the 3D object from state and render it.
        Returns numpy array with size number of viewpoints x self.render_size
        """
        # called with ShapeState instance
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args

        self._build_scene(*parts_positions)
        img_arr = self._render_window_to2D()
        return img_arr
    
    def _render_window_to2D(self):
        """
        Renders the window to 2D black and white image
        Called from render function for each viewpoint
        """
        self.vtkrender_window.Render()
        self.vtkwin_im = vtk.vtkWindowToImageFilter()
        self.vtkwin_im.SetInput(self.vtkrender_window)
        self.vtkwin_im.Update()
        vtk_image = self.vtkwin_im.GetOutput()
        height, width, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
        arr = rgb2gray(arr)
        return arr
    
    def _build_scene(self, parts, positions, scales, viewpoint):
        """
        Places parts scaled to  `scales` to `positions` and 
        adds them to scene.
        Adjusts camera to viewpoint.
        Returns vtkRenderer
        """
        # clear scene
        self.vtkrenderer.RemoveAllViewProps()
        self.vtkrenderer.Clear()
        # add objects
        for part, position, scale in zip(parts, positions, scales):
            actor = vtk.vtkActor()
            actor.SetMapper(self.part_mapper)
            actor.SetPosition(position)
            actor.SetScale(scale)
            self.vtkrenderer.AddActor(actor)
        # calculate camera position
        r, polar_angle, azimuth_angle = viewpoint
        x = r * np.sin(polar_angle) * np.cos(azimuth_angle)
        y = r * np.sin(polar_angle) * np.sin(azimuth_angle)
        z = r * np.cos(polar_angle)
        # rotate camera
        self.vtkcamera.SetPosition(x, y, z)
        # set camera view up
        up_r = 1.0
        # view up is orthogonal to the line of sight
        # if default view up is (0, 0, -1), rotating the camera
        # rotates view up vector as follows
        # up_polar = 90 - camera_polar 
        # up_azimuth = 180 + camera_azimuth
        up_p = (np.pi / 2.0) - polar_angle 
        up_a = np.pi + azimuth_angle
        upx = up_r * np.sin(up_p) * np.cos(up_a)
        upy = up_r * np.sin(up_p) * np.sin(up_a)
        upz = up_r * np.cos(up_p)
        self.vtkcamera.SetViewUp(upx, upy, upz)
        self.vtkcamera.SetFocalPoint(0, 0, 0)
        self.vtkrenderer.SetActiveCamera(self.vtkcamera)

                
    def _view(self, *args):
        """
        Views object in window
        Used for development and testing purposes
        """
        # called with ShapeState instance
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args
        self._build_scene(*parts_positions)
        self.vtkrender_window.Render()
        self.vtkrender_window_interactor.Start()

    def _save_wrl(self, filename, *args):
        """
        Save object to wrl file.
        """
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args
        self._build_scene(*parts_positions)
        self.vtkrender_window.Render()
        self.vtkvrml_exporter.SetInput(self.vtkrender_window)
        self.vtkvrml_exporter.SetFileName(filename)
        self.vtkvrml_exporter.Write()
    
    def _save_obj(self, filename, *args):
        """
        Save object to obj file.
        """
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args
        self._build_scene(*parts_positions)
        self.vtkrender_window.Render()
        self.vtkobj_exporter.SetInput(self.vtkrender_window)
        self.vtkobj_exporter.SetFilePrefix(filename)
        self.vtkobj_exporter.Write()


    def _save_stl(self, filename, *args):
        """
        Save object to stl file.
        """
        return NotImplementedError()
        # TO-DO
        # we can't save the whole scene to a single STL file,
        # we need to figure out how to get around that.
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args
        self._build_scene(*parts_positions)
        self.vtkrender_window.Render()
        self.stl_writer.SetFileName(filename)
        self.stl_writer.SetInput(self.vtkrender_window)
        self.stl_writer.Write()
 

    def save_render(self, filename, *args):
        """
        Save rendered image to disk.
        """
        # called with ShapeState instance
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args

        img = self.render(*parts_positions)
        # make sure that scipy does not normalize the image
        img = scipy.misc.toimage(np.flipud(img), cmin=0, cmax=255)
        img.save(filename)

    def save_image(self, filename, *args):
        """
        Save image of object from canonical view to disk
        """
        # called with ShapeState instance
        if len(args) == 1:
            parts_positions = args[0].convert_to_parts_positions()
        else: # called directly with parts and positions
            parts_positions = args
        self.vtkrender_window.SetSize(self.save_image_size)
        self._build_scene(*parts_positions)
        # set viewpoint to canonical viewpoint
        self.vtkcamera.SetPosition(self.camera_pos)
        self.vtkcamera.SetViewUp(self.camera_up)
        self.vtkcamera.SetFocalPoint(0, 0, 0)
        barr = self._render_window_to2D()
        # make sure that scipy does not normalize the image
        img = scipy.misc.toimage(np.flipud(barr), cmin=0, cmax=255)
        img.save(filename)
    
        
if __name__ == '__main__':
    # ---------------------------------------------------------
    
    forward_model = BDAoOSSVisionForwardModel()
    # rotate camera around vertical axis and generate images
    parts = ['P'] * 2
    positions = [[0,0,0], [-0.33,0.75,0.75]]
    scales = [[1.5,1,1], [.5,.5,.5]]
    polar = 0.0
    azimuth = 0.0
    viewpoint = [np.sqrt(100.0), polar*np.pi/180.0, azimuth*np.pi/180.0]

    forward_model._view(parts, positions, scales, viewpoint)
    forward_model.save_image('test.png', parts, positions, scales, viewpoint)
    forward_model.save_render('render.png', parts, positions, scales, viewpoint)
    forward_model._save_obj('test', parts, positions, scales, viewpoint)
    forward_model._save_wrl('test.wrl', parts, positions, scales, viewpoint)

