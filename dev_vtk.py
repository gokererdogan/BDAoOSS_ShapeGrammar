# -*- coding: utf-8 -*-
'''
Created on May 27, 2015

@author: gerdogan
Goker Erdogan
gokererdogan@gmail.com

VTK development code
Create rectangular prisms and put them together to create objects
'''

# 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

p1Xw = 1 
p1Yw = 1 
p1Zw = 1
p1x = 0
p1y = 0
p1z = 0
part1 = vtk.vtkCubeSource()
part1.SetXLength(p1Xw)
part1.SetYLength(p1Yw)
part1.SetZLength(p1Zw)
part1Output = part1.GetOutput()

p2Xw = 1 
p2Yw = 1 
p2Zw = 1 
# let's put the second part on to the top face of part 1
p2x = 0
p2y = 0
p2z = 1.5 
part2 = vtk.vtkCubeSource()
part2.SetXLength(p2Xw)
part2.SetYLength(p2Yw)
part2.SetZLength(p2Zw)
part2Output = part2.GetOutput()

# Create a mapper and actor
part1Mapper = vtk.vtkPolyDataMapper()
part1Mapper.SetInput(part1Output)
part1Actor = vtk.vtkActor()
part1Actor.SetMapper(part1Mapper)
part1Actor.SetPosition(p1x, p1y, p1z)
part1Actor.SetScale(2, 2, 2)

part2Mapper = vtk.vtkPolyDataMapper()
part2Mapper.SetInput(part2Output)
part2Actor = vtk.vtkActor()
part2Actor.SetMapper(part2Mapper)
part2Actor.SetPosition(p2x, p2y, p2z)

part3Mapper = vtk.vtkPolyDataMapper()
part3Mapper.SetInput(part2Output)
part3Actor = vtk.vtkActor()
part3Actor.SetMapper(part2Mapper)
part3Actor.SetPosition(0, 0, 2.5)
part3Actor.SetScale(2, 1, 1)

camera = vtk.vtkCamera()
camera.SetPosition(5, -5, 5)
camera.SetFocalPoint(0, 0, 0)
camera.SetViewUp(0, 0, 1)

# x, y, z lines
# create source
xl = vtk.vtkLineSource()
xl.SetPoint1(-10,0,0)
xl.SetPoint2(10,0,0)
yl = vtk.vtkLineSource()
yl.SetPoint1(0,-10,0)
yl.SetPoint2(0,10,0)
zl = vtk.vtkLineSource()
zl.SetPoint1(0,0,-10)
zl.SetPoint2(0,0,10)
 
# mapper
mapperx = vtk.vtkPolyDataMapper()
mapperx.SetInput(xl.GetOutput())
mappery = vtk.vtkPolyDataMapper()
mappery.SetInput(yl.GetOutput())
mapperz = vtk.vtkPolyDataMapper()
mapperz.SetInput(zl.GetOutput())

# actor
actorx = vtk.vtkActor()
actorx.SetMapper(mapperx)
actory = vtk.vtkActor()
actory.SetMapper(mappery)
actorz = vtk.vtkActor()
actorz.SetMapper(mapperz)

# color actor
actorx.GetProperty().SetColor(1,0,0)
actory.GetProperty().SetColor(0,1,0)
actorz.GetProperty().SetColor(0,1,1)

# lighting
light1 = vtk.vtkLight()
light1.SetIntensity(.7)
#light.SetLightTypeToSceneLight()
light1.SetPosition(1, -1, 1)
#light.SetFocalPoint(0, 0, 0)
#light.SetPositional(True)
#light.SetConeAngle(60)
light1.SetDiffuseColor(1, 1, 1)
#light.SetAmbientColor(0, 1, 0)

light2 = vtk.vtkLight()
light2.SetIntensity(.7)
#light.SetLightTypeToSceneLight()
light2.SetPosition(-1, -1, 1)
#light.SetFocalPoint(0, 0, 0)
#light.SetPositional(True)
#light.SetConeAngle(60)
light2.SetDiffuseColor(1, 1, 1)
#light.SetAmbientColor(0, 1, 0)

light3 = vtk.vtkLight()
light3.SetIntensity(.7)
#light.SetLightTypeToSceneLight()
light3.SetPosition(-1, -1, -1)
#light.SetFocalPoint(0, 0, 0)
#light.SetPositional(True)
#light.SetConeAngle(60)
light3.SetDiffuseColor(1, 1, 1)
#light.SetAmbientColor(0, 1, 0)


# Visualize
renderer = vtk.vtkRenderer()
# set viewpoint randomly
renderer.SetActiveCamera(camera)
r = np.sqrt(100)
polar_angle = np.random.rand() * 90.0
azimuth_angle = np.random.rand() * 180.0
p_rad = polar_angle * np.pi / 180.0
a_rad = azimuth_angle * np.pi / 180.0
x = r * np.sin(p_rad) * np.cos(a_rad)
y = r * np.sin(p_rad) * np.sin(a_rad)
z = r * np.cos(p_rad)
camera.SetPosition(x, y, z)
camera.SetFocalPoint(0, 0, 0)
up_r = 1.0
up_p = 90 - polar_angle
up_a = 180 + azimuth_angle
up_p_rad = up_p * np.pi / 180.0
up_a_rad = up_a * np.pi / 180.0
upx = up_r * np.sin(up_p_rad) * np.cos(up_a_rad)
upy = up_r * np.sin(up_p_rad) * np.sin(up_a_rad)
upz = up_r * np.cos(up_p_rad)
camera.SetViewUp(upx, upy, upz)

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(600, 600)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

renderer.AddActor(part1Actor)
renderer.AddActor(part2Actor)
renderer.AddActor(part3Actor)
renderer.AddActor(actorx)
renderer.AddActor(actory)
renderer.AddActor(actorz)

#renderer.RemoveAllViewProps()

renderer.SetBackground(0.1, 0.1, 0.1) # Background color
renderer.SetAmbient(.4, .4, .4)
#renderer.TwoSidedLightingOff()
#renderer.LightFollowCameraOff()
renderer.AddLight(light1)
renderer.AddLight(light2)
renderer.AddLight(light3)
renderWindow.Render()

"""
vrml_exporter = vtk.vtkVRMLExporter()
vrml_exporter.SetInput(renderWindow)
vrml_exporter.SetFileName('test.wrl')
vrml_exporter.Write()

obj_exporter = vtk.vtkOBJExporter()
obj_exporter.SetInput(renderWindow)
obj_exporter.SetFilePrefix('test')
obj_exporter.Write()
"""

renderWindowInteractor.Start()

