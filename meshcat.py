#!/usr/bin/env python3

# This examples shows how to load and move a robot in meshcat.
# Note: this feature requires Meshcat to be installed, this can be done using
# pip install --user meshcat
 
import pinocchio as pin
import numpy as np
import sys
import os
from os.path import dirname, join, abspath
 
from pinocchio.visualize import MeshcatVisualizer
 
# Load the URDF model.
model_path = 'models/one_dof/five_segments/'
urdf_path = join(model_path, 'flexible_arm_1dof_5s.urdf')

model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
 
viz = MeshcatVisualizer(model, collision_model, visual_model)
 
# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz.initViewer(open=True)
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install Python meshcat")
    print(err)
    sys.exit(0)
 
# Load the robot in the viewer.
viz.loadViewerModel()
 
# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)
viz.displayCollisions(True)
viz.displayVisuals(False)
 
mesh = visual_model.geometryObjects[0].geometry
mesh.buildConvexRepresentation(True)
convex = mesh.convex
 
if convex is not None:
    placement = pin.SE3.Identity()
    placement.translation[0] = 2.
    geometry = pin.GeometryObject("convex",0,convex,placement)
    geometry.meshColor = np.ones((4))
    visual_model.addGeometryObject(geometry)
 
# Display another robot.
viz2 = MeshcatVisualizer(model, collision_model, visual_model)
viz2.initViewer(viz.viewer)
viz2.loadViewerModel(rootNodeName = "pinocchio2")
q = q0.copy()
q[1] = 1.0
viz2.display(q)
 
