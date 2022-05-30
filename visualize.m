clc; 
clear all; close all;

% path to urdfs
path_to_urdf = 'flexible_arm_v1.urdf';

% Create a robot instance using Matlab Toolbox
robot = importrobot(path_to_urdf);
robot.DataFormat = 'column';
robot.Gravity = [0 0 -9.81];

% q = [0, -0.01, -0.01, -0.01, -0.01]';
% q = [0, 0, 0, 0, pi/2]';
q = [0.84018772 0.39438293 0.78309922 0.79844003 0.91164736]';

getTransform(robot, q,'load')
show(robot, q)

g = gravityTorque(robot, q)