clc; 
clear all; close all;

% path to urdfs
% path_to_urdf = 'models/five_segments/flexible_arm_v1.urdf';
path_to_urdf = 'models/ten_segments/flexible_arm_1dof_10s.urdf';

% Create a robot instance using Matlab Toolbox
robot = importrobot(path_to_urdf);
robot.DataFormat = 'column';
robot.Gravity = [0 0 -9.81];

% q = [0, -0.01, -0.01, -0.01, -0.01]';
% q = [0, 0, 0, 0, pi/2]';
% q = [0.84018772 0.39438293 0.78309922 0.79844003 0.91164736]';
% q = -0.1*rand(robot.NumBodies-1,1);
q = zeros(robot.NumBodies-1,1);
% q = [pi/4, -rand, -rand, -rand, -rand, -rand]';

% Compute gravity and an equilibrium postion of passive joints
g = gravityTorque(robot, q);

% K = diag([100., 100.]);
% q_eq = -inv(K)*g(2:end);
% q = [q(1); q_eq];

getTransform(robot, q,'load')
show(robot, q)

