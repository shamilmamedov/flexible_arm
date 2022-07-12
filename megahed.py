#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag


def rotz(q):
    c = np.cos(q)
    s = np.sin(q)
    R = np.array([[c, -s, 0.],
                  [s, c, 0.],
                  [0., 0., 1.]])
    return R

def one_link_rigid_ode(t, x, tau, I, m, r_com):
    theta = x[0,0]
    dtheta = x[1,0]
    g = 0
    return np.array([[dtheta], 
            [1/(I + m*r_com**2)*(tau - m*g*r_com*np.cos(theta))]])

def two_element_flexible_ode(t, x, tau, m, EA, EI, L):
    """ State is
        x = (q; dq)
        q = [theta1 x2 y2 theta2 x3 y3 theta3]
        dq =  [dtheta1 dx2 dy2 dtheta2 dx3 dy3 dtheta3]
    """
    # Get mass matrix
    Ml_2seg = get_lumped_masses(m, L/2)
    M = np.zeros((9,9))
    M[:6,:6] = Ml_2seg
    M[3:,3:] += Ml_2seg
    M = M[2:, 2:]
    # M[0,0] += m/2*(L/2)**2/12


    # Get stiffmess matrix
    K_2seg = get_stiffness_params(EA, EI, L/2)

    nx = x.size
    disp = x[:nx//2]
    thetas = disp[0::3]
    xs = disp[1::3]
    ys = disp[2::3]

    vel = x[nx//2:].reshape(-1,1)

    # Compute spring forces
    du21 = np.cos(thetas[0])*xs[0] + np.sin(thetas[0])*ys[0] - L/2
    dv21 = -np.sin(thetas[0])*xs[0] + np.cos(thetas[0])*ys[0]
    dtheta21 = thetas[1] - thetas[0]

    Fu1_1 = K_2seg["Ka"]*du21
    Fu2_1 = -K_2seg["Ka"]*du21
    # Fv1_1 = K_2seg["Kb1"]*dv21 - K_2seg["Kb2"]*dtheta21
    # Fv2_1 = -(K_2seg["Kb1"]*dv21 - K_2seg["Kb2"]*dtheta21)
    # Mz1_1 = -K_2seg["Kb2"]*dv21 + K_2seg["Kb3"]*dtheta21
    Fv1_1 = K_2seg["Kb1"]*dv21 + K_2seg["Kb2"]*dtheta21
    Fv2_1 = -(K_2seg["Kb1"]*dv21 + K_2seg["Kb2"]*dtheta21)
    Mz1_1 = K_2seg["Kb2"]*dv21 + K_2seg["Kb3"]*dtheta21
    Mz2_1 = -Mz1_1 - Fv1_1*L/2

    du32 = np.cos(thetas[1])*(xs[1] - xs[0]) + np.sin(thetas[1])*(ys[1] - ys[0]) - L/2
    dv32 = -np.sin(thetas[1])*(xs[1] - xs[0]) + np.cos(thetas[1])*(ys[1] - ys[0])
    dtheta32 = thetas[2] - thetas[1]

    Fu2_2 = K_2seg["Ka"]*du32
    Fu3_2 = -K_2seg["Ka"]*du32
    # Fv2_2 = K_2seg["Kb1"]*dv32 - K_2seg["Kb2"]*dtheta32
    # Fv3_2 = -(K_2seg["Kb1"]*dv32 - K_2seg["Kb2"]*dtheta32)
    # Mz2_2 = -K_2seg["Kb2"]*dv32 + K_2seg["Kb3"]*dtheta32
    Fv2_2 = K_2seg["Kb1"]*dv32 + K_2seg["Kb2"]*dtheta32
    Fv3_2 = -(K_2seg["Kb1"]*dv32 + K_2seg["Kb2"]*dtheta32)
    Mz2_2 = K_2seg["Kb2"]*dv32 + K_2seg["Kb3"]*dtheta32
    Mz3_2 = -Mz2_2 - Fv3_2*L/2

    Fuv1_1 = np.array([[0., 0., tau]]).T # !!!!!!1
    Fuv2_1 = np.array([[Fu2_1, Fv2_1, Mz2_1]]).T
    Fuv2_2 =  np.array([[Fu2_2, Fv2_2, Mz2_2]]).T
    Fuv3_2 = np.array([[Fu3_2, Fv3_2, Mz3_2]]).T

    Fxy1 = rotz(thetas[0]) @ Fuv1_1
    Fxy2 = rotz(thetas[0]) @ Fuv2_1 + rotz(thetas[1]) @ Fuv2_2
    Fxy3 = rotz(thetas[1]) @ Fuv3_2

    F = np.vstack((Fxy1[[2],:], Fxy2, Fxy3))
    accel = np.linalg.inv(M) @ F

    return np.vstack((vel, accel))

def get_stiffness_params(EA, EI, L):
    K = dict()
    K["Ka"] = EA/L
    K["Kb1"] = 12*EI/L**3
    K["Kb2"] = 6*EI/L**2
    K["Kb3"] = 4*EI/L
    # K = np.array([[EA/L, 0., 0., -EA/L, 0., 0.]])
    return K

def get_lumped_masses(m, L):
    # diag = np.array([0.5, 0.5, L**2/24]*2)
    diag = np.array([0.5, 0.5, L**2/6]*2)
    return m*np.diag(diag)

def get_rotation_matrix(theta):
    R = rotz(theta).T
    return block_diag(R, R)

def get_consistent_masses(m, L, theta):
    t1 = np.array([[1/3,    0.,         0.,         1/6,    0.,         0.],
                    [0.,    13/35,      11*L/210,   0.,     9/70,       -13*L/420],
                    [0.,    11*L/210,   L**2/105,   0.,     13*L/420,   -L**2/140],
                    [1/6,   0.,         0.,         1/3,    0.,         0.,],
                    [0.,    9/70,       13*L/420,   0.,     13/35,      -11*L/210],
                    [0.,    -13*L/420,  -L**2/140,  0.,     -11*L/210,  L**2/105]])
    R = get_rotation_matrix(theta)
    return m*R.T @ t1 @ R

def one_link_manipulator():
    # Link parameters
    m = 1 # kg
    L = 1 # m
    r_com = L/2
    EI = 1e+2 # Nm^2
    EA = 1e+3 # N

    # In the paper they do not specify the shape
    # therefore I assume the link to be a rod to calculate the
    # moment of inertia
    I = m*L**2/12    

    # Get stiffness parameters
    K_2seg = get_stiffness_params(EA, EI, L/2)
    K_4seg = get_stiffness_params(EA, EI, L/4)
    K_8seg = get_stiffness_params(EA, EI, L/8)
    
    # Get mass parameters
    Ml_2seg = get_lumped_masses(m, L/2)
    Ml_4seg = get_lumped_masses(m, L/4)
    Ml_8seg = get_lumped_masses(m, L/8)
    Mc_4seg = get_consistent_masses(m, L/4, 0)

    # Input parameter
    tau = 1 # Nm

    # Simulation
    ts = 1e-2
    n_iter = 50
    t = np.arange(0, n_iter+1)*ts
    x_rb = np.zeros((n_iter+1, 2))
    x_rb[0,:] = np.array([0., 0.])
    for k in range(n_iter):
        sol = solve_ivp(one_link_rigid_ode, [0, ts], x_rb[k,:], args=(tau, I, m, r_com), 
                        vectorized=True, rtol=1e-3, atol=1e-6)
        x_rb[k+1,:] = sol.y[:,-1]

    print("Finished simlating rigid body")

    n_iter = 15
    x2element = np.zeros((n_iter+1, 14))
    x2element[0,:] = np.array([0., L/2, 0., 0., L, 0., 0] + [0]*7)

    for k in range(n_iter):
        sol = solve_ivp(two_element_flexible_ode, [0, ts], x2element[k,:], args=(tau, m, EA, EI, L), 
                        method='LSODA', rtol=1e-6, atol=1e-8)
        x2element[k+1,:] = sol.y[:,-1]


    _, ax = plt.subplots()
    ax.plot(t, x_rb[:,0])
    ax.plot(t[:n_iter+1], x2element[:,0], '--')
    ax.grid(alpha=0.25)
    plt.show()


if __name__ == "__main__":
    one_link_manipulator()