#!/usr/bin/env python3

import numpy as np


# def inertial_params_rectangle(a, h, l, rho):
#     """ Inertial parameters of the rectangular beam

#     :parameter a: width of the beam
#     :parameter h: height of the beam
#     :parameter l: length of the beam
#     :parameter rho: density of the material
#     """
#     V = a*h*l
#     m = rho*V

#     # Center of mass
#     rc = l/2

#     # Second moments of inertia
#     Ixx = m/12*(a**2 + h**2)
#     Iyy = m/12*(a**2 + l**2)
#     Izz = m/12*(h**2 + l**2)
#     Ixy = m*a**2/12
#     Ixz = m*h**2/12
#     Iyz = m*l**2/12
#     return m, rc, np.array([Ixx, Ixy, Ixz, Iyy, Iyz, Izz])

def inertial_params_rectangle(a, h, l, rho):
    """ Inertial parameters of the rectangular beam

    :parameter a: width of the beam
    :parameter h: height of the beam
    :parameter l: length of the beam
    :parameter rho: density of the material
    """
    V = a*h*l
    m = rho*V

    # Center of mass
    rc = l/2

    # Second moments of inertia
    Ixx = m/12*l**2
    Iyy = m/12*a**2
    Izz = m/12*h**2
    return m, rc, np.array([Ixx, Iyy, Izz])

def spring_params_rectangle(a, h, l, E, G):
    """ Stiffness and damping parameters of the rectangle
    """
    A = a*h

    Jyy = A*h**2/12
    Jzz = A*a**2/12
    
    kx = G*(Jyy + Jzz)/l
    ky = E*Jyy/l
    kz = E*Jzz/l
    return np.array([kx, ky, kz])


if __name__ == "__main__":
    # Parameters of the beam
    L = 0.5
    a = 0.05
    h = 0.002
    E = 1.9e11
    G = 74e+9
    rho = 7.87e3
    zeta = 5e-3

    m, rc, I = inertial_params_rectangle(a, h, L, rho)
    k = spring_params_rectangle(a, h, L, E, G)
    wn = np.sqrt(k/m)
    d = 2*zeta*wn

    print(f"segement mass = {m}")
    print(f"segement moment of inertia = {I}")
    print(f"spring constants = {k}")
    print(f"natural frequencies = {wn}")
    print(f"damping = {d}")

    # Consider 3 segments
    # rfe_lengths = [L/6, L/3, L/3, L/6]
    # rfe_inertial_params = [inertial_params_rectangle(a, h, x, rho) for x in rfe_lengths]
    # rfe_m = [x[0] for x in rfe_inertial_params]
    # rfe_rc = [x[1] for x in rfe_inertial_params]
    # rfe_I = [x[2] for x in rfe_inertial_params]
    # sde_k = [spring_params_rectangle(a, h, x, E, G) for x in rfe_lengths]
    # rfe_wn = [np.sqrt(k/m) for k, m in zip(sde_k, rfe_m)]
    # sde_d = [2*zeta*wn for wn in rfe_wn]
    # print(sde_d)

    # Consider 5 segments
    n_seg = 10
    rfe_lengths = [L/(2*n_seg)] + [L/n_seg]*(n_seg-1) +  [L/(2*n_seg)] #[L/6, L/3, L/3, L/6]
    rfe_inertial_params = [inertial_params_rectangle(a, h, x, rho) for x in rfe_lengths]
    rfe_m = [x[0] for x in rfe_inertial_params]
    rfe_rc = [x[1] for x in rfe_inertial_params]
    rfe_I = [x[2] for x in rfe_inertial_params]
    sde_k = [spring_params_rectangle(a, h, x, E, G) for x in rfe_lengths]
    rfe_wn = [np.sqrt(k/m) for k, m in zip(sde_k, rfe_m)]
    sde_d = [2*zeta*wn for wn in rfe_wn]
    print(sde_d)

    



