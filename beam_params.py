import numpy as np


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
    rc = np.array([l/2, 0., 0])

    # Second moments of inertia
    Ixx = m/12*(a**2 + h**2)
    Iyy = m/12*(a**2 + l**2)
    Izz = m/12*(l**2 + h**2)
    return m, rc, np.array([Ixx, Iyy, Izz])


def inertial_params_cylinder(d, l, rho):
    """ Inertial parameters of the cylindrical beam

    :parameter d: diameter
    :parameter l: length
    :parameter rho: density
    """
    V = np.pi*d**2*l/4
    m = rho*V

    # Center of mass
    rc = l/2

    # Second moments of inertia
    Ixx = m*d**2/8
    Iyy = m/48*(3*d**2 + 4*l**2)
    Izz = m/48*(3*d**2 + 4*l**2)

    return m, rc, np.array([Ixx, Iyy, Izz])


def spring_params_rectangle(a, h, l, E, G):
    """ Equivalent stiffness parameters of the rectangle
    """
    A = a*h

    Jzz = A*h**2/12
    Jyy = A*a**2/12

    kx = G*(Jyy + Jzz)/l  # True for circular cross section
    ky = E*Jyy/l
    kz = E*Jzz/l
    return np.array([np.nan, ky, kz])


def spring_params_cylinder(d, l, E, G):
    """ Equivalent spring parameters of the cylindtical beam
    """
    r = d/2

    Jxx = np.pi/2*r**4
    Jyy = np.pi/4*r**4
    Jzz = np.pi/4*r**4

    kx = G*Jxx/l  # ??????????
    ky = E*Jyy/l
    kz = E*Jzz/l
    return np.array([np.nan, ky, kz])


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

    # Consider 5 segments
    n_seg = 0
    if n_seg > 0:
        # Get the length of the primary-divison and the mass. These 
        # parameters are used for computing stiffness and damping parameters
        delta_l = L/n_seg
        m_delta_l = inertial_params_rectangle(a, h, delta_l, rho)[0]
        
        # Get the length and inertial parameters of the RFEs from the 
        # secondary division
        rfe_lengths = [L/(2*n_seg)] + [L/n_seg]*(n_seg-1) + \
            [L/(2*n_seg)]  # [L/6, L/3, L/3, L/6]
        rfe_inertial_params = [inertial_params_rectangle(
            a, h, x, rho) for x in rfe_lengths]
        rfe_m = [x[0] for x in rfe_inertial_params]
        rfe_rc = [x[1] for x in rfe_inertial_params]
        rfe_I = [x[2] for x in rfe_inertial_params]

        # Get stiffness and damping parameters
        sde_k = [spring_params_rectangle(a, h, delta_l, E, G)]*n_seg
        delta_l_wn = [np.sqrt(x/m_delta_l) for x in sde_k]
        sde_d = [2*zeta*x for x in delta_l_wn]

        for k, (mk, rck, Ik) in enumerate(zip(rfe_m, rfe_rc, rfe_I)):
            print(f"RFE{[k+1]}")
            print(f"\tmass = {mk}")
            print(f"\tCOM = {rck}")
            print(f"\tmoment of inertia = {Ik}")

        print(f"\nStiffness {n_seg}x = {sde_k[0]}")
        print(f"Damping {n_seg}x = {sde_d[0]}")
    else:
        m, rc, I = inertial_params_rectangle(a, h, L, rho)
        print(f"Link mass = {m}")
        print(f"Link COM = {rc}")
        print(f"Link moment of inertia = {I}")
    