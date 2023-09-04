import numpy as np
import yaml
from dataclasses import dataclass


@dataclass(frozen=True)
class RectangularBeamParams:
    """ Rectangular beam parameters
    """
    a: float # width
    h: float # height
    L: float # length
    rho: float # density
    E: float # Young's modulus
    G: float # shear modulus
    eta: float # normal damping coefficient
    eta_bar: float # tangential damping coefficient


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


def compute_rfe_lengths(L: float, n_seg: int):
    """
    :param L: length of the beam
    :param n_seg: number of segments to divide

    :return: list of lengths of the RFEs
    """
    first_and_last = L/(2*n_seg)
    middle = L/n_seg
    return [first_and_last] + [middle]*(n_seg-1) + [first_and_last]
    

def compute_rfe_inertial_params(
        beam_params: RectangularBeamParams, 
        n_seg: int
):
    """
    :param beam_params: parameters of the beam
    :param n_seg: number of segments to divide

    :return: a tupe of list of masses, COMs, and moments of inertia of the RFEs
    """
    a = beam_params.a
    h = beam_params.h
    rho = beam_params.rho
    L = beam_params.L
    rfe_lengths = compute_rfe_lengths(L, n_seg)
    rfe_inertial_params = [inertial_params_rectangle(
        a, h, x, rho) for x in rfe_lengths]
    
    rfe_m = [x[0] for x in rfe_inertial_params]
    rfe_rc = [x[1] for x in rfe_inertial_params]
    rfe_I = [x[2] for x in rfe_inertial_params]
    return rfe_m, rfe_rc, rfe_I


def compute_flexible_joint_params(
        beam_params: RectangularBeamParams, 
        n_seg: int
):
    """
    NOTE: we consider only bending in longitudinal direction
          and negtect torsion and bending in lateral direction

    :param beam_params: parameters of the beam
    :param n_seg: number of segments to divide

    :return: a tuple of stiffness and damping parameters
    """
    a = beam_params.a
    h = beam_params.h
    L = beam_params.L
    E = beam_params.E
    G = beam_params.G
    eta = beam_params.eta
    eta_bar = beam_params.eta_bar
    delta_l = L/n_seg
    k = [spring_params_rectangle(a, h, delta_l, E, G)[2]]*n_seg
    d = [spring_params_rectangle(a, h, delta_l, eta, eta_bar)[2]]*n_seg
    return k, d

def load_rectangular_beam_params_from_yaml(path: str):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return RectangularBeamParams(**data)


if __name__ == "__main__":
    link_params_path = 'models/three_dof/flexible_link_params.yml'
    beam_params = load_rectangular_beam_params_from_yaml(link_params_path)

    n_seg = 10
    if n_seg > 0:
        k, d = compute_flexible_joint_params(beam_params, n_seg)
        m, rc, I = compute_rfe_inertial_params(beam_params, n_seg)

        for k, (mk, rck, Ik) in enumerate(zip(m, rc, I)):
            print(f"RFE{[k+1]}")
            print(f"\tmass = {mk}")
            print(f"\tCOM = {rck}")
            print(f"\tmoment of inertia = {Ik}")

        print(f"\nStiffness {n_seg}x = {k[0]}")
        print(f"Damping {n_seg}x = {d[0]}")
    else:
        m, rc, I = inertial_params_rectangle(
            beam_params.a, 
            beam_params.h, 
            beam_params.L, 
            beam_params.rho
        )
        print(f"Link mass = {m}")
        print(f"Link COM = {rc}")
        print(f"Link moment of inertia = {I}")