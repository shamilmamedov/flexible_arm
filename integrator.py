import casadi as cs

def RK4(x, u, ode, dt, n):
    """ Numerical RK4 integrator
    """
    h = dt/n
    x_next = x
    for _ in range(n):
        k1 = ode(x_next, u)
        k2 = ode(x_next + h*k1/2, u)
        k3 = ode(x_next + h*k2/2, u)
        k4 = ode(x_next + h*k3, u)
        x_next = x_next + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return x_next


def symbolic_RK4(x, u, ode, n=4):
    """ Creates a symbolic RK4 integrator for
    a given dynamic system
    :parameter x: symbolic vector of states
    :parameter u: symbolic vector of inputs
    :parameter ode: ode of the system
    :parameter n: number of step for RK4 to take
    :return F_rk4: symbolic RK4 integrator
    """
    dt_sym = cs.MX.sym('dt')
    h = dt_sym/n
    x_next = x
    for _ in range(n):
        k1 = ode(x_next, u)
        k2 = ode(x_next + h*k1/2, u)
        k3 = ode(x_next + h*k2/2, u)
        k4 = ode(x_next + h*k3, u)
        x_next = x_next + h/6*(k1 + 2*k2 + 2*k3 + k4)

    F = cs.Function('F', [x, u, dt_sym], [x_next],
                    ['x', 'u', 'dt'], ['x_next'])

    A = cs.jacobian(x_next, x)
    dF_dx = cs.Function('dF_dx', [x, u, dt_sym], [A],
                        ['x', 'u', 'dt'], ['A'])
    return F, dF_dx