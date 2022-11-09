import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import Simulator
from controller import PDController3Dof, FeedforwardController
from flexible_arm_3dof import (SymbolicFlexibleArm3DOF,
                               get_rest_configuration)


def compare_different_discretizations():
    """ Compares models with different discretization -- starting 
    from a rigid body model and until a fine grid of 10 discretization
    points -- in simulatoin. For every simulation the same integrator, 
    controller and initial state are chosen.   
    """
    # Simulation parametes
    ts = 0.001
    n_iter = 1000

    # Create FlexibleArm instance
    n_segs = [0, 1, 2, 3, 5, 10]
    fas = [SymbolicFlexibleArm3DOF(n_seg) for n_seg in n_segs]

    # Initial state; the same for all models
    # active joint configurations
    # NOTE Do all of the should start from the rest position?
    qa = np.array([0., 0., 0.])
    # Q = [get_rest_configuration(qa, n_seg) for n_seg in n_segs]
    Q = [np.zeros((m.nq, 1)) for m in fas]
    dQ = [np.zeros_like(q) for q in Q]
    X0 = [np.vstack((q, dq)) for q, dq in zip(Q, dQ)]

    # Reference; the same for all models but because of
    qa_ref = [1., 0.5, 1.]
    Q_ref = [np.zeros((m.nq, 1)) for m in fas]
    for q_ref, n_seg in zip(Q_ref, n_segs):
        q_ref[:2, 0] = qa_ref[:2]
        q_ref[1 + n_seg + 1] = qa_ref[2]

    # Estimator
    E = None

    # Controller
    # controller = DummyController()
    Kp = (40, 30, 25)
    Kd = (1.5, 0.25, 0.25)

    # Controller for the rigid body approximation
    C_0s = PDController3Dof(Kp, Kd, n_segs[0], Q_ref[0])

    # Simulators
    integrator = 'cvodes'
    # S = [Simulator(fa, c, integrator, E) for fa, c in zip(fas, C)]
    # Simulator for the rigid body approximation
    S_0s = Simulator(fas[0], C_0s, integrator, E)

    # Simulate the rigid body approxmiation
    x_0s, u_0s, y_0s, _ = S_0s.simulate(X0[0].flatten(), ts, n_iter)
    

    # Create feedforward cotnrollers for RFEM models
    C = [FeedforwardController(u_0s) for _ in n_segs[1:]]
    
    # Create simulators for RFEM models
    S = [Simulator(fa, c, integrator, E) for fa, c in zip(fas[1:], C)]

    # Simulate REFEM models
    x_1s, u_1s, y_1s, _ = S[0].simulate(X0[1].flatten(), ts, n_iter)
    x_2s, u_2s, y_2s, _ = S[1].simulate(X0[2].flatten(), ts, n_iter)
    x_3s, u_3s, y_3s, _ = S[2].simulate(X0[3].flatten(), ts, n_iter)
    x_5s, u_5s, y_5s, _ = S[3].simulate(X0[4].flatten(), ts, n_iter)
    x_10s, u_10s, y_10s, _ = S[4].simulate(X0[5].flatten(), ts, n_iter)

    t = np.arange(0, n_iter + 1) * ts

    # Find the norm of the difference betweem 10s and other models
    delta_0s = y_10s - y_0s
    delta_1s = y_10s - y_1s
    delta_2s = y_10s - y_2s
    delta_3s = y_10s - y_3s
    delta_5s = y_10s - y_5s

    norm_delta_0s = np.linalg.norm(delta_0s, axis=0).reshape(1,-1)
    norm_delta_1s = np.linalg.norm(delta_1s, axis=0).reshape(1,-1)
    norm_delta_2s = np.linalg.norm(delta_2s, axis=0).reshape(1,-1)
    norm_delta_3s = np.linalg.norm(delta_3s, axis=0).reshape(1,-1)
    norm_delta_5s = np.linalg.norm(delta_5s, axis=0).reshape(1,-1)

    norm_delta = np.vstack((norm_delta_0s, norm_delta_1s, norm_delta_2s,
                            norm_delta_3s, norm_delta_5s))

    cols = ['|delta qa_1|', '|delta qa_2|', '|delta qa_3|', 
            '|delta dqa_1|', '|delta dqa_2|', '|delta dqa_3|', 
            '|delta ee_x|', '|delta ee_y|', '|delta ee_z|']

    df = pd.DataFrame(norm_delta, columns=cols, index=[0, 1, 2, 3, 5])
    print(df.iloc[:,6:])


    # Plot outputs
    y_lbls = [r'$\mathrm{ee}_x$ [m]', r'$\mathrm{ee}_y$ [m]',
              r'$\mathrm{ee}_z$ [m]']
    legends = [f'{n_seg}s' for n_seg in n_segs]
    _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t, y_0s[:, 6+k], label=legends[0])
        ax.plot(t, y_1s[:, 6+k], label=legends[1])
        ax.plot(t, y_2s[:, 6+k], label=legends[2])
        ax.plot(t, y_3s[:, 6+k], label=legends[3])
        ax.plot(t, y_5s[:, 6+k], ls='--', label=legends[4])
        ax.plot(t, y_10s[:, 6+k], ls='-.', label=legends[5])
        ax.set_ylabel(y_lbls[k])
        ax.grid(alpha=0.5)
        ax.legend()
    ax.set_xlabel('t [s]')
    plt.tight_layout()

    # Plot controls
    y_lbls = [r'$\tau$ [m]', r'$\tau$ [m]', r'$\tau$ [m]']
    _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t[:-1], u_0s[:, k], label=legends[0])
        ax.plot(t[:-1], u_1s[:, k], label=legends[1])
        ax.plot(t[:-1], u_2s[:, k], label=legends[2])
        ax.plot(t[:-1], u_3s[:, k], label=legends[3])
        ax.plot(t[:-1], u_5s[:, k], ls='--', label=legends[4])
        ax.plot(t[:-1], u_10s[:, k], ls='-.', label=legends[5])
        ax.set_ylabel(y_lbls[k])
        # ax.set_xlim([0, 0.025])
        ax.grid(alpha=0.5)
        ax.legend()
    ax.set_xlabel('t [s]')
    plt.tight_layout()

    plt.show()
 
 
if __name__ == "__main__":
    compare_different_discretizations()