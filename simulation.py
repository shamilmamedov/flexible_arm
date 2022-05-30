#!/usr/bin/env python3


class Simulator:
    """ Implements a simulator for FlexibleArm
    """
    def __init__(self) -> None:
        pass

    def __ode(self, t, x):
        """ Wraps ode of the FlexibleArm to match scipy notation
        """