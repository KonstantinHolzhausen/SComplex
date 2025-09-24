import numpy as np
from scipy.integrate import odeint


class ThreeBodySystem():

    def __init__(self, parameters: dict):
        self.pSet = {}
        self.pSet["G"] = parameters["G"]  # gravitational constant
        self.pSet["m1"] = parameters["m1"] # masses
        self.pSet["m2"] = parameters["m2"]
        self.pSet["m3"] = parameters["m3"]

    @staticmethod
    def equation_of_motion(y: np.ndarray, t: float, pSet: dict) -> np.ndarray:
        """
        Right hand side of the ODE

        ARGS: 
            y : np.ndarray (2*#bodies, ), state vector of the three body system
            t : float, time state
            pSet: dict, dictionary containing the parameters of the model

        VALS:
            np.ndarray (2*#bodies, ) derivatives
        """
        # unpack all the state variables
        # phase space body 1
        r_1 = y[0:2]
        v_1 = y[2:4]
        r_2 = y[4:6]
        v_2 = y[6:8]
        r_3 = y[8:10]
        v_3 = y[10:12]

        # unpack parameters
        G = pSet["G"]
        m1 = pSet["m1"]
        m2 = pSet["m2"]
        m3 = pSet["m3"]

        # compute distances
        r12 = r_2 - r_1
        r13 = r_3 - r_1
        r23 = r_3 - r_2
        

        # compute accelerations
        a_1 = G*( m2*r12 / np.linalg.norm(r12)**3 + m3*r13 / np.linalg.norm(r13)**3 )
        a_2 = G*( -m1*r12 / np.linalg.norm(r12)**3 + m3*r23 / np.linalg.norm(r23)**3 )
        a_3 = G*( -m1*r13 / np.linalg.norm(r13)**3 - m2*r23 / np.linalg.norm(r23)**3 )

        # return derivatives
        return  np.array(
            [
                v_1, a_1, \
                v_2, a_2, \
                v_3, a_3
            ]
        ).flatten(order='C')

    def run(self, y0: np.ndarray, T: float, t0: float=0.0, dt: float=None) -> tuple:
        """
        Run the simulation by solving the initial value problem.

        ARGS: 
            y0 : np.ndarray (2*#bodies, ), initial state of the system in position velocity pairs (x0_1, v0_1, ... )
            T : float, simulation duration
            pSet: dict, dictionary containing the parameters of the model
                keys:   "G" - gravitational constant
                        "m_i" - mass value of body i

        VALS:
            tuple - (y: np.ndarray, infodict: dict), solution of the initial value problem, consists of
                y - state trajectory for all time points
                infodict - scipy.integrate infodict
        """

        if dt is None:
            num = 1000
        else:
            num = int ((T - t0) // dt) + 1

        t = np.linspace(t0, T, num=num)

        solution = odeint(
            ThreeBodySystem.equation_of_motion,
            y0,
            t,
            args=(self.pSet,),
        )

        return solution