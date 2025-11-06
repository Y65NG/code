from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestCase:
    # Initial conditions
    vt: float  # velocity (ft/s)
    alpha: float  # angle of attack (rad)
    beta: float  # sideslip angle (rad)
    phi: float  # roll angle (rad)
    theta: float  # pitch angle (rad)
    psi: float  # yaw angle (rad)
    alt: float  # altitude (ft)
    power: float  # engine power (0-10)

    # Position
    pos_n: float = 0.0  # north position (ft)
    pos_e: float = 0.0  # east position (ft)

    # Angular rates (usually start at 0)
    p: float = 0.0  # roll rate (rad/s)
    q: float = 0.0  # pitch rate (rad/s)
    r: float = 0.0  # yaw rate (rad/s)

    def get_states(self) -> list[float]:
        return [
            self.vt,
            self.alpha,
            self.beta,
            self.phi,
            self.theta,
            self.psi,
            self.p,
            self.q,
            self.r,
            self.pos_n,
            self.pos_e,
            self.alt,
            self.power,
        ]


@dataclass
class AcasXuTestCase:
    """Test case for ACASXU scenarios with exactly 2 aircraft: ownship and intruder"""

    # Initial separation distance between ownship and intruder (ft)
    # This is the distance in the horizontal plane
    separation: float = 25000.0

    # Initial conditions for ownship (aircraft 0)
    ownship_vt: float = 807.0  # velocity (ft/s) - typical cruising speed
    ownship_alt: float = 0.0  # altitude (ft) - will use trim state altitude
    ownship_power: float = 9.0  # engine power (0-10)
    ownship_pos_n: float = 0.0  # north position (ft)
    ownship_pos_e: float = 0.0  # east position (ft)
    ownship_psi: float = 0.0  # yaw angle (rad) - 0 is facing north

    # Initial conditions for intruder (aircraft 1)
    intruder_vt: float = 807.0  # velocity (ft/s)
    intruder_alt: float = 0.0  # altitude (ft) - will use trim state altitude
    intruder_power: float = 9.0  # engine power (0-10)
    intruder_psi: float = 0.0  # yaw angle (rad) - random heading
    intruder_placement_angle: float = (
        0.0  # angle (rad) for placing intruder on circle around ownship
    )

    def get_initial_states(self, llc) -> List[float]:
        """
        Generate combined initial state for ownship and intruder.
        Returns concatenated state vector for both aircraft.
        """
        import math
        from aerobench.util import StateIndex, get_state_names

        num_vars = len(get_state_names()) + llc.get_num_integrators()
        rv = []

        # Ownship (aircraft 0)
        # Use equilibrium state (which includes trim pitch angle theta = 0.0389 for level flight)
        ownship_init = list(llc.xequil)
        ownship_init[StateIndex.VT] = self.ownship_vt
        ownship_init[StateIndex.POSN] = self.ownship_pos_n
        ownship_init[StateIndex.POSE] = self.ownship_pos_e
        ownship_init[StateIndex.PSI] = self.ownship_psi
        ownship_init[StateIndex.POW] = self.ownship_power
        # Keep equilibrium theta (0.0389 rad) - this is the trim pitch angle for level flight
        # Keep equilibrium phi (0.0) - wings level
        if self.ownship_alt != 0.0:
            ownship_init[StateIndex.ALT] = self.ownship_alt
        ownship_init += [0] * llc.get_num_integrators()
        rv += ownship_init

        # Intruder (aircraft 1) - positioned at separation distance from ownship
        # Intruder is placed at random angle on circle around ownship (following ACASXU_Dubins pattern)
        intruder_pos_n = self.ownship_pos_n + self.separation * math.cos(
            self.intruder_placement_angle
        )
        intruder_pos_e = self.ownship_pos_e + self.separation * math.sin(
            self.intruder_placement_angle
        )

        # Use equilibrium state (which includes trim pitch angle theta = 0.0389 for level flight)
        intruder_init = list(llc.xequil)
        intruder_init[StateIndex.VT] = self.intruder_vt
        intruder_init[StateIndex.POSN] = intruder_pos_n
        intruder_init[StateIndex.POSE] = intruder_pos_e
        intruder_init[StateIndex.PSI] = self.intruder_psi
        intruder_init[StateIndex.POW] = self.intruder_power
        # Keep equilibrium theta (0.0389 rad) - this is the trim pitch angle for level flight
        # Keep equilibrium phi (0.0) - wings level
        if self.intruder_alt != 0.0:
            intruder_init[StateIndex.ALT] = self.intruder_alt
        intruder_init += [0] * llc.get_num_integrators()
        rv += intruder_init

        return rv


@dataclass
class AcasXuDubinsTestCase:
    """Test case for ACASXU Dubins scenarios with exactly 2 aircraft: ownship and intruder

    Uses simplified Dubins car dynamics (2D only, no altitude).
    State vector: [x1, y1, theta1, x2, y2, theta2, time]
    """

    # Initial separation distance between ownship and intruder (ft)
    # Intruder is placed on circle around ownship
    separation: float = 25000.0

    # Initial conditions for ownship (aircraft 0)
    ownship_x: float = 0.0  # x position (ft) - ownship starts at origin
    ownship_y: float = 0.0  # y position (ft)
    ownship_theta: float = 0.0  # heading angle (rad) - typically π/2 (north)
    ownship_v: float = 800.0  # velocity (ft/s)

    # Initial conditions for intruder (aircraft 1)
    intruder_placement_angle: float = (
        0.0  # angle (rad) for placing intruder on circle around ownship
    )
    intruder_theta: float = 0.0  # heading angle (rad) - random heading
    intruder_v: float = 500.0  # velocity (ft/s)

    # ACASXU-specific parameters
    tau_init: int = 0  # initial time to closest approach (seconds)
    tau_dot: int = 0  # rate of change of tau (-1 for decreasing, 0 for constant)

    # Intruder command list (for simulation)
    # Commands: 0=clear of conflict, 1=weak left, 2=weak right, 3=strong left, 4=strong right
    intruder_cmd_list: Optional[List[int]] = None

    def __post_init__(self):
        """Initialize default intruder command list if not provided"""
        if self.intruder_cmd_list is None:
            # Default: intruder doesn't turn (all clear of conflict commands)
            self.intruder_cmd_list = [0] * 150

    def get_initial_state_vec(self) -> List[float]:
        """
        Generate initial state vector for Dubins simulation.
        Returns [x1, y1, theta1, x2, y2, theta2, time] where time starts at 0.
        """
        import math

        # Ownship starts at origin
        x1 = self.ownship_x
        y1 = self.ownship_y
        theta1 = self.ownship_theta

        # Intruder is placed on circle around ownship
        x2 = x1 + self.separation * math.cos(self.intruder_placement_angle)
        y2 = y1 + self.separation * math.sin(self.intruder_placement_angle)
        theta2 = self.intruder_theta

        # Time starts at 0
        time = 0.0

        return [x1, y1, theta1, x2, y2, theta2, time]
