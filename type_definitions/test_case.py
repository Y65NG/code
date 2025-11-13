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
class TestCase_ACAS:
    # Ownship (main plane) initial conditions - required fields
    ownship_vt: float  # velocity (ft/s)
    ownship_alpha: float  # angle of attack (rad)
    ownship_beta: float  # sideslip angle (rad)
    ownship_phi: float  # roll angle (rad)
    ownship_theta: float  # pitch angle (rad)
    ownship_psi: float  # yaw angle (rad)
    ownship_alt: float  # altitude (ft)
    ownship_power: float  # engine power (0-10)

    # Intruder initial conditions - required fields
    intruder_vt: float  # velocity (ft/s)
    intruder_alpha: float  # angle of attack (rad)
    intruder_beta: float  # sideslip angle (rad)
    intruder_phi: float  # roll angle (rad)
    intruder_theta: float  # pitch angle (rad)
    intruder_psi: float  # yaw angle (rad)
    intruder_alt: float  # altitude (ft)
    intruder_power: float  # engine power (0-10)

    # Ownship optional fields (with defaults)
    ownship_pos_n: float = 0.0  # north position (ft)
    ownship_pos_e: float = 0.0  # east position (ft)
    ownship_p: float = 0.0  # roll rate (rad/s)
    ownship_q: float = 0.0  # pitch rate (rad/s)
    ownship_r: float = 0.0  # yaw rate (rad/s)

    # Intruder optional fields (with defaults)
    intruder_pos_n: float = 0.0  # north position (ft)
    intruder_pos_e: float = 0.0  # east position (ft)
    intruder_p: float = 0.0  # roll rate (rad/s)
    intruder_q: float = 0.0  # pitch rate (rad/s)
    intruder_r: float = 0.0  # yaw rate (rad/s)

    def get_ownship_states(self) -> list[float]:
        """Get state array for ownship aircraft."""
        return [
            self.ownship_vt,
            self.ownship_alpha,
            self.ownship_beta,
            self.ownship_phi,
            self.ownship_theta,
            self.ownship_psi,
            self.ownship_p,
            self.ownship_q,
            self.ownship_r,
            self.ownship_pos_n,
            self.ownship_pos_e,
            self.ownship_alt,
            self.ownship_power,
        ]

    def get_intruder_states(self) -> list[float]:
        """Get state array for intruder aircraft."""
        return [
            self.intruder_vt,
            self.intruder_alpha,
            self.intruder_beta,
            self.intruder_phi,
            self.intruder_theta,
            self.intruder_psi,
            self.intruder_p,
            self.intruder_q,
            self.intruder_r,
            self.intruder_pos_n,
            self.intruder_pos_e,
            self.intruder_alt,
            self.intruder_power,
        ]

    def get_acasxu_states(self, llc=None) -> list[float]:
        """
        Get combined ACASXU state array: [intruder_states (16) + ownship_states (16)].
        Includes integrator states for ACASXU simulator.

        Args:
            llc: LowLevelController instance. If None, one will be created.

        Returns:
            List of 32 floats representing the combined initial state for ACASXU.
        """
        from aerobench_ACASXU.lowlevel.low_level_controller import LowLevelController

        if llc is None:
            llc = LowLevelController()

        num_integrators = llc.get_num_integrators()

        # Get intruder states and add integrators
        intruder_states = self.get_intruder_states()
        intruder_states += [0.0] * num_integrators

        # Get ownship states and add integrators
        ownship_states = self.get_ownship_states()
        ownship_states += [0.0] * num_integrators

        # Concatenate intruder and ownship states
        return intruder_states + ownship_states


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
