from dataclasses import dataclass


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
            self.alt,
            self.power,
            self.pos_n,
            self.pos_e,
            self.p,
            self.q,
            self.r,
        ]
