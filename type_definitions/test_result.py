from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .test_case import TestCase, TestCase_ACAS, AcasXuDubinsTestCase


@dataclass
class TestResult:
    test_case: TestCase

    simulation_failed: bool = False
    crashed: bool = False
    violated_safety_limits: bool = False
    score: float = 0.0

    min_alt: float = 0.0

    trajectory: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TestResult_ACAS:
    test_case: TestCase_ACAS

    simulation_failed: bool = False
    crashed: bool = False
    violated_safety_limits: bool = False
    score: float = 0.0

    min_alt: float = 0.0

    trajectory: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class AcasXuDubinsTestResult:
    """Test result for ACASXU Dubins scenarios with exactly 2 aircraft: ownship and intruder"""

    test_case: AcasXuDubinsTestCase

    simulation_failed: bool = False
    violated_safety_limits: bool = False
    collision_occurred: bool = False
    score: float = 0.0

    # Minimum separation distance in horizontal plane (ft)
    min_separation: float = np.inf

    # ACASXU-specific data
    command_history: Optional[List[int]] = None  # Ownship command history (0-4)
    intruder_command_history: Optional[List[int]] = None  # Intruder command history

    # Trajectories for both aircraft
    # Shape: (num_time_steps, 2, state_features)
    # Features: [x, y, theta] for Dubins (2D only, no altitude)
    trajectories: np.ndarray = field(default_factory=lambda: np.array([]))

    # Individual aircraft trajectories (for compatibility with existing code)
    # List of trajectories: [ownship_traj, intruder_traj]
    # Each trajectory shape: (num_time_steps, 3) where columns are [x, y, theta]
    aircraft_trajectories: List[np.ndarray] = field(default_factory=list)
