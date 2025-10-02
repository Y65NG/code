from dataclasses import dataclass, field
from typing import List
import numpy as np

from .test_case import TestCase


@dataclass
class TestResult:
    test_case: TestCase

    simulation_failed: bool = False
    crashed: bool = False
    violated_safety_limits: bool = False
    score: float = 0.0

    min_alt: float = 0.0

    trajectory: np.ndarray = field(default_factory=lambda: np.array([]))
