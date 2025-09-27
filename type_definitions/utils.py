import numpy as np
from typing import List, Optional
import similaritymeasures

from aerobench.util import StateIndex
from type_definitions.test_case import TestCase
from type_definitions.test_result import TestResult
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from aerobench.run_f16_sim import run_f16_sim


def generate_single_case() -> TestCase:
    vel = np.random.uniform(300, 900)
    alpha = np.deg2rad(np.random.uniform(1.0, 2.0))
    beta = np.deg2rad(np.random.uniform(-5, 5))

    alt = np.random.uniform(4000, 5000)
    phi = np.deg2rad(np.random.uniform(-120, 120))
    theta = np.deg2rad(np.random.uniform(-60, 0))

    psi = 0
    power = 9

    return TestCase(
        vt=vel,
        alpha=alpha,
        beta=beta,
        phi=phi,
        theta=theta,
        psi=psi,
        alt=alt,
        power=power,
    )


def generate_cases(count: int) -> List[TestCase]:
    return [generate_single_case() for _ in range(count)]


def evaluate_cases(cases: List[TestCase]) -> List[TestResult]:
    sim_time = 15

    max_safe_velocity = 2500
    min_safe_velocity = 300

    autopilot = GcasAutopilot(init_mode="roll", stdout=False)

    results = []
    for case in cases:
        try:
            raw_result = run_f16_sim(
                case.get_states(),
                sim_time,
                autopilot,
                extended_states=True,
                print_errors=False,
            )

            states = np.array(raw_result["states"])
            alts = states[:, StateIndex.ALT]
            vels = states[:, StateIndex.VEL]

            result = TestResult(case)
            result.min_alt = np.min(alts)

            if result.min_alt < 0:
                result.crashed = True

            if np.any(vels < min_safe_velocity) or np.any(vels > max_safe_velocity):
                result.violated_safety_limits = True

            if result.crashed:
                result.score -= 500

            for vel in vels:
                if vel < min_safe_velocity or vel > max_safe_velocity:
                    dv = (
                        vel - min_safe_velocity
                        if vel < min_safe_velocity
                        else vel - max_safe_velocity
                    )
                    result.score -= 1 * (0.01 * float(dv))

            # store the trajectory
            result.trajectory = np.column_stack(
                [
                    states[:, StateIndex.POS_N],
                    states[:, StateIndex.POS_E],
                    states[:, StateIndex.ALT],
                    states[:, StateIndex.PHI],
                    states[:, StateIndex.THETA],
                    states[:, StateIndex.PSI],
                ]
            )

            results.append(result)

        except Exception as e:
            print(f"Error evaluating cases: {e}")
            result = TestResult(case, simulation_failed=True)
            results.append(result)

    return results


def frechet_distance(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    return float(similaritymeasures.frechet_dist(trajectory1, trajectory2))
