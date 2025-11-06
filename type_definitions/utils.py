import numpy as np
from typing import List, Optional, Tuple
from numpy.typing import NDArray
import similaritymeasures
from concurrent.futures import ThreadPoolExecutor

from aerobench.util import StateIndex, get_state_names
from type_definitions.test_case import TestCase, AcasXuTestCase, AcasXuDubinsTestCase
from type_definitions.test_result import (
    TestResult,
    AcasXuTestResult,
    AcasXuDubinsTestResult,
)
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from aerobench.examples.acasxu.acasxu_autopilot import AcasXuAutopilot
from aerobench.lowlevel.low_level_controller import LowLevelController
from aerobench.run_f16_sim import run_f16_sim


def generate_single_case() -> TestCase:
    # Velocity: 400-800 ft/s (typical for GCAS scenarios, within 300-2500 safety limit)
    # Alpha: 0-10 deg (wider than trim-only, tests edge cases, within -10 to 45 safety limit)
    # Beta: -20 to 20 deg (wider for realistic scenarios, within ±30 deg safety limit)
    # Altitude: 500-6000 ft (includes below flight deck where GCAS activates at 1000 ft)
    # Phi: -180 to 180 deg (full roll range including inverted scenarios)
    # Theta: -90 to 0 deg (includes very steep dives, typical for GCAS scenarios)

    vel = np.random.uniform(400, 800)
    alpha = np.deg2rad(np.random.uniform(0.0, 10.0))
    beta = np.deg2rad(np.random.uniform(-20, 20))

    alt = np.random.uniform(500, 6000)
    phi = np.deg2rad(np.random.uniform(-180, 180))
    theta = np.deg2rad(np.random.uniform(-90, 0))

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

            # calculate g_force values from Nz_list
            # Nz stands for normal acceleration of the aircraft, which value is zero at 1g
            if "Nz_list" in raw_result and raw_result["Nz_list"] is not None:
                Nz_list = raw_result["Nz_list"]
                g_forces = np.array(Nz_list)
            else:
                g_forces = np.zeros(len(states))

            result.trajectory = np.column_stack(
                [
                    states[:, StateIndex.POS_N],
                    states[:, StateIndex.POS_E],
                    states[:, StateIndex.ALT],
                    states[:, StateIndex.PHI],
                    states[:, StateIndex.THETA],
                    states[:, StateIndex.PSI],
                    g_forces,
                ]
            )

            results.append(result)

        except Exception as e:
            # print(f"Error evaluating cases: {e}")
            result = TestResult(case, simulation_failed=True)
            results.append(result)

    return results


def extract_altitude_trajectory(trajectory: NDArray[np.float64]) -> NDArray[np.float64]:
    altitudes = trajectory[:, 2]
    time_steps = np.arange(len(altitudes), dtype=np.float64)
    return np.column_stack([time_steps, altitudes]).astype(np.float64)


def altitude_frechet_distance(
    trajectory1: NDArray[np.float64], trajectory2: NDArray[np.float64]
) -> float:
    alt_traj1 = extract_altitude_trajectory(trajectory1)
    alt_traj2 = extract_altitude_trajectory(trajectory2)
    return float(similaritymeasures.frechet_dist(alt_traj1, alt_traj2))


def frechet_distance(
    trajectory1: NDArray[np.float64], trajectory2: NDArray[np.float64]
) -> float:
    return float(similaritymeasures.frechet_dist(trajectory1, trajectory2))


def _calculate_distance_pair(args):
    i, j, traj1, traj2 = args
    if i == j:
        return 0.0
    return frechet_distance(traj1, traj2)


def _calculate_altitude_distance_pair(args):
    i, j, traj1, traj2 = args
    if i == j:
        return 0.0
    return altitude_frechet_distance(traj1, traj2)


def _calculate_dubins_distance_pair(args):
    i, j, traj1, traj2 = args
    if i == j:
        return 0.0
    return dubins_frechet_distance(traj1, traj2)


def pairwise_distances(
    trajectories: List[NDArray[np.float64]], n_jobs: Optional[int] = None
) -> NDArray[np.float64]:

    n = len(trajectories)
    if n == 0:
        return np.array([])

    # Initialize distance matrix
    distances = np.zeros((n, n))

    # Prepare arguments for parallel processing
    args_list = []
    for i in range(n):
        for j in range(i, n):
            args_list.append((i, j, trajectories[i], trajectories[j]))

    # Calculate distances in parallel
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_calculate_distance_pair, args_list))

    # Fill the distance matrix
    idx = 0
    for i in range(n):
        for j in range(i, n):
            dist = results[idx]
            distances[i, j] = dist
            distances[j, i] = dist  # Symmetric matrix
            idx += 1

    return distances


def pairwise_altitude_distances(
    trajectories: List[NDArray[np.float64]], n_jobs: Optional[int] = None
) -> NDArray[np.float64]:
    n = len(trajectories)
    if n == 0:
        return np.array([])

    distances = np.zeros((n, n))

    args_list = []
    for i in range(n):
        for j in range(i, n):
            args_list.append((i, j, trajectories[i], trajectories[j]))

    # Calculate distances in parallel
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_calculate_altitude_distance_pair, args_list))

    idx = 0
    for i in range(n):
        for j in range(i, n):
            dist = results[idx]
            distances[i, j] = dist
            distances[j, i] = dist  # Symmetric matrix
            idx += 1

    return distances


def pairwise_dubins_distances(
    combined_trajectories: List[NDArray[np.float64]], n_jobs: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Calculate pairwise Fréchet distances for combined Dubin's trajectories.

    Args:
        combined_trajectories: List of NDArray, each with shape (len, 6) containing
                             [x1, y1, theta1, x2, y2, theta2] for each test case
        n_jobs: Number of parallel jobs (None for auto)

    Returns:
        Symmetric distance matrix of shape (n, n)
    """
    n = len(combined_trajectories)
    if n == 0:
        return np.array([])

    distances = np.zeros((n, n))

    args_list = []
    for i in range(n):
        for j in range(i, n):
            args_list.append((i, j, combined_trajectories[i], combined_trajectories[j]))

    # Calculate distances in parallel
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_calculate_dubins_distance_pair, args_list))

    idx = 0
    for i in range(n):
        for j in range(i, n):
            dist = results[idx]
            distances[i, j] = dist
            distances[j, i] = dist  # Symmetric matrix
            idx += 1

    return distances


def has_unsafe_gforces(trajectory, min_g_force=-1.0, max_g_force=6.0):
    if trajectory.shape[1] < 7:
        return False

    g_forces = trajectory[:, 6]
    return np.any(np.abs(g_forces) > max_g_force) or np.any(g_forces < min_g_force)


def greedy_permutation_clustering(
    distance_matrix: NDArray[np.float64], k_centers: int
) -> Tuple[List[int], List[int]]:
    n_points = distance_matrix.shape[0]

    if k_centers >= n_points:
        return list(range(n_points)), list(range(n_points))

    # Select first center as most central point (minimum sum of distances)
    center_indices = [int(np.argmin(np.sum(distance_matrix, axis=1)))]

    # Select remaining centers
    for _ in range(k_centers - 1):
        # Find point farthest from its nearest center
        max_min_distance = -1
        best_candidate = -1

        for i in range(n_points):
            if i not in center_indices:
                # Find distance to nearest center
                min_distance = min(
                    distance_matrix[i, center] for center in center_indices
                )

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = i

        center_indices.append(int(best_candidate))

    # Assign each point to its nearest center
    cluster_assignments = []
    for i in range(n_points):
        distances_to_centers = [distance_matrix[i, center] for center in center_indices]
        nearest_center_idx = int(np.argmin(distances_to_centers))
        cluster_assignments.append(nearest_center_idx)

    return center_indices, cluster_assignments


def frechet_coverage(
    distance_matrix: NDArray[np.float64],
    bins: int,
    hist_range: Tuple[float, float],
) -> Tuple[float, float]:
    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0] != distance_matrix.shape[1]
    ):
        return 0.0, 0.0
    n = distance_matrix.shape[0]
    if n < 2:
        return 0.0, 0.0

    iu = np.triu_indices(n, k=1)
    d = distance_matrix[iu]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0, 0.0

    # Width still uses robust percentiles (unchanged)
    p5 = np.percentile(d, 5.0)
    p95 = np.percentile(d, 95.0)
    width = max(0.0, float(p95) - float(p5))

    # Uniformity uses FIXED bins and FIXED range for comparability
    lo, hi = hist_range
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return width, 0.0

    counts, _ = np.histogram(d, bins=bins, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        return width, 0.0

    p = counts.astype(np.float64) / total
    p_nonzero = p[p > 0]
    if p_nonzero.size == 0:
        return width, 0.0

    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    max_entropy = np.log(len(p))
    uniformity = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return width, uniformity


# ============================================================================
# ACASXU-specific functions
# ============================================================================


def generate_acasxu_single_case(
    separation_range: Tuple[float, float] = (15000.0, 35000.0),
    vt_range: Tuple[float, float] = (700.0, 900.0),
) -> AcasXuTestCase:
    """
    Generate a single ACASXU test case with random configuration.
    Always generates exactly 2 aircraft: ownship and intruder.

    Follows ACASXU_Dubins pattern:
    - Ownship starts at origin (0, 0) with heading π/2 (north)
    - Intruder is placed at random angle on circle around ownship
    - Intruder heading is completely random

    Args:
        separation_range: Range for initial separation distance (ft)
        vt_range: Range for initial velocity (ft/s)

    Returns:
        AcasXuTestCase with random configuration
    """
    separation = np.random.uniform(separation_range[0], separation_range[1])
    ownship_vt = np.random.uniform(vt_range[0], vt_range[1])
    intruder_vt = np.random.uniform(vt_range[0], vt_range[1])

    # Ownship starts at origin with heading π/2 (north), following ACASXU_Dubins pattern
    ownship_psi = np.pi / 2

    # Intruder is placed at random angle on circle (0 to 2π)
    intruder_placement_angle = np.random.uniform(0, 2 * np.pi)

    # Intruder heading is completely random (0 to 2π)
    intruder_psi = np.random.uniform(0, 2 * np.pi)

    return AcasXuTestCase(
        separation=separation,
        ownship_vt=ownship_vt,
        intruder_vt=intruder_vt,
        ownship_psi=ownship_psi,
        intruder_psi=intruder_psi,
        intruder_placement_angle=intruder_placement_angle,
    )


def generate_acasxu_cases(
    count: int,
    separation_range: Tuple[float, float] = (15000.0, 35000.0),
    vt_range: Tuple[float, float] = (700.0, 900.0),
) -> List[AcasXuTestCase]:
    """
    Generate multiple ACASXU test cases.
    Each case has exactly 2 aircraft: ownship and intruder.

    Args:
        count: Number of test cases to generate
        separation_range: Range for initial separation distance (ft)
        vt_range: Range for initial velocity (ft/s)

    Returns:
        List of AcasXuTestCase objects
    """
    return [
        generate_acasxu_single_case(
            separation_range=separation_range,
            vt_range=vt_range,
        )
        for _ in range(count)
    ]


def evaluate_acasxu_cases(
    cases: List[AcasXuTestCase],
    sim_time: float = 20.0,
    step: float = 1.0 / 30.0,
    stop_on_coc: bool = False,
    extended_states: bool = True,
    print_errors: bool = False,
) -> List[AcasXuTestResult]:
    """
    Evaluate ACASXU test cases by running simulations.
    Always handles exactly 2 aircraft: ownship (index 0) and intruder (index 1).

    Args:
        cases: List of AcasXuTestCase to evaluate
        sim_time: Simulation time in seconds
        step: Simulation time step
        stop_on_coc: Whether to stop simulation on clear of conflict
        extended_states: Whether to collect extended state information
        print_errors: Whether to print simulation errors

    Returns:
        List of AcasXuTestResult objects
    """
    llc = LowLevelController()
    num_vars = len(get_state_names()) + llc.get_num_integrators()

    max_safe_velocity = 2500.0
    min_safe_velocity = 300.0
    min_safe_separation = 500.0  # minimum safe separation distance (ft)
    num_aircraft = 2  # Always 2: ownship and intruder

    results = []

    for case in cases:
        try:
            # Get initial states for both aircraft
            init = case.get_initial_states(llc)

            # Create autopilot (ownship uses ACASXU, intruder does not)
            ap = AcasXuAutopilot(
                init,
                llc,
                num_aircraft_acasxu=1,  # Only ownship uses ACASXU
                stop_on_coc=stop_on_coc,
                stdout=False,
            )

            # Run simulation
            raw_result = run_f16_sim(
                init,
                sim_time,
                ap,
                step=step,
                extended_states=extended_states,
                print_errors=print_errors,
            )

            # Parse results
            states = np.array(raw_result["states"])
            num_time_steps = len(states)

            result = AcasXuTestResult(test_case=case)

            # Store ACASXU-specific history
            result.command_history = ap.command_history
            result.full_history = ap.full_history

            # Extract states for ownship (0) and intruder (1)
            ownship_state = states[:, 0 * num_vars : 1 * num_vars]
            intruder_state = states[:, 1 * num_vars : 2 * num_vars]

            # Calculate metrics per aircraft
            ownship_alts = ownship_state[:, StateIndex.ALT]
            ownship_vels = ownship_state[:, StateIndex.VEL]
            intruder_alts = intruder_state[:, StateIndex.ALT]
            intruder_vels = intruder_state[:, StateIndex.VEL]

            result.min_alt = [
                float(np.min(ownship_alts)),
                float(np.min(intruder_alts)),
            ]

            # Note: We do NOT check for ground crashes (altitude < 0) because
            # altitude doesn't matter for ACASXU collision avoidance logic

            # Check velocity safety limits
            if (
                np.any(ownship_vels < min_safe_velocity)
                or np.any(ownship_vels > max_safe_velocity)
                or np.any(intruder_vels < min_safe_velocity)
                or np.any(intruder_vels > max_safe_velocity)
            ):
                result.violated_safety_limits = True

            # Get g-forces
            if "Nz_list" in raw_result and raw_result["Nz_list"] is not None:
                Nz_list = raw_result["Nz_list"]
                if isinstance(Nz_list[0], tuple):
                    # Multi-aircraft: Nz_list contains tuples
                    ownship_g_forces = np.array([nz[0] for nz in Nz_list])
                    intruder_g_forces = np.array([nz[1] for nz in Nz_list])
                else:
                    # Single aircraft (shouldn't happen for 2 aircraft, but handle it)
                    ownship_g_forces = np.array(Nz_list)
                    intruder_g_forces = np.zeros(num_time_steps)
            else:
                ownship_g_forces = np.zeros(num_time_steps)
                intruder_g_forces = np.zeros(num_time_steps)

            # Build trajectories
            ownship_traj = np.column_stack(
                [
                    ownship_state[:, StateIndex.POS_N],
                    ownship_state[:, StateIndex.POS_E],
                    ownship_state[:, StateIndex.ALT],
                    ownship_state[:, StateIndex.PHI],
                    ownship_state[:, StateIndex.THETA],
                    ownship_state[:, StateIndex.PSI],
                    ownship_g_forces,
                ]
            )

            intruder_traj = np.column_stack(
                [
                    intruder_state[:, StateIndex.POS_N],
                    intruder_state[:, StateIndex.POS_E],
                    intruder_state[:, StateIndex.ALT],
                    intruder_state[:, StateIndex.PHI],
                    intruder_state[:, StateIndex.THETA],
                    intruder_state[:, StateIndex.PSI],
                    intruder_g_forces,
                ]
            )

            result.aircraft_trajectories = [ownship_traj, intruder_traj]

            # Calculate minimum separation distance (horizontal only)
            ownship_pos = ownship_traj[:, :2]  # pos_n, pos_e
            intruder_pos = intruder_traj[:, :2]

            # Calculate distances at each time step
            distances = np.sqrt(np.sum((ownship_pos - intruder_pos) ** 2, axis=1))
            result.min_separation = float(np.min(distances))

            # Check for collision
            if result.min_separation < min_safe_separation:
                result.collision_occurred = True

            # Calculate score
            # Note: No crash penalty since we don't check for ground crashes
            if result.collision_occurred:
                result.score -= 1000
            if result.violated_safety_limits:
                result.score -= 100

            # Penalty for close approaches
            if result.min_separation < min_safe_separation * 2:
                penalty = (min_safe_separation * 2 - result.min_separation) / 100.0
                result.score -= penalty

            # Store trajectories as 3D array (time, aircraft, features)
            max_len = max(len(ownship_traj), len(intruder_traj))
            # Pad trajectories to same length
            if len(ownship_traj) < max_len:
                padding = np.tile(ownship_traj[-1:], (max_len - len(ownship_traj), 1))
                ownship_traj = np.vstack([ownship_traj, padding])
            if len(intruder_traj) < max_len:
                padding = np.tile(intruder_traj[-1:], (max_len - len(intruder_traj), 1))
                intruder_traj = np.vstack([intruder_traj, padding])

            result.trajectories = np.array([ownship_traj, intruder_traj]).transpose(
                1, 0, 2
            )  # (time, aircraft, features)

            results.append(result)

        except Exception as e:
            if print_errors:
                print(f"Error evaluating ACASXU case: {e}")
            result = AcasXuTestResult(test_case=case, simulation_failed=True)
            results.append(result)

    return results


def extract_acasxu_trajectory(
    result: AcasXuTestResult, aircraft_index: int = 0
) -> NDArray[np.float64]:
    """
    Extract trajectory for a specific aircraft from ACASXU result.

    Args:
        result: AcasXuTestResult
        aircraft_index: Index of aircraft (0-based)

    Returns:
        Trajectory array (same format as single-aircraft trajectory)
    """
    if aircraft_index < len(result.aircraft_trajectories):
        return result.aircraft_trajectories[aircraft_index]
    else:
        return np.array([])


# ============================================================================
# ACASXU Dubins-specific functions
# ============================================================================

import os
import sys
from functools import lru_cache
import onnxruntime as ort

# Import Dubins model and methods directly from ACASXU_Dubins
# Add ACASXU_Dubins to path if needed
_acasxu_dubins_path = os.path.join(os.path.dirname(__file__), "../ACASXU_Dubins")
_acasxu_dubins_path = os.path.abspath(_acasxu_dubins_path)
if _acasxu_dubins_path not in sys.path:
    sys.path.insert(0, _acasxu_dubins_path)

# Temporarily change working directory so relative paths work
_original_cwd = os.getcwd()
_simulation_path = os.path.join(_acasxu_dubins_path, "simulation")
if os.path.exists(_simulation_path):
    os.chdir(_simulation_path)

try:
    # Import functions that don't access resources folder
    from simulation.acasxu_dubins import (
        state7_to_state5,
        state7_to_state8,
        get_time_elapse_mat,
        step_state,
        network_index,
        run_network,
    )
finally:
    # Restore original working directory
    os.chdir(_original_cwd)

# Path to ACASXU_Dubins resources
_resources_path = os.path.join(_acasxu_dubins_path, "resources")


@lru_cache(maxsize=None)
def load_networks():
    """Load the 45 neural networks for Dubins simulation using absolute path"""
    nets = []
    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    for last_cmd in range(5):
        for tau in range(9):
            onnx_filename = os.path.join(
                _resources_path,
                f"ACASXU_run2a_{last_cmd + 1}_{tau + 1}_batch_2000.onnx",
            )
            session = ort.InferenceSession(onnx_filename)

            # Warm up the network
            i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
            i.shape = (1, 1, 1, 5)
            session.run(None, {"input": i})

            nets.append((session, range_for_scaling, means_for_scaling))

    return nets


def generate_acasxu_dubins_single_case(
    separation_range: Tuple[float, float] = (15000.0, 35000.0),
    v_own_range: Tuple[float, float] = (100.0, 1200.0),
    v_int_range: Tuple[float, float] = (0.0, 1200.0),
    tau_init: int = 0,
    intruder_can_turn: bool = False,
    num_inputs: int = 150,
    velocity_angle_diff_range: Optional[Tuple[float, float]] = None,
) -> AcasXuDubinsTestCase:
    """
    Generate a single ACASXU Dubins test case with random configuration.
    Always generates exactly 2 aircraft: ownship and intruder.

    Args:
        separation_range: Range for initial separation distance (ft)
        v_own_range: Range for ownship velocity (ft/s)
        v_int_range: Range for intruder velocity (ft/s)
        tau_init: Initial time to closest approach (seconds)
        intruder_can_turn: Whether intruder can turn (otherwise all commands are 0)
        num_inputs: Number of command inputs for intruder
        velocity_angle_diff_range: Optional tuple (min, max) for angle difference between
                                  ownship and intruder velocity headings (radians).
                                  If None, no constraint is applied.

    Returns:
        AcasXuDubinsTestCase with random configuration
    """
    separation = np.random.uniform(separation_range[0], separation_range[1])
    ownship_v = np.random.uniform(v_own_range[0], v_own_range[1])
    intruder_v = np.random.uniform(v_int_range[0], v_int_range[1])

    # Ownship starts at origin with heading π/2 (north)
    ownship_theta = np.pi / 2

    # Intruder is placed at random angle on circle (0 to 2π)
    intruder_placement_angle = np.random.uniform(0, 2 * np.pi)

    # Generate intruder heading based on velocity angle difference constraint
    if velocity_angle_diff_range is not None:
        # Generate angle difference within the specified range
        min_diff, max_diff = velocity_angle_diff_range
        angle_diff = np.random.uniform(min_diff, max_diff)
        
        # Choose direction randomly (clockwise or counterclockwise)
        # This gives us two possible angles: π/2 ± angle_diff
        # We randomly choose one direction, but need to handle wrapping
        direction = np.random.choice([-1, 1])
        intruder_theta = ownship_theta + direction * angle_diff
        
        # Wrap to [0, 2π]
        intruder_theta = intruder_theta % (2 * np.pi)
    else:
        # Intruder heading is completely random (0 to 2π)
        intruder_theta = np.random.uniform(0, 2 * np.pi)

    # Generate intruder command list
    if intruder_can_turn:
        intruder_cmd_list = [np.random.randint(5) for _ in range(num_inputs)]
    else:
        intruder_cmd_list = [0] * num_inputs

    return AcasXuDubinsTestCase(
        separation=separation,
        ownship_x=0.0,
        ownship_y=0.0,
        ownship_theta=ownship_theta,
        ownship_v=ownship_v,
        intruder_placement_angle=intruder_placement_angle,
        intruder_theta=intruder_theta,
        intruder_v=intruder_v,
        tau_init=tau_init,
        tau_dot=-1 if tau_init > 0 else 0,
        intruder_cmd_list=intruder_cmd_list,
    )


def generate_acasxu_dubins_cases(
    count: int,
    separation_range: Tuple[float, float] = (15000.0, 35000.0),
    v_own_range: Tuple[float, float] = (100.0, 1200.0),
    v_int_range: Tuple[float, float] = (0.0, 1200.0),
    tau_init: int = 0,
    intruder_can_turn: bool = False,
    num_inputs: int = 150,
    velocity_angle_diff_range: Optional[Tuple[float, float]] = None,
    max_attempts: Optional[int] = None,
) -> List[AcasXuDubinsTestCase]:
    """
    Generate multiple ACASXU Dubins test cases.
    Each case has exactly 2 aircraft: ownship and intruder.

    Args:
        count: Number of test cases to generate
        separation_range: Range for initial separation distance (ft)
        v_own_range: Range for ownship velocity (ft/s)
        v_int_range: Range for intruder velocity (ft/s)
        tau_init: Initial time to closest approach (seconds)
        intruder_can_turn: Whether intruder can turn
        num_inputs: Number of command inputs for intruder
        velocity_angle_diff_range: Optional tuple (min, max) for angle difference between
                                  ownship and intruder velocity headings (radians).
                                  If None, no constraint is applied.
        max_attempts: Deprecated parameter (kept for backwards compatibility, not used).

    Returns:
        List of AcasXuDubinsTestCase objects
    """
    cases = []
    for _ in range(count):
        case = generate_acasxu_dubins_single_case(
            separation_range=separation_range,
            v_own_range=v_own_range,
            v_int_range=v_int_range,
            tau_init=tau_init,
            intruder_can_turn=intruder_can_turn,
            num_inputs=num_inputs,
            velocity_angle_diff_range=velocity_angle_diff_range,
        )
        cases.append(case)
    
    return cases


def evaluate_acasxu_dubins_cases(
    cases: List[AcasXuDubinsTestCase],
    dt: float = 1.0,
    nn_update_rate: float = 1.0,
    print_errors: bool = False,
) -> List[AcasXuDubinsTestResult]:
    """
    Evaluate ACASXU Dubins test cases by running simulations.
    Always handles exactly 2 aircraft: ownship (index 0) and intruder (index 1).

    Args:
        cases: List of AcasXuDubinsTestCase to evaluate
        dt: Simulation time step (seconds)
        nn_update_rate: Neural network update rate (seconds)
        print_errors: Whether to print simulation errors

    Returns:
        List of AcasXuDubinsTestResult objects
    """
    # Load networks (cached)
    nets = load_networks()

    # Initialize time elapse matrices
    time_elapse_mats = []
    for cmd in range(5):
        time_elapse_mats.append([])
        for int_cmd in range(5):
            mat = get_time_elapse_mat(cmd, dt, int_cmd)
            time_elapse_mats[-1].append(mat)

    min_safe_separation = 500.0  # minimum safe separation distance (ft)

    results = []

    for case in cases:
        try:
            # Get initial state
            state7 = np.array(case.get_initial_state_vec(), dtype=float)
            v_own = case.ownship_v
            v_int = case.intruder_v

            # Initialize simulation state
            command = 0  # initial command (clear of conflict)
            next_nn_update = 0.0
            tau_init = case.tau_init
            tau_dot = case.tau_dot

            # Simulation history
            state_history = [state7.copy()]
            command_history = []
            intruder_command_history = []

            # Intruder command list
            u_list = case.intruder_cmd_list
            if u_list is None:
                u_list = [0] * 150
            u_list_index = 0

            # Simulation parameters
            tmax = len(u_list) * nn_update_rate
            t = 0.0

            # Track minimum separation
            prev_dist_sq = (state7[0] - state7[3]) ** 2 + (state7[1] - state7[4]) ** 2

            if tau_init == 0:
                min_dist_sq = prev_dist_sq
            else:
                min_dist_sq = np.inf

            # Run simulation
            while t + 1e-6 < tmax:
                # Update command if needed
                tol = 1e-6
                if abs(next_nn_update) < tol:
                    # Update ownship command based on ACASXU network
                    tau_now = round(tau_init + tau_dot * state7[-1])

                    # Convert state7 to network input [rho, theta, psi, v_own, v_int]
                    state5 = state7_to_state5(state7, v_own, v_int)

                    rho = state5[0]

                    if rho > 60760:
                        command = 0  # Clear of conflict
                    else:
                        last_command = command
                        ni = network_index(last_command, tau_now)
                        net = nets[ni]

                        res = run_network(net, state5)
                        command = int(np.argmin(res))

                    command_history.append(command)
                    next_nn_update = nn_update_rate

                # Get intruder command
                intruder_cmd = u_list[min(u_list_index, len(u_list) - 1)]
                intruder_command_history.append(intruder_cmd)

                # Step simulation
                time_elapse_mat = time_elapse_mats[command][intruder_cmd]
                state7 = step_state(state7, v_own, v_int, time_elapse_mat, dt)

                state_history.append(state7.copy())
                t += dt
                next_nn_update -= dt

                # Update tau
                tau_now = round(tau_init + tau_dot * state7[-1])

                # Track minimum separation
                cur_dist_sq = (state7[0] - state7[3]) ** 2 + (
                    state7[1] - state7[4]
                ) ** 2

                if tau_now == 0:
                    min_dist_sq = min(min_dist_sq, cur_dist_sq)

                # Early termination if distance is increasing and > 500ft
                if cur_dist_sq > prev_dist_sq and cur_dist_sq > 500**2:
                    break

                if tau_now < 0:
                    break

                prev_dist_sq = cur_dist_sq

                # Advance intruder command index
                if u_list_index < len(u_list) - 1:
                    u_list_index += 1

            # Create result
            result = AcasXuDubinsTestResult(test_case=case)

            # Store command histories
            result.command_history = command_history
            result.intruder_command_history = intruder_command_history

            # Extract trajectories
            state_array = np.array(state_history)

            ownship_traj = np.column_stack(
                [
                    state_array[:, 0],  # x
                    state_array[:, 1],  # y
                    state_array[:, 2],  # theta
                ]
            )

            intruder_traj = np.column_stack(
                [
                    state_array[:, 3],  # x
                    state_array[:, 4],  # y
                    state_array[:, 5],  # theta
                ]
            )

            result.aircraft_trajectories = [ownship_traj, intruder_traj]

            # Calculate minimum separation distance
            ownship_pos = ownship_traj[:, :2]  # x, y
            intruder_pos = intruder_traj[:, :2]

            distances = np.sqrt(np.sum((ownship_pos - intruder_pos) ** 2, axis=1))
            result.min_separation = float(np.min(distances))

            # Check for collision
            if result.min_separation < min_safe_separation:
                result.collision_occurred = True

            # Calculate score
            if result.collision_occurred:
                result.score -= 1000

            # Penalty for close approaches
            if result.min_separation < min_safe_separation * 2:
                penalty = (min_safe_separation * 2 - result.min_separation) / 100.0
                result.score -= penalty

            # Store trajectories as 3D array (time, aircraft, features)
            max_len = max(len(ownship_traj), len(intruder_traj))
            if len(ownship_traj) < max_len:
                padding = np.tile(ownship_traj[-1:], (max_len - len(ownship_traj), 1))
                ownship_traj = np.vstack([ownship_traj, padding])
            if len(intruder_traj) < max_len:
                padding = np.tile(intruder_traj[-1:], (max_len - len(intruder_traj), 1))
                intruder_traj = np.vstack([intruder_traj, padding])

            result.trajectories = np.array([ownship_traj, intruder_traj]).transpose(
                1, 0, 2
            )

            results.append(result)

        except Exception as e:
            if print_errors:
                print(f"Error evaluating ACASXU Dubins case: {e}")
            result = AcasXuDubinsTestResult(test_case=case, simulation_failed=True)
            results.append(result)

    return results


def extract_acasxu_dubins_trajectory(
    result: AcasXuDubinsTestResult, aircraft_index: int = 0
) -> NDArray[np.float64]:
    """
    Extract trajectory for a specific aircraft from ACASXU Dubins result.

    Args:
        result: AcasXuDubinsTestResult
        aircraft_index: Index of aircraft (0-based)

    Returns:
        Trajectory array with shape (num_time_steps, 3) where columns are [x, y, theta]
    """
    if aircraft_index < len(result.aircraft_trajectories):
        return result.aircraft_trajectories[aircraft_index]
    else:
        return np.array([])


def dubins_frechet_distance(
    trajectories1: NDArray[np.float64], trajectories2: NDArray[np.float64]
) -> float:
    """
    Calculate Fréchet distance between two combined Dubin's trajectories.

    Both input trajectories are already combined matrices with shape (len, 6),
    where each row contains [x1, y1, theta1, x2, y2, theta2] representing the combined
    state of both aircraft from a test case.

    Args:
        trajectories1: NDArray of shape (len1, 6) with [x1, y1, theta1, x2, y2, theta2]
                      from first test case
        trajectories2: NDArray of shape (len2, 6) with [x1, y1, theta1, x2, y2, theta2]
                      from second test case

    Returns:
        Fréchet distance between the two combined trajectories as a float
    """
    if len(trajectories1) == 0 or len(trajectories2) == 0:
        return np.inf

    # Both trajectories are already in the combined format (len, 6)
    # Calculate Fréchet distance between them
    # similaritymeasures.frechet_dist expects (time, features) format
    return float(similaritymeasures.frechet_dist(trajectories1, trajectories2))
