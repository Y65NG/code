import numpy as np
from typing import List, Optional, Tuple
from numpy.typing import NDArray
import similaritymeasures
from concurrent.futures import ThreadPoolExecutor

from aerobench_GCAS.util import StateIndex
from type_definitions.test_case import TestCase, TestCase_ACAS
from type_definitions.test_result import TestResult, TestResult_ACAS
from aerobench_GCAS.examples.gcas.gcas_autopilot import GcasAutopilot
from aerobench_ACASXU.examples.acasxu.acasxu_autopilot import AcasXuAutopilot
from aerobench_GCAS.run_f16_sim import run_f16_sim
from aerobench_ACASXU.run_f16_sim import run_f16_sim as run_f16_sim_acasxu
from aerobench_ACASXU.util import StateIndex as StateIndexACASXU, extract_single_result
from aerobench_ACASXU.lowlevel.low_level_controller import LowLevelController


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


def generate_single_case_acasxu() -> TestCase_ACAS:
    """
    Generate initial conditions for ACASXU simulator with ownship and intruder.
    
    Uses the same parameter ranges as generate_single_case() for both aircraft.
    
    Returns:
        TestCase_ACAS object with initial conditions for both ownship and intruder.
    """
    # Generate ownship parameters (same ranges as generate_single_case)
    ownship_vel = np.random.uniform(400, 800)
    ownship_alpha = np.deg2rad(np.random.uniform(0.0, 10.0))
    ownship_beta = np.deg2rad(np.random.uniform(-20, 20))
    ownship_alt = np.random.uniform(500, 6000)
    ownship_phi = np.deg2rad(np.random.uniform(-180, 180))
    ownship_theta = np.deg2rad(np.random.uniform(-90, 0))
    ownship_psi = 0
    ownship_power = 9
    ownship_pos_n = np.random.uniform(-5000, 5000)
    ownship_pos_e = np.random.uniform(-5000, 5000)
    
    # Generate intruder parameters (same ranges as generate_single_case)
    intruder_vel = np.random.uniform(400, 800)
    intruder_alpha = np.deg2rad(np.random.uniform(0.0, 10.0))
    intruder_beta = np.deg2rad(np.random.uniform(-20, 20))
    intruder_alt = np.random.uniform(500, 6000)
    intruder_phi = np.deg2rad(np.random.uniform(-180, 180))
    intruder_theta = np.deg2rad(np.random.uniform(-90, 0))
    intruder_psi = 0
    intruder_power = 9
    # Intruder position: typically positioned north of ownship
    intruder_pos_n = np.random.uniform(15000, 35000)
    intruder_pos_e = np.random.uniform(-5000, 5000)
    
    return TestCase_ACAS(
        ownship_vt=ownship_vel,
        ownship_alpha=ownship_alpha,
        ownship_beta=ownship_beta,
        ownship_phi=ownship_phi,
        ownship_theta=ownship_theta,
        ownship_psi=ownship_psi,
        ownship_alt=ownship_alt,
        ownship_power=ownship_power,
        ownship_pos_n=ownship_pos_n,
        ownship_pos_e=ownship_pos_e,
        intruder_vt=intruder_vel,
        intruder_alpha=intruder_alpha,
        intruder_beta=intruder_beta,
        intruder_phi=intruder_phi,
        intruder_theta=intruder_theta,
        intruder_psi=intruder_psi,
        intruder_alt=intruder_alt,
        intruder_power=intruder_power,
        intruder_pos_n=intruder_pos_n,
        intruder_pos_e=intruder_pos_e,
    )

def generate_cases_acasxu(count: int) -> List[TestCase_ACAS]:
    return [generate_single_case_acasxu() for _ in range(count)]

def evaluate_cases_acasxu(cases: List[TestCase_ACAS]) -> List[TestResult_ACAS]:
    """
    Evaluate TestCase_ACAS objects by running ACASXU simulations.
    
    Args:
        cases: List of TestCase_ACAS objects to evaluate.
    
    Returns:
        List of TestResult_ACAS objects containing simulation results.
    """
    sim_time = 15
    max_safe_velocity = 2500
    min_safe_velocity = 300
    
    llc = LowLevelController()
    
    results = []
    for case in cases:
        try:
            # Get initial state for ACASXU (includes both intruder and ownship)
            init_state = case.get_acasxu_states(llc)
            
            # Create ACASXU autopilot (num_aircraft_acasxu=1 means only ownship uses ACASXU)
            autopilot = AcasXuAutopilot(init_state, llc, num_aircraft_acasxu=1, stdout=False)
            
            # Run simulation
            raw_result = run_f16_sim_acasxu(
                init_state,
                sim_time,
                autopilot,
                extended_states=True,
                print_errors=False,
            )
            
            # Extract states for ownship (index 1) and intruder (index 0)
            ownship_result = extract_single_result(raw_result, 1, llc)  # Ownship is index 1
            intruder_result = extract_single_result(raw_result, 0, llc)  # Intruder is index 0
            
            ownship_states = np.array(ownship_result["states"])
            intruder_states = np.array(intruder_result["states"])
            
            # Check altitudes and velocities for both aircraft
            ownship_alts = ownship_states[:, StateIndexACASXU.ALT]
            ownship_vels = ownship_states[:, StateIndexACASXU.VEL]
            intruder_alts = intruder_states[:, StateIndexACASXU.ALT]
            intruder_vels = intruder_states[:, StateIndexACASXU.VEL]
            
            # Create result object
            result = TestResult_ACAS(case)
            
            # Find minimum altitude (worst case between both aircraft)
            min_alt_ownship = np.min(ownship_alts)
            min_alt_intruder = np.min(intruder_alts)
            result.min_alt = min(min_alt_ownship, min_alt_intruder)
            
            # Check for crashes
            if result.min_alt < 0:
                result.crashed = True
            
            # Check velocity safety limits for both aircraft
            if (np.any(ownship_vels < min_safe_velocity) or np.any(ownship_vels > max_safe_velocity) or
                np.any(intruder_vels < min_safe_velocity) or np.any(intruder_vels > max_safe_velocity)):
                result.violated_safety_limits = True
            
            # Calculate score
            if result.crashed:
                result.score -= 500
            
            # Penalize velocity violations
            for vel in ownship_vels:
                if vel < min_safe_velocity or vel > max_safe_velocity:
                    dv = (
                        vel - min_safe_velocity
                        if vel < min_safe_velocity
                        else vel - max_safe_velocity
                    )
                    result.score -= 1 * (0.01 * float(dv))
            
            for vel in intruder_vels:
                if vel < min_safe_velocity or vel > max_safe_velocity:
                    dv = (
                        vel - min_safe_velocity
                        if vel < min_safe_velocity
                        else vel - max_safe_velocity
                    )
                    result.score -= 1 * (0.01 * float(dv))
            
            # Get g_force values from Nz_list for ownship
            if "Nz_list" in ownship_result and ownship_result["Nz_list"] is not None:
                Nz_list = ownship_result["Nz_list"]
                ownship_g_forces = np.array(Nz_list)
            else:
                ownship_g_forces = np.zeros(len(ownship_states))
            
            # Build trajectory from ownship states (using ownship as primary)
            result.trajectory = np.column_stack(
                [
                    ownship_states[:, StateIndexACASXU.POS_N],
                    ownship_states[:, StateIndexACASXU.POS_E],
                    ownship_states[:, StateIndexACASXU.ALT],
                    ownship_states[:, StateIndexACASXU.PHI],
                    ownship_states[:, StateIndexACASXU.THETA],
                    ownship_states[:, StateIndexACASXU.PSI],
                    ownship_g_forces,
                ]
            )
            
            results.append(result)
            
        except Exception as e:
            # print(f"Error evaluating case: {e}")
            result = TestResult_ACAS(case, simulation_failed=True)
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
