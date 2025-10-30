import numpy as np
from typing import List, Optional, Tuple
from numpy.typing import NDArray
import similaritymeasures
from concurrent.futures import ThreadPoolExecutor

from aerobench.util import StateIndex
from type_definitions.test_case import TestCase
from type_definitions.test_result import TestResult
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from aerobench.run_f16_sim import run_f16_sim


def generate_single_case() -> TestCase:

    # TODO: look for better ranges
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


def frechet_distance(
    trajectory1: NDArray[np.float64], trajectory2: NDArray[np.float64]
) -> float:
    return float(similaritymeasures.frechet_dist(trajectory1, trajectory2))


def _calculate_distance_pair(args):
    i, j, traj1, traj2 = args
    if i == j:
        return 0.0
    return frechet_distance(traj1, traj2)


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
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
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
    width = float(max(0.0, p95 - p5))

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