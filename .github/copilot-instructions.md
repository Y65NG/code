# Copilot Instructions for F-16 Neural Network Test Coverage Research

## Project Overview

This research codebase explores **test case selection methods** (Random, Greedy, CMA-ES) for neural network verification in aerospace systems. It uses F-16 flight simulations with ACASXU (collision avoidance) and GCAS (ground collision avoidance) autopilots, measuring coverage via **trajectory distance metrics** (DTW, Fréchet).

## Architecture

### Core Components

- **`type_definitions/`** - Data structures for test cases and results:
    - `TestCase`, `TestCase_ACAS`, `AcasXuDubinsTestCase` - Initial flight conditions (velocity, angles, altitude, positions)
    - `TestResult`, `TestResult_ACAS` - Simulation outcomes (crashed, trajectory, min_alt, score)
    - `utils.py` - Test generation (`generate_cases`), simulation (`evaluate_cases`), distance calculations (`pairwise_distances`)

- **`aerobench/`** - Third-party F-16 flight dynamics simulator (DO NOT MODIFY):
    - `run_f16_sim.py` - Core simulation runner using scipy RK45 integration
    - `examples/gcas/` - Ground Collision Avoidance System autopilot
    - `examples/acasxu/` - ACAS Xu collision avoidance neural networks
    - Treat as read-only external dependency; wrap calls in `type_definitions/utils.py`

- **`ACASXU_Dubins/`** - Simplified 2D Dubins dynamics for faster experiments

### Key Notebooks

| Notebook              | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `acasxu_f16.ipynb`    | Full F-16 ACASXU experiments with CMA-ES coverage optimization |
| `gcas.ipynb`          | GCAS coverage testing and ML distance predictors               |
| `acasxu_dubins.ipynb` | Simplified Dubins dynamics experiments                         |

## Patterns & Conventions

### Test Case Workflow

```python
# 1. Generate test cases
cases = generate_cases_acasxu(n_cases)  # Returns List[TestCase_ACAS]

# 2. Run simulations (expensive - may fail)
results = evaluate_cases_acasxu(cases)  # Returns List[TestResult_ACAS]
valid_results = [r for r in results if not r.simulation_failed]

# 3. Compute trajectory distances (cache these!)
distance_matrix = pairwise_distances(trajectories, distance_type="dtw")
np.save(f"cache/distance_matrix_seed{seed}_size{len(trajectories)}.npy", distance_matrix)
```

### Feature Engineering for ML Models

Convert test cases to numpy arrays using helper functions like `testcase_to_ndarray()`. Feature vectors include:

- Flight parameters: velocity (ft/s), angles (radians), altitude (ft)
- Position: north/east coordinates or relative positions
- Angular indices for proper angle difference handling: indices `[1,2,3,4,7,8,9,10]`

### Pairwise Feature Pattern for Distance Prediction

ML models predict trajectory distance from test case features. The pattern creates rich pairwise features:

```python
def pair_features_between(feat1: np.ndarray, feat2: np.ndarray) -> np.ndarray:
    """Create ~50 features from two 14-dim test case vectors."""
    feat_diff = feat1 - feat2                    # 14 features: raw differences
    feat_abs_diff = np.abs(feat_diff)            # 14 features: absolute differences
    euclidean_dist = np.linalg.norm(feat_diff)   # 1 feature: overall distance

    # Proper angular differences (wrap to [-pi, pi]) for indices in ANGULAR_INDICES
    angular_diffs = [(diff + np.pi) % (2 * np.pi) - np.pi for diff in angular_subset]  # 8 features

    # Interaction terms for non-linearity
    vt_alt_interaction = np.abs(vt_diff) * np.abs(alt_diff) / 1e6

    return np.concatenate([feat_diff, feat_abs_diff, [euclidean_dist], angular_diffs, ...])
```

**Key insight**: Raw feature differences alone underperform. Add:

- Squared differences for non-linear effects
- Cross-term interactions (velocity × altitude)
- Min/max features for asymmetry detection
- Log-transform targets: `y_log = np.log1p(distances)` for skewed distributions

### Coverage Distance Metric

Lower coverage distance = better test suite coverage:

```python
def compute_coverage_distance(selected_indices, validation_indices, distance_matrix):
    # Sum of min distances from each validation trajectory to nearest selected trajectory
```

## Parameter Bounds

```python
# F-16 flight parameters (imperial units)
BOUNDS_LOWER = [400,  # vt (ft/s)
                np.deg2rad(0),   # alpha
                np.deg2rad(-20), # beta
                np.deg2rad(-180), # phi
                np.deg2rad(-90),  # theta
                500]  # altitude (ft)
```

## Development Tips

### Caching

Distance matrix computation is expensive. Always check cache first:

```python
cache_file = f"cache/distance_matrix_seed{seed}_size{n}.npy"
if os.path.exists(cache_file):
    distance_matrix = np.load(cache_file)
```

### Simulation Robustness

- Simulations can fail silently - always filter: `[r for r in results if not r.simulation_failed]`
- Use `np.random.seed(seed)` for reproducible experiments
- Log transform distance targets for ML models: `y_log = np.log1p(y)`

### Statistical Analysis

Compare methods using Mann-Whitney U tests and Cohen's d effect size. See notebook cells for examples using `scipy.stats.mannwhitneyu`.

## Dependencies

Managed via `pyproject.toml` with `uv`. Key packages: `cma`, `fastdtw`, `scikit-learn`, `scipy`, `torch`
