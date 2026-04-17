"""
Benchmark configuration for the RL Energy Benchmark study.

Defines the algorithm x environment matrix, hyperparameters,
and experimental settings.
"""

# ---------------------------------------------------------------------------
# Algorithms to benchmark (SB3 class names)
# ---------------------------------------------------------------------------
ALGORITHMS = ["PPO", "A2C", "DQN", "SAC", "TD3"]

# ---------------------------------------------------------------------------
# Environments and their properties
# ---------------------------------------------------------------------------
ENVIRONMENTS = {
    "CartPole-v1": {
        "action_space": "discrete",
        "timesteps": 100_000,
        "eval_episodes": 10,
    },
    "LunarLander-v3": {
        "action_space": "discrete",
        "timesteps": 200_000,
        "eval_episodes": 10,
    },
    "Pendulum-v1": {
        "action_space": "continuous",
        "timesteps": 100_000,
        "eval_episodes": 10,
    },
    "BipedalWalker-v3": {
        "action_space": "continuous",
        "timesteps": 500_000,
        "eval_episodes": 10,
    },
}

# ---------------------------------------------------------------------------
# Algorithm compatibility with action spaces
# ---------------------------------------------------------------------------
ALGO_ACTION_SPACE = {
    "PPO":  ["discrete", "continuous"],
    "A2C":  ["discrete", "continuous"],
    "DQN":  ["discrete"],
    "SAC":  ["continuous"],
    "TD3":  ["continuous"],
}

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456]
SAMPLING_INTERVAL_S = 1.0       # Hardware sampling interval (seconds)
CO2_REGION = "world"            # Carbon intensity region for CO2 estimation
DEVICE = "auto"                 # Use GPU when available (we're benchmarking GPU energy)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = "benchmark_results"
SUMMARY_CSV = f"{OUTPUT_DIR}/results_summary.csv"
ENERGY_LOGS_DIR = f"{OUTPUT_DIR}/energy_logs"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"


def get_valid_combinations():
    """
    Generate all valid (algorithm, environment) combinations
    based on action space compatibility.

    Returns:
        List of (algo_name, env_name) tuples.
    """
    combos = []
    for algo in ALGORITHMS:
        for env_name, env_cfg in ENVIRONMENTS.items():
            if env_cfg["action_space"] in ALGO_ACTION_SPACE[algo]:
                combos.append((algo, env_name))
    return combos


def get_full_matrix():
    """
    Generate the full experiment matrix: (algorithm, environment, seed).

    Returns:
        List of (algo_name, env_name, seed) tuples.
    """
    matrix = []
    for algo, env_name in get_valid_combinations():
        for seed in SEEDS:
            matrix.append((algo, env_name, seed))
    return matrix


if __name__ == "__main__":
    combos = get_valid_combinations()
    matrix = get_full_matrix()
    print(f"Valid algorithm x environment combinations: {len(combos)}")
    for algo, env in combos:
        ts = ENVIRONMENTS[env]["timesteps"]
        print(f"  {algo:5s} x {env:20s}  ({ts:>7,} timesteps)")
    print(f"\nTotal runs (x {len(SEEDS)} seeds): {len(matrix)}")
