"""
Run batch simulations and export the collected outputs.
"""

import os

import numpy as np
import pandas as pd
from pyDOE import lhs

from model import BangladeshModel

# ---------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick = 1 minute
RUN_LENGTH = 5 * 24 * 60
PROGRESS_INTERVAL = 500

BASE_SEED = 1234567
NUM_REPLICATES = 1  # 10

NUM_SAMPLES = 10

# Factor ranges (min, max) for bridge failure probabilities
FACTOR_RANGES = {
    'A': [0, 0.10],   # new bridge: low failure probability
    'B': [0.05, 0.20], # older bridge: moderate failure probability
    'C': [0.10, 0.30], # older bridge: higher failure probability
    'D': [0.15, 0.50]  # very old bridge: high failure probability
}

# ---------------------------------------------------------------
# Generate LHS scenarios
# ---------------------------------------------------------------

def generate_scenarios(factor_ranges, num_samples):
    """Generate LHS-sampled scenarios with ordered constraint A < B < C < D."""
    num_factors = len(factor_ranges)
    lhs_samples = lhs(num_factors, samples=num_samples)

    # Scale LHS samples to the defined factor ranges
    scaled_samples = np.zeros_like(lhs_samples)
    for i, (min_val, max_val) in enumerate(factor_ranges.values()):
        scaled_samples[:, i] = min_val + lhs_samples[:, i] * (max_val - min_val)

    df = pd.DataFrame(scaled_samples, columns=factor_ranges.keys())

    # Enforce A < B < C < D for each sample
    df['A'] = np.minimum(df['A'], df['B'] - 0.01)
    df['A'] = np.minimum(df['A'], df['C'] - 0.01)
    df['A'] = np.minimum(df['A'], df['D'] - 0.01)
    df['B'] = np.minimum(df['B'], df['C'] - 0.01)
    df['B'] = np.minimum(df['B'], df['D'] - 0.01)
    df['C'] = np.minimum(df['C'], df['D'] - 0.01)

    scenarios = {
        i: df.iloc[i].to_dict() for i in range(num_samples)
    }
    return scenarios

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    scenarios = generate_scenarios(FACTOR_RANGES, NUM_SAMPLES)

    for scenario_id, params in scenarios.items():
        print(f"Scenario {scenario_id}: {params}")

    base_dir = os.path.dirname(__file__)
    experiments_dir = os.path.normpath(os.path.join(base_dir, "..", "Experiments"))

    for scenario_id in sorted(scenarios.keys()):
        for replicate in range(1, NUM_REPLICATES + 1):
            seed = BASE_SEED + replicate - 1
            sim_model = BangladeshModel(
                scenario_id=scenario_id,
                scenario_probs=scenarios[scenario_id],
                seed=seed
            )

            print(f"SEED {sim_model._seed} | SCENARIO {scenario_id} | REPLICATE {replicate}")

            for i in range(RUN_LENGTH):
                sim_model.step()
                if (i + 1) % PROGRESS_INTERVAL == 0 or i + 1 == RUN_LENGTH:
                    active_trucks = sum(
                        1 for agent in sim_model.schedule._agents.values()
                        if agent.__class__.__name__ == "Vehicle"
                    )
                    print(
                        f"PROGRESS {i + 1}/{RUN_LENGTH}"
                        f" | ACTIVE_TRUCKS {active_trucks}"
                        f" | COMPLETED {len(sim_model.completed_vehicle_times)}"
                    )

            # Export driving times
            output_path = os.path.join(
                experiments_dir,
                f"truck_driving_times_scenario_{scenario_id}_replicate_{replicate}.csv"
            )
            sim_model.export_total_driving_times(output_path=output_path)
            print(f"Driving times saved to: {output_path}")

            # Export infrastructure crossings
            infrastructure_output_path = os.path.join(
                experiments_dir,
                f"infrastructure_crossings_scenario_{scenario_id}_replicate_{replicate}.csv"
            )
            sim_model.export_infrastructure_crossing_summary(output_path=infrastructure_output_path)
            print(f"Infrastructure crossings saved to: {infrastructure_output_path}")
