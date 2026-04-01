import os

from model import BangladeshModel

"""
Run batch simulations and export the collected outputs.
"""

# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick 1 minute
run_length = 5 *24* 60
progress_interval = 500

base_seed = 1234567
num_replicates = 1 #10

import numpy as np
from pyDOE import lhs
import pandas as pd

# Define the number of samples (runs) and factors
num_samples = 10
num_factors = 4  # For example, A, B, C, D
import numpy as np
from pyDOE import lhs
import pandas as pd
import os

# Define the number of samples (runs) and factors
num_samples = 10
num_factors = 4  # A, B, C, D

# Define the ranges for each factor (min, max) with overlap
ranges = {
    'A': [0, 0.10],  # new bridge: low failure probability
    'B': [0.05, 0.20],  # older bridge: moderate failure probability
    'C': [0.10, 0.30],  # older bridge: higher failure probability
    'D': [0.15, 0.50]  # very old bridge: high failure probability
}

# Generate LHS samples
lhs_samples = lhs(num_factors, samples=num_samples)

# Scale LHS samples to the defined factor ranges
scaled_samples = np.zeros_like(lhs_samples)
for i, (min_val, max_val) in enumerate(ranges.values()):
    scaled_samples[:, i] = min_val + lhs_samples[:, i] * (max_val - min_val)

# Convert scaled samples to a DataFrame for easy visualization
df = pd.DataFrame(scaled_samples, columns=ranges.keys())

# Apply logical constraints:
# Ensure A < B < C < D for each sample
df['A'] = np.minimum(df['A'], df['B'] - 0.01)  # Ensure A is always lower than B
df['A'] = np.minimum(df['A'], df['C'] - 0.01)  # Ensure A is always lower than C
df['A'] = np.minimum(df['A'], df['D'] - 0.01)  # Ensure A is always lower than D

df['B'] = np.minimum(df['B'], df['C'] - 0.01)  # Ensure B is always lower than C
df['B'] = np.minimum(df['B'], df['D'] - 0.01)  # Ensure B is always lower than D

df['C'] = np.minimum(df['C'], df['D'] - 0.01)  # Ensure C is always lower than D

print(df)

# Save or use the generated sample for your simulation runs
base_dir = os.path.dirname(__file__)
experiments_dir = os.path.normpath(os.path.join(base_dir, "..", "Experiments"))

# SCENARIOS
SCENARIOS = {
    0: {'A': df['A'][0], 'B': df['B'][0], 'C': df['C'][0], 'D': df['D'][0]},
    1: {'A': df['A'][1], 'B': df['B'][1], 'C': df['C'][1], 'D': df['D'][1]},
    2: {'A': df['A'][2], 'B': df['B'][2], 'C': df['C'][2], 'D': df['D'][2]},
    3: {'A': df['A'][3], 'B': df['B'][3], 'C': df['C'][3], 'D': df['D'][3]},
    4: {'A': df['A'][4], 'B': df['B'][4], 'C': df['C'][4], 'D': df['D'][4]},
    5: {'A': df['A'][5], 'B': df['B'][5], 'C': df['C'][5], 'D': df['D'][5]},
    6: {'A': df['A'][6], 'B': df['B'][6], 'C': df['C'][6], 'D': df['D'][6]},
    7: {'A': df['A'][7], 'B': df['B'][7], 'C': df['C'][7], 'D': df['D'][7]},
    8: {'A': df['A'][8], 'B': df['B'][8], 'C': df['C'][8], 'D': df['D'][8]},
    9: {'A': df['A'][9], 'B': df['B'][9], 'C': df['C'][9], 'D': df['D'][9]}
}

# For each scenario, you can access the parameters like this:
for scenario_id, scenario_params in SCENARIOS.items():
    print(f"Scenario {scenario_id}: {scenario_params}")
# Define the ranges for each factor (min, max) with overlap
ranges = {
    'A': [0, 0.10],  # new bridge: low failure probability
    'B': [0.05, 0.20],  # older bridge: moderate failure probability
    'C': [0.10, 0.30],  # older bridge: higher failure probability
    'D': [0.15, 0.50]  # very old bridge: high failure probability
}

# Generate LHS samples
lhs_samples = lhs(num_factors, samples=num_samples)

# Scale LHS samples to the defined factor ranges
scaled_samples = np.zeros_like(lhs_samples)
for i, (min_val, max_val) in enumerate(ranges.values()):
    scaled_samples[:, i] = min_val + lhs_samples[:, i] * (max_val - min_val)

# Convert scaled samples to a DataFrame for easy visualization
df = pd.DataFrame(scaled_samples, columns=ranges.keys())

# Apply logical constraints:
# Ensure A < B < C < D for each sample
df['A'] = np.minimum(df['A'], df['B'] - 0.01)  # Ensure A is always lower than B
df['A'] = np.minimum(df['A'], df['C'] - 0.01)  # Ensure A is always lower than C
df['A'] = np.minimum(df['A'], df['D'] - 0.01)  # Ensure A is always lower than D

df['B'] = np.minimum(df['B'], df['C'] - 0.01)  # Ensure B is always lower than C
df['B'] = np.minimum(df['B'], df['D'] - 0.01)  # Ensure B is always lower than D

df['C'] = np.minimum(df['C'], df['D'] - 0.01)  # Ensure C is always lower than D

print(df)


base_dir = os.path.dirname(__file__)
experiments_dir = os.path.normpath(os.path.join(base_dir, "..", "Experiments"))

for scenario_id in sorted(SCENARIOS.keys()):
    for replicate in range(1, num_replicates + 1):
        seed = base_seed + replicate - 1
        sim_model = BangladeshModel(
            scenario_id=scenario_id,
            scenario_probs=SCENARIOS[scenario_id],
            seed=seed
        )

        # Check if the seed is set
        print(
            "SEED "
            + str(sim_model._seed)
            + " | SCENARIO "
            + str(scenario_id)
            + " | REPLICATE "
            + str(replicate)
        )

        # One run with given steps
        for i in range(run_length):
            sim_model.step()
            if (i + 1) % progress_interval == 0 or i + 1 == run_length:
                active_trucks = sum(
                    1 for agent in sim_model.schedule._agents.values()
                    if agent.__class__.__name__ == "Vehicle"
                )
                print(
                    "PROGRESS "
                    + str(i + 1)
                    + "/"
                    + str(run_length)
                    + " | ACTIVE_TRUCKS "
                    + str(active_trucks)
                    + " | COMPLETED "
                    + str(len(sim_model.completed_vehicle_times))
                )

        output_path = os.path.join(
            experiments_dir,
            f"truck_driving_times_scenario_{scenario_id}_replicate_{replicate}.csv"
        )
        sim_model.export_total_driving_times(output_path=output_path)
        print("Driving times saved to: " + str(output_path))

        infrastructure_output_path = os.path.join(
            experiments_dir,
            f"infrastructure_crossings_scenario_{scenario_id}_replicate_{replicate}.csv"
        )
        sim_model.export_infrastructure_crossing_summary(output_path=infrastructure_output_path)
        print("Infrastructure crossings saved to: " + str(infrastructure_output_path))
