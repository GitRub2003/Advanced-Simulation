import os

from model import BangladeshModel

"""
Run batch simulations and export the collected outputs.
"""

# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick 1 minute
run_length = 5 * 24 * 60

base_seed = 1234567
num_replicates = 10

SCENARIOS = {
    0: {'A': 0.00, 'B': 0.00, 'C': 0.00, 'D': 0.00},
    1: {'A': 0.00, 'B': 0.00, 'C': 0.00, 'D': 0.05},
    2: {'A': 0.00, 'B': 0.00, 'C': 0.05, 'D': 0.10},
    3: {'A': 0.00, 'B': 0.05, 'C': 0.10, 'D': 0.20},
    4: {'A': 0.05, 'B': 0.10, 'C': 0.20, 'D': 0.40},
   }

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

        output_path = os.path.join(
            experiments_dir,
            f"truck_driving_times_scenario_{scenario_id}_replicate_{replicate}.csv"
        )
        sim_model.export_total_driving_times(output_path=output_path)
        print("Driving times saved to: " + str(output_path))

        bridge_wait_output_path = os.path.join(
            experiments_dir,
            f"bridge_total_wait_times_scenario_{scenario_id}_replicate_{replicate}.csv"
        )
        sim_model.export_bridge_total_wait_times(output_path=bridge_wait_output_path)
        print("Bridge wait times saved to: " + str(bridge_wait_output_path))
