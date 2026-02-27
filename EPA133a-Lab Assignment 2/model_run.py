from pathlib import Path
import pandas as pd
from scenario import SCENARIOS
from model import BangladeshModel


RUN_LENGTH = 5 * 24 * 60
REPLICATIONS = 10
BASE_SEED = 1234567
SCENARIO_IDS = [0, 1, 2, 3, 4, 5]


def run_replication(scenario_id, replication_id, seed):
    sim_model = BangladeshModel(scenario_id=scenario_id, seed=seed, verbose=False)
    for _ in range(RUN_LENGTH):
        sim_model.step()

    rows = []
    for trip in sim_model.completed_trips:
        row = dict(trip)
        row['scenario_id'] = scenario_id
        row['replication_id'] = replication_id
        row['seed'] = seed
        rows.append(row)
    return rows


def run_scenario(scenario_id):
    scenario_rows = []
    for replication_id in range(REPLICATIONS):
        seed = BASE_SEED + scenario_id * 1000 + replication_id
        scenario_rows.extend(run_replication(scenario_id, replication_id, seed))

    scenario_df = pd.DataFrame(scenario_rows)
    if not scenario_df.empty:
        scenario_df['avg_driving_time_replication_min'] = scenario_df.groupby('replication_id')['driving_time_min'].transform('mean')
        scenario_df['avg_delay_time_replication_min'] = scenario_df.groupby('replication_id')['delay_time_min'].transform('mean')
    return scenario_df


def main():
    model_dir = Path(__file__).resolve().parent
    experiment_dir = model_dir.parent / 'experiment'
    experiment_dir.mkdir(parents=True, exist_ok=True)

    for scenario_id in SCENARIO_IDS:
        if scenario_id not in SCENARIOS:
            continue
        scenario_df = run_scenario(scenario_id)
        output_path = experiment_dir / f'scenario{scenario_id}.csv'
        scenario_df.to_csv(output_path, index=False)
        print(f"Saved {output_path} ({len(scenario_df)} rows)")


if __name__ == '__main__':
    main()
