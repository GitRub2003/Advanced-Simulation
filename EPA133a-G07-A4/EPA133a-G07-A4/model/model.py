import os
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import BaseScheduler
from components import Source, Sink, SourceSink, Bridge, Link, Intersection, Vehicle


# ---------------------------------------------------------------
def set_lat_lon_bound(lat_min, lat_max, lon_min, lon_max, edge_ratio=0.02):
    """
    Return a padded bounding box for Mesa's continuous-space visualization.

    The input coordinates come from the network data in decimal degrees.
    A small border is added so agents are not drawn directly on the edge of
    the canvas.
    """

    lat_edge = (lat_max - lat_min) * edge_ratio
    lon_edge = (lon_max - lon_min) * edge_ratio

    x_max = lon_max + lon_edge
    y_max = lat_min - lat_edge
    x_min = lon_min - lon_edge
    y_min = lat_max + lat_edge
    return y_min, y_max, x_min, x_max


# ---------------------------------------------------------------
class BangladeshModel(Model):
    """
    The main (top-level) simulation model

    One tick represents one minute; this can be changed
    but the distance calculation need to be adapted accordingly

    Class Attributes:
    -----------------
    step_time: int
        step_time = 1 # 1 step is 1 min

    route_cache: dict
        Key: (origin, destination)
        Value: the shortest path (Infra component IDs) from an origin to a destination

        Cached shortest paths for the current run only. Routes are not reused
        across runs because broken bridges change edge weights.

    sources: list
        all sources in the network

    sinks: list
        all sinks in the network

    """

    step_time = 1
    steps_per_day = 24 * 60

    file_name = Path(__file__).resolve().parents[1] / 'data' / 'network_model.csv'
    od_matrix_file = Path(__file__).resolve().parents[1] / 'data' / 'od_matrix.csv'
    od_calibration_file = Path(__file__).resolve().parents[1] / 'data' / 'od_calibration_summary.csv'
    od_cache_signature = None
    od_cache = None

    def __init__(self, scenario_id=0, scenario_probs=None, seed=None, x_max=500, y_max=500, x_min=0, y_min=0, demand_scale=1.0):
        """
        Initialize the simulation state and build the network from disk.

        Breakdown probabilities are supplied by the caller so the model stays
        independent from a specific batch-run script.
        """
        super().__init__(seed=seed)
        self.scenario_id = scenario_id
        self.scenario_probs = scenario_probs or {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        self.demand_scale = demand_scale

        self.schedule = BaseScheduler(self)
        self.running = True
        self.route_cache = {}
        self.space = None
        self.sources = []
        self.sinks = []
        self.network = nx.DiGraph()
        self.completed_vehicle_times = []
        self.infrastructure_crossing_stats = {}
        self.endpoint_labels = {}
        self.infrastructure_labels = {}
        self.source_daily_demand = {}
        self.source_destination_weights = {}
        self.od_matrix = pd.DataFrame()
        self.np_random = np.random.default_rng(seed)

        self.generate_model()

    def generate_model(self):
        """
        Build the graph, agents, and continuous space from the CSV input file.

        The column names used here match the labels in the source CSV file.
        """

        df = pd.read_csv(self.file_name)

        roads = df['road'].dropna().unique().tolist()

        df_objects_all = []
        for road in roads:
            # Preserve the CSV ordering so consecutive rows can be linked into a road segment.
            df_objects_on_road = df[df['road'] == road].copy()

            if not df_objects_on_road.empty:
                df_objects_on_road.reset_index(drop=True, inplace=True)
                df_objects_all.append(df_objects_on_road)

                self.add_road_to_network(df_objects_on_road)

        full_network_df = pd.concat(df_objects_all, ignore_index=True)
        self._load_or_build_od_demand(full_network_df)

        # Recombine the selected road data so the global coordinate bounds can be computed.
        df = full_network_df
        y_min, y_max, x_min, x_max = set_lat_lon_bound(
            df['lat'].min(),
            df['lat'].max(),
            df['lon'].min(),
            df['lon'].max(),
            0.05
        )

        # Mesa uses the same coordinates for placement and map-style visualization.
        self.space = ContinuousSpace(x_max, y_max, True, x_min, y_min)

        for df in df_objects_all:
            endpoint_rows = df[
                df['model_type'].astype(str).str.strip().str.lower().isin({'source', 'sink', 'sourcesink'})
            ].index.tolist()
            first_endpoint_index = endpoint_rows[0] if endpoint_rows else None
            last_endpoint_index = endpoint_rows[-1] if endpoint_rows else None
            for row_index, row in df.iterrows():

                # Create the proper agent class for each infrastructure record.
                model_type = row['model_type'].strip()
                agent = None

                name = "" if pd.isna(row['name']) else row['name'].strip()
                if model_type == 'source':
                    agent = Source(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                elif model_type == 'sink':
                    agent = Sink(row['id'], self, row['length'], name, row['road'])
                    self.sinks.append(agent.unique_id)
                    self._attach_sink_remove_hook(agent)
                elif model_type == 'sourcesink':
                    agent = SourceSink(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                    self.sinks.append(agent.unique_id)
                    self._attach_sink_remove_hook(agent)
                elif model_type == 'bridge':
                    agent = Bridge(row['id'], self, row['length'], name, row['road'], row['condition'])
                    infrastructure_label = self._build_infrastructure_label(
                        name=row['name'],
                        lrp=row.get('lrp'),
                        road_name=row['road'],
                        model_type='bridge',
                        unique_id=row['id'],
                    )
                    self.infrastructure_labels[agent.unique_id] = infrastructure_label
                elif model_type == 'link':
                    agent = Link(row['id'], self, row['length'], name, row['road'])
                    infrastructure_label = self._build_infrastructure_label(
                        name=row['name'],
                        lrp=row.get('lrp'),
                        road_name=row['road'],
                        model_type='link',
                        unique_id=row['id'],
                    )
                    self.infrastructure_labels[agent.unique_id] = infrastructure_label
                elif model_type == 'intersection':
                    # Intersections can appear in multiple road definitions, so only add them once.
                    if row['id'] not in self.schedule._agents:
                        agent = Intersection(row['id'], self, row['length'], name, row['road'])

                if agent:
                    self._configure_agent_demand(agent)
                    self.schedule.add(agent)
                    y = row['lat']
                    x = row['lon']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)
                    self._register_endpoint_label(agent.unique_id, row, row_index, first_endpoint_index, last_endpoint_index)

        self.update_network_travel_times()

    @classmethod
    def _network_signature(cls) -> tuple[int, int]:
        stat = cls.file_name.stat()
        return stat.st_mtime_ns, stat.st_size

    @staticmethod
    def _build_endpoint_label_map(df: pd.DataFrame) -> dict[int, str]:
        """
        Build stable human-readable labels for road endpoint ids.
        """
        label_map: dict[int, str] = {}
        for road_name, road_df in df.groupby('road', sort=False):
            road_df = road_df.reset_index(drop=True)
            endpoint_rows = road_df[
                road_df['model_type'].astype(str).str.strip().str.lower().isin({'source', 'sink', 'sourcesink'})
            ].index.tolist()
            first_endpoint_index = endpoint_rows[0] if endpoint_rows else None
            last_endpoint_index = endpoint_rows[-1] if endpoint_rows else None

            for row_index, row in road_df.iterrows():
                model_type = str(row['model_type']).strip().lower()
                if model_type not in {'source', 'sink', 'sourcesink'}:
                    continue

                node_id = int(row['id'])
                if row_index == first_endpoint_index:
                    label_map[node_id] = f"{road_name} start"
                elif row_index == last_endpoint_index:
                    label_map[node_id] = f"{road_name} end"
                else:
                    label_map[node_id] = str(road_name).strip()

        return label_map

    def _load_or_build_od_demand(self, df: pd.DataFrame) -> None:
        """
        Compute one OD matrix from source-sink truck AADT and reuse it across replications.
        """
        signature = self._network_signature()
        if BangladeshModel.od_cache_signature != signature or BangladeshModel.od_cache is None:
            BangladeshModel.od_cache = self._build_od_demand(df)
            BangladeshModel.od_cache_signature = signature

        cache = BangladeshModel.od_cache
        self.od_matrix = cache['od_matrix']
        self.source_daily_demand = cache['source_daily_demand']
        self.source_destination_weights = cache['source_destination_weights']
        self.endpoint_labels.update(cache['endpoint_labels'])

    def _build_od_demand(self, df: pd.DataFrame) -> dict[str, object]:
        """
        Build and calibrate a gravity-style OD matrix from endpoint truck AADT.

        The available data only provides truck AADT at the source/sink endpoints, so those
        values are used as both origin production and destination attraction weights.
        """
        endpoint_df = df[
            df['model_type'].astype(str).str.strip().str.lower().isin({'source', 'sourcesink'})
        ].copy()
        od_columns = ['origin_id', 'destination_id', 'origin_label', 'destination_label', 'cost_minutes', 'daily_trucks', 'probability']
        endpoint_df['id'] = pd.to_numeric(endpoint_df['id'], errors='coerce').astype('Int64')
        endpoint_df['source_total_trucks'] = pd.to_numeric(endpoint_df.get('source_total_trucks'), errors='coerce').fillna(0.0)
        endpoint_df = endpoint_df.dropna(subset=['id']).copy()
        endpoint_df = endpoint_df[endpoint_df['source_total_trucks'] > 0].copy()

        endpoint_labels = self._build_endpoint_label_map(df)
        source_daily_demand = {
            int(row['id']): float(row['source_total_trucks']) / 2.0
            for _, row in endpoint_df.iterrows()
        }
        source_destination_weights: dict[int, dict[int, float]] = {}
        od_rows: list[dict[str, float | int | str]] = []

        if endpoint_df.empty:
            empty_od = pd.DataFrame(columns=od_columns)
            empty_od.to_csv(self.od_matrix_file, index=False)
            return {
                'od_matrix': empty_od,
                'source_daily_demand': source_daily_demand,
                'source_destination_weights': source_destination_weights,
                'endpoint_labels': endpoint_labels,
            }

        baseline_graph = self.network.copy()
        for start_id, end_id in baseline_graph.edges():
            destination_length = float(baseline_graph.nodes[end_id].get('length', 0.0))
            baseline_graph[start_id][end_id]['weight'] = destination_length / Vehicle.speed

        source_ids = [int(node_id) for node_id in endpoint_df['id'].tolist()]
        target_demand = {
            int(row['id']): float(row['source_total_trucks']) / 2.0
            for _, row in endpoint_df.iterrows()
        }

        pair_costs: dict[tuple[int, int], float] = {}
        all_costs: list[float] = []
        for origin_id in source_ids:
            for destination_id in source_ids:
                if destination_id == origin_id:
                    continue
                try:
                    cost = float(nx.shortest_path_length(baseline_graph, origin_id, destination_id, weight='weight'))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                pair_costs[(origin_id, destination_id)] = cost
                if cost > 0:
                    all_costs.append(cost)

        if all_costs:
            median_cost = float(np.median(all_costs))
            beta = math.log(2.0) / median_cost if median_cost > 0 else 0.0
        else:
            beta = 0.0

        prior_matrix = self._build_gravity_prior_matrix(
            source_ids=source_ids,
            target_demand=target_demand,
            pair_costs=pair_costs,
            beta=beta,
        )
        calibrated_matrix, calibration_summary = self._calibrate_od_matrix(
            matrix=prior_matrix,
            source_ids=source_ids,
            target_demand=target_demand,
            endpoint_labels=endpoint_labels,
        )
        calibration_summary.to_csv(self.od_calibration_file, index=False)
        print(f"Wrote OD calibration summary to: {self.od_calibration_file.resolve()}")

        for origin_idx, origin_id in enumerate(source_ids):
            origin_total = float(calibrated_matrix[origin_idx, :].sum())
            if origin_total <= 0:
                source_destination_weights[origin_id] = {}
                source_daily_demand[origin_id] = 0.0
                continue

            source_daily_demand[origin_id] = origin_total
            normalized_weights: dict[int, float] = {}
            for destination_idx, destination_id in enumerate(source_ids):
                if destination_id == origin_id:
                    continue
                daily_trucks = float(calibrated_matrix[origin_idx, destination_idx])
                if daily_trucks <= 0:
                    continue
                probability = daily_trucks / origin_total
                normalized_weights[destination_id] = probability
                od_rows.append(
                    {
                        'origin_id': origin_id,
                        'destination_id': destination_id,
                        'origin_label': endpoint_labels.get(origin_id, str(origin_id)),
                        'destination_label': endpoint_labels.get(destination_id, str(destination_id)),
                        'cost_minutes': pair_costs[(origin_id, destination_id)],
                        'daily_trucks': daily_trucks,
                        'probability': probability,
                    }
                )

            source_destination_weights[origin_id] = normalized_weights

        od_matrix = pd.DataFrame(od_rows, columns=od_columns)
        if not od_matrix.empty:
            od_matrix = od_matrix.sort_values(['origin_id', 'destination_id']).reset_index(drop=True)
        od_matrix.to_csv(self.od_matrix_file, index=False)
        print(f"Wrote OD matrix to: {self.od_matrix_file.resolve()}")

        return {
            'od_matrix': od_matrix,
            'source_daily_demand': source_daily_demand,
            'source_destination_weights': source_destination_weights,
            'endpoint_labels': endpoint_labels,
        }

    def _build_gravity_prior_matrix(
        self,
        source_ids: list[int],
        target_demand: dict[int, float],
        pair_costs: dict[tuple[int, int], float],
        beta: float,
    ) -> np.ndarray:
        """
        Build the unconstrained gravity prior before calibration.
        """
        size = len(source_ids)
        matrix = np.zeros((size, size), dtype=float)

        for origin_idx, origin_id in enumerate(source_ids):
            for destination_idx, destination_id in enumerate(source_ids):
                if destination_id == origin_id:
                    continue
                cost = pair_costs.get((origin_id, destination_id))
                if cost is None:
                    continue
                deterrence = math.exp(-beta * cost) if beta > 0 else 1.0
                matrix[origin_idx, destination_idx] = (
                    float(target_demand[origin_id]) *
                    float(target_demand[destination_id]) *
                    deterrence
                )

        return matrix

    def _simulate_intact_daily_assignment(
        self,
        matrix: np.ndarray,
        source_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate one intact-network daily assignment from the OD matrix.

        With fixed OD trips on an intact shortest-path network, generated counts are the
        row sums and arrived counts are the column sums.
        """
        generated = matrix.sum(axis=1)
        arrived = matrix.sum(axis=0)
        return generated, arrived

    def _calibrate_od_matrix(
        self,
        matrix: np.ndarray,
        source_ids: list[int],
        target_demand: dict[int, float],
        endpoint_labels: dict[int, str],
        max_iterations: int = 30,
        tolerance: float = 0.05,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Iteratively rebalance the OD matrix until intact simulated endpoint flows are close
        to the target truck AADT values.
        """
        calibrated = matrix.copy()
        targets = np.array([float(target_demand[source_id]) for source_id in source_ids], dtype=float)
        epsilon = 1e-9

        for _ in range(max_iterations):
            generated, arrived = self._simulate_intact_daily_assignment(calibrated, source_ids)

            row_scale = np.divide(
                targets,
                np.maximum(generated, epsilon),
                out=np.ones_like(targets),
                where=targets > 0,
            )
            calibrated = (calibrated.T * row_scale).T

            generated, arrived = self._simulate_intact_daily_assignment(calibrated, source_ids)
            col_scale = np.divide(
                targets,
                np.maximum(arrived, epsilon),
                out=np.ones_like(targets),
                where=targets > 0,
            )
            calibrated = calibrated * col_scale

            generated, arrived = self._simulate_intact_daily_assignment(calibrated, source_ids)
            gen_rel_error = np.divide(
                np.abs(generated - targets),
                np.maximum(targets, 1.0),
            )
            arr_rel_error = np.divide(
                np.abs(arrived - targets),
                np.maximum(targets, 1.0),
            )
            max_error = float(max(np.max(gen_rel_error), np.max(arr_rel_error))) if len(gen_rel_error) > 0 else 0.0
            if max_error <= tolerance:
                break

        generated, arrived = self._simulate_intact_daily_assignment(calibrated, source_ids)
        summary_rows = []
        for idx, source_id in enumerate(source_ids):
            target_value = float(targets[idx])
            combined_target = target_value * 2.0
            sim_generated = float(generated[idx])
            sim_arrived = float(arrived[idx])
            sim_total = sim_generated + sim_arrived
            denom = max(target_value, 1.0)
            summary_rows.append(
                {
                    'endpoint_id': source_id,
                    'endpoint_label': endpoint_labels.get(source_id, str(source_id)),
                    'combined_aadt_target': combined_target,
                    'per_direction_target': target_value,
                    'sim_generated_trucks_per_day': sim_generated,
                    'sim_arrived_trucks_per_day': sim_arrived,
                    'sim_total_endpoint_activity': sim_total,
                    'generated_rel_error': abs(sim_generated - target_value) / denom,
                    'arrived_rel_error': abs(sim_arrived - target_value) / denom,
                    'total_activity_rel_error': abs(sim_total - combined_target) / max(combined_target, 1.0),
                }
            )

        summary = pd.DataFrame(summary_rows).sort_values('endpoint_id').reset_index(drop=True)
        return calibrated, summary

    def _configure_agent_demand(self, agent) -> None:
        """
        Attach the cached demand data to source-capable agents.
        """
        if isinstance(agent, Source):
            agent.daily_truck_demand = float(self.source_daily_demand.get(agent.unique_id, 0.0))
            agent.destination_weights = dict(self.source_destination_weights.get(agent.unique_id, {}))

    def _register_endpoint_label(self, unique_id, row, row_index, first_endpoint_index, last_endpoint_index):
        """
        Store a readable label for each road endpoint source/sink.
        """
        model_type = str(row['model_type']).strip().lower()
        if model_type not in {'source', 'sink', 'sourcesink'}:
            return

        road_name = str(row['road']).strip()
        if row_index == first_endpoint_index:
            self.endpoint_labels[unique_id] = f"{road_name} start"
        elif row_index == last_endpoint_index:
            self.endpoint_labels[unique_id] = f"{road_name} end"
        else:
            self.endpoint_labels[unique_id] = road_name

    @staticmethod
    def _build_infrastructure_label(name, lrp, road_name, model_type, unique_id):
        """
        Build a readable label for links/bridges using name + LRP when available.
        """
        name_str = "" if pd.isna(name) else str(name).strip()
        lrp_str = "" if pd.isna(lrp) else str(lrp).strip()
        road_str = "" if pd.isna(road_name) else str(road_name).strip()

        if name_str and lrp_str:
            return f"{name_str} ({lrp_str})"
        if name_str:
            return name_str
        if lrp_str:
            return f"{road_str} {model_type} ({lrp_str})"
        return f"{road_str} {model_type} {unique_id}"

    def add_road_to_network(self, df_objects_on_road):
        """
        Add one road segment to the directed network graph.

        Each infrastructure component is stored as a node. Directed edges are
        added in both directions so route cost can depend on the destination
        node being entered.
        """
        for _, row in df_objects_on_road.iterrows():
            self.network.add_node(
                row['id'],
                road=row['road'],
                model_type=row['model_type'].strip(),
                length=float(row['length']),
                name="" if pd.isna(row['name']) else str(row['name']).strip(),
                condition="" if pd.isna(row.get('condition')) else str(row['condition']).strip()
            )

        for current_row, next_row in zip(
            df_objects_on_road.iloc[:-1].itertuples(index=False),
            df_objects_on_road.iloc[1:].itertuples(index=False)
        ):
            # The traversal cost is attached to the node being entered, so the
            # reverse direction gets its own weight.
            self.network.add_edge(current_row.id, next_row.id, weight=0.0)
            self.network.add_edge(next_row.id, current_row.id, weight=0.0)

    def get_infrastructure_travel_time(self, infra):
        """
        Return the expected time needed to traverse one infrastructure object.

        Base travel time is the infrastructure length divided by truck speed.
        Broken bridges add their mean delay so shortest-path routing reflects
        expected disruption during the current run.
        """
        travel_time = float(infra.length) / Vehicle.speed
        if isinstance(infra, Bridge) and infra.is_broken:
            travel_time += infra.get_average_delay_time()
        return travel_time

    def update_network_travel_times(self):
        """
        Recompute all directed edge weights for the current run.

        Bridge breakdown status is sampled during model construction, so the
        route graph must be updated afterwards to include the extra expected
        delay on broken bridges.
        """
        for start_id, end_id in self.network.edges():
            destination_infra = self.schedule._agents[end_id]  # Access to protected member _agents
            self.network[start_id][end_id]['weight'] = self.get_infrastructure_travel_time(destination_infra)

        # Routes are run-specific because broken bridges change weights.
        self.route_cache.clear()

    def get_random_route(self, source):
        """
        Return a shortest path from the source to a randomly chosen sink.
        """
        possible_sinks = [sink for sink in self.sinks if sink != source]
        if not possible_sinks:
            return None
        sink = self.random.choice(possible_sinks)
        return self.get_shortest_path(source, sink)

    def choose_destination(self, source):
        """
        Choose a destination sink for one generated truck using the cached OD weights.
        """
        destination_weights = self.source_destination_weights.get(source, {})
        if destination_weights:
            destinations = list(destination_weights.keys())
            probabilities = list(destination_weights.values())
            return self.random.choices(destinations, weights=probabilities, k=1)[0]

        possible_sinks = [sink for sink in self.sinks if sink != source]
        if not possible_sinks:
            return None
        return self.random.choice(possible_sinks)

    def get_route(self, source):
        """
        Return a route for a new vehicle leaving the given source.
        """
        destination = self.choose_destination(source)
        if destination is None:
            return None
        return self.get_shortest_path(source, destination)

    def get_shortest_path(self, origin, destination):
        """
        Return the cached or newly computed shortest path between two nodes.
        """
        cache_key = (origin, destination)
        if cache_key not in self.route_cache:
            # Shortest paths are computed on the full network; vehicles may
            # traverse source/sink nodes and only finish at their destination.
            path_ids = nx.shortest_path(self.network, origin, destination, weight='weight')
            self.route_cache[cache_key] = path_ids
        return self.route_cache[cache_key]

    def _attach_sink_remove_hook(self, sink_agent):
        """
        Wrap a sink so completed vehicle travel times are recorded on removal.
        """
        original_remove = sink_agent.remove

        def remove_with_record(vehicle):
            self.record_completed_vehicle(vehicle)
            return original_remove(vehicle)

        sink_agent.remove = remove_with_record

    def record_completed_vehicle(self, vehicle):
        """
        Store the timestamps of a vehicle that finished its route.
        """
        if vehicle.generated_at_step is None or vehicle.removed_at_step is None:
            return

        origin_id = vehicle.generated_by.unique_id if vehicle.generated_by is not None else ''
        destination_id = vehicle.location.unique_id if isinstance(vehicle.location, Sink) else ''

        self.completed_vehicle_times.append(
            {
                'truck_id': vehicle.unique_id,
                'source_id': self.endpoint_labels.get(origin_id, origin_id),
                'sink_id': self.endpoint_labels.get(destination_id, destination_id),
                'generated_at_step': vehicle.generated_at_step,
                'removed_at_step': vehicle.removed_at_step,
                'infra_crossing_count': vehicle.infra_crossing_count,
            }
        )

    def record_infrastructure_crossing(self, infra, vehicle):
        """
        Update per-link/per-bridge counters in constant time.
        """
        infra_id = infra.unique_id
        if infra_id not in self.infrastructure_crossing_stats:
            self.infrastructure_crossing_stats[infra_id] = {
                'infra_id': infra_id,
                'infra_label': self.infrastructure_labels.get(infra_id, infra_id),
                'infra_type': type(infra).__name__.lower(),
                'road_name': getattr(infra, 'road_name', ''),
                'infra_name': getattr(infra, 'name', ''),
                'crossing_count': 0,
                'truck_ids': set(),
            }

        stats = self.infrastructure_crossing_stats[infra_id]
        stats['crossing_count'] += 1
        stats['truck_ids'].add(vehicle.unique_id)

    def calculate_total_driving_times(self):
        """
        Return a DataFrame with total travel time for each completed truck.
        """
        df = pd.DataFrame(self.completed_vehicle_times)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    'truck_id',
                    'source_id',
                    'sink_id',
                    'generated_at_step',
                    'removed_at_step',
                    'infra_crossing_count',
                    'total_driving_time',
                ]
            )

        df['total_driving_time'] = df['removed_at_step'] - df['generated_at_step']
        return df

    def calculate_infrastructure_crossing_summary(self):
        """
        Return a DataFrame with total crossing counts per link/bridge.
        """
        if not self.infrastructure_crossing_stats:
            return pd.DataFrame(
                columns=[
                    'infra_id',
                    'infra_label',
                    'infra_type',
                    'road_name',
                    'infra_name',
                    'crossing_count',
                    'unique_truck_count',
                ]
            )

        summary_rows = []
        for stats in self.infrastructure_crossing_stats.values():
            summary_rows.append(
                {
                    'infra_id': stats['infra_id'],
                    'infra_label': stats['infra_label'],
                    'infra_type': stats['infra_type'],
                    'road_name': stats['road_name'],
                    'infra_name': stats['infra_name'],
                    'crossing_count': stats['crossing_count'],
                    'unique_truck_count': len(stats['truck_ids']),
                }
            )

        summary = pd.DataFrame(summary_rows).sort_values(
            ['crossing_count', 'infra_id'],
            ascending=[False, True]
        ).reset_index(drop=True)
        return summary

    def export_total_driving_times(self, output_path=None):
        """
        Export total travel time per completed truck to CSV.
        """
        if output_path is None:
            output_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'truck_driving_times.csv')
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.calculate_total_driving_times().to_csv(output_path, index=False)
        return output_path

    def export_infrastructure_crossing_summary(self, output_path=None):
        """
        Export aggregated link/bridge crossing counts to CSV.
        """
        if output_path is None:
            output_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'infrastructure_crossings.csv')
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.calculate_infrastructure_crossing_summary().to_csv(output_path, index=False)
        return output_path

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()

# EOF -----------------------------------------------------------
