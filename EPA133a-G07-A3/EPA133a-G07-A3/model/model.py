import os
from pathlib import Path

import networkx as nx
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

    file_name = Path(__file__).resolve().parents[1] / 'data' / 'network_model.csv'

    def __init__(self, scenario_id=0, scenario_probs=None, seed=None, x_max=500, y_max=500, x_min=0, y_min=0):
        """
        Initialize the simulation state and build the network from disk.

        Breakdown probabilities are supplied by the caller so the model stays
        independent from a specific batch-run script.
        """
        super().__init__(seed=seed)
        self.scenario_id = scenario_id
        self.scenario_probs = scenario_probs or {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}

        self.schedule = BaseScheduler(self)
        self.running = True
        self.route_cache = {}
        self.space = None
        self.sources = []
        self.sinks = []
        self.network = nx.DiGraph()
        self.completed_vehicle_times = []

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

        # Recombine the selected road data so the global coordinate bounds can be computed.
        df = pd.concat(df_objects_all)
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
            for _, row in df.iterrows():

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
                elif model_type == 'link':
                    agent = Link(row['id'], self, row['length'], name, row['road'])
                elif model_type == 'intersection':
                    # Intersections can appear in multiple road definitions, so only add them once.
                    if row['id'] not in self.schedule._agents:
                        agent = Intersection(row['id'], self, row['length'], name, row['road'])

                if agent:
                    self.schedule.add(agent)
                    y = row['lat']
                    x = row['lon']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)

        self.update_network_travel_times()

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
        sink = self.random.choice(possible_sinks)
        return self.get_shortest_path(source, sink)

    def get_route(self, source):
        """
        Return a route for a new vehicle leaving the given source.
        """
        return self.get_random_route(source)

    def get_shortest_path(self, origin, destination):
        """
        Return the cached or newly computed shortest path between two nodes.
        """
        cache_key = (origin, destination)
        if cache_key not in self.route_cache:
            # NetworkX handles multi-road routing once the graph has been assembled.
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

        origin_id = ''
        destination_id = ''
        if vehicle.path_ids:
            origin_id = vehicle.path_ids[0]
            destination_id = vehicle.path_ids[-1]

        self.completed_vehicle_times.append(
            {
                'truck_id': vehicle.unique_id,
                'origin_id': origin_id,
                'destination_id': destination_id,
                'generated_at_step': vehicle.generated_at_step,
                'removed_at_step': vehicle.removed_at_step,
            }
        )

    def calculate_total_driving_times(self):
        """
        Return a DataFrame with total travel time for each completed truck.
        """
        df = pd.DataFrame(self.completed_vehicle_times)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    'truck_id',
                    'origin_id',
                    'destination_id',
                    'generated_at_step',
                    'removed_at_step',
                    'total_driving_time',
                ]
            )

        df['total_driving_time'] = df['removed_at_step'] - df['generated_at_step']
        return df

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

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()

# EOF -----------------------------------------------------------
