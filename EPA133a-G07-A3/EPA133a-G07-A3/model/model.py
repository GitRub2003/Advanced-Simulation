import json
from pathlib import Path

import networkx as nx
import pandas as pd
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import BaseScheduler

from components import Source, Sink, SourceSink, Bridge, Link, Intersection


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

    path_ids_dict: defaultdict
        Key: (origin, destination)
        Value: the shortest path (Infra component IDs) from an origin to a destination

        Cached shortest paths between sourcesinks in the network

    sources: list
        all sources in the network

    sinks: list
        all sinks in the network

    """

    step_time = 1

    file_name = Path(__file__).resolve().parents[1] / 'data' / 'network_model.csv'
    path_cache_file = Path(__file__).resolve().parents[1] / 'data' / 'path_ids_dict.json'

    def __init__(self, seed=None, x_max=500, y_max=500, x_min=0, y_min=0):
        """
        Initialize the simulation state and build the network from disk.
        """
        super().__init__(seed=seed)

        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = {}
        self.space = None
        self.sources = []
        self.sinks = []
        self.network = nx.Graph()

        self.load_path_cache()
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
                elif model_type == 'sourcesink':
                    agent = SourceSink(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                    self.sinks.append(agent.unique_id)
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

    def add_road_to_network(self, df_objects_on_road):
        """
        Add one road segment to the network graph and seed obvious path caches.
        """
        path_ids = df_objects_on_road['id'].tolist()

        for _, row in df_objects_on_road.iterrows():
            self.network.add_node(
                row['id'],
                road=row['road'],
                model_type=row['model_type'].strip(),
                length=float(row['length']),
                name="" if pd.isna(row['name']) else str(row['name']).strip()
            )

        for current_row, next_row in zip(
            df_objects_on_road.iloc[:-1].itertuples(index=False),
            df_objects_on_road.iloc[1:].itertuples(index=False)
        ):
            edge_weight = max(float(next_row.length), 1.0)
            self.network.add_edge(current_row.id, next_row.id, weight=edge_weight)

        # A single CSV road is already an ordered path, so cache both travel directions.
        start_id = path_ids[0]
        end_id = path_ids[-1]
        self.path_ids_dict.setdefault((start_id, end_id), path_ids)
        self.path_ids_dict.setdefault((end_id, start_id), list(reversed(path_ids)))

    def load_path_cache(self):
        """
        Load previously computed shortest paths from the JSON cache if available.
        """
        if not self.path_cache_file.exists():
            return

        with self.path_cache_file.open('r', encoding='utf-8') as cache_file:
            cache_entries = json.load(cache_file)

        for entry in cache_entries:
            key = (entry['origin'], entry['destination'])
            self.path_ids_dict[key] = entry['path_ids']

    def save_path_cache(self):
        """
        Persist the path cache so future runs can reuse computed shortest paths.
        """
        cache_entries = [
            {
                'origin': origin,
                'destination': destination,
                'path_ids': path_ids
            }
            for (origin, destination), path_ids in sorted(self.path_ids_dict.items())
        ]

        with self.path_cache_file.open('w', encoding='utf-8') as cache_file:
            json.dump(cache_entries, cache_file, indent=2)

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
        if cache_key not in self.path_ids_dict:
            # NetworkX handles multi-road routing once the graph has been assembled.
            path_ids = nx.shortest_path(self.network, origin, destination, weight='weight')
            self.path_ids_dict[cache_key] = path_ids
            self.save_path_cache()
        return self.path_ids_dict[cache_key]

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()

# EOF -----------------------------------------------------------
