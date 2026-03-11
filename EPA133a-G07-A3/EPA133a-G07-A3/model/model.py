import json
from collections import defaultdict
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
    Set the HTML continuous space canvas bounding box (for visualization)
    give the min and max latitudes and Longitudes in Decimal Degrees (DD)

    Add white borders at edges (default 2%) of the bounding box
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
        generate the simulation model according to the csv file component information

        Warning: the labels are the same as the csv column labels
        """

        df = pd.read_csv(self.file_name)

        roads = df['road'].dropna().unique().tolist()

        df_objects_all = []
        for road in roads:
            # Select all the objects on a particular road in the original order as in the cvs
            df_objects_on_road = df[df['road'] == road].copy()

            if not df_objects_on_road.empty:
                df_objects_on_road.reset_index(drop=True, inplace=True)
                df_objects_all.append(df_objects_on_road)

                self.add_road_to_network(df_objects_on_road)

        # put back to df with selected roads so that min and max and be easily calculated
        df = pd.concat(df_objects_all)
        y_min, y_max, x_min, x_max = set_lat_lon_bound(
            df['lat'].min(),
            df['lat'].max(),
            df['lon'].min(),
            df['lon'].max(),
            0.05
        )

        # ContinuousSpace from the Mesa package;
        # not to be confused with the SimpleContinuousModule visualization
        self.space = ContinuousSpace(x_max, y_max, True, x_min, y_min)

        for df in df_objects_all:
            for _, row in df.iterrows():  # index, row in ...

                # create agents according to model_type
                model_type = row['model_type'].strip()
                agent = None

                name = row['name']
                if pd.isna(name):
                    name = ""
                else:
                    name = name.strip()

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
                    if not row['id'] in self.schedule._agents:
                        agent = Intersection(row['id'], self, row['length'], name, row['road'])

                if agent:
                    self.schedule.add(agent)
                    y = row['lat']
                    x = row['lon']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)

    def add_road_to_network(self, df_objects_on_road):
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

        start_id = path_ids[0]
        end_id = path_ids[-1]
        self.path_ids_dict.setdefault((start_id, end_id), path_ids)
        self.path_ids_dict.setdefault((end_id, start_id), list(reversed(path_ids)))

    def load_path_cache(self):
        if not self.path_cache_file.exists():
            return

        with self.path_cache_file.open('r', encoding='utf-8') as cache_file:
            cache_entries = json.load(cache_file)

        for entry in cache_entries:
            key = (entry['origin'], entry['destination'])
            self.path_ids_dict[key] = entry['path_ids']

    def save_path_cache(self):
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
        pick up a random route given an origin
        """
        while True:
            # different source and sink
            sink = self.random.choice(self.sinks)
            if sink != source:
                break
        return self.get_shortest_path(source, sink)

    def get_route(self, source):
        return self.get_random_route(source)

    def get_shortest_path(self, origin, destination):
        cache_key = (origin, destination)
        if cache_key not in self.path_ids_dict:
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
