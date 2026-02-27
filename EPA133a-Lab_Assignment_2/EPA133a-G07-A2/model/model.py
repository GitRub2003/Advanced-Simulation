from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Source, Sink, SourceSink, Bridge, Link
import pandas as pd
import os
from collections import defaultdict
from scenario import SCENARIOS


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

        Since there is only one road in the Demo, the paths are added with the road info;
        when there is a more complex network layout, the paths need to be managed differently

    sources: list
        all sources in the network

    sinks: list
        all sinks in the network

    """

    step_time = 1

    def __init__(self, scenario_id=0, seed=None, x_max=500, y_max=500, x_min=0, y_min=0):
        super().__init__()
        self.scenario_id = scenario_id
        self.scenario_probs = SCENARIOS[scenario_id]

        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = defaultdict(lambda: pd.Series())
        self.space = None
        self.sources = []
        self.sinks = []
        self.completed_vehicle_times = []

        self.generate_model()

    def generate_model(self):
        """
        generate the simulation model according to the csv file component information

        Warning: the labels are the same as the csv column labels
        """

        df = pd.read_csv('../data/n1_model.csv')
        # Normalize column names for more flexible input data formats
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Ensure required columns exist or derive them from available data.
        if 'id' not in df.columns:
            df = df.reset_index(drop=True)
            df['id'] = df.index.astype(int)

        if 'model_type' not in df.columns:
            df['model_type'] = 'link'
            if 'type' in df.columns:
                bridge_mask = df['type'].astype(str).str.contains('bridge|culvert', case=False, na=False)
                df.loc[bridge_mask, 'model_type'] = 'bridge'

            if 'road' in df.columns:
                for _, group in df.groupby('road', sort=False):
                    if group.empty:
                        continue
                    if 'chainage' in df.columns:
                        ordered = group.sort_values(by=['chainage']).index
                    else:
                        ordered = group.sort_values(by=['id']).index
                    df.loc[ordered[0], 'model_type'] = 'source'
                    df.loc[ordered[-1], 'model_type'] = 'sink'
            else:
                df.loc[df['id'].idxmin(), 'model_type'] = 'source'
                df.loc[df['id'].idxmax(), 'model_type'] = 'sink'

        if 'length' not in df.columns:
            df['length'] = 0.0
            if 'chainage' in df.columns and 'road' in df.columns:
                for _, group in df.groupby('road', sort=False):
                    if group.empty:
                        continue
                    ordered = group.sort_values(by=['chainage'])
                    lengths = ordered['chainage'].diff().fillna(0).abs() * 1000
                    df.loc[ordered.index, 'length'] = lengths
            if 'gap' in df.columns:
                gap = pd.to_numeric(df['gap'], errors='coerce')
                df.loc[gap.notna(), 'length'] = gap[gap.notna()]

        # a list of names of roads to be generated
        roads = ['N1']

        # roads = [
        #     'N1', 'N2', 'N3', 'N4',
        #     'N5', 'N6', 'N7', 'N8'
        # ]

        df_objects_all = []
        for road in roads:

            # be careful with the sorting
            # better remove sorting by id
            # Select all the objects on a particular road
            sort_col = 'id'
            if 'chainage' in df.columns and df['chainage'].notna().any():
                sort_col = 'chainage'
            df_objects_on_road = df[df['road'] == road].sort_values(by=[sort_col])

            if not df_objects_on_road.empty:
                df_objects_all.append(df_objects_on_road)

                # the object IDs on a given road
                path_ids = df_objects_on_road['id']
                # add the path to the path_ids_dict
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids
                # put the path in reversed order and reindex
                path_ids = path_ids[::-1]
                path_ids.reset_index(inplace=True, drop=True)
                # add the path to the path_ids_dict so that the vehicles can drive backwards too
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids

        # put back to df with selected roads so that min and max and be easily calculated
        if not df_objects_all:
            raise ValueError("No objects found for the requested roads; check the input CSV and road names.")
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
            for _, row in df.iterrows():    # index, row in ...

                # create agents according to model_type
                model_type = row['model_type']
                agent = None

                if model_type == 'source':
                    agent = Source(row['id'], self, row['length'], row['name'], row['road'])
                    self.sources.append(agent.unique_id)
                elif model_type == 'sink':
                    agent = Sink(row['id'], self, row['length'], row['name'], row['road'])
                    self.sinks.append(agent.unique_id)
                    self._attach_sink_remove_hook(agent)
                elif model_type == 'sourcesink':
                    agent = SourceSink(row['id'], self, row['length'], row['name'], row['road'])
                    self.sources.append(agent.unique_id)
                    self.sinks.append(agent.unique_id)
                    self._attach_sink_remove_hook(agent)
                elif model_type == 'bridge':
                    agent = Bridge(row['id'], self, row['length'], row['name'], row['road'], row['condition'])
                elif model_type == 'link':
                    agent = Link(row['id'], self, row['length'], row['name'], row['road'])

                if agent:
                    self.schedule.add(agent)
                    y = row['lat']
                    x = row['lon']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)

    def get_random_route(self, source):
        """
        pick up a random route given an origin
        """
        while True:
            # different source and sink
            sink = self.random.choice(self.sinks)
            if sink is not source:
                break
        return self.path_ids_dict[source, sink]

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()

    def _attach_sink_remove_hook(self, sink_agent):
        """
        Wrap sink.remove so completed vehicle timings are recorded in the model.
        """
        original_remove = sink_agent.remove

        def remove_with_record(vehicle):
            self.record_completed_vehicle(vehicle)
            return original_remove(vehicle)

        sink_agent.remove = remove_with_record

    def record_completed_vehicle(self, vehicle):
        """
        Store generated/removed steps for a finished vehicle.
        """
        if vehicle.generated_at_step is None or vehicle.removed_at_step is None:
            return
        self.completed_vehicle_times.append(
            {
                "truck_id": vehicle.unique_id,
                "generated_at_step": vehicle.generated_at_step,
                "removed_at_step": vehicle.removed_at_step,
            }
        )

    def calculate_total_driving_times(self):
        """
        Return a DataFrame with total driving time for each completed truck.
        """
        df = pd.DataFrame(self.completed_vehicle_times)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "truck_id",
                    "generated_at_step",
                    "removed_at_step",
                    "total_driving_time",
                ]
            )
        df["total_driving_time"] = df["removed_at_step"] - df["generated_at_step"]
        return df

    def export_total_driving_times(self, output_path=None):
        """
        Export truck driving times to a CSV file in the data folder.
        """
        if output_path is None:
            output_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "data", "truck_driving_times.csv")
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.calculate_total_driving_times().to_csv(output_path, index=False)
        return output_path

    def calculate_bridge_total_wait_times(self):
        """
        Return a DataFrame with total wait time per bridge for this scenario.
        """
        rows = []
        for agent in self.schedule._agents.values():  # Access to protected member _agents
            if isinstance(agent, Bridge):
                rows.append(
                    {
                        "bridge_id": agent.unique_id,
                        "bridge_name": agent.name,
                        "road_name": agent.road_name,
                        "condition": agent.condition,
                        "total_wait_time": float(getattr(agent, "total_wait_time", 0.0)),
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "bridge_id",
                    "bridge_name",
                    "road_name",
                    "condition",
                    "total_wait_time",
                ]
            )
        return pd.DataFrame(rows).sort_values(by=["bridge_id"]).reset_index(drop=True)

    def export_bridge_total_wait_times(self, output_path=None):
        """
        Export total wait time per bridge to CSV.
        """
        if output_path is None:
            output_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "data", "bridge_total_wait_times.csv")
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.calculate_bridge_total_wait_times().to_csv(output_path, index=False)
        return output_path

# EOF -----------------------------------------------------------
