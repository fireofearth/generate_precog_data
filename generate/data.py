"""Functions and classes for data collection.
"""

import collections
import weakref
import logging
import attrdict
import networkx as nx
import numpy as np
import tensorflow as tf
import carla

import generate.overhead as generate_overhead
import generate.observation as generate_observation
import generate.util as util
import precog.utils.class_util as classu
import precog.utils.tensor_util as tensoru

DEFAULT_PHI_ATTRIBUTES = attrdict.AttrDict({
            "T": 20, "T_past": 10, "B": 1, "A": 5,
            "C": 4, "D": 2, "H": 200, "W": 200})

class LidarParams(object):
    @classu.member_initialize
    def __init__(self,
            meters_max=50,
            pixels_per_meter=2,
            hist_max_per_pixel=25,
            val_obstacle=1.):
        pass

class ESPPhiData(object):
    @classu.member_initialize
    def __init__(self,
            S_past_world_frame=None,
            S_future_world_frame=None,
            yaws=None,
            overhead_features=None,
            agent_presence=None,
            light_strings=None):
        pass

class ScenarioIntersectionLabel(object):
    """Labels samples by proximity of vehicle to intersections."""

    # NONE : str
    #     Vehicle is not near any intersections
    NONE = 'NONE'
    # UNCONTROLLED : str
    #     Vehicle is near an uncontrolled intersection
    UNCONTROLLED = 'UNCONTROLLED'
    # CONTROLLED : str
    #     Vehicle is near a controlled intersection
    CONTROLLED = 'CONTROLLED'

class SampleLabelMap(object):
    """Container of sample labels, categorized by different types."""
    
    @classu.member_initialize
    def __init__(self,
            intersection_type=ScenarioIntersectionLabel.NONE):
        pass

class SampleLabelFilter(object):
    """Container for sample label filter."""

    @classu.member_initialize
    def __init__(self,
            intersection_type=[]):
        """
        Parameters
        ----------
        intersection_type : list of str
        """
        pass

    def contains(self, _type, label):
        """Check whether a label of type _type is in the filter.

        Parameters
        ----------
        _type : str
            Label type to lookup.
        label : str
            Label to check for existence in filter.

        Returns
        -------
        bool
        """
        return label in getattr(self, _type, [])

def create_phi(settings):
    s = settings
    tf.compat.v1.reset_default_graph()
    S_past_world_frame = tf.zeros(
            (s.B, s.A, s.T_past, s.D),
            dtype=tf.float64, name="S_past_world_frame") 
    S_future_world_frame = tf.zeros(
            (s.B, s.A, s.T, s.D),
            dtype=tf.float64, name="S_future_world_frame")
    yaws = tf.zeros(
            (s.B, s.A),
            dtype=tf.float64, name="yaws")
    overhead_features = tf.zeros(
            (s.B, s.H, s.W, s.C),
            dtype=tf.float64, name="overhead_features")
    agent_presence = tf.zeros(
            (s.B, s.A),
            dtype=tf.float64, name="agent_presence")
    light_strings = tf.zeros(
            (s.B,),
            dtype=tf.string, name="light_strings")
    return ESPPhiData(
            S_past_world_frame=S_past_world_frame,
            S_future_world_frame=S_future_world_frame,
            yaws=yaws,
            overhead_features=overhead_features,
            agent_presence=agent_presence,
            light_strings=light_strings)

def create_lidar_blueprint(world):
    bp_library = world.get_blueprint_library()
    """
    sensor.lidar.ray_cast creates a carla.LidarMeasurement per step

    attributes for sensor.lidar.ray_cast
    https://carla.readthedocs.io/en/latest/ref_sensors/#lidar-sensor

    doc for carla.SensorData
    https://carla.readthedocs.io/en/latest/python_api/#carla.SensorData

    doc for carla.LidarMeasurement
    https://carla.readthedocs.io/en/latest/python_api/#carla.LidarMeasurement
    """
    lidar_bp = bp_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '10.0')
    lidar_bp.set_attribute('upper_fov', '10.0')
    lidar_bp.set_attribute('lower_fov', '-30.0')
    return lidar_bp

def create_lidar_blueprint_v2(world):
    """Construct a stronger LIDAR sensor blueprint
    """
    bp_library = world.get_blueprint_library()
    lidar_bp = bp_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '48')
    lidar_bp.set_attribute('range', '70')
    lidar_bp.set_attribute('dropoff_general_rate', '0.25')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '10.0')
    lidar_bp.set_attribute('upper_fov', '30.0')
    lidar_bp.set_attribute('lower_fov', '-30.0')
    return lidar_bp

class IntersectionReader(object):
    """Used to keep track of intersections in map and check whether
    an action is in an intersection.
    """

    # radius used to check whether junction has traffic lights
    FIND_RADIUS = 25
    # radius used to check whether vehicle is in junction
    DETECT_RADIUS = 30

    def __init__(self, carla_world, carla_map, debug=False):
        self._debug = debug
        # get nodes in graph
        topology = carla_map.get_topology()
        G = nx.Graph()
        G.add_edges_from(topology)
        tlights = util.filter_to_list(lambda a: 'traffic_light' in a.type_id,
                carla_world.get_actors())
        junctions = util.get_junctions_from_topology_graph(G)

        tlight_distances = np.zeros((len(tlights), len(junctions),))
        f = lambda j: util.location_to_ndarray(j.bounding_box.location)
        junction_locations = util.map_to_ndarray(f, junctions)
        
        g = lambda tl: util.transform_to_location_ndarray(
                tl.get_transform())
        tlight_locations = util.map_to_ndarray(g, tlights)

        for idx, junction in enumerate(junctions):
            tlight_distances[:,idx] = np.linalg.norm(
                    tlight_locations - junction_locations[idx], axis=1)

        is_controlled_junction = (tlight_distances < self.FIND_RADIUS).any(axis=0)
        is_uncontrolled_junction = np.logical_not(is_controlled_junction)
        self.controlled_junction_locations \
                = junction_locations[is_controlled_junction]
        self.uncontrolled_junction_locations \
                = junction_locations[is_uncontrolled_junction]
    
    def debug_display_intersections(self, carla_world):
        for loc in self.controlled_junction_locations:
            carla_world.debug.draw_string(
                    util.ndarray_to_location(loc) + carla.Location(z=3.0),
                    'o',
                    color=carla.Color(r=255, g=0, b=0, a=100),
                    life_time=10.0)
        for loc in self.uncontrolled_junction_locations:
            carla_world.debug.draw_string(
                    util.ndarray_to_location(loc) + carla.Location(z=3.0),
                    'o',
                    color=carla.Color(r=0, g=255, b=0, a=100),
                    life_time=10.0)

    def at_intersection_to_label(self, actor):
        """Retrieve the label corresponding to the actor's location in the
        map based on proximity to intersections.
        
        Parameters
        ----------
        actor : carla.Actor
        """
        actor_location = util.actor_to_location_ndarray(actor)
        distances_to_uncontrolled = np.linalg.norm(
                self.uncontrolled_junction_locations - actor_location, axis=1)
        if np.any(distances_to_uncontrolled < self.DETECT_RADIUS):
            return ScenarioIntersectionLabel.UNCONTROLLED
        distances_to_controlled = np.linalg.norm(
                self.controlled_junction_locations - actor_location, axis=1)
        if np.any(distances_to_controlled < self.DETECT_RADIUS):
            return ScenarioIntersectionLabel.CONTROLLED
        return ScenarioIntersectionLabel.NONE


class DataCollector(object):
    """Data collector based on DIM."""

    def __init__(self, player_actor,
            intersection_reader=None,
            save_frequency=10,
            save_directory='out',
            burn_frames=60,
            episode=0,
            exclude_samples=SampleLabelFilter(),
            phi_attributes=DEFAULT_PHI_ATTRIBUTES,
            debug=False):
        """
        player_actor : carla.Vehicle
        intersection_reader : IntersectionReader
        exclude_samples : SampleLabelFilter
            Filter to exclude saving samples by label.
        phi_attributes : attrdict.AttrDict
            Attributes T, T_past, B, A, ...etc to construct Phi object.
        """
        self._player = player_actor
        self._intersection_reader = intersection_reader
        self.save_frequency = save_frequency
        self._save_directory = save_directory
        self.burn_frames = burn_frames
        self.episode = episode
        self.exclude_samples = exclude_samples
        self._debug = debug
        self.lidar_params = LidarParams()
        self._phi = create_phi(phi_attributes)
        _, _, self.T_past, _ = tensoru.shape(self._phi.S_past_world_frame)
        self.B, self.A, self.T, self.D = tensoru.shape(self._phi.S_future_world_frame)
        self.B, self.H, self.W, self.C = tensoru.shape(self._phi.overhead_features)
        self._make_sample_name = lambda frame : "ep{:03d}_agent{:03d}_frame{:08d}".format(
                self.episode, self._player.id, frame)
        self._world = self._player.get_world()
        self._other_vehicles = list()
        self._trajectory_size = max(self.T, self.T_past) + 1
        # player_transforms : collections.deque of carla.Transform
        self.player_transforms = collections.deque(
                maxlen=self._trajectory_size)
        # others_transforms : collections.deque of (dict of int : carla.Transform)
        #     Container of dict where key is other vehicle ID and value is
        #     carla.Transform
        self.others_transforms = collections.deque(
                maxlen=self._trajectory_size)
        self.trajectory_feeds = collections.OrderedDict()
        self.lidar_feeds = collections.OrderedDict()
        # _n_feeds : int
        #     Size of trajectory/lidar feed dict
        self._n_feeds = self.T + 1
        self.streaming_generator = generate_observation.StreamingGenerator(
                self._phi)
        self.sensor = self._world.spawn_actor(
                create_lidar_blueprint_v2(self._world),
                carla.Transform(carla.Location(z=2.5)),
                attach_to=self._player,
                attachment_type=carla.AttachmentType.Rigid)
        self._first_frame = None
    
    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: DataCollector._parse_image(weak_self, image))

    def stop_sensor(self):
        """Stop the sensor."""
        self.sensor.stop()

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.sensor.destroy()
        self.sensor = None
    
    def get_player(self):
        return self._player

    def set_vehicles(self, vehicle_ids):
        """Given a list of non-player vehicle IDs retreive the vehicles corr.
        those IDs to watch.
        Used at the start of data collection.
        Do not add the player vehicle ID in the list!
        """
        self._other_vehicles = self._world.get_actors(vehicle_ids)

    def _update_transforms(self):
        """Store player an other vehicle trajectories."""
        self.player_transforms.append(self._player.get_transform())
        others_transform = {}
        for vehicle in self._other_vehicles:
            others_transform[vehicle.id] = vehicle.get_transform()
        self.others_transforms.append(others_transform)

    def _should_save_dataset_sample(self, frame):
        """Check if collector reached a frame where it should save dataset sample.

        Parameters
        ----------
        frame : int
        """

        if len(self.trajectory_feeds) == 0:
            return False
        if frame - next(iter(self.trajectory_feeds)) > self.T:
            """Make sure that we can access past trajectories T steps
            ago relative to current frame."""
            if frame % self.save_frequency == 0:
                """Save dataset every save_frequency steps."""
                if frame - self._first_frame > self.burn_frames:
                    """Skip the first number of burn_frames"""
                    return True
        return False
    
    def _get_sample_labels(self):
        """Get labels for sample collected based on the sensor's current position. 

        Returns
        -------
        SampleLabelMap
        """
        if self._intersection_reader is not None:
            intersection_type_label = self._intersection_reader \
                    .at_intersection_to_label(self._player)
        else:
            intersection_type_label = ScenarioIntersectionLabel.NONE
        return SampleLabelMap(
                intersection_type=intersection_type_label)

    def debug_draw_red_player_bbox(self):
        self._world.debug.draw_box(
                carla.BoundingBox(
                    self._player.get_transform().location,
                    self._player.bounding_box.extent),
                self._player.get_transform().rotation,
                thickness=0.5,
                color=carla.Color(r=255, g=0, b=0, a=255),
                life_time=3.0)

    def debug_draw_green_player_bbox(self):
        self._world.debug.draw_box(
                carla.BoundingBox(
                        self._player.get_transform().location,
                        self._player.bounding_box.extent),
                self._player.get_transform().rotation,
                thickness=0.5,
                color=carla.Color(r=0, g=255, b=0, a=255),
                life_time=3.0)

    def _should_exclude_dataset_sample(self, sample_labels):
        """Check if collector should exclude saving sample at this sensor location.

        Parameters
        ----------
        sample_labels : SampleLabelMap

        Returns
        -------
        bool
        """
        for key, val in vars(sample_labels).items():
            if self.exclude_samples.contains(key, val):
                if self._debug:
                    self.debug_draw_red_player_bbox()
                    logging.debug("filter")
                return True
        
        if self._debug:
            self.debug_draw_green_player_bbox()
            logging.debug("don't filter")
        return False

    def capture_step(self, frame):
        """Have the data collector capture the current snapshot of the simulation.
        
        Parameters
        ----------
        frame : int
            The frame index returned from the latest call to carla.World.tick()
        """
        logging.debug(f"in LidarManager.capture_step() player = {self._player.id} frame = {frame}")
        if self._first_frame is None:
            self._first_frame = frame
        self._update_transforms()
        if len(self.player_transforms) >= self.T_past:
            """Only save trajectory feeds when we have collected at
            least T_past number of player and other vehicle transforms."""
            observation = generate_observation.PlayerObservation(
                    frame, self._phi, self._world, self._other_vehicles,
                    self.player_transforms, self.others_transforms)
            self.streaming_generator.add_feed(
                    frame, observation, self.trajectory_feeds)
            
            if self._should_save_dataset_sample(frame):
                """Save dataset sample if needed."""
                logging.debug(f"saving sample. player = {self._player.id} frame = {frame}")
                sample_labels = self._get_sample_labels()
                if not self._should_exclude_dataset_sample(sample_labels):
                    self.streaming_generator.save_dataset_sample(
                            frame, self.episode, observation,
                            self.trajectory_feeds, self.lidar_feeds,
                            self._player.bounding_box,
                            self.sensor, self.lidar_params,
                            self._save_directory, self._make_sample_name,
                            sample_labels)
        
        if len(self.trajectory_feeds) > self._n_feeds:
            """Remove older frames.
            (frame, feed) is removed in LIFO order."""
            frame, feed = self.trajectory_feeds.popitem(last=False)
            self.lidar_feeds.pop(frame)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        logging.debug(f"in LidarManager._parse_image() player = {self._player.id} frame = {image.frame}")
        self.lidar_feeds[image.frame] = image

# Y_INTERSECTION = 'Y_INTERSECTION'
# X_INTERSECTION = 'X_INTERSECTION'
# T_INTERSECTION = 'T_INTERSECTION'
# TOWN_03_UNCONTROLLED_SPAWN_POINTS = {
#     T_INTERSECTION: {
#         # top-right
#         1: set([
#             82, 111, 150, 145,
#         ]),
#         # top-right
#         2: set([
#             54, 182, 217,
#         ]),
#         # right
#         3: set([
#             21, 22, 23, 24,
#         ]),
#         # bottom-right
#         4: set([
#             87, 89, 201, 224
#         ])
#     }
#     Y_INTERSECTION: {
#         # bottom left
#         1: set([
#             86, 90, 91 173, 198,
#         ]),
#     }
#     X_INTERSECTION: {
#         # top left
#         2: set([
#             56, 57, 241,
#         ])
#     }
# }
# UNCONTROLLED_SPAWN_POINTS = {
#     'Town03': TOWN_03_UNCONTROLLED_SPAWN_POINTS,
# }
