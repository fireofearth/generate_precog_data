import collections
import weakref
import logging
import attrdict
import tensorflow as tf
import carla

import generate.overhead as generate_overhead
import generate.observation as generate_observation
import generate.util as util
import precog.utils.class_util as classu
import precog.utils.tensor_util as tensoru

class LidarParams(object):
    @classu.member_initialize
    def __init__(self, meters_max=50, pixels_per_meter=2, hist_max_per_pixel=25, val_obstacle=1.):
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

def create_phi(settings):
    s = settings
    tf.compat.v1.reset_default_graph()
    S_past_world_frame = tf.zeros((s.B, s.A, s.T_past, s.D), dtype=tf.float64, name="S_past_world_frame") 
    S_future_world_frame = tf.zeros((s.B, s.A, s.T, s.D), dtype=tf.float64, name="S_future_world_frame")
    yaws = tf.zeros((s.B, s.A), dtype=tf.float64, name="yaws")
    overhead_features = tf.zeros((s.B, s.H, s.W, s.C), dtype=tf.float64, name="overhead_features")
    agent_presence = tf.zeros((s.B, s.A), dtype=tf.float64, name="agent_presence")
    light_strings = tf.zeros((s.B,), dtype=tf.string, name="light_strings")
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

class DataCollector(object):
    """Data collector based on DIM."""

    def __init__(self, player_actor,
            save_frequency = 10,
            save_directory='out'):
        self.lidar_params = LidarParams()
        s = attrdict.AttrDict({
            "T": 20, "T_past": 10, "B": 1, "A": 5,
            "C": 4, "D": 2, "H": 200, "W": 200})
        self._phi = create_phi(s)
        _, _, self.T_past, _ = tensoru.shape(self._phi.S_past_world_frame)
        self.B, self.A, self.T, self.D = tensoru.shape(self._phi.S_future_world_frame)
        self.B, self.H, self.W, self.C = tensoru.shape(self._phi.overhead_features)
        self._player = player_actor
        self._save_directory = save_directory
        self._make_sample_name = lambda frame : "agent{:03d}_frame{:08d}".format(
                self._player.id, frame)
        self._world = self._player.get_world()
        self._other_vehicles = list()
        self._trajectory_size = self.T_past + 10
        # player_transforms : collections.deque of carla.Trajectory
        self.player_transforms = collections.deque(
                maxlen=self._trajectory_size)
        # others_transforms : collections.deque
        #    of (dict of int : carla.Trajectory) 
        self.others_transforms = collections.deque(
                maxlen=self._trajectory_size)
        self.trajectory_feeds = collections.OrderedDict()
        self.lidar_feeds = collections.OrderedDict()
        self._n_feeds = self.T + self.T_past + 10
        self.save_frequency = save_frequency
        self.streaming_generator = generate_observation.StreamingGenerator(
                self._phi)
        self.sensor = self._world.spawn_actor(
                create_lidar_blueprint_v2(self._world),
                carla.Transform(carla.Location(z=2.5)),
                attach_to=self._player,
                attachment_type=carla.AttachmentType.Rigid)
    
    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: DataCollector._parse_image(weak_self, image))

    def stop_sensor(self):
        self.sensor.stop()

    def destroy(self):
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
        if len(self.trajectory_feeds) == 0:
            return False
        if frame - next(iter(self.trajectory_feeds)) > self.T:
            """Make sure that we can access past trajectories T steps
            ago relative to current frame."""
            if frame % self.save_frequency == 0:
                """Save dataset every save_frequency steps."""
                return True
        return False

    def capture_step(self, frame):
        logging.debug(f"in LidarManager.capture_step() player = {self._player.id} frame = {frame}")
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
                self.streaming_generator.save_dataset_sample(
                        frame, observation, self.trajectory_feeds,
                        self.lidar_feeds, self._player.bounding_box,
                        self.sensor, self.lidar_params,
                        self._save_directory, self._make_sample_name)
        
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