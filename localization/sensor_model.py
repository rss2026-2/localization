import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D`
# if any error re: scan_simulator_2d occurs

from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_origin = None
        self.map_height = None
        self.map_width = None
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        ####################################

        # rows = z, cols = d
        rows, cols = np.indices(self.sensor_model_table.shape)

        # calculate the four cases
        p_hit = np.exp(-(rows-cols)**2/(2*self.sigma_hit**2))/np.sqrt(2*np.pi*self.sigma_hit**2)
        p_hit /= np.sum(p_hit, axis=0, keepdims=True)
        p_hit *= self.alpha_hit

        mask_short = (cols > 0) & (rows >= 0) & (rows <= cols)
        p_short = np.zeros_like(rows, dtype=float)
        p_short[mask_short] = self.alpha_short * (2.0 / cols[mask_short]) * (
            1.0 - rows[mask_short] / cols[mask_short]
        )

        p_max = np.where(rows == self.table_width - 1, self.alpha_max, 0.0)

        p_rand = self.alpha_rand * np.ones_like(rows) / self.table_width
        self.sensor_model_table = p_hit + p_short + p_max + p_rand

        # normalize
        self.sensor_model_table /= self.sensor_model_table.sum(axis=0, keepdims=True)

        ####################################

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            self.get_logger().info('ERROR: Map has not been set yet. Protected against this error by returning before evaluate.')
            return

        ####################################

        scans = self.scan_sim.scan(particles)

        # Ensure the observation vector matches the number of simulated beams.
        # `scans` is typically shaped (N, num_beams_per_particle).
        obs = np.asarray(observation, dtype=float).reshape(-1)
        num_beams = scans.shape[1] if scans.ndim == 2 else obs.size
        if obs.size != num_beams:
            # Pick evenly spaced beams across the incoming scan.
            # This makes the sensor model robust to different lidar resolutions.
            idx = np.linspace(0, max(obs.size - 1, 0), num_beams).astype(int)
            obs = obs[idx]

        # convert lidar observations and ray tracing scans from meters to pixels
        scans /= self.resolution * self.lidar_scale_to_map_scale # where is map_resolution defined?
        obs /= self.resolution * self.lidar_scale_to_map_scale

        # Handle NaN/Inf ranges (common in LaserScan).
        obs = np.nan_to_num(obs, nan=self.table_width - 1, posinf=self.table_width - 1, neginf=0.0)

        # clip to min and max distances
        scans = np.clip(scans, 0, self.table_width-1).astype(int)
        obs = np.clip(obs, 0, self.table_width-1).astype(int)

        # assign normalized, culmulative sum weights to each particle, combine beam weights
        if scans.ndim == 2 and obs.ndim == 1:
            obs = np.broadcast_to(obs, scans.shape)
        weights = self.sensor_model_table[obs, scans]
        weights = np.prod(weights, axis=1)

        return weights
        ####################################

    def resample(self, particles, weights):
        """
        Resamples the particles given the probability of each particle occuring.
        Applies some small noise to prevent states collapsing. Returns the a Nx3 array of new particles.

        :param particles: An Nx3 matrix of the form:
            [x0 y0 theta0]
            [x1 y0 theta1]
            [    ...     ]
        :param weights: An Nx1 vector that stores the probability of each particle occuring.
        """
        if weights is None:
            return

        weights = weights/sum(weights)
        weights = np.cumsum(weights)

        # resample the particles proportional to their weights
        # here i'm using low variance sampling. read about it here: https://robotics.stackexchange.com/questions/16093/why-does-the-low-variance-resampling-algorithm-for-particle-filters-work#:~:text=Imagine%20laying%20out%20a%20yardstick,many%20offspring%20the%20parents%20produce.
        rng = np.random.default_rng()
        n = len(particles)
        random = rng.uniform(low=0,high=1/n)
        vals = np.arange(1, n + 1)
        pointers = random + (vals - 1)/n
        pointers = np.clip(pointers,0,1)
        indices = np.searchsorted(weights,pointers,side="left")
        sampled_particles = particles[indices]

        # blur the particles after resampling with some gaussian noise
        noise = rng.normal(loc=0.0, scale=[0.05, 0.05, 0.02], size=(n, 3))
        sampled_particles += noise

        if self.map_set and self.map_origin is not None and self.map_height is not None and self.map_width is not None:
            origin_x, origin_y, _ = self.map_origin
            max_x = np.nextafter(origin_x + self.map_width * self.resolution, origin_x)
            max_y = np.nextafter(origin_y + self.map_height * self.resolution, origin_y)
            np.clip(sampled_particles[:, 0], origin_x, max_x, out=sampled_particles[:, 0])
            np.clip(sampled_particles[:, 1], origin_y, max_y, out=sampled_particles[:, 1])
        sampled_particles[:, 2] = (sampled_particles[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

        return sampled_particles

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation

        quat = [origin_o.x, origin_o.y, origin_o.z, origin_o.w]
        yaw = R.from_quat(quat).as_euler("xyz")[2]

        origin = (origin_p.x, origin_p.y, yaw)
        self.map_origin = origin
        self.map_height = map_msg.info.height
        self.map_width = map_msg.info.width

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)

        self.map_set = True
        print("Map initialized")
