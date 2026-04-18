from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose

from rclpy.node import Node
import rclpy

assert rclpy

# added
from scipy.spatial.transform import Rotation as R
import numpy as np
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Float32
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformException
from geometry_msgs.msg import TransformStamped
import numpy as np
from std_msgs.msg import Float32

from viz_utils.visualization_tools import VisualizationTools


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        # -- Declared parameters --
        
        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        self.declare_parameter('num_particles', 100)
        self.num_particles = self.get_parameter("num_particles").value # number of particles we are using

        # Put sensor model on a timer
        self.declare_parameter('timer_period', 1.0)
        timer_period = self.get_parameter('timer_period').get_parameter_value().double_value

        self.declare_parameter('scan_topic', "/scan")
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        
        self.declare_parameter('odom_topic', "/odom")
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        # -- Publishers and subscribers --
        
        self.laser_sub = self.create_subscription(LaserScan, self.scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic,
                                                 self.odom_callback,
                                                 1)

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        self.particle_pub = self.create_publisher(PoseArray, "/particles", 1)

        # Publishers for timing analysis
        self.scan_time_pub = self.create_publisher(Float32, "/timing/sensor_model", 1)
        self.odom_time_pub = self.create_publisher(Float32, "/timing/motion_model", 1)

        # Publishers for cte analysis
        self.cte_pos_pub = self.create_publisher(Float32, "/cross_track_pos_error", 1)
        self.cte_theta_pub = self.create_publisher(Float32, "/cross_track_theta_error", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        # Timer for the sensor model
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # -- Variable initializations --
        
        # Initialize particles to a default pose so callbacks don't crash before /initialpose.
        self.particles = np.zeros((self.num_particles, 3), dtype=float)
        self.updates = 0
        self.thinking_times = []

        self.last_odom_info = None
        self.last_time = None
        self.update_sensor = False

        # # Initialize tf broadcaster and listener
        self.tf_broadcaster = TransformBroadcaster(self)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        self.get_logger().info("=============+LOCALIZATION READY+=============")

    def timer_callback(self):
        """
        Timer callback controlling the update rate of the sensor model to prevent particle collapse.
        """
        # Flip update_sensor to true
        self.update_sensor = True

    def laser_callback(self, laser_msg):
        """
        LaserScan's callback to update particles with sensor model.

        Args:
            laser_msg (_type_): LaserScan
        """
        # Don't run the sensor model if no particles or update_sensor is false
        if self.particles is None or not self.update_sensor:
            return
        observation = laser_msg.ranges
        weights = self.sensor_model.evaluate(self.particles, observation)
        # if weights is None:
        #     return
        self.particles = self.resample(self.particles, weights)

        # Flip update_sensor to false to wait for the timer callback.
        # Computation runs at 60 kHz so this update method is solid
        self.update_sensor = False

        # -- Timing analysis for sensor model --
        
        # # scan_time = self.get_clock().now().from_msg(laser_msg.header.stamp)

        # # 2. Get current time as a ROS Time object
        # current_time = self.get_clock().now()

        # # 3. Subtracting two Time objects returns a Duration object
        # duration = current_time - scan_time

        # # 4. Convert Duration to seconds (as a float)
        # dt = duration.nanoseconds * 1e-9
        # # self.get_logger().info(f'{duration}')

        # self.scan_time_pub.publish(Float32(data=dt))
        # # self.get_logger().info(f'Scan Latency: {dt}s')


    def odom_callback(self, odometry_msg):
        """
        Odometry callback to update particles using the motion model.

        Args:
            odometry_msg (_type_): Odometry
        """
        if self.particles is None:
            return

        # Get the current time
        current_time = odometry_msg.header.stamp.sec + (odometry_msg.header.stamp.nanosec * 1e-9)

        # Don't run the odometry model if we've never set a last time to compare to
        if self.last_time is None:
            self.last_time = current_time
            return

        # Get the change in time from previous to current call to this function
        dt = current_time - self.last_time

        # We are only provided the twist component of odometry, so we should not use the pose component, only twist
        # Get the twist of the robot
        current_odom_twist = odometry_msg.twist.twist
        x_change = dt * current_odom_twist.linear.x
        y_change = dt * current_odom_twist.linear.y
        theta_change = dt * current_odom_twist.angular.z

        # Synthesize the odometry change
        odom_change = np.array([x_change, y_change, theta_change])

        # Evaluate on the motion model
        self.particles = self.motion_model.evaluate(self.particles, odom_change)

        # Set last time to current time
        self.last_time = current_time

        # Update the average pose estimation
        self.update_average()

        # -- Timing analysis for motion model --
        
        # ct = self.get_clock().now().nanoseconds
        # odom_time = odometry_msg.header.stamp.nanosec
        # # Get the change in time from previous to current call to this function
        # dt = (ct - odom_time) * 1e-9
        # self.odom_time_pub.publish(Float32(data =dt))

        # odom_time = self.get_clock().now().from_msg(odometry_msg.header.stamp)

        # # 2. Get current time as a ROS Time object
        # current_time = self.get_clock().now()

        # # 3. Subtracting two Time objects returns a Duration object
        # duration = current_time - odom_time

        # # 4. Convert Duration to seconds (as a float)
        # dt = duration.nanoseconds * 1e-9


        # self.odom_time_pub.publish(Float32(data=dt))
        # self.get_logger().info(f'Transport Latency: {dt}s')


    def pose_callback(self, pose_msg):
        """
        Initializes the particles

        Args:
         - PoseWithCovarianceStamped : whatever pose we set in rviz
        """
        # initializing the particles
         #number of particles

        # initialize with the pose from the message
        pose = pose_msg.pose.pose

        # get the rotation
        rotation_quat = pose.orientation
        rotation_quat_list = [rotation_quat.x, rotation_quat.y, rotation_quat.z, rotation_quat.w]

        theta = R.from_quat(rotation_quat_list).as_euler('xyz')[2]

        x, y = pose.position.x, pose.position.y

        self.particles = np.tile([x, y, theta], (self.num_particles, 1))
        # add some noise to the poses
        noise = np.random.normal(0, 0.5, (self.num_particles, 3))
        self.particles = self.particles + noise
        self.update_average()
        self.get_logger().info("Particles Initialized")

    def update_average(self):
        """
        Publishes the "average" pose of the particles when they are updated from either
        the sensor or motion model.
        """
        if self.particles is None:
            return
        self.visualize_particles()
        # need to define some notion of the average pose
        radians = self.particles[:, 2]
        # Calculate the sum of sin and cos values
        sin_sum = sum([np.sin(rad) for rad in radians])
        cos_sum = sum([np.cos(rad) for rad in radians])

        # Calculate the circular mean using arctan2
        mean_rad = np.arctan2(sin_sum, cos_sum)

        # in simulation
        mean_x, mean_y = np.mean(self.particles[:, :2], axis=0)

        # send out the new messages
        average_pose_estimate = self.create_odom_message(mean_x, mean_y, mean_rad)
        self.odom_pub.publish(average_pose_estimate)

        # -- Broadcasting transform from map to particle filter frame (base_link) --

        # Create the transform message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map' # Transform from map frame
        t.child_frame_id = self.particle_filter_frame # Transform to particle filter frame

        # Set the translation and rotation of the transform based on the average pose estimate (estimate of base_link in map frame)
        t.transform.translation.x = float(mean_x)
        t.transform.translation.y = float(mean_y)
        t.transform.translation.z = 0.0

        q = R.from_euler('z', mean_rad).as_quat()
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

        # # -- CTE analysis computation --
        # # Get the transform from map to base_link for CTE computation
        # try:
        #     ground_truth_trans = self.buffer.lookup_transform('map', 'base_link', rclpy.time.Time(seconds=0))
        # except TransformException as e:
        #     self.get_logger().info(f'Could not transform {t.header.frame_id} to {t.child_frame_id}: {e}')
        #     return

        # ground_truth_x = ground_truth_trans.transform.translation.x
        # ground_truth_y = ground_truth_trans.transform.translation.y
        # ground_truth_quat = ground_truth_trans.transform.rotation
        # rot = R.from_quat([ground_truth_quat.x, ground_truth_quat.y, ground_truth_quat.z, ground_truth_quat.w])
        # ground_truth_theta = rot.as_euler('xyz')[2]

        # cte_x = abs(ground_truth_x - mean_x)
        # cte_y = abs(ground_truth_y - mean_y)

        # cte_pos_msg = Float32()
        # cte_pos = np.sqrt(cte_x**2 + cte_y**2)
        # cte_pos_msg.data = cte_pos
        # self.cte_pos_pub.publish(cte_pos_msg)

        # # Shortest angular arror
        # cte_theta_msg = Float32()
        # cte_theta = abs((ground_truth_theta - mean_rad + np.pi) % (2 * np.pi) - np.pi)
        # cte_theta_msg.data = cte_theta
        # self.cte_theta_pub.publish(cte_theta_msg)


    def create_odom_message(self, x_pos, y_pos, theta):
        msg = Odometry()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = self.particle_filter_frame

        msg.pose.pose.position.x = float(x_pos)
        msg.pose.pose.position.y = float(y_pos)
        msg.pose.pose.position.z = 0.0

        x, y, z, w = R.from_euler('z', theta).as_quat()

        msg.pose.pose.orientation.x = float(x)
        msg.pose.pose.orientation.y = float(y)
        msg.pose.pose.orientation.z = float(z)
        msg.pose.pose.orientation.w = float(w)

        return msg

    def visualize_particles(self):
        """
        Visualizes all the particles as PoseArray messages

        :param particles: a Nx3 array of all particles storing [x,y,theta]
        """
        if self.particles is None:
            return

        stamp = self.get_clock().now().to_msg()
        VisualizationTools.draw_pose_array(self.particles, self.particle_pub, stamp, frame='map')

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
            print("Could not resample nodes because weights were missing.")
            return

        # Get the sum of all the weights
        weight_sum = np.sum(weights)

        # Div by zero check
        if weight_sum == 0.0:
            self.get_logger().warn("All particle weights are zero, resetting to uniform distribution.")
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / weight_sum

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

        sampled_particles[:, 2] = (sampled_particles[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

        return sampled_particles

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
