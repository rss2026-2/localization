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
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('num_particles', 100)

        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, self.scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # added
        self.particle_pub = self.create_publisher(PoseArray, "/particles", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.num_particles = self.get_parameter("num_particles").value # number of particles we are using

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        self.last_odom_info = None

        # Initialize particles to a default pose so callbacks don't crash before /initialpose.
        self.particles = np.zeros((self.num_particles, 3), dtype=float)
        self.updates = 0

    def laser_callback(self, laser_msg):
        """
        LaserScan's callback to update particles with sensor model.

        Args:
            laser_msg (_type_): LaserScan
        """
        if self.particles is None:
            return
        observation = laser_msg.ranges
        weights = self.sensor_model.evaluate(self.particles, observation)
        # if weights is None:
        #     return
        self.particles = self.resample(self.particles, weights)
        self.update_average()
        self.visualize_particles(self.particles)

    def odom_callback(self, odometry_msg):
        """
        Odometry callback to update particles using the motion model.

        Args:
            odometry_msg (_type_): Odometry
        """
        if self.particles is None:
            return
        # Get the current xyz position of the robot
        current_odom_pose = odometry_msg.pose.pose

        # Get the rotation (yaw from the euler angles)
        current_odom_quat = current_odom_pose.orientation
        current_odom_quat_list = [current_odom_quat.x, current_odom_quat.y, current_odom_quat.z, current_odom_quat.w]

        # get it in euler
        r = R.from_quat(current_odom_quat_list)
        current_odom_yaw = r.as_euler('xyz')[2]


        current_odom_info = np.array([current_odom_pose.position.x, current_odom_pose.position.y, current_odom_yaw])

        if self.last_odom_info is None:
            self.last_odom_info = current_odom_info
            return

        # Subtract from last saved odometry to get the change in odometry deltax
        odom_change = current_odom_info - self.last_odom_info
        self.last_odom_info = current_odom_info

        self.particles = self.motion_model.evaluate(self.particles, odom_change)

        self.last_odom_info = current_odom_info
        self.update_average()


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
        self.updates += 1
        self.get_logger().info(f"number of update calls: {self.updates}")
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

    def visualize_particles(self, particles):
        """
        Visualizes all the particles as PoseArray messages

        :param particles: a Nx3 array of all particles storing [x,y,theta]
        """
        if self.particles is None:
            return

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for x, y, theta in particles:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            qx, qy, qz, qw = R.from_euler('z', theta).as_quat()
            pose.orientation.x = float(qx)
            pose.orientation.y = float(qy)
            pose.orientation.z = float(qz)
            pose.orientation.w = float(qw)
            msg.poses.append(pose)
        self.particle_pub.publish(msg)

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

        sampled_particles[:, 2] = (sampled_particles[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

        return sampled_particles

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
