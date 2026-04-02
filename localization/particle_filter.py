from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy

assert rclpy

import math
import numpy as np
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped

def euler_from_quaternion(quat_xyzw):
    x, y, z, w = quat_xyzw

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w

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
        self.declare_parameter('base_link_frame', "base_link_pf") # change to base_link for testing on car
        self.declare_parameter('num_particles', 100)

        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.base_link_frame = self.get_parameter("base_link_frame").get_parameter_value().string_value # added

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
        self.particle_pub = self.create_publisher(PointCloud2, "/particles", 1)

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





    def laser_callback(self, laser_msg):
        """
        LaserScan's callback to update particles with sensor model.

        Args:
            laser_msg (_type_): LaserScan
        """
        observation = laser_msg.ranges
        weights = self.sensor_model.evaluate(self.particles, observation)
        self.particles = self.sensor_model.resample(self.particles,weights)
        self.update_average()

    def odom_callback(self, odometry_msg):
        """
        Odometry callback to update particles using the motion model.

        Args:
            odometry_msg (_type_): Odometry
        """

        # Get the current xyz position of the robot
        current_odom_pose = odometry_msg.pose.pose

        # Get the rotation (yaw from the euler angles)
        current_odom_quat = current_odom_pose.orientation
        current_odom_quat_list = [current_odom_quat.x, current_odom_quat.y, current_odom_quat.z, current_odom_quat.w]
        _, _, current_odom_yaw = euler_from_quaternion(current_odom_quat_list)

        current_odom_info = np.array([current_odom_pose.position.x, current_odom_pose.position.y, current_odom_yaw])

        # Subtract from last saved odometry to get the change in odometry deltax
        odom_change = current_odom_info - self.last_odom_info

        self.particles = self.motion_model.evaluate(self.particles, odom_change)


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
        _, _, theta = euler_from_quaternion(rotation_quat_list)

        x, y = pose.position.x, pose.position.y
        self.particles = np.broadcast_to((x,y,theta), (self.num_particles, 3))

        # add some noise to each of the poses
        noise = np.random.normal(0,1.0, (self.num_particles,3))
        self.particles = self.particles + noise

        self.get_logger().info("Particles Initialized")

    def update_average(self):
        """
        Publishes the "average" pose of the particles when they are updated from either
        the sensor or motion model.
        """
        # need to define some notion of the average pose

        radians = self.particles[:, 2]
        # Calculate the sum of sin and cos values
        sin_sum = sum([np.sin(rad) for rad in radians])
        cos_sum = sum([np.cos(rad) for rad in radians])

        # Calculate the circular mean using arctan2
        mean_rad = np.arctan2(sin_sum, cos_sum)

        # in simulation
        mean_x, mean_y, _ = np.mean(self.particles[:, :2], axis=1)

        # send out the new messages
        average_pose_estimate = self.create_transform_message(mean_x, mean_y, mean_rad)
        self.odom_pub.publish(average_pose_estimate)

    def create_transform_message(self, x_pos, y_pos, theta):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = self.base_link_frame

        t.transform.translation.x = x_pos
        t.transform.translation.y = y_pos
        t.transform.translation.z = 0

        x, y, z, w = quaternion_from_euler(0.0, 0.0, theta)

        t.transform.rotation.x = x
        t.transform.rotation.y = y
        t.transform.rotation.z = z
        t.transform.rotation.w = w

        return t

    def visualize_particles(self, particles):
        """
        Visualize the particles to rviz
        """
        # Get the positions and hard code z to be 0
        positions_3d = np.column_stack(particles[:, :2], np.zeros(particles.shape[0]))

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'

        pc2_msg = point_cloud2.create_cloud_xyz32(header, positions_3d)

        self.particle_pub.publish(pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
