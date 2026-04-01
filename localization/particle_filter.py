from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy

assert rclpy

# added
from tf_transformations import euler_from_quaternion
import numpy as np
from sensor_msgs.msg import PointCloud2

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
        self.declare_parameter('base_link_topic', "/base_link_pf") # change to base_link for testing on car
        self.declare_parameter('num_particles', 100)
        
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("base_link_topic").get_parameter_value().string_value # added

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
        self.particle_pub = self.create_publisher(
            PointCloud2,
            
        )

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
    
    def visualize_particles(self, particles):
        """"
        
        """
        self.particle_pub

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
        mean_positions = np.mean(self.particles[:, :2], axis=1)
        

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
