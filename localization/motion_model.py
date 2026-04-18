import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        
        # -- Declared parameters --
        node.declare_parameter('motion_pos_trans_noise', 0.38)
        motion_pos_trans_noise = node.get_parameter('motion_pos_trans_noise').get_parameter_value().double_value
        
        node.declare_parameter('motion_orient_trans_noise', 0.3)
        motion_orient_trans_noise = node.get_parameter('motion_orient_trans_noise').get_parameter_value().double_value
        
        node.declare_parameter('motion_pos_rot_noise', 0.4)
        motion_pos_rot_noise = node.get_parameter('motion_pos_rot_noise').get_parameter_value().double_value
        
        node.declare_parameter('motion_orient_rot_noise', 0.65)
        motion_orient_rot_noise = node.get_parameter('motion_orient_rot_noise').get_parameter_value().double_value
        
        # Add a forward bias term to address particles appearing behind ground truth due to system latency in real-time runs
        node.declare_parameter('motion_forward_bias', 1.00)
        self.forward_bias = node.get_parameter("motion_forward_bias").get_parameter_value().double_value

        node.declare_parameter('deterministic', False)
        self.is_deterministic = node.get_parameter('deterministic').get_parameter_value().bool_value
        
        # Coefficients to control contribution of each parameter to each noise distribution corresponding to x,y,theta
        self.a1 = motion_pos_trans_noise # Effect of position change on translational noise
        self.a2 = motion_orient_trans_noise # Effect of orientation change on translational noise
        self.a3 = motion_pos_rot_noise # Effect of position change on rotational noise
        self.a4 = motion_orient_rot_noise  # Effect of orientation change on rotational noise
                
        # Note: We have large noise coefficients because our racecar has significant drift when giving a forward command, 
        # causing us to make a lot of micro-rotations to adjust, causing even more input noise into our odometry instead 
        # of just us being able to drive straight and rely on accurate odometry. Therefore we inject more noise than usual
        # to account for this error. We also add forward bias to account for system latency between where our car is and where
        # we're receiving the msgs.

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        # Get each column of odometry
        x_delta = odometry[0]
        y_delta = odometry[1]
        theta_delta = odometry[2]

        # Get each column of particles
        x_old = particles[:, 0]
        y_old = particles[:, 1]
        theta_old = particles[:, 2]
        
        # Add noise if not deterministic
        if not self.is_deterministic:
            
            x_delta = self.forward_bias * x_delta 
            # Determine the standard deviation of the noise distribution for each of x, y, and theta
            
            # Compute standard deviations by adding variances and taking sqrt
            x_std = np.sqrt(self.a1 * x_delta**2 + self.a2 * theta_delta**2)
            y_std = np.sqrt(self.a1 * y_delta**2 + self.a2 * theta_delta**2) # y_delta close to 0 b.c. in local frame of robot
            theta_std = np.sqrt(self.a3 * x_delta**2 + self.a4 * theta_delta**2) # effect of y_delta negligible 
            
            # Prevent errors if odometry is 0
            x_std = max(0.001, x_std)
            y_std = max(0.001, y_std)
            theta_std = max(0.001, theta_std)

            # Generate noise by sampling from gaussian distributions using above computed stds
            # Put them all in one rng.normal call for vectorization. Columns are [x noise, y noise, theta noise]
            rng = np.random.default_rng()
            noise = rng.normal(loc=0.0, scale=[x_std, y_std, theta_std], size=(len(particles), 3))
            
            # Apply noise to local odometry
            x_delta = x_delta + noise[:,0]
            y_delta = y_delta + noise[:,1]
            theta_delta = theta_delta + noise[:,2]
            
        # Using theta_old assumes the car followed a straight path defined by theta_old's orientation, and then instantaneously
        # turned to theta_new. A better estimate uses the average theta value the car would have following the path based on
        # the start and end angles. The equation gets us the orientation halfway between where we started and where we ended.
        theta_avg = theta_old + theta_delta / 2.0 

        # Use formula to get new x values with noise added: x_k = x_{k-1} + x_{delta} * cos(theta_{k-1}) - y_{delta} * sin(theta_{k-1})
        x_new = x_old + x_delta * np.cos(theta_avg) - y_delta * np.sin(theta_avg)

        # Use formula to get new y values with noise added: y_k = x_{k-1} + x_{delta} * sin(theta_{k-1}) + y_{delta} * cos(theta_{k-1})
        y_new = y_old + x_delta * np.sin(theta_avg) + y_delta * np.cos(theta_avg)

        # Use formula to get new theta values with noise added: theta_k = theta_{k-1} + theta_{delta}
        theta_new = theta_old + theta_delta

        # Concatenate the results
        x_new = np.expand_dims(x_new, axis=1)
        y_new = np.expand_dims(y_new, axis=1)
        xy = np.hstack((x_new, y_new))
        new_particles = np.column_stack((xy, theta_new))
        # Normalize theta values between -pi and pi
        new_particles[:, 2] = (new_particles[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

        return new_particles
        ####################################
