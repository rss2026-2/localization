import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.deterministic = False
        self.translational_sigma = 1.0
        self.rotational_sigma = 1.0

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

        ####################################
        # Add noise depending of if the model is deterministic or not
        if self.deterministic:
            NOISE_PROP = 0.0
        else:
            NOISE_PROP = 0.1

        # Get each column of odometry
        x_delta = odometry[0]
        y_delta = odometry[1]
        theta_delta = odometry[2]

        # Get each column of particles
        x_old = particles[:, 0]
        y_old = particles[:, 1]
        theta_old = particles[:, 2]

        # Add noise proportional to the changes in x, y, and theta. Larger absolute values -> greater noise

        # Use formula to get new x values with noise added
        # x_k = x_{k-1} + x_{delta} * cos(theta_{k-1}) - y_{delta} * sin(theta_{k-1})
        rng = np.random.default_rng()

        noise = rng.normal(loc=0.0, scale=[self.translational_sigma, self.translational_sigma, self.rotational_sigma], size=(len(particles), 3))

        x_new = x_old + x_delta * np.cos(theta_old) - y_delta * np.sin(theta_old) + NOISE_PROP * (noise[:,0])

        # Use formula to get new y values with noise added
        # y_k = x_{k-1} + x_{delta} * sin(theta_{k-1}) + y_{delta} * cos(theta_{k-1})
        y_new = y_old + x_delta * np.sin(theta_old) + y_delta * np.cos(theta_old) + NOISE_PROP * (noise[:,1])

        # Use formula to get new theta values with noise added
        # theta_k = theta_{k-1} + theta_{delta}
        theta_new = theta_old + theta_delta + NOISE_PROP * (noise[:,2])

        # Concatenate the results
        x_new = np.expand_dims(x_new, axis=1)
        y_new = np.expand_dims(y_new, axis=1)
        xy = np.hstack((x_new, y_new))
        new_particles = np.column_stack((xy, theta_new))
        new_particles[:, 2] = (new_particles[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

        return new_particles
        ####################################
