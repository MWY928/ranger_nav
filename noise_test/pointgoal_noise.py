import numpy as np

def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

class PointGoalNoise:
    def __init__(
        self,
        sigma_r=0.03,
        sigma_theta=np.deg2rad(2.0),
        drift_r_std=0.002,
        drift_theta_std=np.deg2rad(0.1),
    ):
        self.sigma_r = sigma_r
        self.sigma_theta = sigma_theta
        self.drift_r_std = drift_r_std
        self.drift_theta_std = drift_theta_std
        self.r_bias = 0.0
        self.theta_bias = 0.0

    def reset(self):
        self.r_bias = 0.0
        self.theta_bias = 0.0

    def __call__(self, rho, theta):
        # random-walk drift
        self.r_bias += np.random.randn() * self.drift_r_std
        self.theta_bias += np.random.randn() * self.drift_theta_std

        rho_noisy = rho + self.r_bias + np.random.randn() * self.sigma_r
        theta_noisy = theta + self.theta_bias + np.random.randn() * self.sigma_theta

        rho_noisy = max(0.0, rho_noisy)
        theta_noisy = wrap_angle(theta_noisy)

        return rho_noisy, theta_noisy