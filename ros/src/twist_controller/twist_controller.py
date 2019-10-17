from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MIN_SPEED = 0.1

# These control gains are found experimentally
# VELOCITY_KP = 0.3
# VELOCITY_KI = 0.1
# VELOCITY_KD = 0.0
# MIN_THROTTLE = 0.0
# MAX_THROTTLE = 0.2
# LPF_TAU = 0.5
# LPF_TS = 0.02

THROTTLE_KP = 0.3
THROTTLE_KI = 0.1
THROTTLE_KD = 0.005
MIN_THROTTLE = 0.0
MAX_THROTTLE = 0.2
LPF_TAU = 0.5
LPF_TS = 0.2  # 1/DBW_RATE_HZ

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                 wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, sampling_rate):
        # TODO: Implement
        # pass

        # member variables
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.sampling_rate = sampling_rate

        # update vehicle mass to take into account fuel capacity
        self.vehicle_mass = self.vehicle_mass + (self.fuel_capacity * GAS_DENSITY)

        # Yaw Controller for Steering
        self.yaw_controller = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)

        # PID Controller for Throttle
        # this control gains are found experimentally
        # kp = THROTTLE_KP
        # ki = THROTTLE_KI
        # kd = THROTTLE_KD
        # min_throttle = MIN_THROTTLE
        # max_throttle = MAX_THROTTLE
        # self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)

        self.throttle_controller = PID(THROTTLE_KP, THROTTLE_KI, THROTTLE_KD, -1*self.accel_limit, accel_limit)

        # Velocity Low-Pass Filter
        tau = LPF_TAU  # 1/(2*pi*tau) = cutoff frequency
        ts = LPF_TS  # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        # self.last_time = rospy.get_time()
        self.last_linear_vel = 0.

    def control(self, proposed_linear_vel, proposed_angular_vel, current_linear_vel, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        # if the Drive-By-Wire mode is off (safety driver takes control of the car), we are not in autonomous mode, so
        # we don't want the integral term of PID to controller to accumulate errors, so reset it when the dbw mode
        # is not enabled
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.vel_lpf.reset()
            return 0., 0., 0.

        # clean the current velocity by passing it through the low-pass filter to get rid of the high frequency noises
        current_linear_vel = self.vel_lpf.filt(current_linear_vel)

        # get the steering from the Yaw Controller
        steering = self.yaw_controller.get_steering(proposed_linear_vel, proposed_angular_vel, current_linear_vel)

        # get velocity error between proposed and current velocities
        vel_error = proposed_linear_vel - current_linear_vel
        self.last_linear_vel = current_linear_vel

        # # get sample time
        # current_time = rospy.get_time()
        # sample_time = current_time = self.last_time
        # self.last_time = current_time

        # sample time = 1/rate_hw (50)
        sample_time = 1/float(self.sampling_rate)

        # get the throttle from the Throttle PID controller
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        # if proposed vel is 0 and current vel is small, we probably want to stop
        if proposed_linear_vel == 0. and current_linear_vel < 0.1:
            throttle = 0.
            # braking torque in N*m - to hold the car in place if we are stopped at a light. Acceleration ~1m/s^2.
            # This works for the simulator, but to hold Carla stationary, this has to be changed to ~700 Nm of torque
            brake = 400
        # if throttle is small and velocity error is negative, we are probably going faster than we want to and our pid
        # is letting up on the throttle, but we want to slow down
        elif throttle < 0.1 and vel_error < 0.:
            throttle = 0.
            decel = max(vel_error, self.decel_limit)
            # calculate braking torque (N*m) as desired acceln/decel * vehicle mass * wheel radius. decel here is -ve.
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        rospy.loginfo('proposed_linear_vel: {0}   proposed_angular_vel: {1}   current_linear_vel: {2}'.format(
            proposed_linear_vel, proposed_angular_vel, current_linear_vel))
        rospy.loginfo('Throttle: {0}   Brake: {1}   Steering: {2}'.format(throttle, brake, steering))

        return throttle, brake, steering
