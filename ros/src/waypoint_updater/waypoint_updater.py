#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
from enum import Enum

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

# LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
WAYPOINT_UPDATER_RATE_HZ = 10  # Update rate for waypoint updater

MPH_TO_METERS_PER_SEC_MULTIPLIER = 0.44704
MAX_VELOCITY_PROPORTION = 0.99  # we don't want to be below speed limit

MAX_ACCELERATION = 5.0
MAX_DECELERATION = 5.0

class WaypointUpdaterState(Enum):
    INIT = 1
    CRUISE = 2


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # TODO: Add other member variables you need below
        self.initialization_complete = False
        self.current_state = WaypointUpdaterState.INIT
        self.lane_msg_header = None
        self.base_waypoints = None
        self.base_waypoints_2d = None
        self.base_waypoints_kdtree = None
        self.current_pose = None
        self.current_pose_idx = None
        self.current_velocity_ms = None
        self.stop_line_wp_idx = -1

        self.max_velocity_mph = rospy.get_param('/waypoint_loader/velocity')
        self.max_velocity_ms = self.max_velocity_mph * MPH_TO_METERS_PER_SEC_MULTIPLIER * MAX_VELOCITY_PROPORTION

        rospy.loginfo("=========> WaypointUpdater: Max Velocity: {0} MPH or {1} m/s".format(
            self.max_velocity_mph, self.max_velocity_ms))

        # Add publishers and subscribers
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # run the waypoint updater state machine
        self.run_waypoint_updater_state_machine()

        # rospy.spin()


    def run_waypoint_updater_state_machine(self):

        rate = rospy.Rate(WAYPOINT_UPDATER_RATE_HZ)

        # loop the state machine while the system is running
        while not rospy.is_shutdown():

            if self.current_state == WaypointUpdaterState.INIT:
                # ensure that the initialization is complete before exiting this state
                self.initialization_complete = (self.base_waypoints and self.current_pose)
                if self.initialization_complete:
                    rospy.loginfo("WaypointUpdaterState.INIT ... ")
                    rospy.loginfo("WaypointUpdater initialization complete...")
                    rospy.loginfo("WaypointUpdater switching state to CRUISE State...")
                    self.current_state = WaypointUpdaterState.CRUISE

            elif self.current_state == WaypointUpdaterState.CRUISE:
                # rospy.loginfo("WaypointUpdaterState.CRUISE ... ")
                # if initialization is complete, publish the final waypoints for this cycle
                if self.current_pose_idx:
                    # closest_waypoint_idx = self.get_closest_waypoint_index()
                    # self.publish_final_waypoints(closest_waypoint_idx)
                    self.publish_final_waypoints()

            else:  # switch default state
                self.current_state = WaypointUpdaterState.INIT

            rate.sleep()

    def pose_cb(self, msg):
        '''
        Callback for current position of vehicle messages (/geometry_msgs/PoseStamped topic).
        :param msg: message with current position of the vehicle, provided by the simulator or localization.
        :return: None
        '''
        # TODO: Implement

        # rospy.loginfo("pose_cb called... ")
        self.current_pose = msg
        # also get the current car position index
        if self.current_pose and self.base_waypoints_2d:
            x = self.current_pose.pose.position.x
            y = self.current_pose.pose.position.y
            self.current_pose_idx = self.get_closest_waypoint_index(x, y)


    def waypoints_cb(self, msg):
        # TODO: Implement
        # pass

        '''
        Callback for Waypoints messages (/base_waypoints topic). This is called only once with the base waypoints.
        :param msg: message of type styx_msgs/Lane with base waypoints 
        :return: None
        '''
        # rospy.loginfo("WAYPOINT_UPDATER: waypoints_cb called...")

        self.base_waypoints = msg.waypoints
        if self.base_waypoints is not None:
            # populate the Lane msg header as well (not really required, but to be consistent)
            self.lane_msg_header = msg.header
            # get the 2d (x, y) waypoints
            self.base_waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in self.base_waypoints]
            # build a KDTree for efficient search of nearest waypoint
            self.base_waypoints_kdtree = KDTree(self.base_waypoints_2d)
            rospy.loginfo("WAYPOINT_UPDATER: Base waypoints loaded...")

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # rospy.loginfo("WAYPOINT_UPDATER: traffic_cb called...")

        self.stop_line_wp_idx = msg.data


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def current_velocity_cb(self, msg):
        '''
        Callback of the current velocity msgs (/current_velocity topic) 
        :param msg: message with current velocity in meters per second
        :return: None
        '''
        self.current_velocity_ms = msg.twist.linear.x
        # rospy.loginfo("=========> WaypointUpdater: Current Velocity: {0} m/s".format(self.current_velocity_ms))

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def publish_final_waypoints(self):

        '''
        Publishes the the final waypoints based on current car position wp index
        :return: None
        '''

        # # create a styx_msgs/Lane message
        # lane = Lane()
        # # populate the header (not really required as not used, but just to be consistent)
        # lane.header = self.lane_msg_header
        # # populate the final waypoints, python slicing will take care of array out of bound situation and crop it
        # lane.waypoints = self.base_waypoints[self.current_pose_idx:self.current_pose_idx+LOOKAHEAD_WPS]
        # self.final_waypoints_pub.publish(lane)

        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):

        # make a new lane msg object
        lane = Lane()
        lane.header = self.lane_msg_header

        closest_idx = self.current_pose_idx
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        # get the slice of base waypoints to publish (with or without changing the target velocity.)
        base_wp_slice = self.base_waypoints[closest_idx:farthest_idx]

        # rospy.loginfo("WAYPOINT_UPDATER: Car WP: {0} ** Closest light wp: {1}".format(
        #     self.current_pose_idx, self.stop_line_wp_idx))
        # rospy.loginfo("****** WAYPOINT_UPDATER: farthest_idx: {0}".format(farthest_idx))

        # decelerate = ((self.stop_line_wp_idx != -1) and (self.stop_line_wp_idx < farthest_idx))
        # rospy.loginfo("WAYPOINT_UPDATER: decelerate: {0} ** Cond #1: {1} ** Cond #2: {2}".format(
        #     decelerate, (self.stop_line_wp_idx != -1), (self.stop_line_wp_idx < farthest_idx )))
        #
        # if decelerate:
        #     rospy.loginfo("****** WAYPOINT_UPDATER: Decelerating...")
        #     lane.waypoints = self.decelerate_waypoints(base_wp_slice, closest_idx)
        # else:
        #     lane.waypoints = base_wp_slice

        if self.stop_line_wp_idx == -1 or self.stop_line_wp_idx >= farthest_idx:
            # lane.waypoints = base_wp_slice
            lane.waypoints = self.accelerate_waypoints(base_wp_slice, self.current_velocity_ms, self.max_velocity_ms)
        else:
            # rospy.loginfo("****** WAYPOINT_UPDATER: Decelerating...")
            lane.waypoints = self.decelerate_waypoints(base_wp_slice, closest_idx)

        return lane

    def accelerate_waypoints(self, base_wp_slice, current_velocity, target_velocity):

        # IMPORTANT: we need to make a copy of the base waypoints slice as we don't want to modify the base waypoints
        temp_waypoints = []

        # # we need to stop 2 to 3 waypoints back of the stop line so the front of the car in behind the line
        # # (the pose is for the center of the car)
        # stop_idx = max(self.stop_line_wp_idx - closest_idx - 3, 0)

        # lower speed if we are going beyond max speed
        if current_velocity >= target_velocity:
            for i, wp in enumerate(base_wp_slice):
                p = Waypoint()
                p.pose = wp.pose
                # set the target velocity
                p.twist.twist.linear.x = target_velocity
                temp_waypoints.append(p)

        else:  # accelerate
            # get the total distance in this slice of waypoints
            # distance_to_cover = self.distance(base_wp_slice, 0, len(base_wp_slice)-1)
            # Note: we try to acclerate a little faster, so take 2/3rd of the waypoint slice
            # distance, instead of the full
            distance_to_cover = self.distance(base_wp_slice, 0, int((len(base_wp_slice)-1) * (2.0/3.0)))

            distance_to_cover = distance_to_cover if distance_to_cover > 0 else 1

            acceleration = abs((target_velocity * target_velocity) -
                               (current_velocity * current_velocity)) / (2 * distance_to_cover)

            acceleration = min(acceleration, MAX_ACCELERATION)

            for i, wp in enumerate(base_wp_slice):

                if i == 0:
                    prev_wp_velocity = current_velocity
                else:
                    prev_wp_velocity = base_wp_slice[i-1].twist.twist.linear.x

                velocity = target_velocity
                if i < (len(base_wp_slice) - 1):
                    dist = self.distance(base_wp_slice, i, i+1)
                    velocity = math.sqrt((2 * acceleration * dist) + (prev_wp_velocity * prev_wp_velocity))

                p = Waypoint()
                p.pose = wp.pose
                # set the target velocity
                p.twist.twist.linear.x = min(velocity, target_velocity)
                temp_waypoints.append(p)

        return temp_waypoints

    def decelerate_waypoints(self, base_wp_slice, closest_idx):

        # IMPORTANT: we need to make a copy of the base waypoints slice as we don't want to modify the base waypoints
        temp_waypoints = []

        # we need to stop 2 to 3 waypoints back of the stop line so the front of the car in behind the line
        # (the pose is for the center of the car)
        stop_idx = max(self.stop_line_wp_idx - closest_idx - 3, 0)

        for i, wp in enumerate(base_wp_slice):

            p = Waypoint()
            p.pose = wp.pose

            vel = 0.0

            if i <= stop_idx:
                distance_to_stop_line = self.distance(base_wp_slice, i, stop_idx)
                # calculate the target velocity for this waypoint
                vel = math.sqrt(2 * MAX_DECELERATION * distance_to_stop_line)
                # # vel = math.sqrt(2 * 0.5 * dist)

                # if dist <= 1:
                #     vel = 0.0
                # elif dist <= LOOKAHEAD_WPS/2:
                #     vel = math.sqrt(2 * MAX_DECELERATION * dist)
                # else:
                #     vel = wp.twist.twist.linear.x - (wp.twist.twist.linear.x/dist)

                # reduce velocity a little more to accommodate for any braking delay
                vel = vel * 0.8

                # if the velocity is < 1, just stop
                if vel < 1.0:
                    vel = 0.0

            # set the target velocity
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp_waypoints.append(p)

        return temp_waypoints

    def get_closest_waypoint_index(self, x, y, ahead=True):
        '''
        Returns the index of closest waypoint ahead or behind of the x,y position
        :param x: x coordinate of the position
        :param y: y coordinate of the position
        :param ahead: if True, return waypoint that ahead of the position, else the one behind
        :return: index of the closest waypoint
        '''

        # Note: here we need to check ensure that the waypoint is before the pose as the pose is going to be the
        # position of the traffic light stop line, and we need to stop before the line in case it a red light

        # query for one nearest point to x,y
        # idx 0 contains distance to closest point, idx 1 has the closest point index
        closest_idx = self.base_waypoints_kdtree.query([x, y], 1)[1]

        # check whether the closest point is ahead or behind
        closest_waypoint_coord = self.base_waypoints_2d[closest_idx]
        prev_waypoint_coord = self.base_waypoints_2d[closest_idx-1]

        # equation of the hyperplane throught closest waypoint coordinates
        cl_vect = np.array(closest_waypoint_coord)
        prev_vect = np.array(prev_waypoint_coord)
        curr_pos_vect = np.array([x, y])

        # this dot product will be positive if the closest point is behind the position,
        # and will be negative if its ahead of the position
        val = np.dot((cl_vect - prev_vect), (curr_pos_vect - cl_vect))

        # if we want the closest ahead and the wp is behind the position, get the next wp
        if ahead and val > 0:
            closest_idx = (closest_idx + 1) % len(self.base_waypoints_2d)

        # if we want the closest behind and the wp is ahead of the position, get the previous wp
        if not ahead and val < 0:
            closest_idx = (closest_idx - 1)  # 'idx of -1 is fine as python will wrap back to the last array element'

        return closest_idx


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
