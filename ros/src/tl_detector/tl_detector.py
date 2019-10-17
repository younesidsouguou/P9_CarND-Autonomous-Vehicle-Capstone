#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial.kdtree import KDTree
import numpy as np
from enum import Enum
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3
TL_DETECTOR_RATE_HZ = 50


class TLDetectorState(Enum):
    INIT = 1
    RUN = 2

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.current_pose = None
        self.current_pose_idx = None
        self.base_waypoints = None
        self.base_waypoints_2d = None
        self.base_waypoints_kdtree = None
        self.camera_image = None
        self.has_image = False
        self.all_traffic_lights = []

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.init_complete = False
        self.current_state = TLDetectorState.INIT

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.run_tl_detector_state_machine()

        # rospy.spin()

    def run_tl_detector_state_machine(self):

        rate = rospy.Rate(TL_DETECTOR_RATE_HZ)

        # loop the state machine while the system is running
        while not rospy.is_shutdown():

            if self.current_state == TLDetectorState.INIT:
                # ensure that the initialization is complete before exiting this state
                # rospy.loginfo("TLDetectorState.INIT ... ")

                self.init_complete = (self.base_waypoints and self.current_pose and self.all_traffic_lights)
                if self.init_complete:
                    # rospy.loginfo("TLDetectorState initialization complete...")
                    # rospy.loginfo("TLDetectorState switching state to RUN...")
                    self.current_state = TLDetectorState.RUN

            elif self.current_state == TLDetectorState.RUN:
                # rospy.loginfo("TLDetectorState.RUN ... ")
                # if initialization is complete, publish the final waypoints for this cycle
                if self.init_complete:

                    # TODO: Ashish uncomment camera image check
                    # check whether we have an image
                    if not self.has_image or not self.camera_image:
                        continue

                    stop_line_wp_idx, state = self.process_traffic_lights()

                    '''
                    Publish upcoming red lights at camera frequency.
                    Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
                    of times till we start using it. Otherwise the previous stable state is
                    used.
                    '''
                    if self.state != state:
                        self.state_count = 0
                        self.state = state
                    elif self.state_count >= STATE_COUNT_THRESHOLD:
                        self.last_state = self.state
                        stop_line_wp_idx = stop_line_wp_idx if state == TrafficLight.RED else -1
                        self.last_wp = stop_line_wp_idx
                        self.upcoming_red_light_pub.publish(Int32(stop_line_wp_idx))
                    else:
                        self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                    self.state_count += 1

                    # rospy.loginfo("TL_DETECTOR: Car WP: {0} ** Closest light wp: {1} ** light state: {2}".format(
                    #     self.current_pose_idx, stop_line_wp_idx, state))

            else:  # switch default state
                self.current_state = TLDetectorState.INIT

            rate.sleep()


    def pose_cb(self, msg):
        self.current_pose = msg
        # also get the current car position index
        if self.current_pose and self.base_waypoints_2d:
            x = self.current_pose.pose.position.x
            y = self.current_pose.pose.position.y
            self.current_pose_idx = self.get_closest_waypoint_index(x, y)

    def waypoints_cb(self, msg):
        # self.waypoints = waypoints
        # rospy.loginfo("TL_DETECTOR: waypoints_cb called...")

        self.base_waypoints = msg.waypoints
        if self.base_waypoints is not None:
            # populate the Lane msg header as well (not really required, but to be consistent)
            # self.lane_msg_header = msg.header
            # get the 2d (x, y) waypoints
            self.base_waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in self.base_waypoints]
            # build a KDTree for efficient search of nearest waypoint
            self.base_waypoints_kdtree = KDTree(self.base_waypoints_2d)
            # rospy.loginfo("TL_DETECTOR: Base waypoints loaded...")

    def traffic_cb(self, msg):
        self.all_traffic_lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        # rospy.loginfo("TL_DETECTOR: image_cb called...")

        self.has_image = True
        self.camera_image = msg.data

        ## TODO: Ashish - remove - moved to TL Detector state machine
        # light_wp, state = self.process_traffic_lights()
        #
        # rospy.loginfo("Closest light wp: {0} \t light state: {1}".format(light_wp, state))
        #
        # '''
        # Publish upcoming red lights at camera frequency.
        # Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        # of times till we start using it. Otherwise the previous stable state is
        # used.
        # '''
        # if self.state != state:
        #     self.state_count = 0
        #     self.state = state
        # elif self.state_count >= STATE_COUNT_THRESHOLD:
        #     self.last_state = self.state
        #     light_wp = light_wp if state == TrafficLight.RED else -1
        #     self.last_wp = light_wp
        #     self.upcoming_red_light_pub.publish(Int32(light_wp))
        # else:
        #     self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        # self.state_count += 1

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

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # for testing, just return the light state for now!
        return light.state

        # TODO: ASHISH USE THE TL_CLASSIFIER HERE!!
        # if(not self.has_image):
        #     self.prev_light_loc = None
        #     return False
        #
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #
        # #Get classification
        # return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # light = None

        closest_light = None
        stop_line_wp_idx = -1

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.current_pose_idx:
            # # car_position = self.get_closest_waypoint(self.current_pose.pose)
            # x = self.current_pose.pose.position.x
            # y = self.current_pose.pose.position.y
            # self.current_pose_idx = self.get_closest_waypoint_index(x, y)

            #TODO find the closest visible traffic light (if one exists)

            # we need to find the closest traffic light location, so start with the max difference
            # in distance in terms of the waypoint indices
            diff = len(self.base_waypoints)

            # find the closest traffic light (i.e. stop line of the traffic light) from the list of all traffic lights
            for i, traffic_light in enumerate(self.all_traffic_lights):
                tl_stop_line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint_index(tl_stop_line[0], tl_stop_line[1], ahead=False)

                # find the closest stop line waypoint index
                d = temp_wp_idx - self.current_pose_idx
                if 0 <= d < diff:
                    diff = d
                    closest_light = traffic_light
                    stop_line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return stop_line_wp_idx, state

        return stop_line_wp_idx, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
