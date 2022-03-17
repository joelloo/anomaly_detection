#!/usr/bin/python

import os
import cv2
import time
import datetime
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from clients import MoveBaseClient, FollowTrajectoryClient, PointHeadClient, GraspingClient

class Fetch(object):

    def __init__(self, start_x, start_y, start_theta, frame="map", capture_dir="/root/data/"):
        # Setup clients
        self.base_client = MoveBaseClient()
        self.torso_client = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
        self.head_client = PointHeadClient()
        self.grasping_client = GraspingClient()

        # Move to start location
        self.base_client.goto(start_x, start_y, start_theta, frame)
        self.grasping_client.tuck()

        # Setup camera subscriber
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/head_camera/rgb/image_raw", Image, self._camera_callback)

        # Set directory to capture images
        self.capture_dir = capture_dir
        self.record = False

    def look_at(self, x, y, z, frame, duration=1.0):
        """ Wrapper for looking at a position """
        self.head_client.look_at(x, y, z, frame, duration)

    def goto(self, x, y, theta, frame="map"):
        """ Wrapper for translating to a position """
        self.base_client.goto(x, y, theta, frame)

    def _camera_callback(self, data):
        """ Callback function for images seen by the head camera """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Save cv_image with timestamp
        if self.record:
            filename = "%s.jpeg" % datetime.datetime.fromtimestamp(time.time()).isoformat()
            cv2.imwrite(
                os.path.join(self.capture_dir, filename),
                cv_image
            )
            print("Image saved to %s." % filename)

    def start_record(self):
        self.record = True

if __name__ == "__main__":
    rospy.init_node('fetch_image_collector', anonymous=True)
    agent = Fetch(3.0, 3.0, 0)
    rospy.spin()
