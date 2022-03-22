#!/usr/bin/python

import sys
import termios
import tty
import os
import cv2
import time
import datetime
import rospy
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from clients import MoveBaseClient, FollowTrajectoryClient, PointHeadClient, GraspingClient
from multiprocessing import Lock

from spawn import spawn, rearrange

class Fetch(object):

    def __init__(self, capture_dir="/root/data/"):
        # Setup clients
        self.base_client = MoveBaseClient()
        self.torso_client = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
        self.head_client = PointHeadClient()
        self.grasping_client = GraspingClient()

        self.latest_image = None
        self.lock = Lock()

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


    def save_image(self, prefix):
        self.lock.acquire()
        filename = "%s.jpeg" % datetime.datetime.fromtimestamp(time.time()).isoformat()
        filepath = os.path.join(self.capture_dir, os.path.join(prefix, filename))
        cv2.imwrite(
            filepath,
            self.latest_image
        )
        print("Image saved to %s." % filepath)

        self.lock.release()

    def _camera_callback(self, data):
        """ Callback function for images seen by the head camera """
        self.lock.acquire()
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Center crop image
            w, h = 224, 224
            center = image.shape / 2
            x = center[1] - w/2
            y = center[0] - h/2

            self.latest_image = image[int(y):int(y+h), int(x):int(x+w)]

        except CvBridgeError as e:
            print(e)

        self.lock.release()

    def start_record(self):
        self.record = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", "-r", action="store_true")
    args = parser.parse_args()

    print(args.record)
    rospy.init_node('fetch_image_collector', anonymous=True)
    agent = Fetch()

    # Spawn 10 random objects
    spawn(10)

    cv2.namedWindow("Test")
    count = 0
    while count < 20000:
        key = cv2.waitKey(50)
        if key == ord('q'):
            break

        rearrange(10)

        agent.look_at(3.7, 2, 0, "map")
        agent.save_image("left2")
        agent.look_at(3.7, 1, 0, "map")
        agent.save_image("left1")
        agent.look_at(3.7, 0, 0, "map")
        agent.save_image("middle")
        agent.look_at(3.7, -1, 0, "map")
        agent.save_image("right1")
        agent.look_at(3.7, -2, 0, "map")
        agent.save_image("right2")
        count += 4
        print("Saved %d images" % count)
        
    
