#!/usr/bin/python

import os
import random
import cv2
import time
import datetime
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from clients import MoveBaseClient, FollowTrajectoryClient, PointHeadClient, GraspingClient
from multiprocessing import Lock
import numpy as np

from spawn import spawn, rearrange

from pdb import set_trace as bp

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

    def look_at(self, x, y, z, frame, duration=1.0):
        """ Wrapper for looking at a position """
        self.head_client.look_at(x, y, z, frame, duration)

    def goto(self, x, y, theta, frame="map"):
        """ Wrapper for translating to a position """
        self.base_client.goto(x, y, theta, frame)

    def raise_torso(self, z):
        """ Wrapper for raising the torso to a height """
        self.torso_client.move_to([z, ])

    def move_arm(self, pose):
        self.grasping_client.move_arm(pose)

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
            self.latest_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

        self.lock.release()



if __name__ == "__main__":
    rospy.init_node('fetch_image_collector', anonymous=True)
    agent = Fetch()
    agent.raise_torso(0.3)

    # Spawn 10 random objects
    spawn(10)

    poses = [
        [-1, 0.0 , 1.4 , 0.8, 0, 0.8, np.random.uniform(-1, 1)],
        [-0.2, 0.2 , 0.8 , 0, 0.8, 0.8, np.random.uniform(0.8, 1.2)],
        [0, 0.0 , 0, 0.0, 0, 0.4, np.random.uniform(-1, 1)],
        [0.2, 0.2, -0.8, 0, -0.8, 0.8, np.random.uniform(0.8, 1.2)],
        [1, 0.0 , -1.4 , 0.8, 0, 0.8, np.random.uniform(-1, 1)],
    ]

    # Initialization position
    agent.move_arm([1.32, 0, -1.4, 1.72, 0.0, 1.66, 0.0])
    agent.move_arm([1.32, 0, -1.4, 0, 0.0, 0, 0.0])


    cv2.namedWindow("Test")
    count = 0
    pose_counter = 4
    pose_incre = -1
    while count < 20000:
        key = cv2.waitKey(10)
        if key == ord('q'):
            break

        rearrange(10)

        agent.look_at(3.7, 2, -1, "map")
        agent.move_arm(poses[pose_counter])
        pose_counter += pose_incre
        agent.save_image("left2")
        agent.look_at(3.7, 1, -1, "map")
        agent.move_arm(poses[pose_counter])
        pose_counter += pose_incre
        agent.save_image("left1")
        agent.look_at(3.7, 0, -1, "map")
        agent.move_arm(poses[pose_counter])
        pose_counter += pose_incre
        agent.save_image("middle")
        agent.look_at(3.7, -1, -1, "map")
        agent.move_arm(poses[pose_counter])
        pose_counter += pose_incre
        agent.save_image("right1")
        agent.look_at(3.7, -2, -1, "map")
        agent.move_arm(poses[pose_counter])
        agent.save_image("right2")
        pose_incre *= -1
        count += 4
        print("Saved %d images" % count)
        
    
