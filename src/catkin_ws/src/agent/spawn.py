import random
import numpy as np
import rospy
from gazebo_msgs.srv import SpawnModel, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose

spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
model_state_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

# Boundaries of table with clearance
table_x_bound = (3.6, 4.4)
table_y_bound = (2.6, 3.4)

# Model choices
model_choices = [
    '/root/.gazebo/models/coke_can/model.sdf',
    # '/root/.gazebo/models/beer/model.sdf',
    '/root/.gazebo/models/plastic_cup/model.sdf',
    '/root/.gazebo/models/bowl/model.sdf'
]

def spawn(num_items=10):
    for i in range(num_items):
        # Spawn random items on table
        pose = Pose()
        pose.position.x = np.random.uniform(low=table_x_bound[0], high=table_x_bound[1])
        pose.position.y = np.random.uniform(low=table_y_bound[0], high=table_y_bound[1])
        pose.position.z = 1

        model_file = random.choice(model_choices)
        rospy.wait_for_service("/gazebo/spawn_urdf_model")
        with open(model_file, 'r') as fp:
            spawn_model_client(
                model_name='model%d'%i,
                model_xml=fp.read(),
                initial_pose=pose,
                reference_frame='world'
            )

# Randomly move the items around
def rearrange(num_items=10):
    for i in range(num_items):
        # Select the model
        model = ModelState()
        model.model_name = 'model%d'%i
        model.reference_frame = 'world'

        # Set different pose for selected model
        pose = Pose()
        pose.position.x = np.random.uniform(low=table_x_bound[0], high=table_x_bound[1])
        pose.position.y = np.random.uniform(low=table_y_bound[0], high=table_y_bound[1])
        pose.position.z = 1
        model.pose = pose

        model_state_client(model)

