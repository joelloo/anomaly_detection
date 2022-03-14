# Gazebo Simulation Environment for Fetch

## Pre-requisites
1. Docker-ce: https://docs.docker.com/engine/install/ubuntu/
1. Nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Building the ROS Melodic + Gazebo Environment
1. Build docker image using Dockerfile:
```bash
$host:  docker build -t ros_melodic .
```

## Using the ROS Melodic + Gazebo Environment
1. Run the container and pipe display to docker host:
```bash
$host: ./run.bash
```
This command will launch Fetch in the Gazebo playground demo. 
Note: The launch of Gazebo with playground environment will take a while. Please be patient.
Note: There may be a need to recursively change the ownership of `/tmp/.docker.xauth/` to the user spinning up the container.

2. Activate ros tools:
`hook.bash` provides a convenient script to hook into the docker container's bash environment. Once inside, we can run additional code.
```bash
$host: ./hook.bash
$ros_melodic: source /opt/ros/melodic/setup.bash
$ros_melodic: roslaunch fetch_moveit_config move_group.launch
```

3. Play the disco demo:
In another terminal:
```bash
$host: ./hook.bash
$ros_melodic: source /opt/ros/melodic/setup.bash
$ros_melodic: python ~/sanity/disco.py
```
We can use disco.py as an example to develop other ROS applications.

## Launching Rviz

To run the simulation on Rviz, we need roscore to be running before rviz is launched.
```bash
$ros_melodic: roscore > /dev/null & rosrun rviz rviz
```

Never just run rviz on its own:
```bash
# Don't do this
$ros_melodic: rosrun rviz rviz
```
