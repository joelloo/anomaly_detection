# Gazebo Simulation Environment for Fetch

## Pre-requisites
1. Docker-ce: https://docs.docker.com/engine/install/ubuntu/
1. Nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Building the ROS Melodic + Gazebo Environment
1. Build docker image using Dockerfile:
```bash
$host:  docker build -t ros_melodic .
```
2. Run the container and pipe display to docker host:
```bash
$host: ./run.bash
```

Note: There may be a need to recursively change the ownership of `/tmp/.docker.xauth/` to the user spinning up the container.

3. Activate ros tools:
```bash
$ros_melodic: source /opt/ros/melodic/setup.bash
```

4. Launch the Fetch demo in container:
```bash
$ros_melodic: roslaunch fetch_gazebo simulation.launch
```

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
