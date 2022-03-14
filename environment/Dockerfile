FROM osrf/ros:melodic-desktop-full

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Fetch robot simulation
RUN apt-get update
RUN apt-get install -y ros-$ROS_DISTRO-fetch-gazebo-demo

# Install vim for debugging purposes
RUN apt-get install vim