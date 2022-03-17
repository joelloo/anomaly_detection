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
RUN apt-get install -y vim

# Copy new fuel tool config because the old URL is wrong
COPY ./fuel/config.yaml /usr/share/ignition/fuel_tools/

# Copy a different fetch initialization. This is hacky. Next time make our own launchable package
COPY ./fetch/fetch.launch.xml /opt/ros/melodic/share/fetch_gazebo/launch/include/fetch.launch.xml

# Install joint state publisher gui dependency
RUN apt-get install -y ros-melodic-joint-state-publisher-gui

# Add setup.bash activation
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc