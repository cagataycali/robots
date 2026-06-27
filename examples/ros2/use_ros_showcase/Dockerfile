# Reproducible live showcase for the use_ros tool.
#
# use_ros runs in-process through rclpy, so it needs a sourced ROS 2 distro
# (rclpy + rosidl_runtime_py, which ship with the ros:jazzy image - NOT on PyPI).
# Only strands-agents is pip-installed; the package is put on PYTHONPATH to use
# the local checkout without a full editable install (which would collide with
# the base image's debian-managed numpy and pull heavy torch/lerobot deps).
FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq \
        ros-jazzy-turtlesim python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages -q strands-agents

ENV QT_QPA_PLATFORM=offscreen
ENV PYTHONPATH=/opt/strands-robots
COPY . /opt/strands-robots
WORKDIR /opt/strands-robots
