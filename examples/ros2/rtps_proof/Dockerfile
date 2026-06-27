# Reproducible cross-process proof that use_rtps drives a real ROS 2 node.
#
# Builds on ros:jazzy, compiles CycloneDDS (C lib + idlc) from source so the
# cyclonedds Python binding installs on any arch (no prebuilt wheel needed on
# arm64), installs strands-robots[ros2], and ships turtlesim. The container
# then runs a real turtlesim node AND our bare-cyclonedds use_rtps publisher in
# the same DDS domain - proving our ROS<->DDS mangling + IDL bundle + CDR wire
# layout make a real ROS 2 robot move, with no rclpy on the publish side.
FROM ros:jazzy

ARG CYCLONEDDS_VERSION=0.10.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq \
        ros-jazzy-turtlesim \
        ros-jazzy-rmw-cyclonedds-cpp \
        python3-pip git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build CycloneDDS (C) from source with the IDL compiler so the Python binding
# can build against it on any architecture.
RUN git clone -q --depth 1 --branch ${CYCLONEDDS_VERSION} \
        https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
        -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_IDLC=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/cyclonedds/build --target install -j"$(nproc)" \
    && rm -rf /tmp/cyclonedds

ENV CYCLONEDDS_HOME=/usr/local
ENV CMAKE_PREFIX_PATH=/usr/local
ENV LD_LIBRARY_PATH=/usr/local/lib
# Both turtlesim and our publisher must share one DDS vendor to discover.
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV QT_QPA_PLATFORM=offscreen

# The proof only needs the use_rtps tool + RtpsRobot, whose import chain is
# strands-agents + cyclonedds (the package __init__ is lazy). Install just those
# two - a full editable install of strands-robots would pull torch/lerobot and
# also collides with the base image's debian-managed numpy. Put the checkout on
# PYTHONPATH so `import strands_robots` resolves to the local source.
COPY . /opt/strands-robots
RUN pip install --break-system-packages -q strands-agents 'cyclonedds==0.10.2'

ENV PYTHONPATH=/opt/strands-robots
WORKDIR /opt/strands-robots
