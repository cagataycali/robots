# Chapter 8: Real Hardware

**Time:** 30 minutes · **Hardware:** Required · **Level:** Hardware

Time to go real. Everything you've learned in simulation applies to physical robots — same code, same APIs, real consequences.

---

## The Magic of Auto-Detection

Remember this code?

```python
from strands_robots import Robot

robot = Robot("so100")
```

In simulation, this creates a MuJoCo environment. But if an SO-100 is connected via USB, it detects the hardware and talks to the real servos instead. **Same code, different world.**

---

## Connect Your Robot

### Step 1: Find the Serial Port

```python
from strands import Agent
from strands_robots import serial_tool

agent = Agent(tools=[serial_tool])
agent("List all serial ports")
```

This discovers connected devices. You'll see something like `/dev/ttyACM0` (Linux) or `/dev/cu.usbmodem*` (Mac).

### Step 2: Test the Connection

```python
agent("Ping motor 1 on /dev/ttyACM0")
```

Each servo responds with its ID and current position.

### Step 3: Calibrate

Before first use, calibrate the servos:

```python
agent("Calibrate the SO-100 on /dev/ttyACM0")
```

This records the range of motion for each joint.

---

## Connect Cameras

```python
from strands import Agent
from strands_robots import lerobot_camera

agent = Agent(tools=[lerobot_camera])

# Find cameras
agent("Discover all cameras")

# Test one
agent("Capture a frame from /dev/video0")

# Preview live
agent("Preview camera /dev/video0 for 5 seconds")
```

Supports USB cameras (OpenCV) and Intel RealSense depth cameras.

---

## Full Hardware Setup

```python
from strands import Agent
from strands_robots import Robot, gr00t_inference, lerobot_camera, pose_tool

robot = Robot(
    "so100",
    cameras={
        "front": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 30},
    },
)

agent = Agent(tools=[robot, gr00t_inference, lerobot_camera, pose_tool])

# Save a safe home position first
agent("Save the current position as 'home'")

# Now do tasks
agent("Pick up the red block")

# Always return home when done
agent("Go to home position")
```

!!! warning "Safety First"
    Always save a home position before running autonomous tasks. If anything goes wrong, you can send the robot home. Keep your hand near the emergency stop.

---

## Jetson Deployment

For NVIDIA GR00T inference on Jetson:

```bash
# On the Jetson
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
jetson-containers run $(autotag isaac-gr00t) &
```

Then from your development machine:

```python
policy = create_policy("zmq://jetson-ip:5555")
```

The Jetson runs the heavy neural network inference. Your code runs anywhere.

---

## What You Learned

- ✅ Same `Robot()` code works with real hardware
- ✅ Serial tool for discovering and testing servos
- ✅ Camera setup with OpenCV and RealSense
- ✅ Calibration before first use
- ✅ Safety: home positions and emergency stops
- ✅ Jetson for edge inference

---

**Next:** [Chapter 9: Advanced →](09-advanced.md) — DreamGen, custom policies, mesh networking.
