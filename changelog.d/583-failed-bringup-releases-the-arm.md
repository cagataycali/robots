### Fixed: a real-robot task whose policy cannot load no longer leaves the arm locked

`execute`/`start` connect the arm before the policy is built, and connecting
turns torque on. When the policy then cannot be built or initialized, the arm
this task connected is disconnected again (torque released) and the error says
so. A connection made before the task is left alone, and a rollout that fails
while running still holds its pose.
